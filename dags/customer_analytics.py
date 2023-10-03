from datetime import datetime, date
import os
from textwrap import dedent
from pathlib import Path
import os
import pandas as pd
import numpy as np
import zipfile
from tempfile import TemporaryDirectory
import urllib
import whisper

from astro import sql as aql
from astro.files import File 
from astro.sql.table import Table, Metadata
from airflow.decorators import dag, task, task_group
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from weaviate_provider.operators.weaviate import (
    WeaviateRestoreOperator,
    WeaviateCheckSchemaBranchOperator,
    WeaviateImportDataOperator,
    )
from weaviate.util import generate_uuid5

_WEAVIATE_CONN_ID = 'weaviate_default'
_POSTGRES_CONN_ID = 'postgres_default'
_S3_CONN_ID = 'minio_default'
_DBT_BIN = '/home/astro/.venv/dbt/bin/dbt'

restore_data_uri = 'https://astronomer-demos-public-readonly.s3.us-west-2.amazonaws.com/sissy-g-toys-demo/data'
bucket_names = {'mlflow': 'mlflow-data', 'calls': 'customer-calls', 'weaviate': 'weaviate-backup'}
data_sources = ['ad_spend', 'sessions', 'customers', 'payments', 'subscription_periods', 'customer_conversions', 'orders']
twitter_sources = ['twitter_comments', 'comment_training']
weaviate_class_objects = {'CommentTraining': {'count': 1987}, 'CustomerComment': {'count': 12638}, 'CustomerCall': {'count': 43}}
pg_schema = 'demo'

default_args={
    "weaviate_conn_id": _WEAVIATE_CONN_ID,
}

@dag(schedule=None, start_date=datetime(2023, 1, 1), catchup=False, default_args=default_args)
def customer_analytics():
    
    # @task()
    # def drop_schema():
    #     PostgresHook(_POSTGRES_CONN_ID).run(f'DROP SCHEMA IF EXISTS {pg_schema} CASCADE;')
    #     return pg_schema

    @task()
    def create_buckets(replace_existing=False) -> dict:
        hook = S3Hook(_S3_CONN_ID)

        for bucket_name in list(bucket_names.values()):
            if replace_existing:
                if hook.check_for_bucket(bucket_name):
                    hook.delete_bucket(bucket_name)
            try:
                hook.create_bucket(bucket_name)
            except Exception as e:
                if e.__class__.__name__ == 'botocore.errorfactory.BucketAlreadyOwnedByYou':
                    pass
        
        return bucket_names
        
    @task()
    def download_weaviate_backup() -> str:
        """
        [Weaviate](http://www.weaviate.io) is a vector database which allows us to store a 
        vectorized representation of unstructured data like twitter tweets or audio calls.
        In this demo we use the [OpenAI  embeddings](https://platform.openai.com/docs/guides/embeddings/embeddings) 
        model to build the vectors.  With the vectors we can do sentiment classification 
        based on cosine similarity with a labeled dataset.  

        This demo uses a version of Weaviate running locally in a Docker container.  See the 
        `docker-compose.override.yml` file for details. The Astro CLI will start this container 
        alongside the Airflow webserver, trigger, scheduler and database.

        In order to speed up the demo process the data has already been ingested into weaviate 
        and vectorized.  The data was then backed up and stored in the cloud for easy restore.            
        
        This task will download the backup.zip and make it available in a docker mounted 
        filesystem for the weaviate restore task.  Normally this would be in an cloud storage.
        """
        import urllib
        import zipfile
        import tempfile

        hook = S3Hook(_S3_CONN_ID)

        weaviate_restore_uri = f'{restore_data_uri}/weaviate-backup/backup.zip'

        with TemporaryDirectory() as td:
            zip_path, _ = urllib.request.urlretrieve(weaviate_restore_uri)
            with zipfile.ZipFile(zip_path, "r") as f:
                f.extractall(td)

            for root, dirs, files in os.walk(td, topdown=False):
                for name in files:
                    filename = os.path.join(root, name)

                    hook.load_file(bucket_name=bucket_names['weaviate'],
                                   filename=filename,
                                   key='/'.join(filename.split('/')[3:]))

    _restore_weaviate = WeaviateRestoreOperator(task_id='restore_weaviate',
                                                backend='s3', 
                                                id='backup',
                                                include=list(weaviate_class_objects.keys()),
                                                replace_existing=True)
             
    #structured_data
    @task_group()
    def load_structured_data():
        for source in data_sources:
            aql.load_file(task_id=f'load_{source}',
                input_file = File(f"{restore_data_uri}/{source}.csv"), 
                output_table = Table(name=f'stg_{source}', metadata=Metadata(schema=pg_schema), conn_id=_POSTGRES_CONN_ID)
            )
        
    @task_group()
    def transform_structured():
        
        @aql.dataframe()
        def jaffle_shop(customers_df:pd.DataFrame, orders_df:pd.DataFrame, payments_df:pd.DataFrame):

            orders_df['order_date']=pd.to_datetime(orders_df['order_date'])

            customer_orders_df = orders_df.groupby('customer_id').agg({'order_date': ['min', 'max'], 'order_id': 'count'})
            customer_orders_df.columns = ['first_order', 'most_recent_order', 'number_of_orders']
            
            customer_payments_df = payments_df.merge(orders_df, how='left', on='order_id')\
                                                .groupby('customer_id').sum('amount')['amount'] / 100
                                                
            
            customers = customers_df.merge(customer_orders_df, how='left', on='customer_id')\
                                    .merge(customer_payments_df, how='left', on='customer_id')
            customers.rename({'amount': 'customer_lifetime_value'}, axis=1, inplace=True)
            
            # payment_types = ['credit_card', 'coupon', 'bank_transfer', 'gift_card']
            
            # orders = payments_df.drop('payment_id', axis=1)\
            #                     .pivot_table(index='payment_method', values=payment_types )\
            #                     .agg(F.sum('amount'))\
            #                     .group_by('order_id')\
            #                     .agg({f"'{x}'": "sum" for x in payment_types})\
            #                     .rename({f"SUM('{x.upper()}')": x+'_amount' for x in payment_types})\
            #                     .join(payments_df.group_by('order_id')\
            #                                         .agg(F.sum('amount').alias('total_amount')), on='order_id')\
            #                     .join(orders_df, on='order_id')

            return customers
        
        @aql.dataframe()
        def mrr_playbook(subscription_periods:pd.DataFrame):
            
            subscription_periods['start_date']=pd.to_datetime(subscription_periods['start_date'])
            subscription_periods['end_date']=pd.to_datetime(subscription_periods['end_date'])

            months = pd.date_range('2018-01-01', date.today().strftime(format = '%Y-%m-%d'), freq='MS').to_numpy()
            
            customers = subscription_periods.groupby('customer_id')\
                            .agg({'start_date': 'min', 'end_date': 'max'})\
                            .reset_index()

            customer_months = customers.join(pd.DataFrame(months, columns=['date_month']), how='cross')
            customer_months = customer_months[(customer_months['date_month'] >= customer_months['start_date']) & 
                                              (customer_months['date_month'] < customer_months['end_date'])]\
                                                [['customer_id', 'date_month']]

            customer_revenue_by_month = customer_months.merge(subscription_periods, how='left', on='customer_id')
            customer_revenue_by_month = customer_revenue_by_month[
                                            (customer_revenue_by_month['date_month'] >= customer_revenue_by_month['start_date']) & 
                                            (customer_revenue_by_month['date_month'] < customer_revenue_by_month['end_date'])]\
                                            [['date_month', 'customer_id', 'monthly_amount']]\
                                            .rename({'monthly_amount':'mrr'}, axis=1)

            customer_revenue_by_month = customer_revenue_by_month\
                                            .merge(customer_months, how='right', on=['customer_id', 'date_month'])\
                                            .fillna(0)

            customer_revenue_by_month['is_active'] = customer_revenue_by_month['mrr']>0

            customer_revenue_by_month['first_active_month'] = customer_revenue_by_month.groupby('customer_id')['date_month'].transform('min')
            customer_revenue_by_month['last_active_month'] = customer_revenue_by_month.groupby('customer_id')['date_month'].transform('max')
            customer_revenue_by_month['is_first_month'] = customer_revenue_by_month['first_active_month'] == customer_revenue_by_month['date_month']
            customer_revenue_by_month['is_last_month'] = customer_revenue_by_month['last_active_month'] == customer_revenue_by_month['date_month']

            customer_churn_month = customer_revenue_by_month[customer_revenue_by_month['is_last_month']]
            customer_churn_month['date_month'] = customer_churn_month['date_month'] + pd.DateOffset(months=1)
            customer_churn_month['is_active'] = False
            customer_churn_month['is_first_month'] = False
            customer_churn_month['is_last_month'] = False

            mrr = pd.concat([customer_revenue_by_month, customer_churn_month])
            mrr['id'] = mrr['customer_id'].apply(generate_uuid5)
            mrr['previous_month_is_active'] = mrr.sort_values(['customer_id', 'date_month'])\
                                                    .groupby('customer_id')['is_active']\
                                                    .shift(1)\
                                                    .fillna(False)\
                                                    .reset_index(drop=True)
            
            mrr['previous_month_mrr'] = mrr.sort_values(['customer_id', 'date_month'])\
                                                .groupby('customer_id')['mrr']\
                                                .shift(1)\
                                                .fillna(0)\
                                                .reset_index(drop=True)
            mrr['mrr_change'] = mrr['mrr'] - mrr['previous_month_mrr']

            mrr['change_category'] = np.where(mrr['is_first_month'], 'new',
                                     np.where(mrr['is_active'] & mrr['previous_month_is_active'], 'churn',
                                     np.where(mrr['is_active'] & np.logical_not(mrr['previous_month_is_active']), 'reactivation',
                                     np.where(mrr['mrr_change'] > 0, 'upgrade',
                                     np.where(mrr['mrr_change'] < 0, 'downgrade', np.nan)))))
            mrr['renewal_amount'] = mrr[['mrr','previous_month_mrr']].min(axis=1)

            return mrr

        @aql.dataframe()
        def attribution_playbook(customer_conversions_df:pd.DataFrame, 
                                 sessions_df=pd.DataFrame):
            hook=PostgresHook(_POSTGRES_CONN_ID)
            customers_df = hook.get_pandas_df(f'SELECT * FROM demo.stg_customers;')
            orders_df = hook.get_pandas_df(f'SELECT * FROM demo.stg_orders;')
            payments_df = hook.get_pandas_df(f'SELECT * FROM demo.stg_payments;')
            subscription_periods = hook.get_pandas_df(f'SELECT * FROM demo.stg_subscription_periods;')
            customer_conversions_df = hook.get_pandas_df(f'SELECT * FROM demo.stg_customer_conversions;')
            sessions_df = hook.get_pandas_df(f'SELECT * FROM demo.stg_sessions;')

            attribution_touches = sessions_df.merge(customer_conversions_df)
            attribution_touches['started_at'] = pd.to_datetime(attribution_touches['started_at'])
            attribution_touches['converted_at'] = pd.to_datetime(attribution_touches['converted_at'])

            attribution_touches = attribution_touches[(attribution_touches['started_at'] <= attribution_touches['converted_at'])& \
                                                      (attribution_touches['started_at'] >= attribution_touches['converted_at'] - pd.DateOffset(days=30))]
            
            attribution_touches['total_sessions'] = attribution_touches.groupby('customer_id')['customer_id'].transform('count')
            attribution_touches['session_index'] = attribution_touches.groupby('customer_id')['customer_id'].transform('cumcount')
            attribution_touches['first_touch_points'] = np.where(attribution_touches['session_index'] == 0, 1, 0)
            attribution_touches['last_touch_points'] = np.where(attribution_touches['session_index'] ==  np.add(attribution_touches['total_sessions'], -1), 1, 0)
            attribution_touches['forty_twenty_forty_points'] = np.where(attribution_touches['total_sessions'] == 1, 1,
                                                               np.where(attribution_touches['total_sessions'] == 2, .5,
                                                               np.where(attribution_touches['session_index'] == 1, .4,
                                                               np.where(attribution_touches['session_index'] == attribution_touches['total_sessions'], .4, 
                                                                        np.add(np.divide(.2, attribution_touches['total_sessions']), -2)))))
            attribution_touches['linear_points'] = np.divide(1, attribution_touches['total_sessions'])
            attribution_touches['first_touch_revenue'] = np.multiply(attribution_touches['revenue'], 
                                                                     attribution_touches['first_touch_points'])
            attribution_touches['last_touch_revenue'] = np.multiply(attribution_touches['revenue'], 
                                                                     attribution_touches['last_touch_points'])
            attribution_touches['forty_twenty_forty_revenue'] = np.multiply(attribution_touches['revenue'], 
                                                                     attribution_touches['forty_twenty_forty_points'])
            attribution_touches['linear_revenue'] = np.multiply(attribution_touches['revenue'], 
                                                                     attribution_touches['linear_points'])

            return attribution_touches


    @aql.dataframe()
    def extract_customer_support_calls(bucket_names:dict, replace=False):

        hook = S3Hook(_S3_CONN_ID)

        with TemporaryDirectory() as td:
            zip_path, _ = urllib.request.urlretrieve(restore_data_uri+'/customer_calls.zip')
            with zipfile.ZipFile(zip_path, "r") as f:
                f.extractall(td)

            for file in os.listdir(td+'/customer_calls'):
                try:
                    hook.load_file(filename=td+'/customer_calls/'+file,
                                bucket_name=bucket_names['calls'],
                                key=file,
                                replace=replace)
                except Exception as e:
                    if not replace and 'already exists' in e.args:
                        pass

        df = pd.read_csv(restore_data_uri+'/customer_calls.txt', names=['RELATIVE_PATH'])
        df['CUSTOMER_ID'] = df['RELATIVE_PATH'].apply(lambda x: x.split('-')[0])
        df['FULL_PATH'] = df['RELATIVE_PATH'].apply(lambda x: f"s3://{bucket_names['calls']}/{x}")
        
        return df
    
    _stg_comment_table = aql.load_file(task_id='load_twitter_comments',
        input_file = File(f'{restore_data_uri}/twitter_comments.parquet'),
        output_table = Table(name='stg_twitter_comments', 
                                metadata=Metadata(schema=pg_schema), 
                                conn_id=_POSTGRES_CONN_ID),
        use_native_support=False,
    )

    _stg_training_table = aql.load_file(task_id='load_comment_training',
        input_file = File(f'{restore_data_uri}/comment_training.parquet'), 
        output_table = Table(name='stg_comment_training', 
                                metadata=Metadata(schema=pg_schema), 
                                conn_id=_POSTGRES_CONN_ID),
        use_native_support=False,
    )
        
    @aql.dataframe()
    def transcribe_calls(df:pd.DataFrame):

        hook = S3Hook(_S3_CONN_ID)

        model = whisper.load_model('tiny.en', download_root=os.getcwd())

        with TemporaryDirectory() as tmpdirname:
            for file in hook.list_keys(bucket_names['calls']):
                obj = hook.get_key(key=file,
                                   bucket_name=bucket_names['calls'])
                
                _ = Path(tmpdirname).joinpath(file).write_bytes(obj.get()['Body'].read())
                                        
            df['TRANSCRIPT'] = df.apply(lambda x: model.transcribe(Path(tmpdirname).joinpath(x.RELATIVE_PATH).as_posix(), fp16=False)['text'], axis=1)

        return df

    # @task.weaviate_import()
    # def generate_training_embeddings(df:pd.DataFrame):
    #     import weaviate 
    #     from weaviate.util import generate_uuid5
    #     import numpy as np

    #     df['LABEL'] = df['LABEL'].apply(str)

    #     #openai works best without empty lines or new lines
    #     df = df.replace(r'^\s*$', np.nan, regex=True).dropna()
    #     df['REVIEW_TEXT'] = df['REVIEW_TEXT'].apply(lambda x: x.replace("\n",""))

    #     weaviate_client = weaviate.Client(url=weaviate_conn['url'], 
    #                                         additional_headers=weaviate_conn['headers'])
    #     assert weaviate_client.cluster.get_nodes_status()[0]['status'] == 'HEALTHY' and weaviate_client.is_live()

    #     class_obj = {
    #         "class": "CommentTraining",
    #         "vectorizer": "text2vec-openai",
    #         "properties": [
    #             {
    #                 "name": "rEVIEW_TEXT",
    #                 "dataType": ["text"]
    #             },
    #             {
    #                 "name": "lABEL",
    #                 "dataType": ["string"],
    #                 "moduleConfig": {"text2vec-openai": {"skip": True}}
    #             }
    #         ]
    #     }
    #     try:
    #         weaviate_client.schema.create_class(class_obj)
    #     except Exception as e:
    #         if isinstance(e, weaviate.UnexpectedStatusCodeException) and "already used as a name for an Object class" in e.message:                                
    #             print("schema exists.")
    #         else:
    #             raise e
            
    #     # #For openai subscription without rate limits go the fast route
    #     # uuids=[]
    #     # with client.batch as batch:
    #     #     batch.batch_size=100
    #     #     for properties in df.to_dict(orient='index').items():
    #     #         uuid=client.batch.add_data_object(properties[1], class_obj['class'])
    #     #         uuids.append(uuid)

    #     #For openai with rate limit go the VERY slow route
    #     #Because we restored weaviate from pre-built embeddings this shouldn't be too long.
    #     uuids = []
    #     for row_id, row in df.T.items():
    #         data_object = {'rEVIEW_TEXT': row[0], 'lABEL': row[1]}
    #         uuid = generate_uuid5(data_object, class_obj['class'])
    #         sleep_backoff=.5
    #         success = False
    #         while not success:
    #             try:
    #                 if weaviate_client.data_object.exists(uuid=uuid, class_name=class_obj['class']):
    #                     print(f'UUID {uuid} exists.  Skipping.')
    #                 else:
    #                     uuid = weaviate_client.data_object.create(
    #                         data_object=data_object, 
    #                         uuid=uuid, 
    #                         class_name=class_obj['class'])
    #                     print(f'Added row {row_id} with uuid {uuid}, sleeping for {sleep_backoff} seconds.')
    #                     sleep(sleep_backoff)
    #                 success=True
    #                 uuids.append(uuid)
    #             except Exception as e:
    #                 if isinstance(e, weaviate.UnexpectedStatusCodeException) and "Rate limit reached" in e.message:                                
    #                     sleep_backoff+=1
    #                     print(f'Rate limit reached. Sleeping {sleep_backoff} seconds.')
    #                     sleep(sleep_backoff)
    #                 else:
    #                     raise(e)
        
    #     df['UUID']=uuids

    #     return df

    # @aql.dataframe()
    # def generate_twitter_embeddings(df:pd.DataFrame):
    #     import weaviate 
    #     from weaviate.util import generate_uuid5
    #     import numpy as np

    #     df.columns=['CUSTOMER_ID','DATE','REVIEW_TEXT']
    #     df['CUSTOMER_ID'] = df['CUSTOMER_ID'].apply(str)
    #     df['DATE'] = pd.to_datetime(df['DATE']).dt.strftime("%Y-%m-%dT%H:%M:%S-00:00")

    #     #openai embeddings works best without empty lines or new lines
    #     df = df.replace(r'^\s*$', np.nan, regex=True).dropna()
    #     df['REVIEW_TEXT'] = df['REVIEW_TEXT'].apply(lambda x: x.replace("\n",""))
        
    #     weaviate_client = weaviate.Client(url=weaviate_conn['url'], 
    #                                         additional_headers=weaviate_conn['headers'])
    #     assert weaviate_client.cluster.get_nodes_status()[0]['status'] == 'HEALTHY' and weaviate_client.is_live()

    #     class_obj = {
    #         "class": "CustomerComment",
    #         "vectorizer": "text2vec-openai",
    #         "properties": [
    #             {
    #             "name": "CUSTOMER_ID",
    #             "dataType": ["string"],
    #             "moduleConfig": {"text2vec-openai": {"skip": True}}
    #             },
    #             {
    #             "name": "DATE",
    #             "dataType": ["date"],
    #             "moduleConfig": {"text2vec-openai": {"skip": True}}
    #             },
    #             {
    #             "name": "REVIEW_TEXT",
    #             "dataType": ["text"]
    #             }
    #         ]
    #     }
                    
    #     try:
    #         weaviate_client.schema.create_class(class_obj)
    #     except Exception as e:
    #         if isinstance(e, weaviate.UnexpectedStatusCodeException) and \
    #                 "already used as a name for an Object class" in e.message:                                
    #             print("schema exists.")
    #         else:
    #             raise e
                        
    #     # #For openai subscription without rate limits go the fast route
    #     # uuids=[]
    #     # with client.batch as batch:
    #     #     batch.batch_size=100
    #     #     for properties in df.to_dict(orient='index').items():
    #     #         uuid=client.batch.add_data_object(properties[1], class_obj['class'])
    #     #         uuids.append(uuid)

    #     #For openai with rate limit go the VERY slow route
    #     #Because we restored weaviate from pre-built embeddings this shouldn't be too long.
    #     uuids = []
    #     for row_id, row in df.T.items():
    #         data_object = {'cUSTOMER_ID': row[0], 'dATE': row[1], 'rEVIEW_TEXT': row[2]}
    #         uuid = generate_uuid5(data_object, class_obj['class'])
    #         sleep_backoff=.5
    #         success = False
    #         while not success:
    #             try:
    #                 if weaviate_client.data_object.exists(uuid=uuid, class_name=class_obj['class']):
    #                     print(f'UUID {uuid} exists.  Skipping.')
    #                 else:
    #                     uuid = weaviate_client.data_object.create(
    #                                 data_object=data_object, 
    #                                 uuid=uuid, 
    #                                 class_name=class_obj['class']
    #                             )
                                    
    #                     print(f'Added row {row_id} with uuid {uuid}, sleeping for {sleep_backoff} seconds.')
    #                     sleep(sleep_backoff)
    #                 success=True
    #                 uuids.append(uuid)
    #             except Exception as e:
    #                 if isinstance(e, weaviate.UnexpectedStatusCodeException) and "Rate limit reached" in e.message:                                
    #                     sleep_backoff+=1
    #                     print(f'Rate limit reached. Sleeping {sleep_backoff} seconds.')
    #                     sleep(sleep_backoff)
    #                 else:
    #                     raise(e)

    #     df['UUID']=uuids
    #     df['DATE'] = pd.to_datetime(df['DATE'])

    #     return df
    
    # @task.weaviate_import()
    # def generate_call_embeddings(df:pd.DataFrame):
    #     import weaviate 
    #     from weaviate.util import generate_uuid5
    #     import numpy as np

    #     df['CUSTOMER_ID'] = df['CUSTOMER_ID'].apply(str)

    #     #openai embeddings works best without empty lines or new lines
    #     df = df.replace(r'^\s*$', np.nan, regex=True).dropna()
    #     df['TRANSCRIPT'] = df['TRANSCRIPT'].apply(lambda x: x.replace("\n",""))

    #     weaviate_client = weaviate.Client(url=weaviate_conn['url'], 
    #                                         additional_headers=weaviate_conn['headers'])
    #     assert weaviate_client.cluster.get_nodes_status()[0]['status'] == 'HEALTHY' and weaviate_client.is_live()

    #     class_obj = {
    #         "class": "CustomerCall",
    #         "vectorizer": "text2vec-openai",
    #         "properties": [
    #             {
    #             "name": "CUSTOMER_ID",
    #             "dataType": ["string"],
    #             "moduleConfig": {"text2vec-openai": {"skip": True}}
    #             },
    #             {
    #             "name": "RELATIVE_PATH",
    #             "dataType": ["string"],
    #             "moduleConfig": {"text2vec-openai": {"skip": True}}
    #             },
    #             {
    #             "name": "TRANSCRIPT",
    #             "dataType": ["text"]
    #             }
    #         ]
    #     }
        
    #     try:
    #         weaviate_client.schema.create_class(class_obj)
    #     except Exception as e:
    #         if isinstance(e, weaviate.UnexpectedStatusCodeException) and \
    #                 "already used as a name for an Object class" in e.message:                                
    #             print("schema exists.")
            
    #     # #For openai subscription without rate limits go the fast route
    #     # uuids=[]
    #     # with client.batch as batch:
    #     #     batch.batch_size=100
    #     #     for properties in df.to_dict(orient='index').items():
    #     #         uuid=client.batch.add_data_object(properties[1], class_obj['class'])
    #     #         uuids.append(uuid)

    #     #For openai with rate limit go the VERY slow route
    #     #Because we restored weaviate from pre-built embeddings this shouldn't be too long.
    #     uuids = []
    #     for row_id, row in df.T.items():
    #         data_object = {'cUSTOMER_ID': row[0], 'rELATIVE_PATH': row[1], 'tRANSCRIPT': row[2]}
    #         uuid = generate_uuid5(data_object, class_obj['class'])
    #         sleep_backoff=.5
    #         success = False
    #         while not success:
    #             try:
    #                 if weaviate_client.data_object.exists(uuid=uuid, class_name=class_obj['class']):
    #                     print(f'UUID {uuid} exists.  Skipping.')
    #                 else:
    #                     uuid = weaviate_client.data_object.create(
    #                                 data_object=data_object, 
    #                                 uuid=uuid, 
    #                                 class_name=class_obj['class']
    #                             )   
    #                     print(f'Added row {row_id} with uuid {uuid}, sleeping for {sleep_backoff} seconds.')
    #                     sleep(sleep_backoff)
    #                 success=True
    #                 uuids.append(uuid)
    #             except Exception as e:
    #                 if isinstance(e, weaviate.UnexpectedStatusCodeException) and "Rate limit reached" in e.message:                                
    #                     sleep_backoff+=1
    #                     print(f'Rate limit reached. Sleeping {sleep_backoff} seconds.')
    #                     sleep(sleep_backoff)
    #                 else:
    #                     raise(e)
        
    #     df['UUID']=uuids
        
    #     return df
                    
    # @aql.dataframe()
    # def train_sentiment_classifier(df:pd.DataFrame, weaviate_conn:dict):
    #     from sklearn.model_selection import train_test_split 
    #     from mlflow.tensorflow import log_model
    #     import mlflow
    #     import numpy as np
    #     from tensorflow.keras.models import Sequential
    #     from tensorflow.keras import layers
    #     import weaviate 

    #     weaviate_client = weaviate.Client(url=weaviate_conn['url'], 
    #                                       additional_headers=weaviate_conn['headers'])
    #     assert weaviate_client.cluster.get_nodes_status()[0]['status'] == 'HEALTHY' and weaviate_client.is_live()

    #     df['LABEL'] = df['LABEL'].apply(int)
    #     df['VECTOR'] = df.apply(lambda x: weaviate_client.data_object.get(class_name='CommentTraining', 
    #                                                                       uuid=x.UUID, with_vector=True)['vector'], 
    #                                                                       axis=1)

    #     with mlflow.start_run(run_name='tf_sentiment') as run:

    #         X_train, X_test, y_train, y_test = train_test_split(df['VECTOR'], df['LABEL'], test_size=.3, random_state=1883)
    #         X_train = np.array(X_train.values.tolist())
    #         y_train = np.array(y_train.values.tolist())
    #         X_test = np.array(X_test.values.tolist())
    #         y_test = np.array(y_test.values.tolist())
        
    #         model = Sequential()
    #         model.add(layers.Dense(1, activation='sigmoid'))
    #         model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])
    #         model.fit(X_train, y_train, epochs=70, validation_data=(X_test, y_test))
    
    #         mflow_model_info = log_model(model=model, artifact_path='sentiment_classifier')
    #         model_uri = mflow_model_info.model_uri
        
    #     return model_uri
    
    # #score_sentiment
    # @aql.dataframe()
    # def call_sentiment(df:pd.DataFrame, model_uri:str, weaviate_conn:dict):
    #     import weaviate
    #     from mlflow.tensorflow import load_model
    #     import numpy as np

    #     model = load_model(model_uri=model_uri)

    #     weaviate_client = weaviate.Client(url=weaviate_conn['url'], 
    #                                       additional_headers=weaviate_conn['headers'])
        
    #     df['VECTOR'] = df.apply(lambda x: weaviate_client.data_object.get(class_name='CustomerCall', 
    #                                                                         uuid=x.UUID, with_vector=True)['vector'], 
    #                                                                         axis=1)
        
    #     df['SENTIMENT'] = model.predict(np.stack(df['VECTOR'].values))

    #     df.columns = [col.lower() for col in df.columns]

    #     return df
    
    # @aql.dataframe()
    # def twitter_sentiment(df:pd.DataFrame, model_uri:str, weaviate_conn:dict):
    #     import weaviate
    #     from mlflow.tensorflow import load_model
    #     import numpy as np

    #     model = load_model(model_uri=model_uri)

    #     weaviate_client = weaviate.Client(url=weaviate_conn['url'], 
    #                                       additional_headers=weaviate_conn['headers'])
        
    #     df['VECTOR'] = df.apply(lambda x: weaviate_client.data_object.get(class_name='CustomerComment', 
    #                                                                         uuid=x.UUID, with_vector=True)['vector'], 
    #                                                                         axis=1)
        
    #     df['SENTIMENT'] = model.predict(np.stack(df['VECTOR'].values))

    #     df.columns = [col.lower() for col in df.columns]

    #     return df

    # @aql.dataframe()
    # def create_sentiment_table(pred_calls_df:pd.DataFrame, pred_comment_df:pd.DataFrame):

    #     sentiment_df = pred_calls_df.groupby('customer_id').agg(calls_sentiment=pd.NamedAgg(column='sentiment', aggfunc='mean'))\
    #                         .join(pred_comment_df.groupby('customer_id').agg(comments_sentiment=pd.NamedAgg(column='sentiment', aggfunc='mean')), 
    #                             how='right')\
    #                         .fillna(0)\
    #                         .eval('sentiment_score = (calls_sentiment + comments_sentiment)/2')
        
    #     sentiment_df['sentiment_bucket']=pd.qcut(sentiment_df['sentiment_score'], q=10, precision=4, labels=False)
    #     sentiment_df.reset_index(inplace=True)

    #     return sentiment_df
    
    # @aql.dataframe()
    # def create_ad_spend_table(ad_spend_df:pd.DataFrame):

    #     ad_spend_df = ad_spend_df[['utm_medium', 'revenue']].dropna().groupby('utm_medium').sum('revenue')
    #     ad_spend_df.reset_index(inplace=True)
    #     ad_spend_df.columns=['Medium','Revenue']
    
    #     return ad_spend_df

    # @aql.dataframe()
    # def create_clv_table(customers_df:pd.DataFrame, sentiment_df:pd.DataFrame):
        
    #     customers_df = customers_df.dropna(subset=['customer_lifetime_value'])
    #     customers_df['customer_id'] = customers_df['customer_id'].apply(str)
    #     customers_df['name'] = customers_df[['first_name', 'last_name']].agg(' '.join, axis=1)
    #     customers_df['clv'] = customers_df['customer_lifetime_value'].round(2)

    #     clv_df = customers_df.set_index('customer_id').join(sentiment_df.set_index('customer_id')).reset_index()
    
    #     return clv_df[['customer_id', 'name', 'first_order', 'most_recent_order', 'number_of_orders', 'clv', 'sentiment_score']]
    
    # @aql.dataframe()
    # def create_churn_table(customers_df:pd.DataFrame, sentiment_df:pd.DataFrame, rev_df:pd.DataFrame):
        
    #     customers_df['customer_id'] = customers_df['customer_id'].apply(str)
    #     customers_df['name'] = customers_df[['first_name', 'last_name']].agg(' '.join, axis=1)
    #     customers_df['clv'] = customers_df['customer_lifetime_value'].round(2)
    #     customers_df = customers_df.dropna(subset=['customer_lifetime_value'])[['customer_id', 'name', 'clv']].set_index('customer_id')
    #     rev_df['customer_id'] = rev_df['customer_id'].apply(str)
    #     rev_df = rev_df[['customer_id', 'last_active_month', 'change_category']].set_index('customer_id')
    #     sentiment_df = sentiment_df[['customer_id', 'sentiment_score']].set_index('customer_id')

    #     churn_df = customers_df.join(rev_df, how='right')\
    #                             .join(sentiment_df, how='left')\
    #                             .dropna(subset=['clv'])
                                
    #     churn_df = churn_df[churn_df['change_category'] == 'churn']
        
    #     return churn_df.reset_index()

    # _pg_schema = drop_schema() 

    _bucket_names = create_buckets(replace_existing=True) 

    _download_weaviate_backup = download_weaviate_backup()
    
    _stg_calls_table = extract_customer_support_calls(bucket_names=_bucket_names,
                                                      replace=False,
                                                      output_table=Table(name='stg_customer_calls', 
                                                                            metadata=Metadata(schema=pg_schema), 
                                                                            conn_id=_POSTGRES_CONN_ID))

    _stg_calls_table = transcribe_calls(df=_stg_calls_table,
                                        output_table=Table(name='stg_customer_calls', 
                                                            metadata=Metadata(schema=pg_schema), 
                                                            conn_id=_POSTGRES_CONN_ID))
    
    # _training_table = generate_training_embeddings(df=_stg_training_table, 
    #                                                output_table=Table(name='training_table', 
    #                                                                   metadata=Metadata(schema=pg_schema), 
    #                                                                   conn_id=_POSTGRES_CONN_ID))
    # _comment_table = generate_twitter_embeddings(df=_stg_comment_table, 
    #                                              output_table=Table(name='comment_table', 
    #                                                                 metadata=Metadata(schema=pg_schema), 
    #                                                                 conn_id=_POSTGRES_CONN_ID))
    # _calls_table = generate_call_embeddings(df=_stg_calls_table,
    #                                         output_table=Table(name='calls_table', 
    #                                                            metadata=Metadata(schema=pg_schema), 
    #                                                            conn_id=_POSTGRES_CONN_ID))

    # _model_uri = train_sentiment_classifier(df=_training_table, weaviate_conn=_weaviate_conn)

    # _pred_calls_table = call_sentiment(df=_calls_table,
    #                                    model_uri=_model_uri,
    #                                    weaviate_conn=_weaviate_conn,
    #                                    output_table=Table(name='pred_customer_calls', 
    #                                                            metadata=Metadata(schema=pg_schema), 
    #                                                            conn_id=_POSTGRES_CONN_ID))
    
    # _pred_comment_table = twitter_sentiment(df=_comment_table,
    #                                         model_uri=_model_uri,
    #                                         weaviate_conn=_weaviate_conn,
    #                                         output_table=Table(name='pred_twitter_comments', 
    #                                                            metadata=Metadata(schema=pg_schema), 
    #                                                            conn_id=_POSTGRES_CONN_ID))

    
    # _sentiment_table = create_sentiment_table(pred_calls_df=_pred_calls_table,
    #                                           pred_comment_df=_pred_comment_table,
    #                                           output_table=Table(name='pres_sentiment', metadata=Metadata(schema=pg_schema), conn_id=_POSTGRES_CONN_ID))
    
    # _ad_spend_table = create_ad_spend_table(ad_spend_df=Table(name='attribution_touches', metadata=Metadata(schema=pg_schema), conn_id=_POSTGRES_CONN_ID),
    #                                         output_table=Table(name='pres_ad_spend', metadata=Metadata(schema=pg_schema), conn_id=_POSTGRES_CONN_ID))
    
    # create_clv_table(customers_df=Table(name='customers', metadata=Metadata(schema=pg_schema), conn_id=_POSTGRES_CONN_ID),
    #                  sentiment_df=_sentiment_table,
    #                  output_table=Table(name='pres_clv', metadata=Metadata(schema=pg_schema), conn_id=_POSTGRES_CONN_ID))
    
    # create_churn_table(customers_df=Table(name='customers', metadata=Metadata(schema=pg_schema), conn_id=_POSTGRES_CONN_ID),
    #                  sentiment_df=_sentiment_table,
    #                  rev_df=Table(name='mrr', metadata=Metadata(schema=pg_schema), conn_id=_POSTGRES_CONN_ID),
    #                  output_table=Table(name='pres_churn', metadata=Metadata(schema=pg_schema), conn_id=_POSTGRES_CONN_ID))
    
    # _pg_schema >> \
    load_structured_data()  #>> \
            # transform_structured() >> \
                # _ad_spend_table
    
    _download_weaviate_backup >> _restore_weaviate >> [_stg_calls_table, _stg_comment_table, _stg_training_table]
    #  [_training_table, _comment_table, _calls_table]

customer_analytics()

def test():
    hook=PostgresHook(_POSTGRES_CONN_ID)
    
    pred_comment_df=hook.get_pandas_df(f'SELECT * FROM pred_twitter_comments;')
    pred_calls_df=hook.get_pandas_df(f'SELECT * FROM pred_customer_calls;')

    customers_df = hook.get_pandas_df(f'SELECT * FROM demo.stg_customers;')
    orders_df = hook.get_pandas_df(f'SELECT * FROM demo.stg_orders;')
    payments_df = hook.get_pandas_df(f'SELECT * FROM demo.stg_payments;')
    
    ad_spend_df = hook.get_pandas_df(f'SELECT * FROM attribution_touches;')
    rev_df = hook.get_pandas_df(f'SELECT * FROM mrr;')