from datetime import datetime, date
import os
import hashlib
from pathlib import Path
import os
import pandas as pd
import numpy as np
import zipfile
from tempfile import TemporaryDirectory
import urllib
import whisper
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split 
from mlflow.lightgbm import log_model, load_model
import mlflow

from astro import sql as aql
from astro.files import File 
from astro.sql.table import Table, Metadata
from airflow.decorators import dag, task, task_group
from airflow.operators.smooth import SmoothOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from weaviate_provider.hooks.weaviate import WeaviateHook
from weaviate_provider.operators.weaviate import (
    WeaviateRestoreOperator,
    WeaviateCreateSchemaOperator,
    WeaviateCheckSchemaBranchOperator,
    WeaviateCheckSchemaOperator,
    WeaviateImportDataOperator,
    WeaviateRetrieveAllOperator,
    )
from weaviate.util import generate_uuid5

_WEAVIATE_CONN_ID = 'weaviate_default'
_POSTGRES_CONN_ID = 'postgres_default'
_S3_CONN_ID = 'minio_default'
_DBT_BIN = '/home/astro/.venv/dbt/bin/dbt'

restore_data_uri = 'https://astronomer-demos-public-readonly.s3.us-west-2.amazonaws.com/sissy-g-toys-demo/data'
bucket_names = {'mlflow': 'mlflow-data', 'calls': 'customer-calls', 'weaviate': 'weaviate-backup', 'xcom': 'local-xcom'}
data_sources = ['ad_spend', 'sessions', 'customers', 'payments', 'subscription_periods', 'customer_conversions', 'orders']
twitter_sources = ['twitter_comments', 'comment_training']
weaviate_class_objects = {'CommentTraining': {'count': 1987}, 'CustomerComment': {'count': 12638}, 'CustomerCall': {'count': 43}}
pg_schema = 'demo'

weaviate_client = WeaviateHook(_WEAVIATE_CONN_ID).get_conn()
s3_hook = S3Hook(_S3_CONN_ID)

default_args={
    "weaviate_conn_id": _WEAVIATE_CONN_ID,
}

@dag(schedule=None, start_date=datetime(2023, 1, 1), catchup=False, default_args=default_args)
def customer_analytics():
    
    @task()
    def create_buckets(replace_existing=False) -> dict:

        for bucket_name in list(bucket_names.values()):
            if replace_existing:
                if s3_hook.check_for_bucket(bucket_name):
                    s3_hook.delete_bucket(bucket_name=bucket_name, force_delete=True)
            try:
                s3_hook.create_bucket(bucket_name)
            except Exception as e:
                if e.__class__.__name__ == 'botocore.errorfactory.BucketAlreadyOwnedByYou':
                    pass
        
        return bucket_names
        
    _bucket_names = create_buckets(replace_existing=True) 

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
        
        weaviate_restore_uri = f"{restore_data_uri}/{bucket_names['weaviate']}/backup.zip"

        with TemporaryDirectory() as td:
            zip_path, _ = urllib.request.urlretrieve(weaviate_restore_uri)
            with zipfile.ZipFile(zip_path, "r") as f:
                f.extractall(td)

            for root, dirs, files in os.walk(td, topdown=False):
                for name in files:
                    filename = os.path.join(root, name)

                    s3_hook.load_file(bucket_name=bucket_names['weaviate'],
                                      filename=filename,
                                      key='/'.join(filename.split('/')[3:]))

    _download_weaviate_backup = download_weaviate_backup()

    _create_schema = WeaviateCreateSchemaOperator(task_id='create_schema',
                                                  class_object_data='file://include/weaviate_schema.json',
                                                  existing='replace')

    _restore_weaviate = WeaviateRestoreOperator(task_id='restore_weaviate',
                                                backend='s3', 
                                                id='backup',
                                                include=list(weaviate_class_objects.keys()),
                                                replace_existing=True)
             
    _check_schema = WeaviateCheckSchemaBranchOperator(task_id='check_schema', 
                                                      weaviate_conn_id=_WEAVIATE_CONN_ID,
                                                      class_object_data='file://include/weaviate_schema.json',
                                                      follow_task_ids_if_true=["generate_training_embeddings",
                                                                               "generate_twitter_embeddings",
                                                                               "generate_call_embeddings"
                                                                               ],
                                                      follow_task_ids_if_false=["alert_schema"])

    _alert_schema = SmoothOperator(task_id='alert_schema')

    @task_group()
    def load_structured_data():
        for source in data_sources:
            aql.load_file(task_id=f'load_{source}',
                input_file = File(f"{restore_data_uri}/{source}.csv"), 
                output_table = Table(name=f'stg_{source}', metadata=Metadata(schema=pg_schema), conn_id=_POSTGRES_CONN_ID)
            )
    
    _load_structured_data = load_structured_data()

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
            mrr['id'] = mrr['customer_id'].apply(lambda x: hashlib.md5(str(x).encode()).hexdigest())
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
            
            mrr.reset_index(inplace=True, drop=True)

            return mrr

        @aql.dataframe()
        def attribution_playbook(customer_conversions_df:pd.DataFrame, sessions_df:pd.DataFrame):
            
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
            
            attribution_touches.reset_index(inplace=True, drop=True)

            return attribution_touches

        _customers = jaffle_shop(customers_df=Table(name='stg_customers', 
                                                    metadata=Metadata(schema=pg_schema), 
                                                    conn_id=_POSTGRES_CONN_ID),
                                 orders_df=Table(name='stg_orders', 
                                                 metadata=Metadata(schema=pg_schema), 
                                                 conn_id=_POSTGRES_CONN_ID),
                                 payments_df=Table(name='stg_payments', 
                                                   metadata=Metadata(schema=pg_schema), 
                                                   conn_id=_POSTGRES_CONN_ID),
                                 output_table=Table(name='customers', 
                                                    metadata=Metadata(schema=pg_schema),
                                                    conn_id=_POSTGRES_CONN_ID))
        
        _mrr = mrr_playbook(subscription_periods=Table(name='stg_subscription_periods', 
                                                       metadata=Metadata(schema=pg_schema), 
                                                       conn_id=_POSTGRES_CONN_ID),
                            output_table=Table(name='mrr', 
                                               metadata=Metadata(schema=pg_schema), 
                                               conn_id=_POSTGRES_CONN_ID))
        
        _attribution_touches = attribution_playbook(customer_conversions_df=Table(name='stg_customer_conversions', 
                                                                                  metadata=Metadata(schema=pg_schema), 
                                                                                  conn_id=_POSTGRES_CONN_ID),
                                                    sessions_df=Table(name='stg_sessions', 
                                                                      metadata=Metadata(schema=pg_schema), 
                                                                      conn_id=_POSTGRES_CONN_ID),
                                                    output_table=Table(name='attribution_touches', 
                                                                             metadata=Metadata(schema=pg_schema), 
                                                                             conn_id=_POSTGRES_CONN_ID))
        
        return _customers, _mrr, _attribution_touches

    _customers, _mrr, _attribution_touches = transform_structured()

    @task_group()
    def load_unstructured_data():
        @aql.dataframe()
        def extract_customer_support_calls(bucket_names:dict, replace=False):

            with TemporaryDirectory() as td:
                zip_path, _ = urllib.request.urlretrieve(restore_data_uri+'/customer_calls.zip')
                with zipfile.ZipFile(zip_path, "r") as f:
                    f.extractall(td)

                for file in os.listdir(td+'/customer_calls'):
                    try:
                        s3_hook.load_file(filename=td+'/customer_calls/'+file,
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
        
        _stg_calls_table = extract_customer_support_calls(bucket_names=_bucket_names,
                                                          replace=False,
                                                          output_table=Table(name='stg_customer_calls', 
                                                                             metadata=Metadata(schema=pg_schema), 
                                                                             conn_id=_POSTGRES_CONN_ID))
        
        _stg_comment_table = aql.load_file(task_id='load_twitter_comments',
                                           input_file = File(f'{restore_data_uri}/twitter_comments.parquet'))

        _stg_training_table = aql.load_file(task_id='load_comment_training',
                                            input_file = File(f'{restore_data_uri}/comment_training.parquet'))
            
        return _stg_calls_table, _stg_comment_table, _stg_training_table
    
    _stg_calls_table, _stg_comment_table, _stg_training_table = load_unstructured_data()    
    
    @aql.dataframe()
    def transcribe_calls(df:pd.DataFrame):

        model = whisper.load_model('tiny.en', download_root=os.getcwd())
        files = s3_hook.list_keys(bucket_names['calls'])

        with TemporaryDirectory() as tmpdirname:
            for file in files :
                obj = s3_hook.get_key(key=file,
                                   bucket_name=bucket_names['calls'])
                
                Path(tmpdirname).joinpath(file).write_bytes(obj.get()['Body'].read())
                                        
            df['TRANSCRIPT'] = df.apply(lambda x: model.transcribe(Path(tmpdirname).joinpath(x.RELATIVE_PATH).as_posix(), fp16=False)['text'], axis=1)
        
        df.columns=['rELATIVE_PATH', 'cUSTOMER_ID', 'fULL_PATH',  'tRANSCRIPT']
        df['cUSTOMER_ID'] = df['cUSTOMER_ID'].apply(str)

        df = df.replace(r'^\s*$', np.nan, regex=True).dropna()
        df['tRANSCRIPT'] = df['tRANSCRIPT'].apply(lambda x: x.replace("\n",""))

        df['UUID'] = df[['cUSTOMER_ID', 'rELATIVE_PATH', 'tRANSCRIPT']]\
                            .apply(lambda x: generate_uuid5(x.to_dict(), 'CustomerCall'), axis=1)

        return df
    
    _stg_transcribed_calls = transcribe_calls(df=_stg_calls_table)

    _calls_table = WeaviateImportDataOperator(task_id='generate_call_embeddings', 
                                                           data=_stg_transcribed_calls, 
                                                           uuid_column='UUID', 
                                                           class_name='CustomerCall', 
                                                           existing='skip', 
                                                           batched_mode=True, 
                                                           batch_size=1000)

    @task.weaviate_import
    def generate_training_embeddings(train_df:pd.DataFrame):

        train_df.columns=['rEVIEW_TEXT', 'lABEL']
        train_df['lABEL'] = train_df['lABEL'].apply(str)

        #openai works best without empty lines or new lines
        train_df = train_df.replace(r'^\s*$', np.nan, regex=True).dropna()
        train_df['rEVIEW_TEXT'] = train_df['rEVIEW_TEXT'].apply(lambda x: x.replace("\n",""))
        
        train_df['UUID'] = train_df[['rEVIEW_TEXT', 'lABEL']]\
                                .apply(lambda x: generate_uuid5(x.to_dict(), 'CommentTraining'), axis=1)
        
        return {'data': train_df, 
                'uuid_column': 'UUID', 
                'class_name': 'CommentTraining', 
                'existing': 'skip', 
                'batched_mode': True, 
                'batch_size': 1000}

    @task.weaviate_import()
    def generate_twitter_embeddings(tweet_df:pd.DataFrame):

        tweet_df.columns=['cUSTOMER_ID','dATE','rEVIEW_TEXT']
        tweet_df['cUSTOMER_ID'] = tweet_df['cUSTOMER_ID'].apply(str)
        tweet_df['dATE'] = pd.to_datetime(tweet_df['dATE']).dt.strftime("%Y-%m-%dT%H:%M:%S-00:00")

        #openai embeddings works best without empty lines or new lines
        tweet_df = tweet_df.replace(r'^\s*$', np.nan, regex=True).dropna()
        tweet_df['rEVIEW_TEXT'] = tweet_df['rEVIEW_TEXT'].apply(lambda x: x.replace("\n",""))

        tweet_df['UUID'] = tweet_df[['cUSTOMER_ID','dATE','rEVIEW_TEXT']]\
                                .apply(lambda x: generate_uuid5(x.to_dict(), 'CustomerComment'), axis=1)

        return {'data': tweet_df, 
                'uuid_column': 'UUID', 
                'class_name': 'CustomerComment', 
                'existing': 'skip', 
                'batched_mode': True, 
                'batch_size': 1000}
                    
    @aql.dataframe()
    def train_sentiment_classifier():
        
        assert weaviate_client.cluster.get_nodes_status()[0]['status'] == 'HEALTHY' and weaviate_client.is_live()

        results = []
        last_uuid = None
        while True:
            result = weaviate_client.data_object.get(class_name='CommentTraining', 
                                                 after=last_uuid, 
                                                 with_vector=True, 
                                                 limit=100) or {}
            if not result.get('objects'):
                break
            results.extend(result['objects'])
            last_uuid = result['objects'][-1]['id']
        
        df = pd.DataFrame(results)

        df = pd.concat([pd.json_normalize(df['properties']), df['vector']], axis=1)

        df['lABEL'] = df['lABEL'].apply(int)

        with mlflow.start_run(run_name='lgbm_sentiment') as run:

            X_train, X_test, y_train, y_test = train_test_split(df['vector'], df['lABEL'], test_size=.3, random_state=1883)
            X_train = np.array(X_train.values.tolist())
            y_train = np.array(y_train.values.tolist())
            X_test = np.array(X_test.values.tolist())
            y_test = np.array(y_test.values.tolist())
            
            model = LGBMClassifier(random_state=42)
            model.fit(X=X_train, y=y_train, eval_set=(X_test, y_test))
    
            mflow_model_info = log_model(lgb_model=model, artifact_path='sentiment_classifier')
            model_uri = mflow_model_info.model_uri
        
        return model_uri
    
    @aql.dataframe()
    def call_sentiment(df:pd.DataFrame, model_uri:str):

        model = load_model(model_uri=model_uri)
        
        df['cUSTOMER_ID'] = df['cUSTOMER_ID'].apply(str)

        df['vector'] = df['UUID'].apply(lambda x: weaviate_client.data_object.get(uuid=x, with_vector=True, class_name='CustomerCall')['vector'])
        
        df['sentiment'] = model.predict_proba(np.stack(df['vector'].values))[:,1]

        df.columns = [col.lower() for col in df.columns]

        return df
    
    @aql.dataframe()
    def twitter_sentiment(model_uri:str):
        
        model = load_model(model_uri=model_uri)

        results = []
        last_uuid = None
        while True:
            result = weaviate_client.data_object.get(class_name='CustomerComment', 
                                                 after=last_uuid, 
                                                 with_vector=True, 
                                                 limit=100) or {}
            if not result.get('objects'):
                break
            results.extend(result['objects'])
            last_uuid = result['objects'][-1]['id']
        
        df = pd.DataFrame(results)

        df = pd.concat([pd.json_normalize(df['properties']), df['vector']], axis=1)
                
        df['sentiment'] = model.predict_proba(np.stack(df['vector'].values))[:,1]

        df.columns = [col.lower() for col in df.columns]

        return df

    @aql.dataframe()
    def create_sentiment_table(pred_calls_df:pd.DataFrame, pred_comment_df:pd.DataFrame):

        sentiment_df = pred_calls_df.groupby('customer_id').agg(calls_sentiment=pd.NamedAgg(column='sentiment', aggfunc='mean'))\
                            .join(pred_comment_df.groupby('customer_id').agg(comments_sentiment=pd.NamedAgg(column='sentiment', aggfunc='mean')), how='right')\
                            .fillna(0)\
                            .eval('sentiment_score = (calls_sentiment + comments_sentiment)/2')
        
        sentiment_df['sentiment_bucket']=pd.qcut(sentiment_df['sentiment_score'], q=10, precision=4, labels=False, duplicates='drop')
        sentiment_df.reset_index(inplace=True)

        return sentiment_df
    
    @aql.dataframe()
    def create_ad_spend_table(attribution_touches_df:pd.DataFrame):

        ad_spend_df = attribution_touches_df[['utm_medium', 'revenue']].dropna().groupby('utm_medium').sum('revenue')
        ad_spend_df.reset_index(inplace=True)
        ad_spend_df.columns=['Medium','Revenue']
    
        return ad_spend_df

    @aql.dataframe()
    def create_clv_table(customers_df:pd.DataFrame, sentiment_df:pd.DataFrame):
        
        customers_df = customers_df.dropna(subset=['customer_lifetime_value'])
        customers_df['customer_id'] = customers_df['customer_id'].apply(str)
        customers_df['name'] = customers_df[['first_name', 'last_name']].agg(' '.join, axis=1)
        customers_df['clv'] = customers_df['customer_lifetime_value'].round(2)

        clv_df = customers_df.set_index('customer_id').join(sentiment_df.set_index('customer_id')).reset_index()
    
        return clv_df[['customer_id', 'name', 'first_order', 'most_recent_order', 'number_of_orders', 'clv', 'sentiment_score']]
    
    @aql.dataframe()
    def create_churn_table(customers_df:pd.DataFrame, sentiment_df:pd.DataFrame, rev_df:pd.DataFrame):
        
        customers_df['customer_id'] = customers_df['customer_id'].apply(str)
        customers_df['name'] = customers_df[['first_name', 'last_name']].agg(' '.join, axis=1)
        customers_df['clv'] = customers_df['customer_lifetime_value'].round(2)
        customers_df = customers_df.dropna(subset=['customer_lifetime_value'])[['customer_id', 'name', 'clv']].set_index('customer_id')

        rev_df['customer_id'] = rev_df['customer_id'].apply(str)
        rev_df = rev_df[['customer_id', 'first_active_month', 'last_active_month', 'change_category']].set_index('customer_id')

        sentiment_df = sentiment_df[['customer_id', 'sentiment_score']].set_index('customer_id')

        churn_df = customers_df.join(rev_df, how='right')\
                                .join(sentiment_df, how='left')\
                                .dropna(subset=['clv'])
                                
        churn_df = churn_df[churn_df['change_category'] == 'churn']
        churn_df.reset_index(inplace=True)
        churn_df.drop_duplicates(inplace=True)
        
        return churn_df
    
    _training_table = generate_training_embeddings(train_df=_stg_training_table)

    _comment_table = generate_twitter_embeddings(tweet_df=_stg_comment_table)

    _model_uri = train_sentiment_classifier()

    _pred_calls_table = call_sentiment(df=_stg_transcribed_calls,
                                       model_uri=_model_uri,
                                       output_table=Table(name='pred_customer_calls', 
                                                               metadata=Metadata(schema=pg_schema), 
                                                               conn_id=_POSTGRES_CONN_ID))
    
    _pred_comment_table = twitter_sentiment(model_uri=_model_uri,
                                            output_table=Table(name='pred_twitter_comments', 
                                                               metadata=Metadata(schema=pg_schema), 
                                                               conn_id=_POSTGRES_CONN_ID))

    
    _sentiment_table = create_sentiment_table(pred_calls_df=_pred_calls_table,
                                              pred_comment_df=_pred_comment_table,
                                              output_table=Table(name='pres_sentiment', 
                                                                 metadata=Metadata(schema=pg_schema), 
                                                                 conn_id=_POSTGRES_CONN_ID))
    
    _ad_spend_table = create_ad_spend_table(attribution_touches_df=_attribution_touches,
                                            output_table=Table(name='pres_ad_spend', 
                                                               metadata=Metadata(schema=pg_schema), 
                                                               conn_id=_POSTGRES_CONN_ID))
    
    create_clv_table(sentiment_df=_sentiment_table,
                     customers_df=_customers,
                     output_table=Table(name='pres_clv', 
                                        metadata=Metadata(schema=pg_schema), 
                                        conn_id=_POSTGRES_CONN_ID))
    
    create_churn_table(sentiment_df=_sentiment_table,
                       customers_df=_customers,
                       rev_df=_mrr,
                       output_table=Table(name='pres_churn', 
                                          metadata=Metadata(schema=pg_schema), 
                                          conn_id=_POSTGRES_CONN_ID))
    
    _load_structured_data >> [_customers, _mrr, _attribution_touches]

    [_stg_calls_table, _stg_comment_table, _stg_training_table]
    
    _download_weaviate_backup >> _restore_weaviate >> _check_schema 
    _create_schema >> _restore_weaviate
    _check_schema >> [_alert_schema, _training_table, _comment_table, _calls_table]
    _calls_table >> _pred_calls_table
    _comment_table >> _pred_comment_table
    _training_table >> _model_uri

customer_analytics()

def test():
    from airflow.providers.postgres.hooks.postgres import PostgresHook
    hook=PostgresHook(_POSTGRES_CONN_ID)
    customers_df = hook.get_pandas_df(f'SELECT * FROM demo.stg_customers;')
    orders_df = hook.get_pandas_df(f'SELECT * FROM demo.stg_orders;')
    payments_df = hook.get_pandas_df(f'SELECT * FROM demo.stg_payments;')
    subscription_periods = hook.get_pandas_df(f'SELECT * FROM demo.stg_subscription_periods;')
    customer_conversions_df = hook.get_pandas_df(f'SELECT * FROM demo.stg_customer_conversions;')
    sessions_df = hook.get_pandas_df(f'SELECT * FROM demo.stg_sessions;')
    train_df = hook.get_pandas_df(f'SELECT * FROM demo.stg_comment_training;')

    pred_comment_df=hook.get_pandas_df(f'SELECT * FROM demo.pred_twitter_comments;')
    pred_calls_df=hook.get_pandas_df(f'SELECT * FROM demo.pred_customer_calls;')

    calls_df = hook.get_pandas_df(f'SELECT * FROM demo.stg_customer_calls;')
    
    customers_df = hook.get_pandas_df(f'SELECT * FROM demo.customers;')
    sentiment_df = hook.get_pandas_df(f'SELECT * FROM demo.pres_sentiment;')
    rev_df = hook.get_pandas_df(f'SELECT * FROM demo.mrr;')
    attribution_touches_df = hook.get_pandas_df(f'SELECT * FROM demo.attribution_touches;')
    pres_ad_spend = hook.get_pandas_df(f'SELECT * FROM demo.pres_ad_spend;')
    clv = hook.get_pandas_df(f'SELECT * FROM demo.pres_clv;')
    churn = hook.get_pandas_df(f'SELECT * FROM demo.pres_churn;')
    

    WeaviateHook(_WEAVIATE_CONN_ID).get_conn().schema.delete_all()
    WeaviateCreateSchemaOperator(task_id='test', class_object_data='file://include/weaviate_schema.json', existing='replace').execute({})
    WeaviateCheckSchemaOperator(task_id='test', class_object_data='file://include/weaviate_schema.json').execute({})

