from datetime import datetime 
import os
from pathlib import Path
from time import sleep
import os
import warnings
import json
import pandas as pd

from astro import sql as aql
from astro.files import File 
from astro.sql.table import Table, Metadata
from airflow.decorators import dag, task, task_group
from cosmos.task_group import DbtTaskGroup
from airflow.providers.postgres.hooks.postgres import PostgresHook
from great_expectations_provider.operators.great_expectations import GreatExpectationsOperator


_POSTGRES_CONN_ID = 'postgres_default'
_DBT_BIN = '/home/astro/.venv/dbt/bin/dbt'

restore_data_uri = 'https://astronomer-demos-public-readonly.s3.us-west-2.amazonaws.com/sissy-g-toys-demo/data'
bucket_names = {'mlflow': 'mlflow-data', 'calls': 'customer-calls', 'weaviate': 'weaviate-backup'}
data_sources = ['ad_spend', 'sessions', 'customers', 'payments', 'subscription_periods', 'customer_conversions', 'orders']
twitter_sources = ['twitter_comments', 'comment_training']
weaviate_class_objects = {'CommentTraining': {'count': 1987}, 'CustomerComment': {'count': 12638}, 'CustomerCall': {'count': 43}}
minio_conn = json.loads(os.environ['AIRFLOW_CONN_MINIO_DEFAULT'])
weaviate_conn = {'url': 'http://weaviate:8081', 'headers': {"X-OpenAI-Api-Key": os.environ['OPENAI_APIKEY']}}
pg_schema = 'public'

@dag(schedule=None, start_date=datetime(2023, 1, 1), catchup=False)
def customer_analytics():
    
    @task()
    def drop_schema():
        PostgresHook(_POSTGRES_CONN_ID).run(f'DROP SCHEMA IF EXISTS {pg_schema} CASCADE;')
        return pg_schema

    @task()
    def create_minio_buckets(replace_existing=False) -> dict:
        import minio

        minio_client = minio.Minio(
            endpoint=minio_conn['extra']['endpoint'],
            access_key=minio_conn['extra']['aws_access_key_id'],
            secret_key=minio_conn['extra']['aws_secret_access_key'],
            secure=False
        )

        for bucket_name in list(bucket_names.values()):
            if replace_existing:
                try:
                    for object in minio_client.list_objects(bucket_name=bucket_name, recursive=True):
                        minio_client.remove_object(bucket_name=bucket_name, object_name=object.object_name)
                    minio_client.remove_bucket(bucket_name)
                except:
                    pass
            
            sleep(10)

            try:
                minio_client.make_bucket(bucket_name)
            except Exception as e:
                if e.code == 'BucketAlreadyOwnedByYou':
                    print(e.message)
        
        return bucket_names
        
    @task()
    def restore_weaviate(bucket_names:dict, replace_existing=False):
        import urllib
        import zipfile
        import minio
        import weaviate
        import tempfile

        """
        This task exists only to speedup the demo. By restoring prefetched embeddings to weaviate the later tasks 
        will skip embeddings and only make calls to openai for data it hasn't yet embedded.
        """

        weaviate_restore_uri = f'{restore_data_uri}/weaviate-backup/backup.zip'

        weaviate_client = weaviate.Client(url = weaviate_conn['url'], additional_headers=weaviate_conn['headers'])
        assert weaviate_client.cluster.get_nodes_status()[0]['status'] == 'HEALTHY' and weaviate_client.is_live()

        minio_client = minio.Minio(
            endpoint=minio_conn['extra']['endpoint'],
            access_key=minio_conn['extra']['aws_access_key_id'],
            secret_key=minio_conn['extra']['aws_secret_access_key'],
            secure=False
        )

        if replace_existing:
            weaviate_client.schema.delete_all()
        
        else:
            existing_classes = [classes['class'] for classes in weaviate_client.schema.get()['classes']]
            class_collision = set.intersection(set(existing_classes), set(weaviate_class_objects.keys()))
            if class_collision:
                warnings.warn(f'Class objects {class_collision} already exist and replace_existing={replace_existing}. Skipping restore.')
                response = 'skipped'

                return weaviate_conn
        
        with tempfile.TemporaryDirectory() as td:
            zip_path, _ = urllib.request.urlretrieve(weaviate_restore_uri)
            with zipfile.ZipFile(zip_path, "r") as f:
                f.extractall(td)

            for root, dirs, files in os.walk(td, topdown=False):
                for name in files:
                    filename = os.path.join(root, name)

                    minio_client.fput_object(
                        object_name=os.path.relpath(filename, td),
                        file_path=filename,
                        bucket_name=bucket_names['weaviate'],
                    )

        response = weaviate_client.backup.restore(
                backup_id='backup',
                backend="s3",
                include_classes=list(weaviate_class_objects.keys()),
                wait_for_completion=True,
            )
        
        assert response['status'] == 'SUCCESS', 'Weaviate restore did not succeed.'

        #check restore counts
        for class_object in weaviate_class_objects.keys():
            expected_count = weaviate_class_objects[class_object]['count']
            response = weaviate_client.query.aggregate(class_name=class_object).with_meta_count().do()               
            count = response["data"]["Aggregate"][class_object][0]["meta"]["count"]
            assert count == expected_count, f"Class {class_object} check failed. Expected {expected_count} objects.  Found {count}"
        
        return weaviate_conn
         
    #structured_data
    @task_group()
    def load_structured_data():
        for source in data_sources:
            aql.load_file(task_id=f'load_{source}',
                input_file = File(f"{restore_data_uri}/{source}.csv"), 
                output_table = Table(name=f'stg_{source}', metadata=Metadata(schema=pg_schema), conn_id=_POSTGRES_CONN_ID)
            )

    @task_group()
    def data_quality_checks():
        expectations_dir = Path('include/great_expectations').joinpath(f'expectations')
        for project in [x[0] for x in os.walk(expectations_dir)][1:]:
            for expectation in os.listdir(project):
                project = project.split('/')[-1]
                expectation = expectation.split('.')[0]
                GreatExpectationsOperator(
                    task_id=f"ge_{project}_{expectation}",
                    data_context_root_dir='include/great_expectations',
                    conn_id=_POSTGRES_CONN_ID,
                    expectation_suite_name=f"{project}.{expectation}",
                    data_asset_name=f"stg_{expectation}",
                    fail_task_on_validation_failure=False,
                    return_json_dict=True,
                    database='postgres',
                    schema=pg_schema,
                )
        
    @task_group()
    def transform_structured():
        jaffle_shop = DbtTaskGroup(
            dbt_project_name="jaffle_shop",
            dbt_root_path="/usr/local/airflow/include/dbt",
            conn_id=_POSTGRES_CONN_ID,
            dbt_args={
                "dbt_executable_path": _DBT_BIN,
                "schema": pg_schema,
            },
            profile_args={
                "schema": pg_schema,
            },
            test_behavior='after_each',
        )
        
        attribution_playbook = DbtTaskGroup(
            dbt_project_name="attribution_playbook",
            dbt_root_path="/usr/local/airflow/include/dbt",
            conn_id=_POSTGRES_CONN_ID,
            dbt_args={
                "dbt_executable_path": _DBT_BIN,
                "schema": pg_schema,
            },
            test_behavior='after_each',
        )

        mrr_playbook = DbtTaskGroup(
            dbt_project_name="mrr_playbook",
            dbt_root_path="/usr/local/airflow/include/dbt",
            conn_id=_POSTGRES_CONN_ID,
            dbt_args={
                "dbt_executable_path": _DBT_BIN,
                "schema": pg_schema,
            },
            test_behavior='after_each',
        )

    #unstructured_data
    @aql.dataframe()
    def extract_customer_support_calls(pg_schema:str, bucket_names:dict):
        import minio
        import zipfile
        import tempfile
        import urllib

        minio_client = minio.Minio(
            endpoint=minio_conn['extra']['endpoint'],
            access_key=minio_conn['extra']['aws_access_key_id'],
            secret_key=minio_conn['extra']['aws_secret_access_key'],
            secure=False
        )

        with tempfile.TemporaryDirectory() as td:
            zip_path, _ = urllib.request.urlretrieve(restore_data_uri+'/customer_calls.zip')
            with zipfile.ZipFile(zip_path, "r") as f:
                f.extractall(td)

            for file in os.listdir(td+'/customer_calls'):
                minio_client.fput_object(
                        object_name=file,
                        file_path=td+'/customer_calls/'+file,
                        bucket_name=bucket_names['calls'],
                    )


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
        import whisper
        import minio
        import tempfile

        model = whisper.load_model('tiny.en', download_root=os.getcwd())

        minio_client = minio.Minio(
                endpoint=minio_conn['extra']['endpoint'],
                access_key=minio_conn['extra']['aws_access_key_id'],
                secret_key=minio_conn['extra']['aws_secret_access_key'],
                secure=False
            )

        with tempfile.TemporaryDirectory() as tmpdirname:
            for file in minio_client.list_objects(bucket_names['calls']):
                minio_client.fget_object(object_name=file.object_name,
                                            file_path=Path(tmpdirname).joinpath(file.object_name),
                                            bucket_name=bucket_names['calls'])
                                        
            df['TRANSCRIPT'] = df.apply(lambda x: model.transcribe(Path(tmpdirname).joinpath(x.RELATIVE_PATH).as_posix(), fp16=False)['text'], axis=1)

        return df

    @aql.dataframe()
    def generate_training_embeddings(df:pd.DataFrame):
        import weaviate 
        from weaviate.util import generate_uuid5
        import numpy as np

        df['LABEL'] = df['LABEL'].apply(str)

        #openai works best without empty lines or new lines
        df = df.replace(r'^\s*$', np.nan, regex=True).dropna()
        df['REVIEW_TEXT'] = df['REVIEW_TEXT'].apply(lambda x: x.replace("\n",""))

        weaviate_client = weaviate.Client(url=weaviate_conn['url'], 
                                            additional_headers=weaviate_conn['headers'])
        assert weaviate_client.cluster.get_nodes_status()[0]['status'] == 'HEALTHY' and weaviate_client.is_live()

        class_obj = {
            "class": "CommentTraining",
            "vectorizer": "text2vec-openai",
            "properties": [
                {
                    "name": "rEVIEW_TEXT",
                    "dataType": ["text"]
                },
                {
                    "name": "lABEL",
                    "dataType": ["string"],
                    "moduleConfig": {"text2vec-openai": {"skip": True}}
                }
            ]
        }
        try:
            weaviate_client.schema.create_class(class_obj)
        except Exception as e:
            if isinstance(e, weaviate.UnexpectedStatusCodeException) and "already used as a name for an Object class" in e.message:                                
                print("schema exists.")
            else:
                raise e
            
        # #For openai subscription without rate limits go the fast route
        # uuids=[]
        # with client.batch as batch:
        #     batch.batch_size=100
        #     for properties in df.to_dict(orient='index').items():
        #         uuid=client.batch.add_data_object(properties[1], class_obj['class'])
        #         uuids.append(uuid)

        #For openai with rate limit go the VERY slow route
        #Because we restored weaviate from pre-built embeddings this shouldn't be too long.
        uuids = []
        for row_id, row in df.T.items():
            data_object = {'rEVIEW_TEXT': row[0], 'lABEL': row[1]}
            uuid = generate_uuid5(data_object, class_obj['class'])
            sleep_backoff=.5
            success = False
            while not success:
                try:
                    if weaviate_client.data_object.exists(uuid=uuid, class_name=class_obj['class']):
                        print(f'UUID {uuid} exists.  Skipping.')
                    else:
                        uuid = weaviate_client.data_object.create(
                            data_object=data_object, 
                            uuid=uuid, 
                            class_name=class_obj['class'])
                        print(f'Added row {row_id} with uuid {uuid}, sleeping for {sleep_backoff} seconds.')
                        sleep(sleep_backoff)
                    success=True
                    uuids.append(uuid)
                except Exception as e:
                    if isinstance(e, weaviate.UnexpectedStatusCodeException) and "Rate limit reached" in e.message:                                
                        sleep_backoff+=1
                        print(f'Rate limit reached. Sleeping {sleep_backoff} seconds.')
                        sleep(sleep_backoff)
                    else:
                        raise(e)
        
        df['UUID']=uuids

        return df

    @aql.dataframe()
    def generate_twitter_embeddings(df:pd.DataFrame):
        import weaviate 
        from weaviate.util import generate_uuid5
        import numpy as np

        df.columns=['CUSTOMER_ID','DATE','REVIEW_TEXT']
        df['CUSTOMER_ID'] = df['CUSTOMER_ID'].apply(str)
        df['DATE'] = pd.to_datetime(df['DATE']).dt.strftime("%Y-%m-%dT%H:%M:%S-00:00")

        #openai embeddings works best without empty lines or new lines
        df = df.replace(r'^\s*$', np.nan, regex=True).dropna()
        df['REVIEW_TEXT'] = df['REVIEW_TEXT'].apply(lambda x: x.replace("\n",""))
        
        weaviate_client = weaviate.Client(url=weaviate_conn['url'], 
                                            additional_headers=weaviate_conn['headers'])
        assert weaviate_client.cluster.get_nodes_status()[0]['status'] == 'HEALTHY' and weaviate_client.is_live()

        class_obj = {
            "class": "CustomerComment",
            "vectorizer": "text2vec-openai",
            "properties": [
                {
                "name": "CUSTOMER_ID",
                "dataType": ["string"],
                "moduleConfig": {"text2vec-openai": {"skip": True}}
                },
                {
                "name": "DATE",
                "dataType": ["date"],
                "moduleConfig": {"text2vec-openai": {"skip": True}}
                },
                {
                "name": "REVIEW_TEXT",
                "dataType": ["text"]
                }
            ]
        }
                    
        try:
            weaviate_client.schema.create_class(class_obj)
        except Exception as e:
            if isinstance(e, weaviate.UnexpectedStatusCodeException) and \
                    "already used as a name for an Object class" in e.message:                                
                print("schema exists.")
            else:
                raise e
                        
        # #For openai subscription without rate limits go the fast route
        # uuids=[]
        # with client.batch as batch:
        #     batch.batch_size=100
        #     for properties in df.to_dict(orient='index').items():
        #         uuid=client.batch.add_data_object(properties[1], class_obj['class'])
        #         uuids.append(uuid)

        #For openai with rate limit go the VERY slow route
        #Because we restored weaviate from pre-built embeddings this shouldn't be too long.
        uuids = []
        for row_id, row in df.T.items():
            data_object = {'cUSTOMER_ID': row[0], 'dATE': row[1], 'rEVIEW_TEXT': row[2]}
            uuid = generate_uuid5(data_object, class_obj['class'])
            sleep_backoff=.5
            success = False
            while not success:
                try:
                    if weaviate_client.data_object.exists(uuid=uuid, class_name=class_obj['class']):
                        print(f'UUID {uuid} exists.  Skipping.')
                    else:
                        uuid = weaviate_client.data_object.create(
                                    data_object=data_object, 
                                    uuid=uuid, 
                                    class_name=class_obj['class']
                                )
                                    
                        print(f'Added row {row_id} with uuid {uuid}, sleeping for {sleep_backoff} seconds.')
                        sleep(sleep_backoff)
                    success=True
                    uuids.append(uuid)
                except Exception as e:
                    if isinstance(e, weaviate.UnexpectedStatusCodeException) and "Rate limit reached" in e.message:                                
                        sleep_backoff+=1
                        print(f'Rate limit reached. Sleeping {sleep_backoff} seconds.')
                        sleep(sleep_backoff)
                    else:
                        raise(e)

        df['UUID']=uuids
        df['DATE'] = pd.to_datetime(df['DATE'])

        return df
    
    @aql.dataframe()
    def generate_call_embeddings(df:pd.DataFrame):
        import weaviate 
        from weaviate.util import generate_uuid5
        import numpy as np

        df['CUSTOMER_ID'] = df['CUSTOMER_ID'].apply(str)

        #openai embeddings works best without empty lines or new lines
        df = df.replace(r'^\s*$', np.nan, regex=True).dropna()
        df['TRANSCRIPT'] = df['TRANSCRIPT'].apply(lambda x: x.replace("\n",""))

        weaviate_client = weaviate.Client(url=weaviate_conn['url'], 
                                            additional_headers=weaviate_conn['headers'])
        assert weaviate_client.cluster.get_nodes_status()[0]['status'] == 'HEALTHY' and weaviate_client.is_live()

        class_obj = {
            "class": "CustomerCall",
            "vectorizer": "text2vec-openai",
            "properties": [
                {
                "name": "CUSTOMER_ID",
                "dataType": ["string"],
                "moduleConfig": {"text2vec-openai": {"skip": True}}
                },
                {
                "name": "RELATIVE_PATH",
                "dataType": ["string"],
                "moduleConfig": {"text2vec-openai": {"skip": True}}
                },
                {
                "name": "TRANSCRIPT",
                "dataType": ["text"]
                }
            ]
        }
        
        try:
            weaviate_client.schema.create_class(class_obj)
        except Exception as e:
            if isinstance(e, weaviate.UnexpectedStatusCodeException) and \
                    "already used as a name for an Object class" in e.message:                                
                print("schema exists.")
            
        # #For openai subscription without rate limits go the fast route
        # uuids=[]
        # with client.batch as batch:
        #     batch.batch_size=100
        #     for properties in df.to_dict(orient='index').items():
        #         uuid=client.batch.add_data_object(properties[1], class_obj['class'])
        #         uuids.append(uuid)

        #For openai with rate limit go the VERY slow route
        #Because we restored weaviate from pre-built embeddings this shouldn't be too long.
        uuids = []
        for row_id, row in df.T.items():
            data_object = {'cUSTOMER_ID': row[0], 'rELATIVE_PATH': row[1], 'tRANSCRIPT': row[2]}
            uuid = generate_uuid5(data_object, class_obj['class'])
            sleep_backoff=.5
            success = False
            while not success:
                try:
                    if weaviate_client.data_object.exists(uuid=uuid, class_name=class_obj['class']):
                        print(f'UUID {uuid} exists.  Skipping.')
                    else:
                        uuid = weaviate_client.data_object.create(
                                    data_object=data_object, 
                                    uuid=uuid, 
                                    class_name=class_obj['class']
                                )   
                        print(f'Added row {row_id} with uuid {uuid}, sleeping for {sleep_backoff} seconds.')
                        sleep(sleep_backoff)
                    success=True
                    uuids.append(uuid)
                except Exception as e:
                    if isinstance(e, weaviate.UnexpectedStatusCodeException) and "Rate limit reached" in e.message:                                
                        sleep_backoff+=1
                        print(f'Rate limit reached. Sleeping {sleep_backoff} seconds.')
                        sleep(sleep_backoff)
                    else:
                        raise(e)
        
        df['UUID']=uuids
        
        return df
                    
    

            # return _training_table, _comment_table, _calls_table

        # _training_table, _comment_table, _calls_table = generate_embeddings() 
            
        # return _training_table, _comment_table, _calls_table
        
    @aql.dataframe()
    def train_sentiment_classifier(df:pd.DataFrame, weaviate_conn:dict):
        from sklearn.model_selection import train_test_split 
        from mlflow.tensorflow import log_model
        import mlflow
        import numpy as np
        from tensorflow.keras.models import Sequential
        from tensorflow.keras import layers
        import weaviate 

        weaviate_client = weaviate.Client(url=weaviate_conn['url'], 
                                          additional_headers=weaviate_conn['headers'])
        assert weaviate_client.cluster.get_nodes_status()[0]['status'] == 'HEALTHY' and weaviate_client.is_live()

        df['LABEL'] = df['LABEL'].apply(int)
        df['VECTOR'] = df.apply(lambda x: weaviate_client.data_object.get(class_name='CommentTraining', 
                                                                          uuid=x.UUID, with_vector=True)['vector'], 
                                                                          axis=1)

        with mlflow.start_run(run_name='tf_sentiment') as run:

            X_train, X_test, y_train, y_test = train_test_split(df['VECTOR'], df['LABEL'], test_size=.3, random_state=1883)
            X_train = np.array(X_train.values.tolist())
            y_train = np.array(y_train.values.tolist())
            X_test = np.array(X_test.values.tolist())
            y_test = np.array(y_test.values.tolist())
        
            model = Sequential()
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=70, validation_data=(X_test, y_test))
    
            mflow_model_info = log_model(model=model, artifact_path='sentiment_classifier')
            model_uri = mflow_model_info.model_uri
        
        return model_uri
    
    #score_sentiment
    @aql.dataframe()
    def call_sentiment(df:pd.DataFrame, model_uri:str, weaviate_conn:dict):
        import weaviate
        from mlflow.tensorflow import load_model
        import numpy as np

        model = load_model(model_uri=model_uri)

        weaviate_client = weaviate.Client(url=weaviate_conn['url'], 
                                          additional_headers=weaviate_conn['headers'])
        
        df['VECTOR'] = df.apply(lambda x: weaviate_client.data_object.get(class_name='CustomerCall', 
                                                                            uuid=x.UUID, with_vector=True)['vector'], 
                                                                            axis=1)
        
        df['SENTIMENT'] = model.predict(np.stack(df['VECTOR'].values))

        df.columns = [col.lower() for col in df.columns]

        return df
    
    @aql.dataframe()
    def twitter_sentiment(df:pd.DataFrame, model_uri:str, weaviate_conn:dict):
        import weaviate
        from mlflow.tensorflow import load_model
        import numpy as np

        model = load_model(model_uri=model_uri)

        weaviate_client = weaviate.Client(url=weaviate_conn['url'], 
                                          additional_headers=weaviate_conn['headers'])
        
        df['VECTOR'] = df.apply(lambda x: weaviate_client.data_object.get(class_name='CustomerComment', 
                                                                            uuid=x.UUID, with_vector=True)['vector'], 
                                                                            axis=1)
        
        df['SENTIMENT'] = model.predict(np.stack(df['VECTOR'].values))

        df.columns = [col.lower() for col in df.columns]

        return df

    @aql.dataframe()
    def create_sentiment_table(pred_calls_df:pd.DataFrame, pred_comment_df:pd.DataFrame):

        sentiment_df = pred_calls_df.groupby('customer_id').agg(calls_sentiment=pd.NamedAgg(column='sentiment', aggfunc='mean'))\
                            .join(pred_comment_df.groupby('customer_id').agg(comments_sentiment=pd.NamedAgg(column='sentiment', aggfunc='mean')), 
                                how='right')\
                            .fillna(0)\
                            .eval('sentiment_score = (calls_sentiment + comments_sentiment)/2')
        
        sentiment_df['sentiment_bucket']=pd.qcut(sentiment_df['sentiment_score'], q=10, precision=4, labels=False)
        sentiment_df.reset_index(inplace=True)

        return sentiment_df
    
    @aql.dataframe()
    def create_ad_spend_table(ad_spend_df:pd.DataFrame):

        ad_spend_df = ad_spend_df[['utm_medium', 'revenue']].dropna().groupby('utm_medium').sum('revenue')
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
        rev_df = rev_df[['customer_id', 'last_active_month', 'change_category']].set_index('customer_id')
        sentiment_df = sentiment_df[['customer_id', 'sentiment_score']].set_index('customer_id')

        churn_df = customers_df.join(rev_df, how='right')\
                                .join(sentiment_df, how='left')\
                                .dropna(subset=['clv'])
                                
        churn_df = churn_df[churn_df['change_category'] == 'churn']
        
        return churn_df.reset_index()

    @task()
    def backup_weaviate(weaviate_class_objects:list, services:dict, weaviate_backup_bucket:str, replace_existing=False) -> str:
        import weaviate 
        import minio

        weaviate_client = weaviate.Client(url=services['weaviate_conn']['dns_name'], 
                                            additional_headers=services['weaviate_conn']['headers'])
        
        if replace_existing:
            minio_client = minio.Minio(
                endpoint=services['minio_conn']['dns_name'],
                access_key=services['minio_conn']['aws_access_key_id'],
                secret_key=services['minio_conn']['aws_secret_access_key'],
                secure=False
            )
            for obj in minio_client.list_objects(bucket_name=weaviate_backup_bucket, prefix='backup', recursive=True):
                minio_client.remove_object(bucket_name=weaviate_backup_bucket, object_name=obj.object_name)
            

        response = weaviate_client.backup.create(
                backup_id='backup',
                backend="s3",
                include_classes=list(weaviate_class_objects.keys()),
                wait_for_completion=True,
            )
        return services

    _pg_schema = drop_schema() 

    _bucket_names = create_minio_buckets(replace_existing=True) 

    _weaviate_conn = restore_weaviate(bucket_names=_bucket_names,
                                          replace_existing=True)
    
    _stg_calls_table = extract_customer_support_calls(pg_schema=_pg_schema,
                                                      bucket_names=_bucket_names,
                                                      output_table=Table(name='stg_customer_calls', 
                                                                            metadata=Metadata(schema=pg_schema), 
                                                                            conn_id=_POSTGRES_CONN_ID))

    _stg_calls_table = transcribe_calls(df=_stg_calls_table,
                                        output_table=Table(name='stg_customer_calls', 
                                                            metadata=Metadata(schema=pg_schema), 
                                                            conn_id=_POSTGRES_CONN_ID))
    
    _training_table = generate_training_embeddings(df=_stg_training_table, 
                                                   output_table=Table(name='training_table', 
                                                                      metadata=Metadata(schema=pg_schema), 
                                                                      conn_id=_POSTGRES_CONN_ID))
    _comment_table = generate_twitter_embeddings(df=_stg_comment_table, 
                                                 output_table=Table(name='comment_table', 
                                                                    metadata=Metadata(schema=pg_schema), 
                                                                    conn_id=_POSTGRES_CONN_ID))
    _calls_table = generate_call_embeddings(df=_stg_calls_table,
                                            output_table=Table(name='calls_table', 
                                                               metadata=Metadata(schema=pg_schema), 
                                                               conn_id=_POSTGRES_CONN_ID))

    _model_uri = train_sentiment_classifier(df=_training_table, weaviate_conn=_weaviate_conn)

    _pred_calls_table = call_sentiment(df=_calls_table,
                                       model_uri=_model_uri,
                                       weaviate_conn=_weaviate_conn,
                                       output_table=Table(name='pred_customer_calls', 
                                                               metadata=Metadata(schema=pg_schema), 
                                                               conn_id=_POSTGRES_CONN_ID))
    
    _pred_comment_table = twitter_sentiment(df=_comment_table,
                                            model_uri=_model_uri,
                                            weaviate_conn=_weaviate_conn,
                                            output_table=Table(name='pred_twitter_comments', 
                                                               metadata=Metadata(schema=pg_schema), 
                                                               conn_id=_POSTGRES_CONN_ID))

    
    _sentiment_table = create_sentiment_table(pred_calls_df=_pred_calls_table,
                                              pred_comment_df=_pred_comment_table,
                                              output_table=Table(name='pres_sentiment', metadata=Metadata(schema=pg_schema), conn_id=_POSTGRES_CONN_ID))
    
    _ad_spend_table = create_ad_spend_table(ad_spend_df=Table(name='attribution_touches', metadata=Metadata(schema=pg_schema), conn_id=_POSTGRES_CONN_ID),
                                            output_table=Table(name='pres_ad_spend', metadata=Metadata(schema=pg_schema), conn_id=_POSTGRES_CONN_ID))
    
    create_clv_table(customers_df=Table(name='customers', metadata=Metadata(schema=pg_schema), conn_id=_POSTGRES_CONN_ID),
                     sentiment_df=_sentiment_table,
                     output_table=Table(name='pres_clv', metadata=Metadata(schema=pg_schema), conn_id=_POSTGRES_CONN_ID))
    
    create_churn_table(customers_df=Table(name='customers', metadata=Metadata(schema=pg_schema), conn_id=_POSTGRES_CONN_ID),
                     sentiment_df=_sentiment_table,
                     rev_df=Table(name='mrr', metadata=Metadata(schema=pg_schema), conn_id=_POSTGRES_CONN_ID),
                     output_table=Table(name='pres_churn', metadata=Metadata(schema=pg_schema), conn_id=_POSTGRES_CONN_ID))
    
    _pg_schema >> \
        load_structured_data()  >> \
            data_quality_checks() >> \
                transform_structured() >> \
                    _ad_spend_table
    
    _pg_schema >> [_stg_calls_table, _stg_comment_table, _stg_training_table]
    _weaviate_conn >> [_training_table, _comment_table, _calls_table]

customer_analytics()

def test():
    hook=PostgresHook(_POSTGRES_CONN_ID)
    pred_comment_df=hook.get_pandas_df(f'SELECT * FROM pred_twitter_comments;')
    pred_calls_df=hook.get_pandas_df(f'SELECT * FROM pred_customer_calls;')
    customers_df = hook.get_pandas_df(f'SELECT * FROM customers;')
    ad_spend_df = hook.get_pandas_df(f'SELECT * FROM attribution_touches;')
    rev_df = hook.get_pandas_df(f'SELECT * FROM mrr;')