from datetime import datetime 
import os
from pathlib import Path
import tempfile
import requests
import pandas as pd
import numpy as np
import whisper
import weaviate
from weaviate.util import generate_uuid5
from time import sleep

from astro import sql as aql 
from astro.files import File 
from astro.sql.table import Table 
from airflow.decorators import dag, task, task_group
from cosmos.providers.dbt.task_group import DbtTaskGroup
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from great_expectations_provider.operators.great_expectations import GreatExpectationsOperator
from airflow.models.baseoperator import chain

_POSTGRES_CONN_ID = 'postgres_default'
_MINIO_CONN_ID = 'minio_default'
restore_data_uri = 'https://astronomer-demos-public.s3.us-west-2.amazonaws.com/sissy-g-toys-demo/data'
bucket_names = {'raw': 'raw-data', 'calls': 'customer-calls', 'weaviate': 'weaviate-backup'}
local_data_dir = 'include/data'
calls_file='include/data/customer_calls.csv'
weaviate_endpoint_url = os.environ['WEAVIATE_ENDPOINT_URL']
openai_apikey = os.environ['OPENAI_APIKEY']

hubspot_sources = ['ad_spend']
segment_sources = ['sessions']
sfdc_sources = ['customers', 'payments', 'subscription_periods', 'customer_conversions', 'orders']
twitter_sources = ['twitter_comments', 'comment_training']
weaviate_class_objects = {'CommentTraining': {'count': 1987}, 'CustomerComment': {'count': 12638}, 'CustomerCall': {'count': 43}}

_DBT_BIN = '/home/astro/.venv/dbt/bin/dbt'

@dag(schedule=None, start_date=datetime(2023, 1, 1), catchup=False)
def customer_analytics():
    
    @task_group()
    def enter():

        @task()
        def drop_schema(schema='tmp_astro'):
            PostgresHook(_POSTGRES_CONN_ID).run(f'DROP SCHEMA IF EXISTS {schema} CASCADE;')

        @task()
        def create_minio_buckets(bucket_names:str) -> dict:
            s3_hook = S3Hook(_MINIO_CONN_ID)

            for bucket_name in list(bucket_names.values()):
                try:
                    s3_hook.create_bucket(bucket_name)
                except Exception as e:
                    if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
                        print(e.response['Error']['Message'])
            
            return bucket_names

        @task()
        def restore_weaviate(class_objects:dict, restore_data_uri:str, weaviate_backup_bucket:str, weaviate_endpoint_url:str, s3_conn_id: str, replace_existing=False):
            import weaviate 
            from weaviate.exceptions import UnexpectedStatusCodeException
            import tempfile
            import urllib
            import zipfile
            import os
            import warnings
            from airflow.providers.amazon.aws.hooks.s3 import S3Hook

            weaviate_restore_uri = f'{restore_data_uri}/weaviate-backup/backup.zip'
            client = weaviate.Client(url = weaviate_endpoint_url)
            s3_hook = S3Hook(s3_conn_id)

            if replace_existing:
                client.schema.delete_all()
            
            else:
                existing_classes = [classes['class'] for classes in client.schema.get()['classes']]
                class_collision = set.intersection(set(existing_classes), set(class_objects.keys()))
                if class_collision:
                    warnings.warn(f'Class objects {class_collision} already exist and replace_existing={replace_existing}. Skipping restore.')
                    response = 'skipped'

                    return weaviate_endpoint_url
            
            with tempfile.TemporaryDirectory() as td:
                zip_path, _ = urllib.request.urlretrieve(weaviate_restore_uri)
                with zipfile.ZipFile(zip_path, "r") as f:
                    f.extractall(td)

                for root, dirs, files in os.walk(td, topdown=False):
                    for name in files:
                        filename = os.path.join(root, name)

                        s3_hook.load_file(
                            filename=filename, 
                            key=os.path.relpath(filename, td), 
                            bucket_name=weaviate_backup_bucket,
                            replace=True,
                        )

            try:
                response = client.backup.restore(
                        backup_id='backup',
                        backend="s3",
                        include_classes=list(class_objects.keys()),
                        wait_for_completion=True,
                    )
            except Exception as e:
                if 'already exists' in e.message:
                    warnings.warn(f'One or more class objects {class_objects} already exist and replace_existing={replace_existing}. Skipping restore.')
                    response = 'skipped'
            
            return weaviate_endpoint_url

        @task()
        def check_weaviate(class_objects:dict, weaviate_endpoint_url:str):
            import weaviate 

            client = weaviate.Client(url = weaviate_endpoint_url)

            for class_object in class_objects.keys():
                expected_count = class_objects[class_object]['count']
                response = client.query.aggregate(class_name=class_object).with_meta_count().do()               
                count = response["data"]["Aggregate"][class_object][0]["meta"]["count"]
                assert count == expected_count, f"Class {class_object} check failed. Expected {expected_count} objects.  Found {count}"
            
            return weaviate_endpoint_url

        drop_schema()
        
        _bucket_names = create_minio_buckets(bucket_names)

        _weaviate_backup_bucket = _bucket_names['weaviate']

        _weaviate_endpoint_url = restore_weaviate(
            class_objects=weaviate_class_objects, 
            restore_data_uri=restore_data_uri,
            weaviate_backup_bucket=_weaviate_backup_bucket,
            weaviate_endpoint_url=weaviate_endpoint_url,
            s3_conn_id=_MINIO_CONN_ID,
            replace_existing=True)
                
        _weaviate_endpoint_url = check_weaviate(
            class_objects=weaviate_class_objects, 
            weaviate_endpoint_url=_weaviate_endpoint_url)

        return _bucket_names, _weaviate_endpoint_url

    @task_group()
    def structured_data(bucket_names:dict):

        @task_group()
        def extract_structured_data(raw_bucket_name:str):
            @task()
            def extract_SFDC_data(raw_bucket_name:str):
                s3hook = S3Hook(_MINIO_CONN_ID)
                for source in sfdc_sources:    
                    s3hook.load_file(
                        filename=f"{local_data_dir}/{source}.csv", 
                        key=f'{source}.csv', 
                        bucket_name=raw_bucket_name,
                        replace=True,
                    )
            
            @task()
            def extract_hubspot_data(raw_bucket_name:str):
                s3hook = S3Hook(_MINIO_CONN_ID)
                for source in hubspot_sources:    
                    s3hook.load_file(
                        filename=f"{local_data_dir}/{source}.csv", 
                        key=f'{source}.csv', 
                        bucket_name=raw_bucket_name,
                        replace=True,
                    )

            @task()
            def extract_segment_data(raw_bucket_name:str):
                s3hook = S3Hook(_MINIO_CONN_ID)
                for source in segment_sources:    
                    s3hook.load_file(
                        filename=f"{local_data_dir}/{source}.csv", 
                        key=f'{source}.csv', 
                        bucket_name=raw_bucket_name,
                        replace=True,
                    )
            
            [extract_SFDC_data(raw_bucket_name=raw_bucket_name), 
             extract_hubspot_data(raw_bucket_name=raw_bucket_name), 
             extract_segment_data(raw_bucket_name=raw_bucket_name)]

        @task_group()
        def load_structured_data(raw_bucket_name:str):
            for source in sfdc_sources:
                aql.load_file(task_id=f'load_{source}',
                    input_file = File(f"S3://{raw_bucket_name}/{source}.csv", conn_id=_MINIO_CONN_ID), 
                    output_table = Table(name=f'STG_{source.upper()}', conn_id=_POSTGRES_CONN_ID),
                    if_exists='append'
                )
            
            for source in hubspot_sources:
                aql.load_file(task_id=f'load_{source}',
                    input_file = File(f"S3://{raw_bucket_name}/{source}.csv", conn_id=_MINIO_CONN_ID), 
                    output_table = Table(name=f'STG_{source.upper()}', conn_id=_POSTGRES_CONN_ID),
                    if_exists='append'
                )
            
            for source in segment_sources:
                aql.load_file(task_id=f'load_{source}',
                    input_file = File(f"S3://{raw_bucket_name}/{source}.csv", conn_id=_MINIO_CONN_ID), 
                    output_table = Table(name=f'STG_{source.upper()}', conn_id=_POSTGRES_CONN_ID),
                    if_exists='append'
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
                        schema='tmp_astro',
                    )
            
        @task_group()
        def transform_structured():
            jaffle_shop = DbtTaskGroup(
                dbt_project_name="jaffle_shop",
                dbt_root_path="/usr/local/airflow/include/dbt",
                conn_id=_POSTGRES_CONN_ID,
                dbt_args={
                    "dbt_executable_path": _DBT_BIN,
                    "schema": "tmp_astro",
                },
                test_behavior="after_all",
            )
            
            attribution_playbook = DbtTaskGroup(
                dbt_project_name="attribution_playbook",
                dbt_root_path="/usr/local/airflow/include/dbt",
                conn_id=_POSTGRES_CONN_ID,
                dbt_args={
                    "dbt_executable_path": _DBT_BIN,
                    "schema": "tmp_astro",
                },
            )

            mrr_playbook = DbtTaskGroup(
                dbt_project_name="mrr_playbook",
                dbt_root_path="/usr/local/airflow/include/dbt",
                conn_id=_POSTGRES_CONN_ID,
                dbt_args={
                    "dbt_executable_path": _DBT_BIN,
                    "schema": "tmp_astro",
                },
            )

        extract_structured_data(raw_bucket_name=bucket_names['raw']) >> \
            load_structured_data(raw_bucket_name=bucket_names['raw']) >> \
                data_quality_checks() >> \
                    transform_structured()

    @task_group()
    def unstructured_data(bucket_names:dict, weaviate_endpoint_url:str):

        @task_group()
        def extract_unstructured_data(raw_bucket_name:str, calls_bucket_name:str):
            
            s3hook = S3Hook(_MINIO_CONN_ID)

            @task()
            def extract_twitter(raw_bucket_name:str):
                for source in twitter_sources:
                    s3hook.load_file(
                        filename=f"{local_data_dir}/{source}.parquet", 
                        key=f'{source}.parquet', 
                        bucket_name=raw_bucket_name,
                        replace=True,
                    )
        
            @aql.dataframe(conn_id=_POSTGRES_CONN_ID, database='postgres', schema='tmp_astro')
            def extract_customer_support_calls(calls_bucket_name:str, output_table=Table(name='STG_CUSTOMER_CALLS')):

                calls_df = pd.read_csv(calls_file)
                calls_df['CUSTOMER_ID'] = calls_df['RELATIVE_PATH'].apply(lambda x: x.split('-')[0])
                calls_df['FULL_PATH'] = calls_df['RELATIVE_PATH'].apply(lambda x: f's3://{calls_bucket_name}/{x}')

                for file in calls_df['RELATIVE_PATH']:
                    with tempfile.NamedTemporaryFile() as tf:
                        tf.write(requests.get(f'{restore_data_uri}/audio/{file}').content)
                        s3hook.load_file(
                            filename=tf.name, 
                            key=file, 
                            bucket_name=calls_bucket_name,
                            replace=True,
                        )
                
                return calls_df

            extract_customer_support_calls(calls_bucket_name=calls_bucket_name)
            extract_twitter(raw_bucket_name=raw_bucket_name)
        
        @task_group()
        def load_unstructured_data(raw_bucket_name:str, calls_bucket_name:str):
        
            aql.load_file(task_id='load_twitter_comments',
                input_file = File(f"S3://{raw_bucket_name}/twitter_comments.parquet", conn_id=_MINIO_CONN_ID),
                output_table = Table(name='STG_twitter_comments'.upper(), conn_id=_POSTGRES_CONN_ID),
                use_native_support=False,
            )

            aql.load_file(task_id='load_comment_training',
                input_file = File(f"S3://{calls_bucket_name}/comment_training.parquet", conn_id=_MINIO_CONN_ID), 
                output_table = Table(name='STG_comment_training'.upper(), conn_id=_POSTGRES_CONN_ID),
                use_native_support=False,
            )
        
            return bucket_names
        
        @task()
        def transcribe_calls(calls_df:pd.DataFrame):

            model = whisper.load_model("tiny.en")

            with tempfile.TemporaryDirectory() as tmpdirname:
                calls_df.apply(lambda x: Path(tmpdirname)\
                                        .joinpath(x.RELATIVE_PATH)\
                                        .write_bytes(requests.get(x.PRESIGNED_URL).content), axis=1)
                
                calls_df['TRANSCRIPT'] = calls_df.apply(lambda x: model.transcribe(Path(tmpdirname)
                                                                            .joinpath(x.RELATIVE_PATH).as_posix())['text'], axis=1)

            # snowflake_hook.run('CREATE OR REPLACE TABLE STG_CUSTOMER_CALLS (CUSTOMER_ID number, \
            #                                                                 RELATIVE_PATH varchar(60), \
            #                                                                 TRANSCRIPT varchar(4000)) ')

            # write_pandas(snowflake_hook.get_conn(), calls_df[['CUSTOMER_ID', 'RELATIVE_PATH', 'TRANSCRIPT']], 'STG_CUSTOMER_CALLS')

            # return 'success'

        @task_group()
        def generate_embeddings(openai_apikey:str, weaviate_endpoint_url:str):
            
            @aql.dataframe()
            def generate_training_embeddings(comment_training_df:pd.DataFrame, openai_apikey:str, weaviate_endpoint_url:str):
            
                comment_training_df['LABEL'] = comment_training_df['LABEL'].apply(str)

                #openai works best without empty lines or new lines
                comment_training_df = comment_training_df.replace(r'^\s*$', np.nan, regex=True).dropna()
                comment_training_df['REVIEW_TEXT'] = comment_training_df['REVIEW_TEXT'].apply(lambda x: x.replace("\n",""))

                client = weaviate.Client(
                    url = weaviate_endpoint_url, 
                        additional_headers = {
                            "X-OpenAI-Api-Key": openai_apikey
                    }
                )

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
                    client.schema.create_class(class_obj)
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
                for row_id, row in comment_training_df.T.items():
                    data_object = {'rEVIEW_TEXT': row[0], 'lABEL': row[1]}
                    uuid = generate_uuid5(data_object, class_obj['class'])
                    sleep_backoff=.5
                    success = False
                    while not success:
                        try:
                            if client.data_object.exists(uuid=uuid, class_name=class_obj['class']):
                                print(f'UUID {uuid} exists.  Skipping.')
                            else:
                                uuid = client.data_object.create(
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
                
                comment_training_df['UUID']=uuids
                
                # check load
                # df = hook.get_pandas_df('SELECT * FROM COMMENT_TRAINING;')
                # df['VECTOR'] = df.apply(lambda x: client.data_object.get(class_name=class_obj['class'], uuid=x.UUID, with_vector=True)['vector'], axis=1)
                # df.to_parquet('include/data/comment_training_jic.parquet')

                return comment_training_df

            @aql.dataframe()
            def generate_twitter_embeddings(comment_df:pd.DataFrame, openai_apikey:str, weaviate_endpoint_url:str):

                comment_df['CUSTOMER_ID'] = comment_df['CUSTOMER_ID'].apply(str)
                comment_df['DATE'] = pd.to_datetime(comment_df['DATE']).dt.strftime("%Y-%m-%dT%H:%M:%S-00:00")

                #openai embeddings works best without empty lines or new lines
                comment_df = comment_df.replace(r'^\s*$', np.nan, regex=True).dropna()
                comment_df['REVIEW_TEXT'] = comment_df['REVIEW_TEXT'].apply(lambda x: x.replace("\n",""))

                client = weaviate.Client(
                    url = weaviate_endpoint_url, 
                        additional_headers = {
                            "X-OpenAI-Api-Key": openai_apikey
                    }
                )
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
                    client.schema.create_class(class_obj)
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
                for row_id, row in comment_df.T.items():
                    data_object = {'cUSTOMER_ID': row[0], 'dATE': row[1], 'rEVIEW_TEXT': row[2]}
                    uuid = generate_uuid5(data_object, class_obj['class'])
                    sleep_backoff=.5
                    success = False
                    while not success:
                        try:
                            if client.data_object.exists(uuid=uuid, class_name=class_obj['class']):
                                print(f'UUID {uuid} exists.  Skipping.')
                            else:
                                uuid = client.data_object.create(
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

                comment_df['UUID']=uuids

                # check embeddings
                # df = hook.get_pandas_df('SELECT * FROM TWITTER_COMMENTS;')
                # df['VECTOR'] = df.apply(lambda x: client.data_object.get(class_name=class_obj['class'], uuid=x.UUID, with_vector=True)['vector'], axis=1)
                # df.to_parquet('include/data/twitter_jic.parquet')

                return comment_df
            
            @aql.dataframe()
            def generate_call_embeddings(call_df:pd.DataFrame, openai_apikey:str, weaviate_endpoint_url:str):

                call_df['CUSTOMER_ID'] = call_df['CUSTOMER_ID'].apply(str)

                #openai embeddings works best without empty lines or new lines
                call_df = call_df.replace(r'^\s*$', np.nan, regex=True).dropna()
                call_df['TRANSCRIPT'] = call_df['TRANSCRIPT'].apply(lambda x: x.replace("\n",""))

                client = weaviate.Client(
                    url = weaviate_endpoint_url, 
                        additional_headers = {
                            "X-OpenAI-Api-Key": openai_apikey
                    }
                )
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
                    client.schema.create_class(class_obj)
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
                for row_id, row in call_df.T.items():
                    data_object = {'cUSTOMER_ID': row[0], 'rELATIVE_PATH': row[1], 'tRANSCRIPT': row[2]}
                    uuid = generate_uuid5(data_object, class_obj['class'])
                    sleep_backoff=.5
                    success = False
                    while not success:
                        try:
                            if client.data_object.exists(uuid=uuid, class_name=class_obj['class']):
                                print(f'UUID {uuid} exists.  Skipping.')
                            else:
                                uuid = client.data_object.create(
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
                
                call_df['UUID']=uuids
                
                return call_df
            
            generate_training_embeddings(comment_training_df=Table(name='STG_COMMENT_TRAINING', conn_id=_POSTGRES_CONN_ID), 
                                         openai_apikey=openai_apikey, 
                                         weaviate_endpoint_url=weaviate_endpoint_url)
            generate_twitter_embeddings(comment_df=Table(name='STG_COMMENT_TRAINING', conn_id=_POSTGRES_CONN_ID), 
                                        openai_apikey=openai_apikey, 
                                        weaviate_endpoint_url=weaviate_endpoint_url)
            generate_call_embeddings(call_df=Table(name='STG_COMMENT_TRAINING', conn_id=_POSTGRES_CONN_ID), 
                                     openai_apikey=openai_apikey, 
                                     weaviate_endpoint_url=weaviate_endpoint_url)
        
        extract_unstructured_data(raw_bucket_name=bucket_names['raw'], calls_bucket_name=bucket_names['calls']) 
        # >> \
            # load_unstructured_data(raw_bucket_name=bucket_names['raw'], calls_bucket_name=bucket_names['calls'])
        
        # transcribe_calls(calls_df=_calls_df)
        # generate_embeddings(openai_apikey=openai_apikey, weaviate_endpoint_url=weaviate_endpoint_url)
        
    @task() 
    def train_sentiment_classifier(
        weaviate_endpoint_url:str, 
        mlflow_tracking_uri:str,
        mlflow_s3_endpoint_url:str, 
        aws_access_key_id:str, 
        aws_secret_access_key:str
    ) -> str:

        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split 
        import mlflow
        from mlflow.keras import log_model
        from keras.models import Sequential
        from keras import layers
        import weaviate
        import os
        
        os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key_id
        os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_access_key
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = mlflow_s3_endpoint_url
        os.environ['MLFLOW_TRACKING_URI'] = mlflow_tracking_uri

        df = hook.get_pandas_df('SELECT * FROM COMMENT_TRAINING;') 
        df['LABEL'] = df['LABEL'].apply(int)
        df = df.replace(r'^\s*$', np.nan, regex=True).dropna()
        df['REVIEW_TEXT'] = df['REVIEW_TEXT'].apply(lambda x: x.replace("\n",""))
        
        #read from pre-existing embeddings in weaviate
        client = weaviate.Client(url = weaviate_endpoint_url)
        df['VECTOR'] = df.apply(lambda x: client.data_object.get(class_name='CommentTraining', uuid=x.UUID, with_vector=True)['vector'], axis=1)

        with mlflow.start_run(run_name='keras_pretrained_embeddings') as run:

            X_train, X_test, y_train, y_test = train_test_split(df['VECTOR'], df['LABEL'], test_size=.3, random_state=1883)
            X_train = np.array(X_train.values.tolist())
            y_train = np.array(y_train.values.tolist())
            X_test = np.array(X_test.values.tolist())
            y_test = np.array(y_test.values.tolist())
            
            model = Sequential()
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=70,validation_data=(X_test, y_test))

            mflow_model_info = log_model(model=model, artifact_path='sentiment_classifier')
            model_uri = mflow_model_info.model_uri

        return model_uri

    @task_group()
    def score_sentiment(
        weaviate_endpoint_url:str, 
        registry_name:str,
        model_id:str,
    ):

        @task() 
        def call_sentiment(
            weaviate_endpoint_url:str,  
            registry_name:str,
            model_id:str,
        ):
            import weaviate
            import numpy as np

            pred_table_name = 'PRED_CUSTOMER_CALLS'

            registry = model_registry.ModelRegistry(name=registry_name)

            df = ('CUSTOMER_CALLS').to_pandas()

            client = weaviate.Client(url = weaviate_endpoint_url)
            df['VECTOR'] = df.apply(lambda x: client.data_object.get(class_name='CustomerCall', uuid=x.UUID, with_vector=True)['vector'], axis=1)

            metrics = registry.get_metrics(id=model_id)
            model = registry.load_model(id=model_id)
            
            df['SENTIMENT'] = model.predict(np.stack(df['VECTOR'].values))

            on.write_pandas(df, table_name=pred_table_name, auto_create_table=True, overwrite=True)

            return df

        @task() 
        def twitter_sentiment(
            weaviate_endpoint_url:str, 
            registry_name:str,
            model_id:str,
        ):
            import weaviate
            import numpy as np

            pred_table_name = 'PRED_TWITTER_COMMENTS'

            registry = model_registry.ModelRegistry(name=registry_name)
            
            df = ession.table('TWITTER_COMMENTS').to_pandas()

            client = weaviate.Client(url = weaviate_endpoint_url)
            df['VECTOR'] = df.apply(lambda x: client.data_object.get(class_name='CustomerComment', uuid=x.UUID, with_vector=True)['vector'], axis=1)

            metrics = registry.get_metrics(id=model_id)
            model = registry.load_model(id=model_id)
            
            df['SENTIMENT'] = model.predict(np.stack(df['VECTOR'].values))

            ssion.write_pandas(df, table_name=pred_table_name, auto_create_table=True, overwrite=True)

            return df

        # [call_sentiment(
        #     weaviate_endpoint_url=weaviate_endpoint_url, 
        #     registry_name=registry_name,
        #     model_id=model_id,
        #     ), 
        #  twitter_sentiment(
        #     weaviate_endpoint_url=weaviate_endpoint_url, 
        #     registry_name=registry_name,
        #     model_id=model_id,
        #     )
        # ]

    @task_group()
    def exit(weaviate_endpoint_url:str, weaviate_backup_bucket:str):
        @task()
        def backup_weaviate(class_objects:list, weaviate_endpoint_url:str, weaviate_backup_bucket:str, s3_conn_id: str, replace_existing=False) -> str:
            import weaviate 
            from weaviate.exceptions import UnexpectedStatusCodeException
            from airflow.providers.amazon.aws.hooks.s3 import S3Hook

            client = weaviate.Client(url = weaviate_endpoint_url)

            if replace_existing:
                s3_hook = S3Hook(s3_conn_id)

                _ = s3_hook.delete_objects(
                        keys=s3_hook.list_keys(bucket_name=weaviate_backup_bucket, prefix='backup/'),
                        bucket=weaviate_backup_bucket,
                    )

            response = client.backup.create(
                    backup_id='backup',
                    backend="s3",
                    include_classes=list(class_objects.keys()),
                    wait_for_completion=True,
                )
            
            return response

        backup_weaviate(
            class_objects=weaviate_class_objects,
            weaviate_endpoint_url=weaviate_endpoint_url, 
            weaviate_backup_bucket=weaviate_backup_bucket,
            s3_conn_id=_MINIO_CONN_ID,
            replace_existing=True
        )
    _bucket_names, _weaviate_endpoint_url = enter() #, _registry_name

    _structured_data = structured_data(bucket_names=_bucket_names)
    _unstructured_data = unstructured_data(bucket_names=_bucket_names, weaviate_endpoint_url=_weaviate_endpoint_url)
    # model_id = train_sentiment_classifier(
    #     weaviate_endpoint_url=_weaviate_endpoint_url,
    #     registry_name=_registry_name
    #     )
    # predictions = score_sentiment(
    #         weaviate_endpoint_url=_weaviate_endpoint_url, 
    #         registry_name=_registry_name,
    #         model_id=model_id
    #         )
    # _exit = exit(
    #     weaviate_endpoint_url=_weaviate_endpoint_url, 
    #     weaviate_backup_bucket=_bucket_names['weaviate'])
    
    # chain([_structured_data, _unstructured_data] >> model_id >> predictions >> _exit)

customer_analytics()
