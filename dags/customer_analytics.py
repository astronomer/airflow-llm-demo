from datetime import datetime 
import os
from pathlib import Path
import tempfile
import requests

from astro import sql as aql 
from astro.files import File 
from astro.sql.table import Table 
from airflow.decorators import dag, task, task_group
from airflow.utils.task_group import TaskGroup
from cosmos.providers.dbt.task_group import DbtTaskGroup
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from great_expectations_provider.operators.great_expectations import GreatExpectationsOperator
from airflow.utils.helpers import chain

_POSTGRES_CONN = 'POSTGRES_DEFAULT'

@dag(schedule_interval=None, start_date=datetime(2023, 1, 1), catchup=False, )
def customer_analytics():
    
    restore_data_uri = 'https://astronomer-demos-public.s3.us-west-2.amazonaws.com/sissy-g-toys-demo/data/'
    raw_data_bucket = 'raw-data'
    xcom_bucket = 'xcom_bucket'
    customer_call_bucket = 'customer-calls'
    mlflow_bucket = 'mlflow-data'
    weaviate_backup_bucket = 'weaviate-backup'
    local_data_dir = 'include/data'
    hubspot_sources = ['ad_spend']
    segment_sources = ['sessions']
    sfdc_sources = ['customers', 'payments', 'subscription_periods', 'customer_conversions', 'orders']
    twitter_sources = ['twitter_comments', 'comment_training']
    support_calls = ['audio1596680176.wav']
    weaviate_endpoint_url = os.environ['WEAVIATE_ENDPOINT_URL'] #'http://weaviate.<DATABASE>.<SCHEMA>:8081/'
    mlflow_tracking_uri = os.environ['MLFLOW_TRACKING_URI'] #'http://mlflow.<DATABASE>.<SCHEMA>:5000'
    mlflow_s3_endpoint_url = os.environ['MLFLOW_S3_ENDPOINT_URL'] #'http://minio.<DATABASE>.<SCHEMA>:9000'
    openai_apikey = os.environ['OPENAI_APIKEY']

    @task_group()
    def enter():
        
        @task()
        def create_minio_buckets():
            s3_hook = S3Hook()

            buckets = [raw_data_bucket, customer_call_bucket, mlflow_bucket, weaviate_backup_bucket, xcom_bucket]

            for bucket in buckets:
                try:
                    s3_hook.create_bucket(buckets[0])
                except Exception as e:
                    if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
                        print(e.response['Error']['Message'])

        @task.virtualenv(requirements='weaviate-client')
        def restore_weaviate(weaviate_endpoint_url:str, class_objects:list, weaviate_restore_uri:str, weaviate_backup_bucket:str, replace_existing=False):
            import weaviate 
            from weaviate.exceptions import UnexpectedStatusCodeException
            import tempfile
            import urllib
            import zipfile
            import os
            import warnings
            from airflow.providers.amazon.aws.hooks.s3 import S3Hook

            client = weaviate.Client(url = weaviate_endpoint_url)
            s3_hook = S3Hook()

            if replace_existing:
                client.schema.delete_all()
            
            else:
                existing_classes = [classes['class'] for classes in client.schema.get()['classes']]
                class_collision = set.intersection(set(existing_classes), set(class_objects))
                if class_collision:
                    warnings.warn(f'Class objects {class_collision} already exist and replace_existing={replace_existing}. Skipping restore.')
                    response = 'skipped'

                    return response 
            
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
                        include_classes=class_objects,
                        wait_for_completion=True,
                    )
                
                return response
            
            except Exception as e:
                if 'already exists' in e.message:
                    warnings.warn(f'One or more class objects {class_objects} already exist and replace_existing={replace_existing}. Skipping restore.')
                    response = 'skipped'
                    return response
    
    @task_group()
    def structured_data():

        @task_group()
        def extract_structured_data():
            @task()
            def extract_SFDC_data():
                s3hook = S3Hook()
                for source in sfdc_sources:    
                    s3hook.load_file(
                        filename=f'{local_data_dir}/{source}.csv', 
                        key=f'{source}.csv', 
                        bucket_name=raw_data_bucket,
                        replace=True,
                    )
            
            @task()
            def extract_hubspot_data():
                s3hook = S3Hook()
                for source in hubspot_sources:    
                    s3hook.load_file(
                        filename=f'{local_data_dir}/{source}.csv', 
                        key=f'{source}.csv', 
                        bucket_name=raw_data_bucket,
                        replace=True,
                    )

            @task()
            def extract_segment_data():
                s3hook = S3Hook()
                for source in segment_sources:    
                    s3hook.load_file(
                        filename=f'{local_data_dir}/{source}.csv', 
                        key=f'{source}.csv', 
                        bucket_name=raw_data_bucket,
                        replace=True,
                    )
            
            [extract_SFDC_data(), extract_hubspot_data(), extract_segment_data()]

        @task_group()
        def load_structured_data():
            for source in sfdc_sources:
                aql.load_file(task_id=f'load_{source}',
                    input_file = File(f"S3://{raw_data_bucket}/{source}.csv"), 
                    output_table = Table(name=f'STG_{source.upper()}', conn_id=_POSTGRES_CONN)
                )
            
            for source in hubspot_sources:
                aql.load_file(task_id=f'load_{source}',
                    input_file = File(f"S3://{raw_data_bucket}/{source}.csv"), 
                    output_table = Table(name=f'STG_{source.upper()}', conn_id=_POSTGRES_CONN)
                )
            
            for source in segment_sources:
                aql.load_file(task_id=f'load_{source}',
                    input_file = File(f"S3://{raw_data_bucket}/{source}.csv"), 
                    output_table = Table(name=f'STG_{source.upper()}', conn_id=_POSTGRES_CONN)
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
                        conn_id=_SNOWFLAKE_CONN,
                        expectation_suite_name=f"{project}.{expectation}",
                        data_asset_name=f"STG_{expectation.upper()}",
                        fail_task_on_validation_failure=False,
                        return_json_dict=True,
                    )
            
        @task_group()
        def transform_structured():
            jaffle_shop = DbtTaskGroup(
                dbt_project_name="jaffle_shop",
                dbt_root_path="/usr/local/airflow/include/dbt",
                conn_id=_SNOWFLAKE_CONN,
                dbt_args={"dbt_executable_path": "/home/astro/.venv/dbt/bin/dbt"},
                test_behavior="after_all",
            )
            
            attribution_playbook = DbtTaskGroup(
                dbt_project_name="attribution_playbook",
                dbt_root_path="/usr/local/airflow/include/dbt",
                conn_id=_SNOWFLAKE_CONN,
                dbt_args={"dbt_executable_path": "/home/astro/.venv/dbt/bin/dbt"},
            )

            mrr_playbook = DbtTaskGroup(
                dbt_project_name="mrr_playbook",
                dbt_root_path="/usr/local/airflow/include/dbt",
                conn_id=_SNOWFLAKE_CONN,
                dbt_args={"dbt_executable_path": "/home/astro/.venv/dbt/bin/dbt"},
            )

        extract_structured_data() >> load_structured_data() >> data_quality_checks() >> transform_structured()

    @task_group()
    def unstructured_data():
        @task_group()
        def extract_unstructured_data():
            s3hook = S3Hook()

            @task()
            def extract_twitter():
                for source in twitter_sources:
                    s3hook.load_file(
                        filename=f'{local_data_dir}/{source}.parquet', 
                        key=f'{source}.parquet', 
                        bucket_name=raw_data_bucket,
                        replace=True,
                    )
            
            @task()
            def extract_customer_support_calls():        
                for file in support_calls:
                    with tempfile.NamedTemporaryFile() as tf:
                        tf.write(requests.get(restore_data_uri+'audio/'+file).content)
                        s3hook.load_file(
                            filename=tf.name, 
                            key=file, 
                            bucket_name=customer_call_bucket,
                            replace=True,
                        )

            [extract_twitter(), extract_customer_support_calls()]
        
        @task_group()
        def load_unstructured_data():
            
            @task()
            def load_support_calls():
                snowflake_hook = SnowflakeHook()
                s3hook = S3Hook()

                snowflake_hook.run(f"CREATE OR REPLACE STAGE {customer_call_directory_stage} DIRECTORY = (ENABLE = TRUE) ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');")

                for file in support_calls + ['audio1596680176.wav']:
                    tmp_file = s3hook.download_file(file, bucket_name=customer_call_bucket, preserve_file_name=True)
                    snowflake_hook.run(f'PUT file://{tmp_file} @{customer_call_directory_stage} SOURCE_COMPRESSION = NONE OVERWRITE = TRUE AUTO_COMPRESS = FALSE;')
                    os.remove(tmp_file)

                snowflake_hook.run(f'ALTER STAGE {customer_call_directory_stage} REFRESH;')

            load_support_calls()
            
            aql.load_file(task_id='load_twitter_comments',
                # input_file = File(f"S3://raw-data/twitter_comments.parquet"),  #TODO: https://github.com/astronomer/astro-sdk/issues/1755
                input_file = File('include/data/twitter_comments.parquet'),
                output_table = Table(name='STG_twitter_comments'.upper(), conn_id=_SNOWFLAKE_CONN)
            )

            aql.load_file(task_id='load_comment_training',
                # input_file = File(f"S3://raw-data/comment_training.parquet"), #TODO: https://github.com/astronomer/astro-sdk/issues/1755
                input_file = File('include/data/comment_training.parquet'),
                output_table = Table(name='STG_comment_training'.upper(), conn_id=_SNOWFLAKE_CONN)
            )
        
        @snowservices_python(runner_endpoint=runner_endpoint, python='3.8', requirements=['numpy','torch', 'tqdm', 'more-itertools', 'transformers>=4.19.0', 'ffmpeg-python==0.2.0', 'openai-whisper==v20230308'], snowflake_conn_id=_SNOWFLAKE_CONN)
        def transcribe_calls(customer_call_directory_stage:str):
            import whisper
            from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
            import requests
            import tempfile
            from snowflake.connector.pandas_tools import write_pandas
            import pandas as pd
            from pathlib import Path


            #Fake associate with customers
            calls = pd.DataFrame({'CUSTOMER_ID': [53], 'DATE': ['2018-01-07']})

            model = whisper.load_model("tiny.en")

            snowflake_hook=SnowflakeHook()
            calls_directory_df = snowflake_hook.get_pandas_df(f'SELECT *, get_presigned_url(@{customer_call_directory_stage}, LIST_DIR_TABLE.RELATIVE_PATH) as presigned_url FROM DIRECTORY( @{customer_call_directory_stage});')
            calls = pd.concat([calls, calls_directory_df], axis=1)

            with tempfile.TemporaryDirectory() as tmpdirname:
                calls.apply(lambda x: Path(tmpdirname)\
                                        .joinpath(x.RELATIVE_PATH)\
                                        .write_bytes(requests.get(x.PRESIGNED_URL).content), axis=1)
                
                calls['TRANSCRIPT'] = calls.apply(lambda x: model.transcribe(Path(tmpdirname)
                                                                            .joinpath(x.RELATIVE_PATH).as_posix())['text'], axis=1)

            snowflake_hook.run('CREATE OR REPLACE TABLE STG_CUSTOMER_CALLS (CUSTOMER_ID number, \
                                                                            DATE date, \
                                                                            RELATIVE_PATH varchar(60), \
                                                                            TRANSCRIPT varchar(4000)) ')

            write_pandas(snowflake_hook.get_conn(), calls[['CUSTOMER_ID', 'DATE', 'RELATIVE_PATH', 'TRANSCRIPT']], 'STG_CUSTOMER_CALLS')

            return 'success'

        @task_group()
        def generate_embeddings(openai_apikey:str, weaviate_endpoint_url:str):
            
            @snowservices_python(runner_endpoint=runner_endpoint, requirements=['weaviate-client'], snowflake_conn_id=_SNOWFLAKE_CONN)
            def generate_training_embeddings(openai_apikey:str, weaviate_endpoint_url:str):
                import weaviate
                from weaviate.util import generate_uuid5
                from snowflake.connector.pandas_tools import write_pandas
                from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
                from time import sleep
                import numpy as np

                hook = SnowflakeHook()

                df = hook.get_pandas_df('SELECT * FROM STG_COMMENT_TRAINING;')
                df['LABEL'] = df['LABEL'].apply(str)

                #openai works best without empty lines or new lines
                df = df.replace(r'^\s*$', np.nan, regex=True).dropna()
                df['REVIEW_TEXT'] = df['REVIEW_TEXT'].apply(lambda x: x.replace("\n",""))

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
                uuids = []
                for row_id, row in df.T.items():
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
                
                df['UUID']=uuids
                
                longest_text = df['REVIEW_TEXT'].apply(len).max()

                hook.run(f'CREATE OR REPLACE TABLE COMMENT_TRAINING (REVIEW_TEXT varchar({longest_text}), \
                                                                    LABEL varchar(1), \
                                                                    UUID varchar(36)) ')

                write_pandas(hook.get_conn(), df, 'COMMENT_TRAINING')

                # df = hook.get_pandas_df('SELECT * FROM COMMENT_TRAINING;')
                # df['VECTOR'] = df.apply(lambda x: client.data_object.get(class_name=class_obj['class'], uuid=x.UUID, with_vector=True)['vector'], axis=1)
                # df.to_parquet('include/data/comment_training_jic.parquet')

                return 'success'

            @snowservices_python(runner_endpoint=runner_endpoint, requirements=['weaviate-client'], snowflake_conn_id=_SNOWFLAKE_CONN)
            def generate_twitter_embeddings(openai_apikey:str, weaviate_endpoint_url:str):
                import weaviate
                from weaviate.util import generate_uuid5
                from snowflake.connector.pandas_tools import write_pandas
                from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
                from time import sleep
                import numpy as np
                import pandas as pd

                hook = SnowflakeHook()

                df = hook.get_pandas_df('SELECT * FROM STG_TWITTER_COMMENTS;')
                df['CUSTOMER_ID'] = df['CUSTOMER_ID'].apply(str)
                df['DATE'] = pd.to_datetime(df['DATE']).dt.strftime("%Y-%m-%dT%H:%M:%S-00:00")

                #openai embeddings works best without empty lines or new lines
                df = df.replace(r'^\s*$', np.nan, regex=True).dropna()
                df['REVIEW_TEXT'] = df['REVIEW_TEXT'].apply(lambda x: x.replace("\n",""))

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
                uuids = []
                for row_id, row in df.T.items():
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

                df['UUID']=uuids
                
                longest_text = df['REVIEW_TEXT'].apply(len).max()

                hook.run(f'CREATE OR REPLACE TABLE TWITTER_COMMENTS (CUSTOMER_ID varchar(36), \
                                                                    DATE date, \
                                                                    REVIEW_TEXT varchar({longest_text}), \
                                                                    UUID varchar(36));')

                write_pandas(hook.get_conn(), df, 'TWITTER_COMMENTS')

                # df = hook.get_pandas_df('SELECT * FROM TWITTER_COMMENTS;')
                # df['VECTOR'] = df.apply(lambda x: client.data_object.get(class_name=class_obj['class'], uuid=x.UUID, with_vector=True)['vector'], axis=1)
                # df.to_parquet('include/data/twitter_jic.parquet')

                return 'success'
            
            @snowservices_python(runner_endpoint=runner_endpoint, requirements=['weaviate-client'], snowflake_conn_id=_SNOWFLAKE_CONN)
            def generate_call_embeddings(openai_apikey:str, weaviate_endpoint_url:str):
                import weaviate
                from weaviate.util import generate_uuid5
                from snowflake.connector.pandas_tools import write_pandas
                from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
                from time import sleep
                import numpy as np
                import pandas as pd

                snowflake_hook = SnowflakeHook()

                df = snowflake_hook.get_pandas_df('SELECT * FROM STG_CUSTOMER_CALLS;')
                df['CUSTOMER_ID'] = df['CUSTOMER_ID'].apply(str)
                df['DATE'] = pd.to_datetime(df['DATE']).dt.strftime("%Y-%m-%dT%H:%M:%S-00:00")

                #openai embeddings works best without empty lines or new lines
                df = df.replace(r'^\s*$', np.nan, regex=True).dropna()
                df['TRANSCRIPT'] = df['TRANSCRIPT'].apply(lambda x: x.replace("\n",""))

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
                        "name": "DATE",
                        "dataType": ["date"],
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
                uuids = []
                for row_id, row in df.T.items():
                    data_object = {'cUSTOMER_ID': row[0], 'dATE': row[1], 'rELATIVE_PATH': row[2], 'tRANSCRIPT': row[3]}
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
                
                df['UUID']=uuids
                
                longest_text = df['TRANSCRIPT'].apply(len).max()

                snowflake_hook.run(f'CREATE OR REPLACE TABLE CUSTOMER_CALLS (CUSTOMER_ID varchar(36), \
                                                                DATE date, \
                                                                RELATIVE_PATH varchar(20), \
                                                                TRANSCRIPT varchar({longest_text}), \
                                                                UUID varchar(36));')

                write_pandas(snowflake_hook.get_conn(), df, 'CUSTOMER_CALLS')

                # df['VECTOR'] = df.apply(lambda x: client.data_object.get(class_name=class_obj['class'], uuid=x.UUID, with_vector=True)['vector'], axis=1)
                # df.to_parquet('include/data/call_jic.parquet')

                return 'success'
            
            #tasks will run serialy due to api rate limits
            [generate_training_embeddings(openai_apikey=openai_apikey, weaviate_endpoint_url=weaviate_endpoint_url) >> \
            generate_twitter_embeddings(openai_apikey=openai_apikey, weaviate_endpoint_url=weaviate_endpoint_url) >> \
            generate_call_embeddings(openai_apikey=openai_apikey, weaviate_endpoint_url=weaviate_endpoint_url)]
        
        extract_unstructured_data() >> \
        load_unstructured_data() >> \
        transcribe_calls(customer_call_directory_stage=customer_call_directory_stage) >> \
        generate_embeddings(openai_apikey=openai_apikey, weaviate_endpoint_url=weaviate_endpoint_url)
        
    @snowservices_python(runner_endpoint=runner_endpoint, requirements=['mlflow', 'boto3', 'tensorflow', 'scikit-learn', 'keras', 'weaviate-client'], snowflake_conn_id=_SNOWFLAKE_CONN)
    def train_sentiment_classifier(
        weaviate_endpoint_url:str, 
        mlflow_tracking_uri:str,
        mlflow_s3_endpoint_url:str, 
        aws_access_key_id:str, 
        aws_secret_access_key:str
    ) -> str:

        from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split 
        import mlflow
        from mlflow.keras import log_model
        from keras.models import Sequential
        from keras import layers
        import weaviate
        import os

        hook = SnowflakeHook()
        
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
        mlflow_tracking_uri:str,
        mlflow_s3_endpoint_url:str, 
        aws_access_key_id:str, 
        aws_secret_access_key:str,
        model_uri:str
    ):

        @snowservices_python(runner_endpoint=runner_endpoint, requirements=['mlflow', 'keras', 'tensorflow', 'weaviate-client'], snowflake_conn_id=_SNOWFLAKE_CONN)
        def call_sentiment(
            weaviate_endpoint_url:str,  
            mlflow_tracking_uri:str,
            mlflow_s3_endpoint_url:str, 
            aws_access_key_id:str, 
            aws_secret_access_key:str,
            model_uri:str
        ):
            from mlflow.keras import load_model
            from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
            import weaviate
            import numpy as np
            from snowflake.connector.pandas_tools import write_pandas
            import os

            os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key_id
            os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_access_key
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = mlflow_s3_endpoint_url
            os.environ['MLFLOW_TRACKING_URI'] = mlflow_tracking_uri

            snowflake_hook = SnowflakeHook()
            df = snowflake_hook.get_pandas_df('SELECT * FROM CUSTOMER_CALLS;') 

            client = weaviate.Client(url = weaviate_endpoint_url)
            df['VECTOR'] = df.apply(lambda x: client.data_object.get(class_name='CustomerCall', uuid=x.UUID, with_vector=True)['vector'], axis=1)

            model = load_model(model_uri=model_uri)
            
            df['SENTIMENT'] = model.predict(np.stack(df['VECTOR'].values))

            longest_text = df['TRANSCRIPT'].apply(len).max()

            snowflake_hook.run(f'CREATE OR REPLACE TABLE PRED_CUSTOMER_CALLS (CUSTOMER_ID varchar(36), \
                                                                    DATE date, \
                                                                    RELATIVE_PATH varchar(20), \
                                                                    TRANSCRIPT varchar({longest_text}), \
                                                                    UUID varchar(36),\
                                                                    SENTIMENT float);')

            write_pandas(
                snowflake_hook.get_conn(), 
                df[['CUSTOMER_ID', 'DATE', 'RELATIVE_PATH', 'TRANSCRIPT', 'UUID', 'SENTIMENT']], 
                'PRED_CUSTOMER_CALLS')

            return 'PRED_CUSTOMER_CALLS'

        @snowservices_python(runner_endpoint=runner_endpoint, requirements=['mlflow', 'keras', 'tensorflow', 'weaviate-client'], snowflake_conn_id=_SNOWFLAKE_CONN)
        def twitter_sentiment(
            weaviate_endpoint_url:str, 
            mlflow_tracking_uri:str,
            mlflow_s3_endpoint_url:str, 
            aws_access_key_id:str, 
            aws_secret_access_key:str,
            model_uri:str
        ):
            from mlflow.keras import load_model
            from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
            import weaviate
            import numpy as np
            from snowflake.connector.pandas_tools import write_pandas
            import os
            
            os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key_id
            os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_access_key
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = mlflow_s3_endpoint_url
            os.environ['MLFLOW_TRACKING_URI'] = mlflow_tracking_uri

            snowflake_hook = SnowflakeHook()
            df = snowflake_hook.get_pandas_df('SELECT * FROM TWITTER_COMMENTS;') 

            client = weaviate.Client(url = weaviate_endpoint_url)
            df['VECTOR'] = df.apply(lambda x: client.data_object.get(class_name='CustomerComment', uuid=x.UUID, with_vector=True)['vector'], axis=1)

            model = load_model(model_uri=model_uri)
            
            df['SENTIMENT'] = model.predict(np.stack(df['VECTOR'].values))

            longest_text = df['REVIEW_TEXT'].apply(len).max()

            snowflake_hook.run(f'CREATE OR REPLACE TABLE PRED_TWITTER_COMMENTS (CUSTOMER_ID varchar(36), \
                                                                      DATE date, \
                                                                      REVIEW_TEXT varchar({longest_text}), \
                                                                      UUID varchar(36),\
                                                                      SENTIMENT float);')

            write_pandas(
                snowflake_hook.get_conn(), 
                df[['CUSTOMER_ID', 'DATE', 'REVIEW_TEXT', 'UUID', 'SENTIMENT']], 
                'PRED_TWITTER_COMMENTS')

            return 'PRED_TWITTER_COMMENTS'

        [call_sentiment(
            weaviate_endpoint_url=weaviate_endpoint_url, 
            mlflow_tracking_uri=mlflow_tracking_uri,            
            mlflow_s3_endpoint_url=mlflow_s3_endpoint_url, 
            aws_access_key_id=aws_access_key_id, 
            aws_secret_access_key=aws_secret_access_key,
            model_uri=model_uri
            ), 
         twitter_sentiment(
            weaviate_endpoint_url=weaviate_endpoint_url, 
            mlflow_tracking_uri=mlflow_tracking_uri,            
            mlflow_s3_endpoint_url=mlflow_s3_endpoint_url, 
            aws_access_key_id=aws_access_key_id, 
            aws_secret_access_key=aws_secret_access_key,
            model_uri=model_uri
            )
        ]

    @task_group()
    def exit():
        @task.virtualenv(requirements='weaviate-client')
        def backup_weaviate(weaviate_endpoint_url:str, class_objects:list, weaviate_backup_bucket:str, replace_existing=False) -> str:
            import weaviate 
            from weaviate.exceptions import UnexpectedStatusCodeException
            from airflow.providers.amazon.aws.hooks.s3 import S3Hook

            client = weaviate.Client(url = weaviate_endpoint_url)

            if replace_existing:
                s3_hook = S3Hook()

                _ = s3_hook.delete_objects(
                        keys=s3_hook.list_keys(bucket_name=weaviate_backup_bucket, prefix='backup/'),
                        bucket=weaviate_backup_bucket,
                    )

            response = client.backup.create(
                    backup_id='backup',
                    backend="s3",
                    include_classes=class_objects,
                    wait_for_completion=True,
                )
            
            return response
    
        @task()
        def pause_snowservices() -> str:
            ss_hook = SnowServicesHook(snowflake_conn_id = _SNOWFLAKE_CONN)

            for service in snowservice_names:
                _ = ss_hook.suspend_service(service_name=service)

        backup_weaviate(
            weaviate_endpoint_url=weaviate_endpoint_url, 
            class_objects=['CommentTraining', 'CustomerComment', 'CustomerCall'],
            weaviate_backup_bucket=weaviate_backup_bucket,
            replace_existing=True
        ) >> \
        pause_snowservices() >> \
        aql.cleanup()


    _train_sentiment_classifier = train_sentiment_classifier(
        weaviate_endpoint_url=weaviate_endpoint_url,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_s3_endpoint_url=mlflow_s3_endpoint_url,
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
    )
    _score_sentiment = score_sentiment(
            weaviate_endpoint_url=weaviate_endpoint_url, 
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_s3_endpoint_url=mlflow_s3_endpoint_url,
            aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
            model_uri=_train_sentiment_classifier
        )
    
    enter() >> [structured_data(), unstructured_data()] >> _train_sentiment_classifier >> _score_sentiment >> exit()

customer_analytics()

# from include.helpers import cleanup_snowflake
# cleanup_snowflake(database='', schema='')