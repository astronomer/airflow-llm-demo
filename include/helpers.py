from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook

def cleanup_snowflake(database:str = None, schema:str = None, snowflake_conn_id = 'snowflake_default', drop_list = ['PRED_', '_TMP_', 'STG_', 'TWITTER_', 'XCOM_', 'CUSTOMER_', 'COMMENT_', 'CALL_']):
    hook = SnowflakeHook(snowflake_conn_id)
    conn_params = hook._get_conn_params()
    hook.database = database or conn_params['database']
    hook.schema = schema or conn_params['schema']

    print(f"WARNING: DANGER ZONE!  This will drop all stages and views in schema {hook.database}.{hook.schema} and all tables with names containing {drop_list}")
    prompt = input("Are you sure you want to do this?: N/y: ")

    if prompt.upper() == 'Y':
        prompt = input("Are you REALLY sure?: N/y: ")

        if prompt.upper() == 'Y':

            tables = hook.get_records(f'USE DATABASE {hook.database}; USE SCHEMA {hook.schema}; SHOW TABLES')
            tables = [table_name[1] for table_name in tables]
            drop_tables =  [drop_table for drop_table in tables if any(drop_table for j in drop_list if str(j) in drop_table)]

            if len(drop_tables) > 0: 
                prompt = input(f"Are you REALLY sure you want to drop these tables {drop_tables} in schema {hook.database}.{hook.schema}?: N/y: ")
                if prompt.upper() == 'Y':
                    for table in drop_tables:
                        hook.run(f'USE DATABASE {hook.database}; USE SCHEMA {hook.schema}; DROP TABLE IF EXISTS {table}')
                        print(f'dropped table {hook.database}.{hook.schema}.{table}')
                else: 
                    print("Not dropping tables.")
            else:
                print('No tables to drop.')
            
            stages = hook.get_records(f'USE DATABASE {hook.database}; USE SCHEMA {hook.schema}; SHOW STAGES')
            stages = [stage_name[1] for stage_name in stages]

            if len(stages) > 0: 
                prompt = input(f"Are you REALLY sure you want to drop these stages {stages} in schema {hook.database}.{hook.schema}?: N/y: ")
                if prompt.upper() == 'Y':
                    for stage in stages:
                        hook.run(f'USE DATABASE {hook.database}; USE SCHEMA {hook.schema}; DROP STAGE IF EXISTS {stage}')
                        print(f'dropped stage: {hook.database}.{hook.schema}.{stage}')
                else: 
                    print("Not dropping stages.")
            else:
                print('No stages to drop.')
            
            views = hook.get_records(f'USE DATABASE {hook.database}; USE SCHEMA {hook.schema}; SHOW VIEWS')
            views = [view_name[1] for view_name in views]
            
            if len(views) > 0: 
                prompt = input(f"Are you REALLY sure you want to drop these views {views} in schema {hook.database}.{hook.schema}?: N/y: ")
                if prompt.upper() == 'Y':
                    for view in views:
                        hook.run(f'USE DATABASE {hook.database}; USE SCHEMA {hook.schema}; DROP VIEW IF EXISTS {view}')
                        print(f'dropped view: {hook.database}.{hook.schema}.{view}')
                else: 
                    print("Not dropping views.")
            else:
                print('No views to drop.')
        
        else: 
            print("Not dropping objects.")
    else: 
            print("Not dropping objects.")

    return 

