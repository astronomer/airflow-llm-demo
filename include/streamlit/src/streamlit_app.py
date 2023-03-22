import os
import altair as alt
import pandas as pd
import streamlit as st
from snowflake.snowpark import Session, Window
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import functions as F
from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
import weaviate
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode

st.set_page_config(layout="wide")

#Connections and Clients
hook = SnowflakeHook()

if "snowflake_conn_params" not in st.session_state:
    snowflake_conn_params = hook._get_conn_params()
    st.session_state['snowflake_conn_params'] = snowflake_conn_params
else: 
    snowflake_conn_params = st.session_state['snowflake_conn_params']

if "snowpark_session" not in st.session_state:
  snowpark_session = Session.builder.configs(snowflake_conn_params).create()
  st.session_state['snowpark_session'] = snowpark_session
else:
  snowpark_session = st.session_state['snowpark_session']

if "weaviate_client" not in st.session_state:
    weaviate_client = weaviate.Client(
            url = os.environ['WEAVIATE_ENDPOINT_URL'], 
                additional_headers = {
                    "X-OpenAI-Api-Key": os.environ['OPENAI_APIKEY']
                }
            )
    st.session_state['weaviate_client'] = weaviate_client
else:
  weaviate_client = st.session_state['weaviate_client']    


#Setup data sources
attribution_df = snowpark_session.table('ATTRIBUTION_TOUCHES')
customers_df = snowpark_session.table('CUSTOMERS')\
                                .with_column('CLV', F.round(F.col('CUSTOMER_LIFETIME_VALUE'), 2))
churn_month_df = snowpark_session.table('CUSTOMER_CHURN_MONTH')
rev_month_df = snowpark_session.table('CUSTOMER_REVENUE_BY_MONTH')
rev_df = snowpark_session.table('MRR').drop(['ID'])
orders_df = snowpark_session.table('ORDERS')
calls_df = snowpark_session.table('PRED_CUSTOMER_CALLS')
comments_df = snowpark_session.table('PRED_TWITTER_COMMENTS')

sentiment_df =  calls_df.group_by(F.col('CUSTOMER_ID'))\
                        .agg(F.avg('SENTIMENT').alias('CALLS_SENTIMENT'))\
                        .join(comments_df.group_by(F.col('CUSTOMER_ID'))\
                                .agg(F.avg('SENTIMENT').alias('COMMENTS_SENTIMENT')), 
                            on='CUSTOMER_ID',
                            how='right')\
                        .fillna(0, subset=['CALLS_SENTIMENT'])\
                        .with_column('SENTIMENT_SCORE', F.round((F.col('CALLS_SENTIMENT') + F.col('COMMENTS_SENTIMENT'))/2, 4))\
                        .with_column('SENTIMENT_BUCKET', F.call_builtin('WIDTH_BUCKET', F.col('SENTIMENT_SCORE'), 0, 1, 10))

@st.cache_data
def ad_spend_df():
    return attribution_df.select(['UTM_MEDIUM', 'REVENUE'])\
                                .dropna()\
                                .group_by(F.col('UTM_MEDIUM'))\
                                .sum(F.col('REVENUE'))\
                                .rename('SUM(REVENUE)', 'Revenue')\
                                .rename('UTM_MEDIUM', 'Medium').to_pandas()
_ad_spend_df = ad_spend_df()

@st.cache_data
def clv_df(row_limit=10):
    df = customers_df.dropna(subset=['CLV'])\
                     .join(sentiment_df, 'CUSTOMER_ID', how='left')\
                     .sort(F.col('CLV'), ascending=False)\
                     .limit(row_limit)\
                     .to_pandas()
    df['NAME']=df['FIRST_NAME']+' '+df['LAST_NAME']
    df = df[['CUSTOMER_ID', 'NAME', 'FIRST_ORDER', 'MOST_RECENT_ORDER', 'NUMBER_OF_ORDERS', 'CLV', 'SENTIMENT_SCORE']]
    df.columns = [col.replace('_', ' ') for col in df.columns]
    df.set_index('CUSTOMER ID', drop=True, inplace=True)
    return df
_clv_df = clv_df()

@st.cache_data
def churn_df(row_limit=10):
    df = customers_df.select('CUSTOMER_ID', 'FIRST_NAME', 'LAST_NAME', 'CLV')\
                     .join(rev_df.select('CUSTOMER_ID', 'FIRST_ACTIVE_MONTH', 'LAST_ACTIVE_MONTH', 'CHANGE_CATEGORY'), on='CUSTOMER_ID', how='right')\
                     .join(sentiment_df, 'CUSTOMER_ID', how='left')\
                     .dropna(subset=['CLV'])\
                     .filter(F.col('CHANGE_CATEGORY') == 'churn')\
                     .sort(F.col('LAST_ACTIVE_MONTH'), ascending=False)\
                     .limit(row_limit)\
                     .to_pandas()
    df['NAME']=df['FIRST_NAME']+' '+df['LAST_NAME']
    df = df[['CUSTOMER_ID', 'NAME', 'CLV', 'LAST_ACTIVE_MONTH', 'SENTIMENT_SCORE']]
    df.columns = [col.replace('_', ' ') for col in df.columns]
    
    return df
_churn_df = churn_df()

def aggrid_interactive_table(df: pd.DataFrame, height=400):
    options = GridOptionsBuilder.from_dataframe(
        df, 
        enableRowGroup=True, 
        enableValue=True, 
        enablePivot=True,
    )

    options.configure_side_bar()

    options.configure_selection("single")
    selection = AgGrid(
        df,
        height=height,
        fit_columns_on_grid_load=True,
        enable_enterprise_modules=True,
        gridOptions=options.build(),
        theme="streamlit",
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=True,
    )

    return selection

with st.container():              
    title_col, logo_col = st.columns([8,2])  
    with title_col:
        st.title("GroundTruth: Customer Analytics")
        st.subheader("Powered by SnowServices and Streamlit")
    with logo_col:
        st.image('logo_small.png')

marketing_tab, sales_tab, product_tab = st.tabs(["Marketing", "Sales", "Product"])

with marketing_tab:
    with st.container():
        st.header("Return on Ad Spend")
        st.subheader("Digital Channel Revenue")

        medium_list = [i.UTM_MEDIUM for i in attribution_df.select(F.col('UTM_MEDIUM')).na.fill('NONE').distinct().collect()]
        
        st.bar_chart(_ad_spend_df, x="MEDIUM", y="REVENUE")

        st.subheader('Revenue by Ad Medium')
        st.table(attribution_df.select(['UTM_MEDIUM', 'UTM_SOURCE', 'REVENUE'])\
                                .pivot(F.col('UTM_MEDIUM'), medium_list)\
                                .sum('REVENUE')\
                                .rename('UTM_SOURCE', 'Source')\
        )

with sales_tab:
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.header("Top Customers by Customer Lifetime Value")
            st.dataframe(_clv_df.style.background_gradient(cmap='Reds', subset=['SENTIMENT SCORE'], axis=None))

        with col2:
            st.header("Churned Customers")
            # st.experimental_data_editor(
                # _churn_df.style.background_gradient(cmap='Reds', subset=['SENTIMENT SCORE'], axis=None), 
                # on_change=None, 
                # args=None, 
                # kwargs=None
            # )          
            selection_churn = aggrid_interactive_table(_churn_df, height=400)

    with st.container():
        # selection_churn = aggrid_interactive_table(_churn_df)

        if len(selection_churn['selected_rows']) > 0:
            selected_customer_id = selection_churn['selected_rows'][0]['CUSTOMER ID']
            
            selected_calls_df = calls_df.filter(F.col('CUSTOMER_ID') == selected_customer_id)\
                                        .drop(F.col('UUID'))\
                                        .sort(F.col('SENTIMENT'), ascending=False)\
                                        .to_pandas()
            selected_comments_df = comments_df.filter(F.col('CUSTOMER_ID') == selected_customer_id)\
                                            .drop(F.col('UUID'))\
                                            .sort(F.col('SENTIMENT'), ascending=False)\
                                            .limit(5)\
                                            .to_pandas()
            
            st.header('Customer Support Calls')
            selection_call = aggrid_interactive_table(selected_calls_df, height=100)

            if len(selection_call['selected_rows']) > 0:
                st.subheader('Call Transcription')
                st.write(selection_call['selected_rows'][0]['TRANSCRIPT'])
                st.subheader('Call Audio')
                audio_url = hook.get_first(f"SELECT get_presigned_url(@call_stage, LIST_DIR_TABLE.RELATIVE_PATH) as presigned_url \
                                             FROM DIRECTORY( @call_stage) \
                                             WHERE RELATIVE_PATH = '{selection_call['selected_rows'][0]['RELATIVE_PATH']}';")[0]
                st.audio(audio_url, format='audio/wav') 

            st.header('Customer Social Media Comments')
            selection_comment = aggrid_interactive_table(selected_comments_df, height=200)
    
            if len(selection_comment['selected_rows']) > 0:
                st.subheader('Twitter comment text')
                st.write(selection_comment['selected_rows'][0]['REVIEW_TEXT'])

with product_tab:
    with st.container():
        st.header('Product Research')
        search_text = st.text_input('Customer Feedback Keyword Search', value="")

        if search_text:
            st.write("Showing vector search results for: "+search_text)

            nearText = {"concepts": [search_text]}

            col1, col2 = st.columns(2)
            with col1:
                comments_result = (
                    weaviate_client.query
                    .get("CustomerComment", ["cUSTOMER_ID", "dATE", "rEVIEW_TEXT"])
                    .with_additional(['id'])
                    .with_near_text(nearText)
                    .with_limit(3)
                    .do()
                )
                
                near_comments_df = pd.DataFrame(comments_result['data']['Get']['CustomerComment'])\
                        .rename(columns={'cUSTOMER_ID':'CUSTOMER ID', 'dATE':'DATE', 'rEVIEW_TEXT':'TEXT'})
                near_comments_df['UUID'] = near_comments_df['_additional'].apply(lambda x: x['id'])
                near_comments_df.drop('_additional', axis=1, inplace=True)

                st.subheader('Customer Twitter Comments')
                selection_comment = aggrid_interactive_table(near_comments_df, height=100)
                
                if len(selection_comment['selected_rows']) > 0:
                    st.write(selection_comment['selected_rows'][0]['TEXT'])

                    st.subheader('Named Entities Extracted: ')
                    result = (
                        weaviate_client.query
                            .get("CustomerComment", ["rEVIEW_TEXT", "_additional {tokens ( properties: [\"rEVIEW_TEXT\"], limit: 1, certainty: 0.5) {entity property word certainty startPosition endPosition }}"])
                            .with_where(
                                {
                                    'path': ["id"],
                                    'operator': 'Equal',
                                    'valueString': selection_comment['selected_rows'][0]['UUID']
                                }
                            )
                            # .with_additional(
                            #     {
                            #         'tokens': ( properties: [\"rEVIEW_TEXT\"], limit: 1, certainty: 0.5) {entity property word certainty startPosition endPosition }
                            #     }
                            # )
                            .do()
                    )
                    NER_string = ''
                    tokens = result['data']['Get']['CustomerComment'][0]['_additional']['tokens']
                    for token in range(len(tokens)):
                        NER_string = NER_string + tokens[token]['word'] + ' : ' + tokens[token]['entity'] + ', '

                    st.write(NER_string)

            with col2:
                call_result = (
                    weaviate_client.query
                    .get("CustomerCall", ["cUSTOMER_ID", "dATE", "tRANSCRIPT"])
                    .with_additional(['id'])
                    .with_near_text(nearText)
                    .with_limit(3)
                    .do()
                )

                near_calls_df = pd.DataFrame(call_result['data']['Get']['CustomerCall'])\
                        .rename(columns={'cUSTOMER_ID':'CUSTOMER ID', 'dATE':'DATE', 'tRANSCRIPT':'TEXT'})
                near_calls_df['UUID'] = near_calls_df['_additional'].apply(lambda x: x['id'])
                near_calls_df.drop('_additional', axis=1, inplace=True)

                st.subheader('Customer Calls')
                selection_call = aggrid_interactive_table(near_calls_df, height=100)

                if len(selection_call['selected_rows']) > 0:
                    st.write(selection_call['selected_rows'][0]['TEXT'])

                    st.subheader('Named Entities Extracted: ')
                    result = (
                        weaviate_client.query
                            .get("CustomerCall", ["tRANSCRIPT", "_additional {tokens ( properties: [\"tRANSCRIPT\"], limit: 1, certainty: 0.7) {entity property word certainty startPosition endPosition }}"])
                            .with_where(
                                {
                                    'path': ["id"],
                                    'operator': 'Equal',
                                    'valueString': selection_call['selected_rows'][0]['UUID']
                                }
                            )
                            .do()
                    )

                    NER_string = ''
                    tokens = result['data']['Get']['CustomerCall'][0]['_additional']['tokens']
                    for token in range(len(tokens)):
                        NER_string = NER_string + tokens[token]['word'] + ' : ' + tokens[token]['entity'] + ', '

                    st.write(NER_string)
                    


    with st.container():
        st.header('QNA Search')
        search_question = st.text_area('Customer Feedback QNA Search', value="")

        if search_question:
            st.write("Showing QNA search results for:  "+search_question)

            col1, col2 = st.columns(2)
            with col1:
                col1.subheader('Twitter Comment Results')
                comment_ask = {
                    "question": search_question,
                    "properties": ["rEVIEW_TEXT"]
                }

                result = (
                weaviate_client.query
                    .get("CustomerComment", ["rEVIEW_TEXT", "_additional {answer {hasAnswer property result startPosition endPosition} }"])
                    .with_ask(comment_ask)
                    .with_limit(1)
                    .do()
                )

                if result['data']['Get']['CustomerComment']: 
                    # if result['data']['Get']['CustomerComment'][0]['_additional']['answer']['hasAnswer']:
                    st.write(result['data']['Get']['CustomerComment'][0]['rEVIEW_TEXT'])

            with col2:
                col2.subheader('Customer Call Results')
                call_ask = {
                    "question": search_question,
                    "properties": ["tRANSCRIPT"]
                }

                result = (
                weaviate_client.query
                    .get("CustomerCall", ["tRANSCRIPT", "_additional {answer {hasAnswer property result startPosition endPosition} }"])
                    .with_ask(call_ask)
                    .with_limit(1)
                    .do()
                )

                if result['data']['Get']['CustomerCall']: 
                    # if result['data']['Get']['CustomerCall'][0]['_additional']['answer']['hasAnswer']:
                    st.write(result['data']['Get']['CustomerCall'][0]['tRANSCRIPT'])


    

# uuids = [row['UUID'] for row in comments_df.select('UUID').collect()]

# result = weaviate_client.query.get("CustomerComment", ["cUSTOMER_ID", "dATE", "rEVIEW_TEXT"]).do()

# weaviate_client.query.get("CustomerComment", ["cUSTOMER_ID", "dATE", "rEVIEW_TEXT"]).with_additional(['id']).do()

# for uuid in uuids:
#     len(weaviate_client.data_object.get(uuid=uuid, with_vector=True)['vector'])

# weaviate_client.data_object.get(uuid=uuids[12345], with_vector=False)['properties'].keys()



        
    # nearText = {"concepts": ["magic"]}
    # result = (
    #     client.query
    #     .get("CustomerComment", ["customer_id", "date", "review_text"])
    #     .with_near_text(nearText)
    #     .with_limit(3)
    #     .do()
    # )
    # print(json.dumps(result, indent=4))

    # test_vector = client.data_object.get(uuid=uuids[57], with_vector=True)
    # test_vector['properties']
    # len(test_vector['vector'])
    # result = (
    #     client.query
    #     .get("CustomerComment", ["customer_id", "date", "review_text"])
    #     .with_near_vector({
    #         "vector": test_vector['vector'],
    #         "certainty": 0.5
    #     })
    #     .with_limit(4)
    #     .with_additional(['certainty'])
    #     .do()
    # )
    # print(json.dumps(result, indent=4))

    # where_filter = {
    #     "path": ["customer_id"],
    #     "operator": "Equal",
    #     "valueNumber": 25,
    # }

    # result = (
    #     client.query
    #     .get("CustomerComment", ["customer_id", "date", "review_text"])
    #     .with_where(where_filter)
    #     .do()
    # )
    # print(json.dumps(result, indent=4))



# audio_file = open('audio1.wav', 'rb')
                # audio_bytes = audio_file.read()
                # st.audio(audio_bytes, format='audio/wav')


# selection_call = {
                #     'selected_rows': {
                #         0:
                #             {
                #                 "_selectedRowNodeInfo":{
                #                     "nodeRowIndex":0,
                #                     "nodeId":"0"
                #                 },
                #                 "CUSTOMER_ID":"54",
                #                 "DATE":"2018-01-07T00:00:00.000",
                #                 "RELATIVE_PATH":"MLKH_Sr37.wav",
                #                 "TRANSCRIPT":" Today Giza is a suburb of roughly growing Cairo, the largest city in Africa and the fifth largest in the world.",
                #                 "SENTIMENT":0.7652763128
                #             }
                #     }
                # }
                