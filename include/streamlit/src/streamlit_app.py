import os
import pandas as pd
import streamlit as st
import weaviate
from sqlalchemy import create_engine
from weaviate_provider.hooks.weaviate import WeaviateHook
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode

st.set_page_config(layout="wide")

_WEAVIATE_CONN_ID = 'weaviate_default'
_S3_CONN_ID = 'minio_default'

if 'psql_engine' not in st.session_state:
    psql_engine = create_engine(st.secrets['postgres']['url'], connect_args={'options': '-csearch_path=demo'})
    st.session_state['psql_engine'] = psql_engine
else:
    psql_engine = st.session_state['psql_engine']

if "weaviate_client" not in st.session_state:
    weaviate_client = WeaviateHook(_WEAVIATE_CONN_ID).get_conn()
    st.session_state['weaviate_client'] = weaviate_client
else:
  weaviate_client = st.session_state['weaviate_client']    

s3_hook = S3Hook(_S3_CONN_ID)

#Setup data sources

st.cache_data()
def attribution_df():
    return pd.read_sql('select * from attribution_touches', psql_engine)
_attribution_df = attribution_df()

st.cache_data()
def calls_df():
    return pd.read_sql('select * from pred_customer_calls', psql_engine)
_calls_df = calls_df()

st.cache_data()
def comments_df():
    return pd.read_sql('select * from pred_twitter_comments', psql_engine)
_comments_df = comments_df()

@st.cache_data()
def ad_spend_df():
    return pd.read_sql('select * from pres_ad_spend', psql_engine)
_ad_spend_df = ad_spend_df()

@st.cache_data()
def clv_df(row_limit=10):
    df = pd.read_sql('select * from pres_clv', psql_engine)
    df.columns = [col.replace('_', ' ') for col in df.columns]
    df.set_index('customer id', drop=True, inplace=True)
    return df
_clv_df = clv_df()
 
@st.cache_data()
def churn_df(row_limit=10):
    df = pd.read_sql('select * from pres_churn', psql_engine)
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

    options.configure_selection("single")
    selection = AgGrid(
        data=df,
        height=height,
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
        st.subheader("Powered by Apache Airflow, Snowpark Containers and Streamlit")
    with logo_col:
        st.image('logo_small.png')

marketing_tab, sales_tab, product_tab = st.tabs(["Marketing", "Sales", "Product"])

with marketing_tab:
    with st.container():
        st.header("Return on Ad Spend")
        st.subheader("Digital Channel Revenue")

        medium_list = _attribution_df['utm_medium'].fillna('NONE').unique().tolist()
        
        st.bar_chart(_ad_spend_df, x="Medium", y="Revenue")

        st.subheader('Revenue by Ad Medium')
        st.table(pd.pivot_table(_attribution_df[['utm_medium', 'utm_source', 'revenue']], 
                       index='utm_source', 
                       columns='utm_medium', 
                       aggfunc='sum', 
                       margins=True, 
                       margins_name='revenue',
                       fill_value=0)
        )

with sales_tab:
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.header("Top Customers by Customer Lifetime Value")
            st.dataframe(_clv_df.style.background_gradient(cmap='Reds', subset=['sentiment score'], axis=None))

        with col2:
            st.header("Churned Customers")        
            selection_churn = aggrid_interactive_table(_churn_df.sort_values('sentiment score', ascending=False), height=400)

    with st.container():

        if len(selection_churn['selected_rows']) > 0:
            selected_customer_id = selection_churn['selected_rows'][0]['customer id']
            
            selected_calls_df = _calls_df[_calls_df['customer_id'] == selected_customer_id]\
                                        .drop(['vector'], axis=1)\
                                        .sort_values('sentiment', ascending=False)
            
            selected_comments_df = _comments_df[_comments_df['customer_id'] == selected_customer_id]\
                                        .drop(['vector'], axis=1)\
                                        .sort_values('sentiment', ascending=False)
            
            st.header('Customer Support Calls')
            selection_call = aggrid_interactive_table(selected_calls_df, height=200)

            if len(selection_call['selected_rows']) > 0:
                st.subheader('Call Transcription')
                st.write(selection_call['selected_rows'][0]['transcript'])
                st.subheader('Call Audio')
                
                audio_url = selection_call['selected_rows'][0]['full_path']

                response = s3_hook.get_key(bucket_name=audio_url.split('/')[2],
                                           key=audio_url.split('/')[3]).get()['Body']
                
                audio = response.read()
                response.close()

                st.audio(audio, format='audio/wav') 

            st.header('Customer Social Media Comments')
            selection_comment = aggrid_interactive_table(selected_comments_df, height=200)
    
            if len(selection_comment['selected_rows']) > 0:
                st.subheader('Twitter comment text')
                st.write(selection_comment['selected_rows'][0]['review_text'])

with product_tab:
    with st.container():
        st.header('Product Research')
        search_text = st.text_input('Customer Feedback Keyword Search', value="")

        if search_text:
            st.write("Showing vector search results for: "+search_text)

            nearText = {"concepts": [search_text]}

            col1, col2 = st.columns(2)
            with col1:

                comments_result = weaviate_client.query\
                                                 .get("CustomerComment", ["cUSTOMER_ID", "dATE", "rEVIEW_TEXT"])\
                                                 .with_additional(['id'])\
                                                 .with_near_text(nearText)\
                                                 .with_limit(3)\
                                                 .do()
                
                if comments_result.get('errors'):
                    for error in comments_result['errors']:
                        if 'no api key found' or 'remote client vectorize: failed with status: 401 error' in error['message']:
                            raise Exception('Cannot vectorize.  Check the OPENAI_API key as environment variable.')
                
                near_comments_df = pd.DataFrame(comments_result['data']['Get']['CustomerComment'])\
                        .rename(columns={'cUSTOMER_ID':'CUSTOMER ID', 'dATE':'DATE', 'rEVIEW_TEXT':'TEXT'})
                near_comments_df['UUID'] = near_comments_df['_additional'].apply(lambda x: x['id'])
                near_comments_df.drop('_additional', axis=1, inplace=True)

                st.subheader('Customer Twitter Comments')
                selection_comment = aggrid_interactive_table(near_comments_df, height=100)
                
                if len(selection_comment['selected_rows']) > 0:
                    st.write(selection_comment['selected_rows'][0]['TEXT'])

                    # st.subheader('Named Entities Extracted: ')
                    # result = weaviate_client.query\
                    #         .get("CustomerComment", ["rEVIEW_TEXT", 
                    #                                  "_additional {tokens ( properties: [\"rEVIEW_TEXT\"], limit: 1, certainty: 0.5) {entity property word certainty startPosition endPosition }}"])\
                    #         .with_where(
                    #             {
                    #                 'path': ["id"],
                    #                 'operator': 'Equal',
                    #                 'valueString': selection_comment['selected_rows'][0]['UUID']
                    #             }
                    #         )\
                    #         .do()
                    
                    # NER_string = ''
                    # tokens = result['data']['Get']['CustomerComment'][0]['_additional']['tokens']
                    # for token in range(len(tokens)):
                    #     NER_string = NER_string + tokens[token]['word'] + ' : ' + tokens[token]['entity'] + ', '

                    # st.write(NER_string)

            with col2:
                call_result = weaviate_client.query\
                                             .get("CustomerCall", ["cUSTOMER_ID", "tRANSCRIPT"])\
                                             .with_additional(['id'])\
                                             .with_near_text(nearText)\
                                             .with_limit(3)\
                                             .do()\
                
                near_calls_df = pd.DataFrame(call_result['data']['Get']['CustomerCall'])\
                        .rename(columns={'cUSTOMER_ID':'CUSTOMER ID', 'tRANSCRIPT':'TEXT'})
                near_calls_df['UUID'] = near_calls_df['_additional'].apply(lambda x: x['id'])
                near_calls_df.drop('_additional', axis=1, inplace=True)

                st.subheader('Customer Calls')
                selection_call = aggrid_interactive_table(near_calls_df, height=100)

                if len(selection_call['selected_rows']) > 0:
                    st.write(selection_call['selected_rows'][0]['TEXT'])

                    # st.subheader('Named Entities Extracted: ')
                    # result = weaviate_client.query\
                    #             .get("CustomerCall", 
                    #                  ["tRANSCRIPT", 
                    #                   "_additional {tokens ( properties: [\"tRANSCRIPT\"], limit: 1, certainty: 0.7) {entity property word certainty startPosition endPosition }}"])\
                    #             .with_where(
                    #                 {
                    #                     'path': ["id"],
                    #                     'operator': 'Equal',
                    #                     'valueString': selection_call['selected_rows'][0]['UUID']
                    #                 }
                    #             )\
                    #             .do()
                    
                    # NER_string = ''
                    # tokens = result['data']['Get']['CustomerCall'][0]['_additional']['tokens']
                    # for token in range(len(tokens)):
                    #     NER_string = NER_string + tokens[token]['word'] + ' : ' + tokens[token]['entity'] + ', '

                    # st.write(NER_string)
                    


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

                result = weaviate_client.query\
                            .get("CustomerComment", ["rEVIEW_TEXT", 
                                                     "_additional {answer {hasAnswer property result startPosition endPosition} }"])\
                            .with_ask(comment_ask)\
                            .with_limit(1)\
                            .do()
                
                if result.get('errors'):
                        for error in result['errors']:
                            if 'no api key found' or 'remote client vectorize: failed with status: 401 error' in error['message']:
                                raise Exception('Cannot vectorize.  Check the OPENAI_API key as environment variable.')

                if result['data']['Get']['CustomerComment']: 
                    st.write(result['data']['Get']['CustomerComment'][0]['rEVIEW_TEXT'])

            with col2:
                col2.subheader('Customer Call Results')
                call_ask = {
                    "question": search_question,
                    "properties": ["tRANSCRIPT"]
                }

                result = weaviate_client.query\
                            .get("CustomerCall", 
                                ["tRANSCRIPT", 
                                "_additional {answer {hasAnswer property result startPosition endPosition} }"])\
                            .with_ask(call_ask)\
                            .with_limit(1)\
                            .do()

                if result['data']['Get']['CustomerCall']: 
                    st.write(result['data']['Get']['CustomerCall'][0]['tRANSCRIPT'])
