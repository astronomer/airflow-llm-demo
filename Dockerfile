# syntax=quay.io/astronomer/airflow-extensions:latest

FROM quay.io/astronomer/astro-runtime:9.1.0-base

COPY include/airflow_provider_weaviate-0.0.1-py3-none-any.whl /tmp

#need virtualenv for dbt due to protobuf dependency collision with streamlit
PYENV 3.9 dbt requirements-dbt.txt