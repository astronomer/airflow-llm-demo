# syntax=quay.io/astronomer/airflow-extensions:latest

FROM quay.io/astronomer/astro-runtime:7.4.2-base

COPY include/airflow_provider_great_expectations-0.2.6-py3-none-any.whl /tmp

PYENV 3.8 dbt requirements-dbt.txt