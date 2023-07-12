# syntax=quay.io/astronomer/airflow-extensions:latest

FROM quay.io/astronomer/astro-runtime:8.7.0-base

COPY include/airflow_provider_great_expectations-0.2.6-py3-none-any.whl /tmp

PYENV 3.9 dbt requirements-dbt.txt