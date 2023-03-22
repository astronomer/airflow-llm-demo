# syntax=quay.io/astronomer/airflow-extensions:v1.0.0-alpha.3

FROM quay.io/astronomer/astro-runtime:7.4.1

PYENV 3.9 dbt requirements-dbt.txt

PYENV 3.8 snowpark requirements-snowpark.txt