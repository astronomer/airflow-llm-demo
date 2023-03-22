# syntax=quay.io/astronomer/airflow-extensions:v1.0.0-alpha.2

# Prebuilt Docker image is available at mpgregor/snowservices-streamlit:latest
# TODO: move to quay
# To build from scratch run
# docker buildx build --load --no-cache --platform linux/arm64/v8,linux/amd64 --tag snowservices-streamlit:latest .

# FROM quay.io/astronomer/astro-runtime:7.4.1
FROM python:3.9-slim

USER root
RUN useradd -ms /bin/bash astro
PYENV 3.8 snowpark requirements-snowpark.txt

WORKDIR /app
COPY src .

EXPOSE 8501

CMD ["/home/astro/.venv/snowpark/bin/python", "-m", "streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
