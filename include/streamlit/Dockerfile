FROM python:3.9-slim-bullseye

COPY requirements.txt /tmp

RUN apt-get update && \
    apt-get install -y build-essential && \
    useradd -ms /bin/bash streamlit
USER streamlit
RUN pip install -r /tmp/requirements.txt

WORKDIR /app
COPY src .

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
