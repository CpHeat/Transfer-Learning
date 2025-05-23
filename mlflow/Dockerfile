FROM python:3.13.3-slim-bullseye

RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install "mlflow[auth]" pymysql boto3 cryptography