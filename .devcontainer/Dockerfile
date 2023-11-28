FROM mcr.microsoft.com/devcontainers/python:3.10
# Install the xz-utils package
RUN apt-get update && apt-get install -y xz-utils
RUN pip install --upgrade pip
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt --no-cache-dir && rm /tmp/requirements.txt