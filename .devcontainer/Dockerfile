# Use Python 3.10 base image
FROM mcr.microsoft.com/devcontainers/python:3.10

# Set environment variables
ENV PIP_DISABLE_PIP_VERSION_CHECK=on

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /rag_app

# Copy and install Python requirements
COPY requirements.txt /rag_app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /rag_app/

# Set entrypoint
CMD ["bash"]
