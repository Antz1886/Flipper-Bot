# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install base Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Ensure TradingAgents is present and fully installed; auto-clone if submodule was not initialized on host
RUN if [ ! -f "TradingAgents/pyproject.toml" ]; then \
        rm -rf TradingAgents && \
        git clone --depth 1 https://github.com/TauricResearch/TradingAgents.git TradingAgents; \
    fi && \
    pip install --no-cache-dir -e ./TradingAgents

# Create the data and logs directories if they don't exist
RUN mkdir -p data logs models

# Command to run the bot
CMD ["python", "main.py"]
