# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH "/app:/app/TradingAgents:${PYTHONPATH}"

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

# Install all Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Clone and install TradingAgents framework
RUN git clone --depth 1 https://github.com/TauricResearch/TradingAgents.git /app/TradingAgents && \
    pip install --no-cache-dir -e /app/TradingAgents

# Copy the rest of the application code
COPY . .

# Create data, logs, and models directories
RUN mkdir -p data logs models

# Command to run the bot
CMD ["python", "main.py"]
