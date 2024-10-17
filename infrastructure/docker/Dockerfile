# Stage 1: Build bluevi-gpt app
FROM python:3.11-slim AS app
USER root
# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app
# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Remove cache
RUN find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs -r rm -rf

# Copy the rest of the application code
COPY . .

# Command to run the application
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "-b", "0.0.0.0:8000"]