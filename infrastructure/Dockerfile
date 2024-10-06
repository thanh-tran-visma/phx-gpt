# Stage 1: Build bluevi-gpt app
FROM python:3.11-slim AS app
USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy the rest of the application code
COPY . .

# Load and download the model
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
ARG MODEL_NAME
ENV MODEL_NAME=${MODEL_NAME}
RUN python -c "from app.model.model import BlueViGptModel; BlueViGptModel().load_model()"

# Command to run the application
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "-b", "0.0.0.0:8000"]

# Final Nginx image
FROM nginx:1.25.4 AS nginx

# Copy SSL certificates
COPY infrastructure/certs/* /etc/nginx/certs/
COPY infrastructure/nginx/nginx.conf /etc/nginx/nginx.conf
COPY infrastructure/nginx/conf.d/ /etc/nginx/conf.d/

EXPOSE 80
EXPOSE 443

CMD ["nginx", "-g", "daemon off;"]
