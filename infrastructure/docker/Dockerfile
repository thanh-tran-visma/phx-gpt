FROM python:3.11-slim AS app
# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Create a virtual environment
RUN python -m venv /opt/venv

# Install dependencies in the virtual environment
RUN /opt/venv/bin/pip install --upgrade pip
COPY requirements.txt .
RUN /opt/venv/bin/pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Remove cache
RUN find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs -r rm -rf

# Copy the rest of the application code
COPY . .

# Load and download the model using secrets for HF_TOKEN and MODEL_NAME
RUN --mount=type=secret,id=hf_token \
    --mount=type=secret,id=model_name \
    bash -c "HF_TOKEN=$(cat /run/secrets/hf_token) \
    MODEL_NAME=$(cat /run/secrets/model_name) \
    /opt/venv/bin/python -c 'from app.llm.llm_model import BlueViGptModel; BlueViGptModel().load_model()'"

# Command to run the application using the virtual environment
CMD ["/opt/venv/bin/gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "-b", "0.0.0.0:8000"]