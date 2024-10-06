.PHONY: venv install dev tests docker-remove clear-cache

# Define the virtual environment directory
VENV_DIR=env

# Create a virtual environment
venv:
	@if [ ! -d $(VENV_DIR) ]; then \
		echo "Creating Python virtual environment..."; \
		python -m venv $(VENV_DIR); \
	else \
		echo "Virtual environment already exists."; \
	fi

# Install requirements after activating the virtual environment
install:
	@echo "Installing requirements..."
	$(VENV_DIR)/bin/pip install -r requirements.txt

# Run the application
dev:
	@gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app -b ${BLUEVI_GPT}:8000 --keyfile infrastructure/certs/dotweb.test.key --certfile infrastructure/certs/dotweb.test.crt

# Run the tests
tests:
	@echo "Running tests..."
	@$(VENV_DIR)/bin/python -m pytest tests/

# Remove stopped Docker containers
docker-remove:
	@echo "Removing stopped Docker containers..."
	docker-compose down --rmi all  # Removes images as well, use with caution

# Clear cache
clear-cache:
	@echo "Removing cache..."
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs -r rm -rf
