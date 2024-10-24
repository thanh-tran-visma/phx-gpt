.PHONY: venv install dev tests docker-remove clear-cache lint

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
	@gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app -b 0.0.0.0:8000

# Run the tests
tests:
	@echo "Running tests..."
	@$(VENV_DIR)/bin/python -m pytest tests/

# Clear cache
clear-cache:
	@echo "Removing cache..."
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs -r rm -rf

# Lint the project using flake8
lint:
	@echo "Running Flake8 linter..."
	@$(VENV_DIR)/bin/flake8 .

# Automatically fix linting issues using black
lint-fix:
	@echo "Fixing linting issues with black..."
	@$(VENV_DIR)/bin/black .
