# Llama-3.1-8B-blueVi

## Overview

This project involves training or fine-tuning the **Llama-3.1-8B** model.

## Hugging Face Model

- **Inference Model**: [Llama3.1-8B-blueVi-GPT](https://huggingface.co/ThanhTranVisma/Llama3.1-8B-blueVi-GPT)
- **Lora (For continue training**[ThanhTranVisma/Llama3.1-8B-blueVi-GPT-lora](https://huggingface.co/ThanhTranVisma/Llama3.1-8B-blueVi-GPT-lora)

## Hugging Face Access Token Required

You will need a Hugging Face access token to run this project. You can get your token at [Hugging Face Tokens](https://huggingface.co/settings/tokens).

### How to Setup Locally

1. **Clone the Repository**: Clone the repository and navigate to the project folder.

2. **Create Python Virtual Environment, Activate it, and Install Requirements**
    ```bash
    make venv
    source env/bin/activate
    make install
    ```

3. **Copy Environment Variables**: Copy the environment variables example file and adjust the Hugging Face token with your own.
    ```bash
    cp .env.example .env
    vim .env
    ```

4. **Add the Database Host to `/etc/hosts`**:
    * Add the following line to your `/etc/hosts` file (Linux/MAC) or `C:\Windows\System32\drivers\etc\hosts` (Windows):
    ```bash
    127.0.0.1 gpt.dotweb.test
    127.0.0.1 gpt_db.dotweb.test
    ```
   
5. **Build docker containers**:
    * We need to use port 445 as well as port 3308 for db, so we will not have conflicts with the DWC-PHX application
    ```bash
    docker compose build
    ```
   
6. **Migration**:
    * Create a new database migration script
    ```bash
    alembic revision --autogenerate -m "init-db"
    ```
   
   * Run migration
   ```bash
    alembic upgrade head
    ```
7. **Run the Tests**:
    Run the tests to verify everything is working:
    ```bash
    make tests
    ```

## Linting

To ensure code quality and adherence to style guidelines, we use `flake8` and `black`. Follow these steps to run the linters:

1. **Run Flake8**:
   * `flake8` checks your code for style guide enforcement. To run `flake8`, execute the following command:
    ```bash
    make lint
    ```

2. **To format the code automatically, run**:
    ```bash
    make lint-fix
    ```