# Llama-3.1-8B-blueVi

## Overview

This project involves training or fine-tuning the **Llama-3.1-8B** model.

## Hugging Face Model

- **Base Model**: [Llama3.1-8B-blueVi-GPT-base](https://huggingface.co/ThanhTranVisma/Llama3.1-8B-blueVi-GPT-base)
- **LORA**: [Llama3.1-8B-blueVi-GPT-lora](https://huggingface.co/ThanhTranVisma/Llama3.1-8B-blueVi-GPT-lora)

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

5. **Add the Database Host to `/etc/hosts`**:
    Add the following line to your `/etc/hosts` file (Linux/MAC) or `C:\Windows\System32\drivers\etc\hosts` (Windows):
    ```bash
    127.0.0.1 gpt.dotweb.test
    127.0.0.1 gpt_db.dotweb.test
    ```

6. **Build docker containers**:
    We need to use port 445 as well as port 3308 for db, so we will not have conflicts with the DWC-PHX application
    ```bash
    docker compose build
    ```

7. **Migration**:
    ```bash
    alembic upgrade head
    ```
    FOR UPDATING DATABASE
    ```bash
    alembic revision --autogenerate -m "your-message"
    ```
8. **Run the Tests**:
    Run the tests to verify everything is working:
    ```bash
    make tests
    ```