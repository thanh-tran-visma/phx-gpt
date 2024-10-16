#!/bin/bash

# Wait for MySQL to be healthy
echo "Waiting for MySQL to be healthy..."
until mysqladmin ping -h gpt_db -u gpt_user -p${DB_PASSWORD} --silent; do
  sleep 1
done

# Run migrations
echo "Running migrations..."
python /app/database/migrate.py

# Start the application with Gunicorn
echo "Starting the application..."
exec gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app -b 0.0.0.0:8000
