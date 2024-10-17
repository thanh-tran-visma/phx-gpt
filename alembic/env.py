from dotenv import load_dotenv
load_dotenv()

import os
from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
from app.database.base import Base

# This is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config
# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Add your model's MetaData object here for 'autogenerate' support
# This ensures that Alembic has access to your models' metadata for autogeneration
target_metadata = Base.metadata 

# Fetching the database URL from environment variables
# (You can also use a library like `dotenv` to load these variables from a `.env` file)
def get_database_url():
    db_username = os.getenv("DB_USERNAME")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")  # Removed the default value
    db_name = os.getenv("DB_DATABASE")

    if not db_username or not db_password or not db_host or not db_port or not db_name:
        raise ValueError("One or more required environment variables are missing")

    try:
        db_port = int(db_port)  # Ensure DB_PORT is converted to an integer
    except ValueError:
        raise ValueError(f"Invalid DB_PORT value: {db_port}")

    return f"mysql+mysqlconnector://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"

# Set SQLAlchemy connection URL dynamically using environment variables
config.set_section_option('alembic', 'sqlalchemy.url', get_database_url())

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = get_database_url()  # Fetch the URL from environment variables
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
