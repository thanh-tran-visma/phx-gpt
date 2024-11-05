from app.config.config_env import (
    DB_USERNAME,
    DB_PASSWORD,
    DB_HOST,
    DB_PORT,
    DB_DATABASE,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from .base import Base
from typing import Generator, Optional


class Database:
    def __init__(self) -> None:
        self.engine = None
        self.Session: Optional[sessionmaker] = None
        self.connect()

    def connect(self) -> None:
        """Establish a connection to the database."""
        if self.engine is None:
            # Create connection string for SQLAlchemy
            db_url = f"mysql+mysqlconnector://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DATABASE}"
            # Create an engine instance
            self.engine = create_engine(db_url, echo=True)

            # Create a session maker for connecting to the DB
            self.Session = sessionmaker(bind=self.engine)

            # Bind the engine to the metadata of the Base class
            Base.metadata.create_all(bind=self.engine)

            print("Connected to MySQL database using SQLAlchemy")

    def get_session(self) -> Session:
        """Get a new database session."""
        if self.Session is None:
            raise Exception("Database is not connected.")
        return self.Session()

    def disconnect(self):
        if self.engine:
            self.engine.dispose()


# Dependency to get a database session
def get_db() -> Generator[Session, None, None]:
    database = Database()
    db: Session = database.get_session()
    try:
        yield db
    finally:
        db.close()
