from app.config.env_config import (
    DB_USERNAME,
    DB_PASSWORD,
    DB_HOST,
    DB_PORT,
    DB_DATABASE,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .base import Base


class Database:
    def __init__(self):
        self.engine = None
        self.Session = None

    def connect(self):
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

    @staticmethod
    def close():
        """Close the database connection."""
        # No need to manage connection explicitly, as sessions are created and closed in execute_query
        print(
            "No direct connection to close. Session management is handled by execute_query."
        )

    def execute_query(self, query, params=None):
        """Execute a SQL query and return the result."""
        self.connect()
        session = self.Session()
        try:
            result = session.execute(query, params)
            session.commit()
            print("Query executed successfully")
            return result.fetchall()  # Return all results
        except Exception as e:
            print(f"Error: {e}")
            session.rollback()
            return None  # Return None on error
        finally:
            session.close()


# Create an instance of the Database class
database = Database()


# Dependency to get a database session
def get_db():
    database.connect()
    db = database.Session()
    try:
        yield db
    finally:
        db.close()
