import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from .base import Base

class Database:
    def __init__(self):
        self.engine = None
        self.Session = None
        self.connection = None
        self.connect()

    def connect(self):
        # Create connection string for SQLAlchemy
        db_url = f"mysql+mysqlconnector://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_DATABASE')}"
        
        # Create an engine instance
        self.engine = create_engine(db_url, echo=True)
        
        # Create a session maker for connecting to the DB
        self.Session = sessionmaker(bind=self.engine)
        
        # Bind the engine to the metadata of the Base class
        Base.metadata.create_all(bind=self.engine)
        
        print("Connected to MySQL database using SQLAlchemy")

    def close(self):
        if self.connection:
            self.connection.close()
            print("MySQL connection closed")

    def execute_query(self, query, params=None):
        session = self.Session()
        try:
            result = session.execute(query, params)
            session.commit()
            print("Query executed successfully")
        except Exception as e:
            print(f"Error: {e}")
            session.rollback()
        finally:
            session.close()

# Create an instance of the Database class
database = Database()

# Dependency to get a database session
def get_db():
    db = database.Session()
    try:
        yield db
    finally:
        db.close()