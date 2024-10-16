from app.database.database import Database

class Migrator:
    def __init__(self):
        self.db = Database()

    def create_users_table(self):
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            role ENUM('user') NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        );
        '''
        self.db.execute_query(create_table_query)

    def create_gpt_table(self):
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS gpt (
            id INT AUTO_INCREMENT PRIMARY KEY,
            role ENUM('gpt') NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        );
        '''
        self.db.execute_query(create_table_query)

    def create_content_table(self):
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS content (
            id INT AUTO_INCREMENT PRIMARY KEY,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        '''
        self.db.execute_query(create_table_query)

    def create_conversations_table(self):
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS conversations (
            id INT AUTO_INCREMENT PRIMARY KEY,
            thread_id INT,
            sender_id INT,
            gpt_sender_id INT,
            content_id INT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            end_at TIMESTAMP,
            FOREIGN KEY (sender_id) REFERENCES users(id),
            FOREIGN KEY (gpt_sender_id) REFERENCES gpt(id),
            FOREIGN KEY (content_id) REFERENCES content(id)
        );
        '''
        self.db.execute_query(create_table_query)

    def run_migrations(self):
        self.db.connect()
        self.create_users_table()
        self.create_gpt_table()
        self.create_content_table()
        self.create_conversations_table()
        self.db.close()

if __name__ == "__main__":
    migrator = Migrator()
    migrator.run_migrations()
