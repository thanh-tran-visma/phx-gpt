from app.database.database import Database

class User:
    def __init__(self, id=None, role=None, created_at=None, updated_at=None):
        self.id = id
        self.role = role
        self.created_at = created_at
        self.updated_at = updated_at

    @staticmethod
    def insert_user(role):
        insert_query = '''
        INSERT INTO users (role) VALUES (%s);
        '''
        db = Database()
        db.connect()
        db.execute_query(insert_query, (role,))
        db.close()

    @staticmethod
    def get_user_by_id(user_id):
        select_query = '''
        SELECT * FROM users WHERE id = %s;
        '''
        db = Database()
        db.connect()
        cursor = db.connection.cursor(dictionary=True)
        cursor.execute(select_query, (user_id,))
        user = cursor.fetchone()
        cursor.close()
        db.close()
        return user

class Gpt:
    def __init__(self, id=None, role=None, created_at=None, updated_at=None):
        self.id = id
        self.role = role
        self.created_at = created_at
        self.updated_at = updated_at

    @staticmethod
    def insert_gpt():
        insert_query = '''
        INSERT INTO gpt (role) VALUES ('gpt');
        '''
        db = Database()
        db.connect()
        db.execute_query(insert_query)
        db.close()

    @staticmethod
    def get_gpt_by_id(gpt_id):
        select_query = '''
        SELECT * FROM gpt WHERE id = %s;
        '''
        db = Database()
        db.connect()
        cursor = db.connection.cursor(dictionary=True)
        cursor.execute(select_query, (gpt_id,))
        gpt = cursor.fetchone()
        cursor.close()
        db.close()
        return gpt

class Content:
    def __init__(self, id=None, content=None, created_at=None):
        self.id = id
        self.content = content
        self.created_at = created_at

    @staticmethod
    def insert_content(content):
        insert_query = '''
        INSERT INTO content (content) VALUES (%s);
        '''
        db = Database()
        db.connect()
        db.execute_query(insert_query, (content,))
        db.close()

    @staticmethod
    def get_content_by_id(content_id):
        select_query = '''
        SELECT * FROM content WHERE id = %s;
        '''
        db = Database()
        db.connect()
        cursor = db.connection.cursor(dictionary=True)
        cursor.execute(select_query, (content_id,))
        content = cursor.fetchone()
        cursor.close()
        db.close()
        return content

class Conversation:
    def __init__(self, id=None, thread_id=None, sender_id=None, gpt_sender_id=None, content_id=None, created_at=None, end_at=None):
        self.id = id
        self.thread_id = thread_id
        self.sender_id = sender_id
        self.gpt_sender_id = gpt_sender_id
        self.content_id = content_id
        self.created_at = created_at
        self.end_at = end_at

    @staticmethod
    def insert_conversation(thread_id, sender_id, gpt_sender_id, content_id):
        insert_query = '''
        INSERT INTO conversations (thread_id, sender_id, gpt_sender_id, content_id) 
        VALUES (%s, %s, %s, %s);
        '''
        db = Database()
        db.connect()
        db.execute_query(insert_query, (thread_id, sender_id, gpt_sender_id, content_id))
        db.close()

    @staticmethod
    def get_conversation_by_id(conversation_id):
        select_query = '''
        SELECT * FROM conversations WHERE id = %s;
        '''
        db = Database()
        db.connect()
        cursor = db.connection.cursor(dictionary=True)
        cursor.execute(select_query, (conversation_id,))
        conversation = cursor.fetchone()
        cursor.close()
        db.close()
        return conversation
