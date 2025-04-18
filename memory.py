from datetime import datetime
import sqlite3
import uuid
import json

class ChatHistoryDB:
    """
    A class to handle saving and managing chatbot history in an SQLite database.
    """


    def __init__(self, db_path):
        """
        Initialize the ChatHistoryDB instance.

        Args:
            db_path (str): Path to the SQLite database file.
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.table_name = None
        self.cursor = self.conn.cursor()

    def create_table(self, table_name):
        """
        Create a table for storing chat history if it doesn't exist.

        Args:
            table_name (str): Name of the table to store chat history.
        """
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT NOT NULL,
                create_date TEXT NOT NULL,
                full_body TEXT NOT NULL
            )
        """)
        self.conn.commit()
        self.table_name = table_name

    def save_chat_history(self, chat_history):
        """
        Save chatbot history into the specified table.

        Args:
            chat_history (list of dict): List of chat messages as dictionaries with keys 'uuid', 'create_date', and 'full_body'.
        """
        
        for chat in chat_history:
            # Check if the chat already exists based on the UUID
            self.cursor.execute(f"""
            SELECT id FROM {self.table_name} WHERE uuid = ?
            """, (str(chat["conversation_id"]),))
            result = self.cursor.fetchone()

            if result:
                # Update the existing chat
                self.cursor.execute(f"""
                    UPDATE {self.table_name}
                    SET create_date = ?, full_body = ?
                    WHERE uuid = ?
                    """, (chat["create_date"], json.dumps(chat, default=str), str(chat["conversation_id"])))
            else:
                # Insert a new chat
                self.cursor.execute(f"""
                    INSERT INTO {self.table_name} (uuid, create_date, full_body)
                    VALUES (?, ?, ?)
                    """, (str(chat["conversation_id"]), chat["create_date"], json.dumps(chat, default=str)))
        self.conn.commit()

    def read_all_chats(self):
        """
        Read all chat history from the database.

        Returns:
            list of dict: List of chat messages as dictionaries with keys 'uuid', 'create_date', and 'full_body'.
        """
        self.cursor.execute(f"SELECT uuid, create_date, full_body FROM {self.table_name}")
        rows = self.cursor.fetchall()
        chat_history = []
        for row in rows:
            chat_history.append(
                json.loads(row[2])
            )
        return chat_history
        
    def close_connection(self):
        """
        Close the database connection.
        """
        self.conn.close()

        def read_all_chats(self):
            """
            Read all chat history from the database.

            Returns:
                list of dict: List of chat messages as dictionaries with keys 'uuid', 'create_date', and 'full_body'.
            """
            self.cursor.execute(f"SELECT uuid, create_date, full_body FROM {self.table_name}")
            rows = self.cursor.fetchall()
            chat_history = []
            for row in rows:
                chat_history.append(
                    json.loads(row[2])
                )
            return chat_history
