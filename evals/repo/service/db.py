"""SQLite data access layer."""
import sqlite3


class Store:
    def __init__(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        self.conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, item TEXT)")
        self.conn.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, kind TEXT)")
        self.conn.commit()

    def add_user(self, name: str) -> None:
        cur = self.conn.cursor()
        params = (name,)
        cur.execute("INSERT INTO users (name) VALUES (?)", params)
        self.conn.commit()

    def add_order(self, item: str) -> None:
        cur = self.conn.cursor()
        params = (item,)
        cur.execute("INSERT INTO orders (item) VALUES (?)", params)
        self.conn.commit()

    def add_event(self, kind: str) -> None:
        cur = self.conn.cursor()
        params = (kind,)
        cur.execute("INSERT INTO events (kind) VALUES (?)", params)
        self.conn.commit()

    def get_user(self, user_id: int):
        cur = self.conn.cursor()
        params = (user_id,)
        cur.execute("SELECT id, name FROM users WHERE id = ?", params)
        return cur.fetchone()

    def get_order(self, order_id: int):
        cur = self.conn.cursor()
        params = (order_id,)
        cur.execute("SELECT id, item FROM orders WHERE id = ?", params)
        return cur.fetchone()

    def get_event(self, event_id: int):
        cur = self.conn.cursor()
        params = (event_id,)
        cur.execute("SELECT id, kind FROM events WHERE id = ?", params)
        return cur.fetchone()
