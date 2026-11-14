import os
import psycopg2

from dotenv import load_dotenv

load_dotenv()


def get_db_connection():

    try:
        connection = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )

        return connection

    except psycopg2.Error as e:
        print(f"Database connection failed: {e}")
        raise


def test_connection():

    connection = get_db_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT
                current_database(),
                current_setting('port'),
                version()
        """)

        result = cursor.fetchone()

        print("\n========== DATABASE CONNECTED ==========")
        print("Database:", result[0])
        print("Port:", result[1])
        print("PostgreSQL:", result[2])

        cursor.close()

    finally:
        connection.close()


if __name__ == "__main__":
    test_connection()