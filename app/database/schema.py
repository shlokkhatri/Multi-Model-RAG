from app.database.db_connection import get_connection


def create_tables():

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("""
            CREATE EXTENSION IF NOT EXISTS vector;

            CREATE TABLE IF NOT EXISTS document_chunks (

                id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

                chunk_id TEXT NOT NULL,

                document_name TEXT NOT NULL,

                page_number INTEGER,

                content_type TEXT NOT NULL,

                content TEXT NOT NULL,

                image_path TEXT,

                embedding VECTOR(1536) NOT NULL,

                metadata JSONB DEFAULT '{}'::jsonb,

                created_at TIMESTAMP WITH TIME ZONE
                DEFAULT CURRENT_TIMESTAMP,

                CONSTRAINT unique_document_chunk
                UNIQUE (document_name, chunk_id)
            );
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding
            ON document_chunks
            USING hnsw (embedding vector_cosine_ops);
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_document_chunks_content_type
            ON document_chunks(content_type);
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_document_chunks_page
            ON document_chunks(page_number);
        """)

        connection.commit()

        print("Database schema verified successfully!")

    except Exception as e:

        connection.rollback()
        print(f"Schema creation failed: {e}")
        raise

    finally:

        cursor.close()
        connection.close()


if __name__ == "__main__":
    create_tables()