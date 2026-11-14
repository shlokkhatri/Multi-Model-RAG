import json

from psycopg2.extras import Json

from app.database.db_connection import get_connection


def insert_chunks(chunks):

    if not chunks:
        print("No chunks available for insertion.")
        return

    connection = get_connection()

    try:

        cursor = connection.cursor()

        insert_query = """
            INSERT INTO document_chunks (

                chunk_id,
                document_name,
                page_number,
                content_type,
                content,
                image_path,
                embedding,
                metadata

            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)

            ON CONFLICT (document_name, chunk_id)

            DO UPDATE SET

                page_number = EXCLUDED.page_number,
                content_type = EXCLUDED.content_type,
                content = EXCLUDED.content,
                image_path = EXCLUDED.image_path,
                embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata;
        """

        inserted_count = 0

        for chunk in chunks:

            embedding = chunk["embedding"]

            if len(embedding) != 1536:
                raise ValueError(
                    f"Invalid embedding dimensions for "
                    f"{chunk['chunk_id']}: {len(embedding)}"
                )

            metadata = dict(chunk.get("metadata") or {})

            # Preserve source information.
            metadata["source_file"] = chunk.get("source_file")

            image_path = (
                metadata.get("image_path")
                or metadata.get("visual_path")
                or metadata.get("file_path")
                or metadata.get("image_file")
            )

            content = chunk["content"]

            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)

            cursor.execute(
                insert_query,
                (
                    chunk["chunk_id"],
                    chunk["document_name"],
                    chunk.get("page_number"),
                    chunk["content_type"],
                    content,
                    image_path,
                    embedding,
                    Json(metadata)
                )
            )

            inserted_count += 1

        connection.commit()

        print("\n========== INGESTION COMPLETE ==========")
        print(f"Processed chunks: {inserted_count}")
        print("PostgreSQL storage successful!")

    except Exception as e:

        connection.rollback()

        print(f"Data insertion failed: {e}")

        raise

    finally:

        cursor.close()
        connection.close()