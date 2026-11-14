import json
from pathlib import Path
from psycopg2.extras import Json
from pgvector.psycopg2 import register_vector
from app.database.db_connection import get_db_connection

PROCESSED_DIR = Path(
    "data/processed/Global_Renewable_Energy_Outlook_220MB"
)

UNIFIED_CHUNKS_FILE = PROCESSED_DIR / "unified_chunks.json"
EMBEDDED_CHUNKS_FILE = PROCESSED_DIR / "embedded_chunks.json"

EXPECTED_DIMENSION = 1536

def load_json(file_path):

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found {file_path}"
        )
    with open(file_path,"r",encoding = "utf-8") as file:
        return json.load(file)

# Load Chunks and velidate them

def prepared_chunks():

    print("--- LOADING MULTIMODEL CHUNKS ---")

    unified_data = load_json(UNIFIED_CHUNKS_FILE)
    embedded_data = load_json(EMBEDDED_CHUNKS_FILE)

    unified_chunks = unified_data["chunks"]
    embedded_chunks = embedded_data["chunks"]

    print("UNIFIED CHUNKS : ",len(unified_chunks))
    print("EMBEDDED CHUNKS : ",len(embedded_chunks))

    embedding_map = {}

    for chunk in embedded_chunks:
        
        chunk_id = chunk.get("chunk_id")
        embedding = chunk.get("embedding")

        if not chunk_id:
            raise ValueError(
                "Embedding chunk missing chunk_id"
            )

        if chunk_id in embedding_map:
            raise ValueError(
                f"Duplicate Chunk embrdding : {chunk_id}"
            )

        if not isinstance(embedding,list):
            raise ValueError(
                f"Invalid embedding diamentions for {chunk_id}"
            )

        if len(embedding) != EXPECTED_DIMENSION:
            raise ValueError(
                f"Invalid embedding dimensions for {chunk_id}. "
                f"Expected {EXPECTED_DIMENSION}, got {len(embedding)}"
            )

        embedding_map[chunk_id] = embedding

    prepared_chunks = []

    for chunk in unified_chunks:

        chunk_id = chunk.get("chunk_id")

        if chunk_id not in embedding_map:
            raise ValueError(
                f"Missing embedding for chunk : {chunk_id}"
            )

        embedding = embedding_map[chunk_id]

        prepared_chunks.append({
            "chunk_id" : chunk_id,
            "document_name" : chunk.get("document_name"),
            "page_number" : chunk.get("page_number"),
            "content_type" : chunk.get("content_type"),
            "content" : chunk.get("content"),
            "image_path" : (
                chunk.get("image_path") 
                or chunk.get("metadata",{}).get("image_path") 
            ),
            "embedding" : embedding,
            "metadata" : {
                **chunk.get("metadata", {}),
                "source_file" : chunk.get("source_file")
            }
        })

    unified_ids = {
        chunk["chunk_id"] for chunk in unified_chunks
    }

    embedded_ids = set(embedding_map.keys())

    if unified_ids != embedded_ids:

        missing = unified_ids - embedded_ids
        extra = embedded_ids - unified_ids

        raise ValueError(
            f"Chunk ID mismatch.\n"
            f"Missing embeddings: {missing}\n"
            f"Extra embeddings: {extra}"
        )

    print("\nPrepared chunks:", len(prepared_chunks))
    print("Embedding dimensions:", EXPECTED_DIMENSION)

    return prepared_chunks


def insert_chunks():

    chunks = prepared_chunks()

    connection = get_db_connection()

    try:

        # Register pgvector
        register_vector(connection)

        print("\nPGVECTOR REGISTERED SUCCESSFULLY!")

        cursor = connection.cursor()

        cursor.execute("""
            SELECT
                table_schema,
                table_name
            FROM information_schema.tables
            WHERE table_type = 'BASE TABLE'
            ORDER BY table_schema, table_name;
        """)

        tables = cursor.fetchall()

        print("\n========== AVAILABLE DATABASE TABLES ==========")

        for table in tables:
            print(f"{table[0]}.{table[1]}")

        cursor.execute("""
            SELECT
                current_database(),
                current_schema(),
                current_setting('search_path');
        """)

        print("\nDATABASE DETAILS:", cursor.fetchone())

        inserted = 0
        skipped = 0

        print("\n========== INSERTING INTO POSTGRESQL ==========")

        for index, chunk in enumerate(chunks, start=1):

            cursor.execute(
                """
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
                ON CONFLICT (chunk_id) DO NOTHING
                RETURNING id;
                """,
                (
                    chunk["chunk_id"],
                    chunk["document_name"],
                    chunk["page_number"],
                    chunk["content_type"],
                    chunk["content"],
                    chunk["image_path"],
                    chunk["embedding"],
                    Json(chunk["metadata"])
                )
            )

            result = cursor.fetchone()

            if result:
                inserted += 1
                status = "INSERTED"
            else:
                skipped += 1
                status = "SKIPPED"

            print(
                f"[{index}/{len(chunks)}] "
                f"{chunk['chunk_id']} {status}"
            )

        connection.commit()

        print("\n========== INSERTION COMPLETED ==========")
        print("Total chunks:", len(chunks))
        print("Inserted:", inserted)
        print("Skipped:", skipped)

        cursor.close()

    except Exception as e:

        connection.rollback()
        print("Insertion failed:", e)
        raise

    finally:

        connection.close()

if __name__ == "__main__":
    insert_chunks()