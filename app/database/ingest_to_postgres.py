import json

from pathlib import Path

from app.database.schema import create_tables
from app.database.vector_store import insert_chunks


# Existing processed document directory

PROCESSED_DIR = Path(
    "data/processed/Global_Renewable_Energy_Outlook_220MB"
)

EMBEDDING_FILE = PROCESSED_DIR / "embedded_chunks.json"


def load_embeddings():

    print("\n========== LOADING EMBEDDINGS ==========\n")

    if not EMBEDDING_FILE.exists():

        raise FileNotFoundError(
            f"Embedding file not found: {EMBEDDING_FILE}"
        )

    with open(
        EMBEDDING_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    chunks = data.get("chunks", [])

    print("Document:", data.get("document_name"))
    print("Embedding model:", data.get("embedding_model"))
    print("Declared dimensions:", data.get("embedding_dimension"))
    print("Total chunks:", len(chunks))

    return data, chunks


def validate_embeddings(data, chunks):

    print("\n========== VALIDATING DATA ==========\n")

    if not chunks:
        raise ValueError("No chunks found!")

    if data.get("embedding_dimension") != 1536:
        raise ValueError("Unexpected embedding dimension!")

    chunk_ids = set()

    for index, chunk in enumerate(chunks, start=1):

        required_fields = [
            "chunk_id",
            "document_name",
            "content_type",
            "content",
            "embedding"
        ]

        for field in required_fields:

            if field not in chunk or chunk[field] is None:
                raise ValueError(
                    f"Chunk {index} is missing {field}"
                )

        if chunk["chunk_id"] in chunk_ids:
            raise ValueError(
                f"Duplicate chunk ID: {chunk['chunk_id']}"
            )

        chunk_ids.add(chunk["chunk_id"])

        embedding = chunk["embedding"]

        if len(embedding) != 1536:
            raise ValueError(
                f"Invalid dimensions in {chunk['chunk_id']}"
            )

        if not all(
            isinstance(value, (int, float))
            for value in embedding
        ):
            raise ValueError(
                f"Invalid embedding values in {chunk['chunk_id']}"
            )

    print("Validation successful!")
    print("Valid chunks:", len(chunks))
    print("Embedding dimensions: 1536")


def main():

    # 1. Load existing embeddings

    data, chunks = load_embeddings()

    # 2. Validate all chunks

    validate_embeddings(data, chunks)

    # 3. Verify database schema

    create_tables()

    # 4. Insert existing embeddings

    insert_chunks(chunks)

    print("\n========== PIPELINE FINISHED ==========")


if __name__ == "__main__":

    main()