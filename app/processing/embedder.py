import os
import json
import time

from pathlib import Path
from dotenv import load_dotenv

from google import genai
from google.genai import types

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PDF_NAME = "Global_Renewable_Energy_Outlook_220MB"

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / PDF_NAME
)

INPUT_FILE = PROCESSED_DIR / "unified_chunks.json"

OUTPUT_FILE = PROCESSED_DIR / "embedded_chunks.json"

CHECKPOINT_FILE = PROCESSED_DIR / "embedding_checkpoint.json"

MODEL_NAME = "gemini-embedding-001"

EMBEDDING_DIMENSION = 1536

MAX_RETRIES = 5

RETRY_DELAY = 2



load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=API_KEY)

# JSON UTILITIES

def load_json(file_path):

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data, file_path):

    temp_file = file_path.with_suffix(
        file_path.suffix + ".tmp"
    )

    with open(temp_file, "w", encoding="utf-8") as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )

    temp_file.replace(file_path)


# -----GENERATE EMBEDDING-----

def generate_embedding(content):

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = client.models.embed_content(

                model=MODEL_NAME,

                contents=content,

                config=types.EmbedContentConfig(

                    task_type="RETRIEVAL_DOCUMENT",

                    output_dimensionality=EMBEDDING_DIMENSION

                )
            )

            embedding = response.embeddings[0].values

            if len(embedding) != EMBEDDING_DIMENSION:

                raise ValueError(
                    f"Expected {EMBEDDING_DIMENSION} dimensions, "
                    f"got {len(embedding)}"
                )

            return embedding

        except Exception as error:

            print(
                f"\nEmbedding attempt {attempt}/{MAX_RETRIES} failed."
            )

            print(f"Error: {error}")

            if attempt == MAX_RETRIES:
                raise

            wait_time = min(
                RETRY_DELAY * (2 ** (attempt - 1)),
                30
            )

            print(f"Retrying in {wait_time} seconds...")

            time.sleep(wait_time)



def load_checkpoint():

    if not CHECKPOINT_FILE.exists():

        return {}

    checkpoint_data = load_json(CHECKPOINT_FILE)

    # Prevent reusing embeddings from another model
    if checkpoint_data.get("model") != MODEL_NAME:

        raise ValueError(
            "Checkpoint model does not match current embedding model."
        )

    if checkpoint_data.get("dimension") != EMBEDDING_DIMENSION:

        raise ValueError(
            "Checkpoint embedding dimension does not match."
        )

    print("\nExisting checkpoint found.")

    return checkpoint_data.get("chunks", {})


def save_checkpoint(embedded_chunks):

    checkpoint_data = {

        "model": MODEL_NAME,

        "dimension": EMBEDDING_DIMENSION,

        "chunks": embedded_chunks

    }

    save_json(
        checkpoint_data,
        CHECKPOINT_FILE
    )


# PROCESS CHUNKS

def embed_chunks():

    print("\n========== EMBEDDING GENERATION STARTED ==========\n")

    data = load_json(INPUT_FILE)

    chunks = data.get("chunks", [])

    if not chunks:

        raise ValueError(
            "No chunks found in unified_chunks.json"
        )

    # Load previous progress
    embedded_chunks = load_checkpoint()

    total_chunks = len(chunks)

    successful = 0

    skipped = 0

    print(f"Total chunks: {total_chunks}")

    print(f"Already embedded: {len(embedded_chunks)}")

    print("\nStarting embedding process...\n")

    for index, chunk in enumerate(chunks, start=1):

        chunk_id = chunk.get("chunk_id")

        content = chunk.get("content", "")

        if not chunk_id:

            print(f"Skipping chunk {index}: Missing chunk ID")

            continue

        if not isinstance(content, str) or not content.strip():

            print(f"Skipping {chunk_id}: Empty content")

            continue

        # Skip already processed chunks
        if chunk_id in embedded_chunks:

            skipped += 1

            print(
                f"[{index}/{total_chunks}] "
                f"Already embedded: {chunk_id}"
            )

            continue

        print(
            f"\n[{index}/{total_chunks}] "
            f"Processing: {chunk_id}"
        )

        print(
            f"Content type: {chunk.get('content_type')}"
        )

        print(
            f"Page: {chunk.get('page_number')}"
        )

        try:

            embedding = generate_embedding(content)

            # Preserve all original chunk information
            embedded_chunk = {

                **chunk,

                "embedding": embedding

            }

            embedded_chunks[chunk_id] = embedded_chunk

            # Save progress immediately
            save_checkpoint(embedded_chunks)

            successful += 1

            print(
                f"Embedding generated successfully. "
                f"Dimensions: {len(embedding)}"
            )

            time.sleep(0.3)

        except Exception as error:

            print(
                f"\nFailed to process {chunk_id}: {error}"
            )

            print(
                "Progress has been saved. "
                "You can run the script again to resume."
            )

            raise

    # Maintain original chunk order in final output
    final_chunks = [
        embedded_chunks[chunk["chunk_id"]]
        for chunk in chunks
        if chunk.get("chunk_id") in embedded_chunks
    ]

    output = {

        "document_name": data.get("document_name"),

        "total_chunks": len(final_chunks),

        "embedding_model": MODEL_NAME,

        "embedding_dimension": EMBEDDING_DIMENSION,

        "chunk_types": data.get("chunk_types", {}),

        "chunks": final_chunks

    }

    save_json(output, OUTPUT_FILE)

    print("\n========== EMBEDDING COMPLETED ==========")

    print(f"Total chunks: {total_chunks}")

    print(f"Newly embedded: {successful}")

    print(f"Skipped from checkpoint: {skipped}")

    print(f"Total embedded: {len(final_chunks)}")

    print(f"Embedding model: {MODEL_NAME}")

    print(f"Embedding dimensions: {EMBEDDING_DIMENSION}")

    print(f"\nOutput saved at:\n{OUTPUT_FILE}")

    print(
        f"\nCheckpoint saved at:\n{CHECKPOINT_FILE}"
    )


if __name__ == "__main__":
    embed_chunks()