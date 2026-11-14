import os
import time

from pathlib import Path
from dotenv import load_dotenv

from google import genai
from google.genai import types


# PROJECT CONFIGURATION

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

MODEL_NAME = "gemini-embedding-001"

EMBEDDING_DIMENSION = 1536

MAX_RETRIES = 5

RETRY_DELAY = 2


# GEMINI CLIENT

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=API_KEY)


# GENERATE QUERY EMBEDDING

def generate_query_embedding(query: str) -> list:

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query cannot be empty.")

    query = query.strip()

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = client.models.embed_content(

                model=MODEL_NAME,

                contents=query,

                config=types.EmbedContentConfig(

                    task_type="RETRIEVAL_QUERY",

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
                f"\nQuery embedding attempt "
                f"{attempt}/{MAX_RETRIES} failed."
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


# TEST QUERY EMBEDDING

def main():

    print("\n---QUERY EMBEDDING TEST---\n")

    test_query = (
        "What are the global renewable energy "
        "capacity projections for 2050?"
    )

    print(f"Test Query: {test_query}")

    embedding = generate_query_embedding(test_query)

    print("\n========== RESULTS ==========")

    print(f"Model: {MODEL_NAME}")

    print(f"Embedding Dimensions: {len(embedding)}")

    print(f"First 5 Values: {embedding[:5]}")

    print("\nSUCCESS: Query embedding generated!")

if __name__ == "__main__":

    main()