import json

from pgvector.psycopg2 import register_vector
from pgvector import Vector

from app.database.db_connection import get_db_connection
from app.retrieval.query_embedder import generate_query_embedding


DEFAULT_TOP_K = 5


# ==========================================
# RETRIEVE SIMILAR CHUNKS
# ==========================================

def retrieve_similar_chunks(
    query_embedding,
    top_k=DEFAULT_TOP_K,
    content_type=None
):

    """
    Retrieve relevant document chunks using
    PostgreSQL vector similarity search.

    Supports:
        - Text
        - Tables
        - Images
        - Graphs
        - Visual descriptions
    """

    if query_embedding is None or len(query_embedding) == 0:

        raise ValueError("Query embedding cannot be empty.")

    if len(query_embedding) != 1536:

        raise ValueError(
            f"Expected 1536 dimensions, got {len(query_embedding)}"
        )

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:

        raise ValueError("top_k must be a positive integer.")

    connection = get_db_connection()

    try:

        register_vector(connection)

        cursor = connection.cursor()

        query_vector = Vector(query_embedding)

        sql = """
            SELECT
                chunk_id,
                document_name,
                page_number,
                content_type,
                content,
                image_path,
                metadata,

                1 - (embedding <=> %s) AS similarity

            FROM public.document_chunks

            WHERE embedding IS NOT NULL
        """

        params = [query_vector]

        if content_type:

            sql += """
                AND content_type = %s
            """

            params.append(content_type)

        sql += """
            ORDER BY embedding <=> %s

            LIMIT %s;
        """

        params.extend([query_vector, top_k])

        cursor.execute(sql, params)

        rows = cursor.fetchall()

        results = []

        for row in rows:

            results.append({

                "chunk_id": row[0],

                "document_name": row[1],

                "page_number": row[2],

                "content_type": row[3],

                "content": row[4],

                "image_path": row[5],

                "metadata": row[6],

                "similarity": float(row[7])

            })

        cursor.close()

        return results

    except Exception as error:

        print(f"Retrieval error: {error}")

        raise

    finally:

        connection.close()


# ==========================================
# RETRIEVE USING ACTUAL USER QUESTION
# ==========================================

def retrieve_by_query(
    query,
    top_k=DEFAULT_TOP_K,
    content_type=None
):

    """
    Complete retrieval pipeline:

    1. Accept user question.
    2. Generate query embedding.
    3. Search PostgreSQL.
    4. Return relevant chunks.
    """

    if not isinstance(query, str) or not query.strip():

        raise ValueError("User query cannot be empty.")

    print("\n========== QUERY RETRIEVAL STARTED ==========\n")

    print("User Query:", query)

    # STEP 1: Generate query embedding

    print("\nGenerating query embedding...")

    query_embedding = generate_query_embedding(query)

    print(
        f"Query embedding generated: {len(query_embedding)} dimensions"
    )

    # STEP 2: Retrieve relevant chunks

    print("\nSearching PostgreSQL...")

    results = retrieve_similar_chunks(

        query_embedding=query_embedding,

        top_k=top_k,

        content_type=content_type

    )

    print(f"\nRetrieved {len(results)} chunks.")

    print("\n========== RETRIEVAL COMPLETED ==========\n")

    return results


# ==========================================
# TEST ACTUAL USER QUERY
# ==========================================

def test_real_query_retrieval():

    print("\n========== REAL QUERY RETRIEVAL TEST ==========\n")

    test_query = (
        "What are the global renewable energy "
        "capacity projections for 2050?"
    )

    results = retrieve_by_query(

        query=test_query,

        top_k=5

    )

    print("\n========== RETRIEVED RESULTS ==========")

    for index, result in enumerate(results, start=1):

        print(f"\nResult {index}")

        print("Chunk ID:", result["chunk_id"])

        print("Document:", result["document_name"])

        print("Page:", result["page_number"])

        print("Content Type:", result["content_type"])

        print("Similarity:", round(result["similarity"], 4))

        print("Image Path:", result["image_path"])

        print("Metadata:", result["metadata"])

        print(
            "Content:",
            str(result["content"])[:300]
        )

    print("\nTotal Retrieved:", len(results))

    print("\n==============================================")


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    test_real_query_retrieval()