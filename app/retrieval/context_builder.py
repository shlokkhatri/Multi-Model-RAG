import json

from pathlib import Path

from app.retrieval.vector_retriever import retrieve_by_query


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def normalize_metadata(metadata):

    """
    Ensure metadata is always a dictionary.
    """

    if isinstance(metadata, dict):
        return metadata

    if isinstance(metadata, str):

        try:
            return json.loads(metadata)

        except json.JSONDecodeError:
            return {"raw_metadata": metadata}

    return {}


def resolve_image_path(image_path):

    """
    Resolve image paths relative to project root.
    """

    if not image_path:
        return None

    path = Path(image_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if path.exists():
        return str(path.resolve())

    return None


# ==========================================
# BUILD MULTIMODAL CONTEXT
# ==========================================

def build_multimodal_context(retrieved_chunks):

    """
    Organize retrieved chunks into structured
    multimodal context.

    Supports:
        - Text
        - Tables
        - Embedded images
        - Graphs
        - Charts
        - Visual descriptions
    """

    if not retrieved_chunks:

        raise ValueError(
            "No retrieved chunks provided."
        )

    context = {

        "total_chunks": len(retrieved_chunks),

        "pages": {},

        "text_chunks": [],

        "table_chunks": [],

        "visual_chunks": [],

        "image_paths": [],

        "text_context": ""

    }

    seen_images = set()

    for chunk in retrieved_chunks:

        chunk_id = chunk.get("chunk_id")

        page_number = chunk.get("page_number")

        content_type = str(
            chunk.get("content_type", "unknown")
        ).lower()

        content = str(
            chunk.get("content") or ""
        )

        image_path = chunk.get("image_path")

        metadata = normalize_metadata(
            chunk.get("metadata")
        )

        similarity = chunk.get("similarity", 0.0)

        # ==================================
        # CHUNK INFORMATION
        # ==================================

        chunk_info = {

            "chunk_id": chunk_id,

            "page_number": page_number,

            "content_type": content_type,

            "content": content,

            "image_path": image_path,

            "metadata": metadata,

            "similarity": similarity

        }

        # ==================================
        # PAGE ORGANIZATION
        # ==================================

        page_key = str(
            page_number if page_number is not None
            else "unknown"
        )

        if page_key not in context["pages"]:

            context["pages"][page_key] = {

                "page_number": page_number,

                "chunks": []

            }

        context["pages"][page_key]["chunks"].append(
            chunk_info
        )

        # ==================================
        # CLASSIFY CONTENT
        # ==================================

        if content_type in (
            "text",
            "paragraph",
            "document_text"
        ):

            context["text_chunks"].append(
                chunk_info
            )

        elif content_type in (
            "table",
            "tabular",
            "table_description"
        ):

            context["table_chunks"].append(
                chunk_info
            )

        elif content_type in (
            "embedded_image",
            "image",
            "graph",
            "chart",
            "visual",
            "visual_description"
        ):

            context["visual_chunks"].append(
                chunk_info
            )

        else:

            # Preserve unknown content types.
            context["text_chunks"].append(
                chunk_info
            )

        # ==================================
        # IMAGE PATH COLLECTION
        # ==================================

        if image_path:

            resolved_path = resolve_image_path(
                image_path
            )

            if resolved_path:

                if resolved_path not in seen_images:

                    context["image_paths"].append({

                        "chunk_id": chunk_id,

                        "page_number": page_number,

                        "image_path": resolved_path,

                        "content_type": content_type,

                        "description": content

                    })

                    seen_images.add(resolved_path)

                else:

                    print(
                        f"Duplicate image skipped: {image_path}"
                    )

            else:

                print(
                    f"WARNING: Image not found: {image_path}"
                )

    # ==========================================
    # GENERATE TEXT CONTEXT
    # ==========================================

    context_parts = []

    for page_key, page_data in context["pages"].items():

        context_parts.append(
            f"\n========== PAGE {page_key} ==========\n"
        )

        for chunk in page_data["chunks"]:

            content_type = chunk["content_type"]

            context_parts.append(

                f"\nContent Type: {content_type}\n"

                f"Chunk ID: {chunk['chunk_id']}\n"

                f"Similarity: {chunk['similarity']}\n"

                f"Content:\n{chunk['content']}\n"

            )

            if chunk["image_path"]:

                context_parts.append(
                    f"Image Reference: {chunk['image_path']}\n"
                )

    context["text_context"] = "\n".join(
        context_parts
    )

    return context


# ==========================================
# DISPLAY CONTEXT SUMMARY
# ==========================================

def display_context_summary(context):

    print("\n========== CONTEXT BUILDER RESULTS ==========")

    print(
        "\nTotal Retrieved Chunks:",
        context["total_chunks"]
    )

    print(
        "Total Pages:",
        len(context["pages"])
    )

    print(
        "Text Chunks:",
        len(context["text_chunks"])
    )

    print(
        "Table Chunks:",
        len(context["table_chunks"])
    )

    print(
        "Visual Chunks:",
        len(context["visual_chunks"])
    )

    print(
        "Available Images:",
        len(context["image_paths"])
    )

    print("\n---------- PAGE DETAILS ----------")

    for page_number, page_data in context["pages"].items():

        print(
            f"\nPage {page_number}: "
            f"{len(page_data['chunks'])} chunks"
        )

        for chunk in page_data["chunks"]:

            print(
                f"  - {chunk['chunk_id']} "
                f"| {chunk['content_type']}"
            )

    print("\n---------- IMAGE DETAILS ----------")

    for image in context["image_paths"]:

        print(
            "\nImage:",
            image["image_path"]
        )

        print(
            "Page:",
            image["page_number"]
        )

    print("\n============================================")


# ==========================================
# TEST CONTEXT BUILDER
# ==========================================

def test_context_builder():

    print("\n========== CONTEXT BUILDER TEST ==========\n")

    query = (
        "What are the global renewable energy "
        "capacity projections for 2050?"
    )

    # STEP 1: Retrieve relevant chunks

    print("Step 1: Retrieving relevant chunks...")

    retrieved_chunks = retrieve_by_query(

        query=query,

        top_k=5

    )

    # STEP 2: Build context

    print("\nStep 2: Building multimodal context...")

    context = build_multimodal_context(
        retrieved_chunks
    )

    # STEP 3: Display results

    display_context_summary(context)

    # STEP 4: Preview context

    print("\n---------- CONTEXT PREVIEW ----------")

    print(
        context["text_context"][:2000]
    )

    print("\nSUCCESS: Context building completed!")


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    test_context_builder()