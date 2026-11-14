import json
import re
from pathlib import Path
from typing import Any

# ----- Configuration -----

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_NAME = "Global_Renewable_Energy_Outlook_220MB"

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / PDF_NAME
)

TEXT_FILE = PROCESSED_DIR / "extracted_data.json"

TABLE_FILE = PROCESSED_DIR / "tables.json"

VISUAL_FILE = PROCESSED_DIR / "visual_descriptions.json"

OUTPUT_FILE = PROCESSED_DIR / "unified_chunks.json"

# Chunking configuration

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 500

# ----- JSON Loader -----

def load_json(file_path: Path) -> Any:

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

# ----- Text Chunking -----

def split_text (
        text: str,
        chunk_size:int = CHUNK_SIZE,
        overlap:int = CHUNK_OVERLAP,
):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be smaller then Chunk size"
        )
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size,len(text))

        if end < len(text):
            boundry = max(
                text.rfind(". ", start, end),
                text.rfind("? ", start, end),
                text.rfind("! ", start, end),
                text.rfind(" ", start, end)
            )

            if boundry > start:
                end = boundry + 1

        chunk = text[start:end].strip()

        if chunk :
            chunks.append(chunk)

        if end >= len(text) : 
            break

        start = max(end - overlap,start + 1)

    return chunks

# --- Process Text ---

def process_text(data,document_name):
    chunks = []

    pages = data.get("pages", [])
    
    document_name = data.get(
        "filename",
        PDF_NAME
    )


    for page in pages:
        page_number = page.get("page_number")

        page_text = page.get("text") or ""

        if not isinstance(page_text, str):
            page_text = str(page_text)

        if not page_text.strip():
            continue

        text_chunks = split_text(page_text)

        for index, text_chunk in enumerate(
            text_chunks,
            start = 1
        ):
            chunks.append({
                "chunk_id" : (f"text_page_{page_number}_{index}"),
                "document_name": document_name,
                "page_number" : page_number,
                "content_type" : "text", 
                "content" : text_chunk,
                "source_file" : "extracted_data.json",
                "metadata" : {
                    "chunk_index" : index,
                    "total_page_chunks" : len(text_chunks),
                    "has_extracted_text" : page.get(
                        "has_extracted_text",
                        True
                    )
                }
            })
    print(f"Text Chunk Created : {len(chunks)}")
    return chunks

# --- Process Tables ---

def process_table(data,document_name):
    chunks = []

    tables = data.get("tables", [])
    for index, table in enumerate(tables,start=1):
        page_number = table.get("page")

        title = table.get("title", f"Table {index}")

        columns = table.get("columns", [])

        rows = table.get("rows", [])

        content_parts = [

                f"Table: {title}",

                f"Page Number: {page_number}",

                "Columns: " + ", ".join(
                    str(column) for column in columns
                )

            ]

        content_parts.append("Table data : ")

        for row_index, row in enumerate(rows, start = 1):
            if isinstance(row,dict):
                row_values = [
                    f"{key} : {value}"
                    for key,value in row.items()
                ]
            elif isinstance(row,list) :
                row_values = [
                    f"{columns[i] if i < len(columns) else f'Column {i+1}'}: {value}"
                    for i,value in enumerate(row)
                ]
            else :
                row_values = [str(row)]

            content_parts.append(
                f"Row {row_index} : " + ", ".join(row_values)
            )
        table_content = "\n".join(content_parts)

        chunks.append({
            "chunk_id" : f"table_page_{page_number}_{index}",
            "document_name" : document_name,
            "page_number" : page_number,
            "content_type" : "table",
            "content" : table_content,
            "source_file" : "tables.json",
            "metadata" : {
                "table_id" : f"table_{index}",
                "title" : title,
                "columns" : columns,
                "total_row" : len(rows)
            }
        })

    print(f"Table Chunks created {len(chunks)}")
    return chunks

# --- Process Visuals ---

def process_visuals(data,document_name) :
    chunks = []

    for visual in data:
        visual_id = visual.get("visual_id")

        page_number = visual.get("page_number")

        if page_number is None:
            match = re.search(r"page_(\d+)", str(visual_id))

            if match:
                page_number = int(match.group(1))

        visual_type = visual.get("visual_type", "image")

        description = visual.get("description","")

        if not description.strip():
            continue

        if visual_type.lower() == "vector_page" :
            continue

        chunks.append({
            "chunk_id" : (f"visual_{page_number}_{visual_id}"),
            "document_name" : document_name,
            "content_type" : visual_type,
            "page_number": page_number,
            "content" : description,
            "source_file" : "visual_description.json",
            "metadata" : {
                "visual_id" : visual_id,
                "visual_type" : visual_type,
                "image_path" : visual.get("image_path")
            }
        })
    print(f"Visual Chunks created : {len(chunks)}")

    return chunks

# --- Create Unified Chunks ---

def create_unified_chunks():

    print("----STARTING UNIFIED CHUNKING----")

    print("\nLoading json files...........")

    text_data = load_json(TEXT_FILE)
    table_data = load_json(TABLE_FILE)
    visual_data = load_json(VISUAL_FILE)

    # Define document name
    document_name = text_data.get(
        "filename",
        f"{PDF_NAME}.pdf"
    )

    print("\nProcessing Text..........")
    text_chunks = process_text(text_data, document_name)

    print("\nProcessing Tables..........")
    table_chunks = process_table(table_data, document_name)

    print("\nProcessing Visuals..........")
    visual_chunks = process_visuals(visual_data, document_name)

    # Combine all chunks
    unified_chunks = (
        text_chunks
        + table_chunks
        + visual_chunks
    )

    # Sort by page number
    unified_chunks.sort(
        key=lambda chunk: (
            chunk.get("page_number") or 0,
            chunk.get("chunk_id", "")
        )
    )

    output = {
        "document_name": document_name,
        "total_chunks": len(unified_chunks),

        "chunk_types": {
            "text": len(text_chunks),
            "table": len(table_chunks),
            "visual": len(visual_chunks)
        },

        "chunks": unified_chunks
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n----UNIFIED CHUNKING COMPLETED----")

    print(f"Total Chunks: {len(unified_chunks)}")
    print(f"Text Chunks: {len(text_chunks)}")
    print(f"Table Chunks: {len(table_chunks)}")
    print(f"Visual Chunks: {len(visual_chunks)}")

    print(f"\nOutput saved at:\n{OUTPUT_FILE}")

if __name__ == "__main__" :
    create_unified_chunks(  )