from pathlib import Path
import json
import shutil

PDF_NAME = "Global_Renewable_Energy_Outlook_220MB"

OUTPUT_DIR = Path("data/processed") / PDF_NAME
PAGES_DIR = OUTPUT_DIR / "pages"
VISUALS_DIR = OUTPUT_DIR / "visuals"

# Pages identified during our earlier PDF inspection

VECTOR_PAGES = [3, 5, 7, 8, 10, 11]

def extract_vector_pages():

    VISUALS_DIR.mkdir(parents=True, exist_ok=True)

    metadata = []

    print("\n----- VECTOR VISUAL EXTRACTION STARTED -----")

    for page_number in VECTOR_PAGES:

        source_path = PAGES_DIR / f"page_{page_number}.png"

        if not source_path.exists():
            print(f"Missing page image: {source_path}")
            continue

        output_name = f"page_{page_number}_vector_page.png"
        output_path = VISUALS_DIR / output_name

        shutil.copy2(source_path, output_path)

        metadata.append({
            "visual_id": f"page_{page_number}_vector",
            "page_number": page_number,
            "visual_type": "vector_page",
            "image_path": str(output_path),
            "description": (
                "Full-page image preserved for visual analysis "
                "of vector-based charts, diagrams, and drawings."
            )
        })

        print(f"Preserved vector page: {page_number}")

    metadata_path = OUTPUT_DIR / "vector_visuals.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)

    print("\n----- EXTRACTION COMPLETED -----")
    print("Total pages preserved:", len(metadata))
    print("Metadata saved:", metadata_path)

if __name__ == "__main__":
    extract_vector_pages()