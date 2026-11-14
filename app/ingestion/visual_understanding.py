import os
import time
import json
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("Gemini's API KEY not Found")

client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-flash"
PDF_NAME = "Global_Renewable_Energy_Outlook_220MB"
OUTPUT_DIR  = Path("data/processed") / PDF_NAME
VISUALS_JSON = OUTPUT_DIR / "visuals.json"
VECTOR_JSON = OUTPUT_DIR / "vector_visuals.json"
OUTPUT_JSON = OUTPUT_DIR / "visual_descriptions.json"
EXTRACTED_JSON = OUTPUT_DIR / "extracted_data.json"

VISUAL_PROMPT = """
You are an expert document visual analyst.

Analyze the provided image carefully.

Identify the visual type:
- Graph or chart
- Table
- Diagram
- Map
- Photograph
- Infographic
- Other visual

Extract all relevant information visible in the image.

For graphs and charts:
- Identify the chart type.
- Explain the axes and units.
- Extract visible values and labels.
- Describe trends and comparisons.

For tables:
- Identify column headings.
- Extract visible rows and values.

For maps:
- Identify geographical regions.
- Explain the legend, colors, and patterns.
- Describe visible geographical differences.

For diagrams:
- Identify components.
- Explain connections and relationships.

For other images:
- Describe visible objects and their significance.

IMPORTANT RULES:
1. Use only information visible in the image.
2. Never invent values, labels, or facts.
3. If something is unreadable, explicitly mention it.
4. Preserve important numerical values.
5. Write a detailed description suitable for a RAG system.

Return only the description in plain text.
"""

def load_visual_metadata():

    all_visuals = []

    for json_path in [VISUALS_JSON, VECTOR_JSON]:

        if not json_path.exists():
            print(f"Metadata file not found: {json_path}")
            continue

        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        all_visuals.extend(data)

    return all_visuals

def load_page_context():

    if not EXTRACTED_JSON.exists():
        raise FileNotFoundError(
            f"Extracted data not found: {EXTRACTED_JSON}"
        )

    with open(EXTRACTED_JSON, "r", encoding="utf-8") as file:
        data = json.load(file)

    page_context = {}

    for page in data.get("pages", []):

        page_number = page.get("page_number")
        page_text = page.get("text", "")

        page_context[page_number] = page_text

    return page_context

def analyze_visuals(visual, page_context):

    image_path = Path(visual["image_path"])

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return None

    page_number = visual["page_number"]
    context = page_context.get(page_number, "")

    print(f"Analyzing: {image_path.name}")
    print(f"Page Number: {page_number}")

    if not context.strip():
        print(f"Warning: No extracted text found for page {page_number}")

    try:

        uploaded_file = client.files.upload(
            file=str(image_path)
        )

        prompt = f"""
You are an expert multimodal document analyst.

Analyze the provided map image using the extracted text
from the same PDF page.

MAP ANALYSIS RULES:

1. Identify the map's subject and geographical coverage.
2. Explain the legend, colors, and patterns.
3. Connect visible map information with the page text.
4. Identify relevant captions, headings, and statistics.
5. Describe geographical differences and patterns.
6. Preserve important numerical values.
7. Never invent missing information.
8. If the legend or meaning of a color is unavailable,
   explicitly mention it.

EXTRACTED TEXT FROM PAGE {page_number}:

{context}

Return a detailed description suitable for a multimodal RAG system.

Use only information supported by the image and extracted page text.
"""

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                prompt,
                uploaded_file
            ],
            config=types.GenerateContentConfig(
                temperature=0.2
            )
        )

        description = response.text

        return {
            "visual_id": visual["visual_id"],
            "page_number": page_number,
            "visual_type": visual["visual_type"],
            "image_path": str(image_path),
            "description": description
        }

    except Exception as error:

        print(f"Failed to analyze {image_path.name}: {error}")
        return None


def generate_visual_descriptions():

    visuals = load_visual_metadata()

    # Only process map images from pages 13 to 22.
    map_visuals = [
        visual for visual in visuals
        if 13 <= visual["page_number"] <= 22
        and "vector" not in visual["visual_type"].lower()
    ]

    print("\nTotal Map Visuals Found:", len(map_visuals))

    page_context = load_page_context()

    # Load existing descriptions.
    if OUTPUT_JSON.exists():

        with open(OUTPUT_JSON, "r", encoding="utf-8") as file:
            descriptions = json.load(file)

    else:
        descriptions = []

    # Preserve descriptions of all other visuals.
    descriptions = [
        item for item in descriptions
        if not (
            13 <= item["page_number"] <= 22
            and "vector" not in item["visual_type"].lower()
        )
    ]

    for index, visual in enumerate(map_visuals, start=1):

        print(f"\nProcessing Map: {index}/{len(map_visuals)}")

        result = analyze_visuals(
            visual,
            page_context
        )

        if result:

            descriptions.append(result)

            # Save immediately.
            with open(OUTPUT_JSON, "w", encoding="utf-8") as file:

                json.dump(
                    descriptions,
                    file,
                    indent=4,
                    ensure_ascii=False
                )

            print("Description updated:", visual["visual_id"])

        time.sleep(2)

    print("\n----- MAP UNDERSTANDING COMPLETED -----")
    print("Total descriptions:", len(descriptions))
    print("Saved at:", OUTPUT_JSON)

if __name__ == "__main__":
    generate_visual_descriptions()