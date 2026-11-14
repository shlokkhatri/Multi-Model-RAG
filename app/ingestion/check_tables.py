from pathlib import Path
import pymupdf

PDF_PATH = Path(
    "data/Global_Renewable_Energy_Outlook_220MB.pdf"
)

# PDF pages containing tables, using 1-based page numbers.
TABLE_PAGES = [3, 7, 8, 10, 11]

with pymupdf.open(PDF_PATH) as document:
    for page_number in TABLE_PAGES:
        page = document[page_number - 1]

        print("\n" + "=" * 70)
        print(f"PDF PAGE {page_number}")
        print("=" * 70)

        words = page.get_text("words")
        rows = {}

        for word in words:
            x0, y0, x1, y1, text = word[:5]

            # Group words by their approximate vertical position.
            y_key = round(y0)

            rows.setdefault(y_key, []).append({
                "x": x0,
                "text": text
            })

        for y in sorted(rows):
            row = sorted(
                rows[y],
                key=lambda item: item["x"]
            )

            print(
                f"Y={y:>3}: "
                + " | ".join(
                    f"{item['x']:.1f}:{item['text']}"
                    for item in row
                )
            )
