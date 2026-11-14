from pathlib import Path
import json
import pymupdf


PDF_PATH = Path("data/Global_Renewable_Energy_Outlook_220MB.pdf")

OUTPUT_PATH = Path(
    "data/processed/Global_Renewable_Energy_Outlook_220MB/"
    "tables.json"
)


def get_page_rows(page):
    """Group PDF words into rows based on their vertical coordinates."""
    rows = {}

    for word in page.get_text("words"):
        x0, y0, x1, y1, text = word[:5]

        row_key = round(y0)

        rows.setdefault(row_key, []).append({
            "x": x0,
            "text": text
        })

    for row in rows.values():
        row.sort(key=lambda item: item["x"])

    return rows


def get_text_between(row, start_x, end_x):
    """Extract words whose starting X coordinates fall in a column."""
    return " ".join(
        item["text"]
        for item in row
        if start_x <= item["x"] < end_x
    ).strip()


def extract_table(
    page,
    title,
    page_number,
    columns,
    start_y,
    end_y,
    skip_first_row=False
):
    """
    Extract a table using predefined column boundaries.

    columns format:
        [{"name": "Column name", "start": 70, "end": 180}, ...]
    """
    rows = get_page_rows(page)
    extracted_rows = []

    for y in sorted(rows):
        if not start_y <= y <= end_y:
            continue

        source_row = rows[y]

        values = [
            get_text_between(
                source_row,
                column["start"],
                column["end"]
            )
            for column in columns
        ]

        # Ignore rows that contain no data.
        if not any(values):
            continue

        # Ignore repeated header rows.
        if values == [column["name"] for column in columns]:
            continue

        extracted_rows.append({
            column["name"]: value
            for column, value in zip(columns, values)
        })

    # Skip the first extracted row only when required.
    if skip_first_row and extracted_rows:
        extracted_rows = extracted_rows[1:]

    return {
        "title": title,
        "page": page_number,
        "columns": [column["name"] for column in columns],
        "rows": extracted_rows
    }


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with pymupdf.open(PDF_PATH) as document:
        tables = []

        # TABLE 1 — PDF page 3
        tables.append(extract_table(
            document[2],
            "Table 1. Key indicators",
            3,
            [
                {"name": "Indicator", "start": 70, "end": 290},
                {"name": "2024", "start": 290, "end": 360},
                {"name": "2025", "start": 360, "end": 440},
                {"name": "Change", "start": 440, "end": 530},
            ],
            510, 660
        ))

        # TABLE 2 — PDF page 5
        tables.append(extract_table(
            document[4],
            "Table 2. Installed capacity and projected growth",
            5,
            [
                {"name": "Technology", "start": 70, "end": 180},
                {"name": "2015_GW", "start": 180, "end": 245},
                {"name": "2020_GW", "start": 245, "end": 315},
                {"name": "2025_GW", "start": 315, "end": 383},
                {"name": "2030E_GW", "start": 383, "end": 455},
                {"name": "CAGR_25_30", "start": 455, "end": 530},
            ],
            130, 235
        ))

        # TABLE 3 — PDF page 5
        tables.append(extract_table(
            document[4],
            "Table 3. Regional summary, 2025",
            5,
            [
                {"name": "Region", "start": 70, "end": 195},
                {"name": "Capacity_GW", "start": 195, "end": 270},
                {"name": "Share", "start": 270, "end": 335},
                {"name": "Top_technology", "start": 335, "end": 425},
                {"name": "Growth_outlook", "start": 425, "end": 540},
            ],
            290, 395
        ))

        # TABLE 4 — PDF page 7
        tables.append(extract_table(
            document[6],
            "Table 4. Levelized cost of electricity",
            7,
            [
                {"name": "Technology", "start": 90, "end": 245},
                {"name": "LCOE_2010_USD_MWh", "start": 245, "end": 315},
                {"name": "LCOE_2025_USD_MWh", "start": 315, "end": 380},
                {"name": "Change", "start": 380, "end": 450},
                {"name": "Capacity_factor", "start": 450, "end": 530},
            ],
            120, 255,
            skip_first_row=True
        ))

        # TABLE 5 — PDF page 8
        tables.append(extract_table(
            document[7],
            "Table 5. Investment by segment",
            8,
            [
                {"name": "Segment", "start": 85, "end": 195},
                {"name": "2020_USD_bn", "start": 195, "end": 235},
                {"name": "2021_USD_bn", "start": 235, "end": 282},
                {"name": "2022_USD_bn", "start": 282, "end": 329},
                {"name": "2023_USD_bn", "start": 329, "end": 376},
                {"name": "2024_USD_bn", "start": 376, "end": 423},
                {"name": "2025_USD_bn", "start": 423, "end": 470},
                {"name": "CAGR", "start": 470, "end": 530},
            ],
            430, 545,
            skip_first_row=True
        ))

        # TABLE 6 — PDF page 10
        tables.append(extract_table(
            document[9],
            "Table 6. Scenario comparison",
            10,
            [
                {"name": "Metric", "start": 70, "end": 250},
                {"name": "Stated_Policies", "start": 250, "end": 345},
                {"name": "Accelerated_Transition", "start": 345, "end": 455},
                {"name": "Net_Zero_2050", "start": 455, "end": 535},
            ],
            430, 565,
            skip_first_row=True
        ))

        # TABLE 7 — PDF page 11
        tables.append(extract_table(
            document[10],
            "Table 7. Principal risks",
            11,
            [
                {"name": "Risk", "start": 75, "end": 210},
                {"name": "Likelihood", "start": 210, "end": 275},
                {"name": "Impact", "start": 275, "end": 330},
                {"name": "Mitigation", "start": 330, "end": 540},
            ],
            130, 290
        ))

    output = {
        "source_pdf": PDF_PATH.name,
        "table_count": len(tables),
        "tables": tables
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print(f"\nExtracted {len(tables)} tables.")
    print(f"Saved to: {OUTPUT_PATH}\n")

    total_rows = 0

    for table in tables:
        row_count = len(table["rows"])
        total_rows += row_count

        print(
            f"{table['title']} | "
            f"PDF page {table['page']} | "
            f"{row_count} rows"
        )

        for row in table["rows"]:
            print("  ", row)

        print()

    print(f"TOTAL EXTRACTED ROWS: {total_rows}")


if __name__ == "__main__":
    main()