from pathlib import Path
import json
import pymupdf


def inspect_pdf(pdf_path: str) -> dict:
    """
    Extract text from a PDF page by page.

    Also render every page as an image for later
    multimodal processing.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    output_dir = Path("data/processed") / pdf_path.stem
    pages_dir = output_dir / "pages"

    pages_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "filename": pdf_path.name,
        "total_pages": 0,
        "pages": []
    }

    with pymupdf.open(pdf_path) as document:

        if document.needs_pass:
            raise ValueError(
                "This PDF is password-protected."
            )

        result["total_pages"] = len(document)

        for page_index, page in enumerate(document):

            page_number = page_index + 1

            # Extract page text
            text = page.get_text("text")

            # Render page as an image
            image_path = (
                pages_dir / f"page_{page_number}.png"
            )

            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(1.5, 1.5),
                alpha=False
            )

            pixmap.save(str(image_path))

            result["pages"].append({
                "page_number": page_number,
                "text": text,
                "image_path": str(image_path),
                "has_extracted_text": bool(text.strip())
            })

    return result


if __name__ == "__main__":

    pdf_path = input("Enter your PDF path: ").strip()
    pdf_path = pdf_path.strip('"')

    try:
        data = inspect_pdf(pdf_path)

        print("\nPDF processing successful!")
        print("Filename:", data["filename"])
        print("Total pages:", data["total_pages"])

        for page in data["pages"]:

            print(f"\n--- Page {page['page_number']} ---")

            print(
                page["text"][:300]
                or "[No text extracted]"
            )

            print("Page image:", page["image_path"])

        # Save extracted information
        output_json = (
            Path("data/processed")
            / Path(pdf_path).stem
            / "extracted_data.json"
        )

        output_json.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            output_json,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                data,
                file,
                indent=4,
                ensure_ascii=False
            )

        print(
            f"\nExtracted data saved successfully: {output_json}"
        )

    except Exception as error:
        print("PDF processing failed:", error)