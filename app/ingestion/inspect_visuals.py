from pathlib import Path
import pymupdf
import json


def extract_visuals(pdf_path):

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    # Output directories
    output_dir = (
        Path("data/processed") / pdf_path.stem
    )

    visuals_dir = output_dir / "visuals"

    visuals_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    visuals_metadata = []

    with pymupdf.open(pdf_path) as document:

        print("\n----- VISUAL EXTRACTION STARTED -----")

        for page_index, page in enumerate(document):

            page_number = page_index + 1

            images = page.get_images(full=True)

            print(
                f"\nProcessing Page {page_number}..."
            )

            for image_index, image in enumerate(images):

                xref = image[0]

                # Extract original image
                image_data = document.extract_image(xref)

                image_bytes = image_data["image"]

                image_extension = image_data["ext"]

                image_width = image_data["width"]
                image_height = image_data["height"]

                # Get image location on PDF page
                rectangles = page.get_image_rects(xref)

                for occurrence, rect in enumerate(rectangles):

                    image_name = (
                        f"page_{page_number}_"
                        f"visual_{image_index + 1}_"
                        f"{occurrence + 1}."
                        f"{image_extension}"
                    )

                    image_path = visuals_dir / image_name

                    # Save original image
                    with open(image_path, "wb") as file:
                        file.write(image_bytes)

                    # Save metadata
                    visual_info = {

                        "visual_id": (
                            f"page_{page_number}_"
                            f"visual_{image_index + 1}_"
                            f"{occurrence + 1}"
                        ),

                        "page_number": page_number,

                        "visual_type": "embedded_image",

                        "image_path": str(image_path),

                        "width": image_width,

                        "height": image_height,

                        "format": image_extension,

                        "coordinates": {

                            "x0": rect.x0,
                            "y0": rect.y0,
                            "x1": rect.x1,
                            "y1": rect.y1

                        },

                        "xref": xref
                    }

                    visuals_metadata.append(visual_info)

                    print(
                        f"  Extracted: {image_name}"
                    )

        # Save metadata
        metadata_path = output_dir / "visuals.json"

        with open(
            metadata_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                visuals_metadata,
                file,
                indent=4,
                ensure_ascii=False
            )

        print("\n----- EXTRACTION COMPLETED -----")

        print(
            "Total images extracted:",
            len(visuals_metadata)
        )

        print(
            "Images saved at:",
            visuals_dir
        )

        print(
            "Metadata saved at:",
            metadata_path
        )

    return visuals_metadata


if __name__ == "__main__":

    pdf_path = input(
        "Enter PDF path: "
    ).strip().strip('"')

    try:

        extract_visuals(pdf_path)

    except Exception as error:

        print(
            "\nVisual extraction failed:",
            error
        )