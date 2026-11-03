
import pymupdf
from pathlib import Path
import json


class PDFParser:

    def __init__(self, pdf_path):
        self.pdf_path = Path(pdf_path)

        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found: {self.pdf_path}"
            )

    def extract_text(self):

        document = pymupdf.open(self.pdf_path)

        extracted_pages = []

        for page_number, page in enumerate(document, start=1):

            text = page.get_text("text")

            page_data = {
                "page_number": page_number,
                "text": text,
                "source": self.pdf_path.name
            }

            extracted_pages.append(page_data)

        document.close()

        return extracted_pages

    def save_extracted_text(self, extracted_pages):

        output_dir = Path("data/extracted")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "extracted_text.json"

        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(
                extracted_pages,
                file,
                indent=4,
                ensure_ascii=False
            )

        print(f"Extracted text saved to: {output_file}")


if __name__ == "__main__":

    pdf_path = "data/raw_pdfs/tsla-20260331-gen_copy.pdf"

    parser = PDFParser(pdf_path)

    pages = parser.extract_text()

    parser.save_extracted_text(pages)

    print("\nPDF Extraction Summary")
    print("----------------------")
    print("Total pages:", len(pages))

    for page in pages[:3]:
        print(f"\nPage {page['page_number']}")
        print(page["text"][:500])
