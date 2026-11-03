
from pathlib import Path
from docling.document_converter import DocumentConverter


class DoclingParser:

    def __init__(self, pdf_path):
        self.pdf_path = Path(pdf_path)

        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found: {self.pdf_path}"
            )

        self.converter = DocumentConverter()

    def extract_document(self):

        print("Starting Docling document conversion...")

        result = self.converter.convert(str(self.pdf_path))

        document = result.document

        print("Document conversion completed!")

        return document

    def save_markdown(self, document):

        output_dir = Path("data/extracted")
        output_dir.mkdir(parents=True, exist_ok=True)

        markdown = document.export_to_markdown()

        output_file = output_dir / "docling_extracted.md"

        output_file.write_text(
            markdown,
            encoding="utf-8"
        )

        print(f"Markdown saved to: {output_file}")

        return markdown


if __name__ == "__main__":

    pdf_path = "data/raw_pdfs/tsla-20260331-gen_copy.pdf"

    parser = DoclingParser(pdf_path)

    document = parser.extract_document()

    markdown = parser.save_markdown(document)

    print("\nExtraction Summary")
    print("------------------")
    print("Extracted characters:", len(markdown))

    print("\nFirst 2000 characters:")
    print(markdown[:2000])
