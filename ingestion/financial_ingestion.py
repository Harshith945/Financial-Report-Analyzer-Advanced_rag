
import json
import uuid
import time

from pathlib import Path
from functools import lru_cache

from docling.document_converter import DocumentConverter

from ingestion.metadata_extractor import FinancialMetadataExtractor


@lru_cache(maxsize=1)
def get_document_converter():
    """
    Initialize Docling converter once and reuse it
    for subsequent PDF uploads.
    """

    print("\nInitializing Docling converter...")

    converter = DocumentConverter()

    print("Docling converter initialized!")

    return converter


class FinancialDocumentIngestion:

    def __init__(self, pdf_path, report_id=None):

        self.pdf_path = Path(pdf_path)

        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found: {self.pdf_path}"
            )

        if self.pdf_path.suffix.lower() != ".pdf":
            raise ValueError("Only PDF files are supported.")

        # Generate a unique ID for each report.
        self.report_id = (
            report_id or f"report_{uuid.uuid4().hex[:12]}"
        )

        # Reuse the cached Docling converter.
        self.converter = get_document_converter()

        self.metadata_extractor = FinancialMetadataExtractor()

    def process_document(self):

        print("\nStarting document processing...")
        print("Source:", self.pdf_path.name)
        print("Report ID:", self.report_id)

        # --------------------------------
        # Convert PDF using Docling
        # --------------------------------

        print("\nConverting PDF using Docling...")

        start_time = time.perf_counter()

        result = self.converter.convert(
            str(self.pdf_path)
        )

        docling_time = time.perf_counter() - start_time

        print("Document processing completed!")

        print(
            f"Docling conversion time: {docling_time:.2f} seconds"
        )

        document = result.document

        extracted_data = {

            "report_id": self.report_id,

            "source": self.pdf_path.name,

            "total_pages": len(document.pages),

            "report_metadata": {
                "company": None,
                "ticker": None,
                "report_type": None,
                "fiscal_year": None,
                "fiscal_quarter": None,
                "period_end": None
            },

            "text_elements": [],

            "tables": []
        }

        # --------------------------------
        # Extract text elements
        # --------------------------------

        print("\nExtracting text elements...")

        text_start = time.perf_counter()

        for item in document.texts:

            page_number = None

            if item.prov:
                page_number = item.prov[0].page_no

            extracted_data["text_elements"].append({

                "report_id": self.report_id,

                "text": item.text,

                "page_number": page_number,

                "content_type": type(item).__name__

            })

        text_time = time.perf_counter() - text_start

        print(
            "Text elements extracted:",
            len(extracted_data["text_elements"])
        )

        print(
            f"Text extraction time: {text_time:.2f} seconds"
        )

        # --------------------------------
        # Extract tables
        # --------------------------------

        print("\nExtracting financial tables...")

        table_start = time.perf_counter()

        for index, table in enumerate(document.tables):

            page_number = None

            if table.prov:
                page_number = table.prov[0].page_no

            try:

                dataframe = table.export_to_dataframe(
                    doc=document
                )

                extracted_data["tables"].append({

                    "report_id": self.report_id,

                    "table_id": index + 1,

                    "page_number": page_number,

                    "rows": dataframe.fillna("").values.tolist(),

                    "columns": dataframe.columns.tolist()

                })

            except Exception as error:

                print(
                    f"Table {index + 1} extraction failed: {error}"
                )

        table_time = time.perf_counter() - table_start

        print(
            "Tables extracted:",
            len(extracted_data["tables"])
        )

        print(
            f"Table extraction time: {table_time:.2f} seconds"
        )

        # --------------------------------
        # Extract report metadata
        # --------------------------------

        print("\nIdentifying report metadata...")

        metadata_start = time.perf_counter()

        metadata = self.metadata_extractor.extract_metadata(
            extracted_data
        )

        extracted_data["report_metadata"] = metadata

        metadata_time = time.perf_counter() - metadata_start

        print("\nMetadata identified:")

        for key, value in metadata.items():

            print(f"{key}: {value}")

        print(
            f"Metadata extraction time: {metadata_time:.2f} seconds"
        )

        # --------------------------------
        # Processing summary
        # --------------------------------

        print("\nPROCESSING TIME SUMMARY")
        print("=" * 45)

        print(f"Docling conversion: {docling_time:.2f} seconds")
        print(f"Text extraction: {text_time:.2f} seconds")
        print(f"Table extraction: {table_time:.2f} seconds")
        print(f"Metadata extraction: {metadata_time:.2f} seconds")

        return extracted_data

    # --------------------------------
    # Save extraction separately
    # --------------------------------

    def save_results(self, extracted_data):

        output_dir = (
            Path("data/extracted") / self.report_id
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_file = (
            output_dir / "financial_extraction.json"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                extracted_data,
                file,
                indent=4,
                ensure_ascii=False
            )

        print(
            f"\nExtraction saved to: {output_file}"
        )

        return output_file


# --------------------------------
# Test the ingestion pipeline
# --------------------------------

if __name__ == "__main__":

    pdf_path = (
        "data/raw_pdfs/tsla-20260331-gen_copy.pdf"
    )

    ingestion = FinancialDocumentIngestion(pdf_path)

    data = ingestion.process_document()

    saved_file = ingestion.save_results(data)

    print("\nEXTRACTION SUMMARY")
    print("------------------")

    print("Report ID:", data["report_id"])

    print("Source:", data["source"])

    print("Pages:", data["total_pages"])

    print(
        "Text elements:",
        len(data["text_elements"])
    )

    print(
        "Tables:",
        len(data["tables"])
    )

    print(
        "Company:",
        data["report_metadata"]["company"]
    )

    print(
        "Reporting period:",
        data["report_metadata"]["fiscal_quarter"],
        data["report_metadata"]["fiscal_year"]
    )

    print("Saved file:", saved_file)



