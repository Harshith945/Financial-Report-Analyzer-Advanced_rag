
import json
import re
import time

from pathlib import Path
from collections import defaultdict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

from model_cache import get_embedding_model


class FinancialTextChunker:

    def __init__(self, extraction_path=None):

        self.input_path = (
            Path(extraction_path)
            if extraction_path
            else None
        )

        self.output_root = Path("data/processed")

        # Reuse the shared cached embedding model.
        self.embeddings = get_embedding_model()

        # Semantic chunker for narrative text.
        self.semantic_splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
            buffer_size=1
        )

        # Recursive splitter for financial tables.
        self.table_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    # --------------------------------
    # Load extracted report
    # --------------------------------

    def load_extracted_data(self):

        if self.input_path is None:
            raise ValueError(
                "Please provide an extraction JSON path."
            )

        if not self.input_path.exists():
            raise FileNotFoundError(
                f"Extraction file not found: {self.input_path}"
            )

        with open(
            self.input_path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    # --------------------------------
    # Prepare report metadata
    # --------------------------------

    def get_report_metadata(self, data):

        metadata = data.get("report_metadata", {})

        report_metadata = {}

        for key, value in metadata.items():

            if value is not None:
                report_metadata[key] = value

        return report_metadata

    # --------------------------------
    # Create common chunk metadata
    # --------------------------------

    def create_base_metadata(
        self,
        report_id,
        source,
        report_metadata
    ):

        metadata = {
            "report_id": report_id,
            "source": source
        }

        metadata.update(report_metadata)

        return metadata

    # --------------------------------
    # Text cleaning
    # --------------------------------

    def clean_text(self, text):

        text = str(text).strip()

        text = re.sub(r"\s+", " ", text)

        return text

    # --------------------------------
    # Create narrative documents
    # --------------------------------

    def create_text_documents(
        self,
        text_elements,
        base_metadata
    ):

        page_blocks = defaultdict(list)

        current_sections = {}

        for item in text_elements:

            text = self.clean_text(
                item.get("text", "")
            )

            if not text:
                continue

            if text.lower() == "table of contents":
                continue

            if re.fullmatch(r"\d+", text):
                continue

            page_number = item.get("page_number")

            content_type = item.get(
                "content_type",
                "TextItem"
            )

            if page_number not in current_sections:
                current_sections[page_number] = "General"

            if content_type == "SectionHeaderItem":
                current_sections[page_number] = text

            section = current_sections[page_number]

            page_blocks[
                (page_number, section)
            ].append(text)

        documents = []

        for (page_number, section), texts in page_blocks.items():

            combined_text = "\n".join(texts)

            metadata = base_metadata.copy()

            metadata.update({
                "section": section,
                "content_type": "text"
            })

            if page_number is not None:
                metadata["page_number"] = page_number

            documents.append(
                Document(
                    page_content=combined_text,
                    metadata=metadata
                )
            )

        return documents

    # --------------------------------
    # Convert financial table to text
    # --------------------------------

    def convert_table_to_text(self, table):

        page_number = table.get("page_number")

        table_name = (
            f"Financial Table {table['table_id']}"
        )

        columns = table.get("columns", [])
        rows = table.get("rows", [])

        periods = [
            self.clean_text(column)
            for column in columns[1:]
        ]

        lines = [
            f"Financial statement: {table_name}",
            f"PDF page number: {page_number}",
            "Reporting periods: " + " | ".join(periods)
        ]

        current_category = None

        for row in rows:

            if not row:
                continue

            label = self.clean_text(row[0])

            if not label:
                continue

            values = [
                self.clean_text(value)
                for value in row[1:]
            ]

            if not any(values):

                current_category = label

                lines.append(
                    f"\nCategory: {current_category}"
                )

                continue

            row_details = []

            for index, value in enumerate(values):

                if not value:
                    continue

                period = (
                    periods[index]
                    if index < len(periods)
                    else f"Period {index + 1}"
                )

                row_details.append(
                    f"{period}: {value}"
                )

            if row_details:

                line = (
                    f"{label}: "
                    + "; ".join(row_details)
                )

                lines.append(line)

        return "\n".join(lines), table_name

    # --------------------------------
    # Create table documents
    # --------------------------------

    def create_table_documents(
        self,
        tables,
        base_metadata
    ):

        documents = []

        for table in tables:

            table_text, table_name = (
                self.convert_table_to_text(table)
            )

            metadata = base_metadata.copy()

            metadata.update({
                "section": table_name,
                "table_id": table["table_id"],
                "content_type": "financial_table"
            })

            page_number = table.get("page_number")

            if page_number is not None:
                metadata["page_number"] = page_number

            documents.append(
                Document(
                    page_content=table_text,
                    metadata=metadata
                )
            )

        return documents

    # --------------------------------
    # Remove unwanted and duplicate text
    # --------------------------------

    def remove_unwanted_text_documents(
        self,
        documents,
        company
    ):

        cleaned_documents = []

        seen = set()

        company_name = (
            self.clean_text(company).lower()
            if company
            else None
        )

        for doc in documents:

            text = " ".join(
                doc.page_content.split()
            )

            if company_name and text.lower() == company_name:
                continue

            key = (
                text,
                doc.metadata.get("report_id"),
                doc.metadata.get("content_type"),
                doc.metadata.get("section")
            )

            if key in seen:
                continue

            seen.add(key)

            cleaned_documents.append(doc)

        return cleaned_documents

    # --------------------------------
    # Create chunks
    # --------------------------------

    def create_chunks(
        self,
        text_documents,
        table_documents
    ):

        # Semantic chunking for narrative text.
        semantic_chunks = (
            self.semantic_splitter.split_documents(
                text_documents
            )
        )

        # Recursive chunking for financial tables.
        table_chunks = (
            self.table_splitter.split_documents(
                table_documents
            )
        )

        print(
            "Semantic text chunks:",
            len(semantic_chunks)
        )

        print(
            "Financial table chunks:",
            len(table_chunks)
        )

        return semantic_chunks + table_chunks

    # --------------------------------
    # Save chunks separately per report
    # --------------------------------

    def save_chunks(self, chunks, report_id):

        output_dir = self.output_root / report_id

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = output_dir / "text_chunks.json"

        output_data = []

        for index, chunk in enumerate(chunks):

            output_data.append({

                "chunk_id": index,

                "report_id": report_id,

                "text": chunk.page_content,

                "metadata": chunk.metadata

            })

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                output_data,
                file,
                indent=4,
                ensure_ascii=False
            )

        print(
            "\nChunks saved to:",
            output_path
        )

        return output_path

    # --------------------------------
    # Main pipeline
    # --------------------------------

    def run(self, extraction_path=None):

        if extraction_path is not None:
            self.input_path = Path(extraction_path)

        total_start = time.perf_counter()

        data = self.load_extracted_data()

        report_id = data["report_id"]

        source = data["source"]

        report_metadata = self.get_report_metadata(data)

        base_metadata = self.create_base_metadata(
            report_id,
            source,
            report_metadata
        )

        print("\nPROCESSING REPORT")
        print("-----------------")

        print("Report ID:", report_id)
        print("Source:", source)

        print("Company:", report_metadata.get("company"))
        print("Ticker:", report_metadata.get("ticker"))
        print("Report type:", report_metadata.get("report_type"))
        print("Fiscal year:", report_metadata.get("fiscal_year"))
        print("Fiscal quarter:", report_metadata.get("fiscal_quarter"))

        # --------------------------------
        # Process narrative text
        # --------------------------------

        text_documents = self.create_text_documents(
            data["text_elements"],
            base_metadata
        )

        text_documents = self.remove_unwanted_text_documents(
            text_documents,
            report_metadata.get("company")
        )

        # --------------------------------
        # Process financial tables
        # --------------------------------

        table_documents = self.create_table_documents(
            data["tables"],
            base_metadata
        )

        print(
            "\nText documents after cleanup:",
            len(text_documents)
        )

        print(
            "Table documents:",
            len(table_documents)
        )

        # --------------------------------
        # Apply chunking strategies
        # --------------------------------

        chunking_start = time.perf_counter()

        chunks = self.create_chunks(
            text_documents,
            table_documents
        )

        chunking_time = (
            time.perf_counter() - chunking_start
        )

        print("\nTotal chunks:", len(chunks))

        print(
            f"Chunking time: {chunking_time:.2f} seconds"
        )

        # --------------------------------
        # Save report-specific chunks
        # --------------------------------

        save_start = time.perf_counter()

        output_path = self.save_chunks(
            chunks,
            report_id
        )

        save_time = time.perf_counter() - save_start

        print(
            f"Chunk saving time: {save_time:.2f} seconds"
        )

        # --------------------------------
        # Preview a narrative chunk
        # --------------------------------

        for chunk in chunks:

            if chunk.metadata["content_type"] == "text":

                print("\nSEMANTIC TEXT CHUNK PREVIEW")
                print("=" * 60)

                print(chunk.page_content[:1000])
                print(chunk.metadata)

                break

        # --------------------------------
        # Preview a financial table chunk
        # --------------------------------

        for chunk in chunks:

            if chunk.metadata["content_type"] == "financial_table":

                print("\nFINANCIAL TABLE CHUNK PREVIEW")
                print("=" * 60)

                print(chunk.page_content[:1000])
                print(chunk.metadata)

                break

        total_time = time.perf_counter() - total_start

        print("\nCHUNKING TIME SUMMARY")
        print("=" * 45)

        print(f"Chunking: {chunking_time:.2f} seconds")
        print(f"Saving: {save_time:.2f} seconds")
        print(f"Total: {total_time:.2f} seconds")

        print("Output file:", output_path)

        return chunks


# --------------------------------
# Test the chunking pipeline
# --------------------------------

if __name__ == "__main__":

    extraction_files = list(
        Path("data/extracted").glob(
            "*/financial_extraction.json"
        )
    )

    if not extraction_files:
        raise FileNotFoundError(
            "No financial extraction files found."
        )

    latest_file = max(
        extraction_files,
        key=lambda path: path.stat().st_mtime
    )

    print("Using extraction:", latest_file)

    chunker = FinancialTextChunker(
        extraction_path=latest_file
    )

    chunker.run()

