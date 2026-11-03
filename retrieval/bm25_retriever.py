
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


class BM25Retriever:

    def __init__(self):

        self.processed_directory = Path("data/processed")

        self.documents = []
        self.bm25 = None

    def tokenize(self, text):
        """
        Extract financial numbers, percentages, and words.
        """

        pattern = (
            r"\$?\(?-?\d[\d,]*(?:\.\d+)?%?\)?"
            r"|[a-zA-Z]+(?:'[a-z]+)?"
        )

        return re.findall(pattern, text.lower())

    def get_report_files(self):
        """
        Find all report-specific chunk files.
        """

        report_files = sorted(
            self.processed_directory.glob("*/text_chunks.json")
        )

        if not report_files:
            raise FileNotFoundError(
                "No report chunk files found in data/processed/"
            )

        return report_files

    def load_chunks(self):
        """
        Load chunks from all reports and preserve metadata.
        """

        self.documents = []

        report_files = self.get_report_files()

        for chunks_path in report_files:

            with open(
                chunks_path,
                "r",
                encoding="utf-8"
            ) as file:

                chunks = json.load(file)

            for chunk in chunks:

                metadata = dict(chunk.get("metadata", {}))

                # Flatten nested report metadata if present.
                report_metadata = metadata.pop(
                    "report_metadata", {}
                )

                if isinstance(report_metadata, dict):
                    metadata.update(report_metadata)

                report_id = (
                    chunk.get("report_id")
                    or metadata.get("report_id")
                    or chunks_path.parent.name
                )

                metadata["report_id"] = str(report_id)
                metadata["chunk_id"] = chunk["chunk_id"]

                # Keep only Chroma-compatible scalar metadata.
                clean_metadata = {
                    key: value
                    for key, value in metadata.items()
                    if value is not None
                    and isinstance(value, (str, int, float, bool))
                }

                document = Document(
                    page_content=chunk["text"],
                    metadata=clean_metadata
                )

                self.documents.append(document)

        print("Report files loaded:", len(report_files))
        print("Total BM25 documents loaded:", len(self.documents))

    def build_index(self):
        """
        Build one BM25 index containing all reports.
        """

        if not self.documents:
            raise ValueError(
                "No documents loaded. Run load_chunks() first."
            )

        tokenized_documents = [
            self.tokenize(document.page_content)
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

        print("BM25 index created successfully!")

    def get_report_ids(self):
        """
        Return unique report IDs in the BM25 index.
        """

        return sorted({
            document.metadata.get("report_id")
            for document in self.documents
            if document.metadata.get("report_id")
        })

    def search(self, query, k=5, report_id=None):
        """
        Search all reports or restrict search to one report.
        """

        if self.bm25 is None:
            raise ValueError(
                "BM25 index is not initialized."
            )

        tokenized_query = self.tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)

        # Select eligible documents.
        candidate_indices = []

        for index, document in enumerate(self.documents):

            document_report_id = document.metadata.get("report_id")

            if (
                report_id is None
                or document_report_id == report_id
            ):
                candidate_indices.append(index)

        if not candidate_indices:
            return []

        # Rank only eligible documents.
        ranked_indices = sorted(
            candidate_indices,
            key=lambda index: scores[index],
            reverse=True
        )[:k]

        return [
            (self.documents[index], scores[index])
            for index in ranked_indices
        ]


if __name__ == "__main__":

    retriever = BM25Retriever()

    retriever.load_chunks()
    retriever.build_index()

    print("\nINDEXED REPORTS")
    print("=" * 60)

    for report_id in retriever.get_report_ids():
        print(report_id)

    query = (
        "Total revenues Three Months Ended "
        "March 31 2026 22387"
    )

    # Search across all reports.
    results = retriever.search(query, k=5)

    print("\nBM25 SEARCH RESULTS")
    print("=" * 60)

    for rank, (document, score) in enumerate(
        results,
        start=1
    ):

        print(f"\nRank: {rank}")
        print("BM25 Score:", round(float(score), 4))
        print("Report ID:", document.metadata.get("report_id"))
        print("Company:", document.metadata.get("company"))
        print("Ticker:", document.metadata.get("ticker"))
        print("Period:", document.metadata.get("period_end"))
        print("Page:", document.metadata.get("page_number"))
        print("Section:", document.metadata.get("section"))
        print("Content:")
        print(document.page_content[:500])

    # Test report-specific retrieval.
    if retriever.get_report_ids():

        selected_report = retriever.get_report_ids()[0]

        filtered_results = retriever.search(
            query,
            k=5,
            report_id=selected_report
        )

        print("\nREPORT-SPECIFIC SEARCH")
        print("=" * 60)

        for document, score in filtered_results:

            print(
                document.metadata.get("report_id"),
                document.metadata.get("page_number"),
                round(float(score), 4)
            )

