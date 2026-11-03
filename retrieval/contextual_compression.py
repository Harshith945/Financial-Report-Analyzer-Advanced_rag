
import re

from langchain_core.documents import Document

from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import FinancialReranker


class ContextualCompressor:

    def __init__(self, cross_encoder):

        # Reuse the existing CrossEncoder model.
        self.model = cross_encoder

    def split_into_units(self, text):
        """
        Split document content into lines.
        Split long narrative lines into sentences.
        """

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        units = []

        for line in lines:

            if len(line) > 300:

                sentences = re.split(
                    r"(?<=[.!?])\s+",
                    line
                )

                units.extend(
                    sentence.strip()
                    for sentence in sentences
                    if sentence.strip()
                )

            else:
                units.append(line)

        return units

    def is_context_unit(self, unit):
        """
        Identify important financial context and headers.
        """

        context_prefixes = (
            "Financial statement:",
            "Financial Table",
            "PDF page number:",
            "Reporting periods:",
            "Category:"
        )

        return unit.startswith(context_prefixes)

    def identify_metric(self, query):
        """
        Identify the financial metric requested by the user.
        """

        query = query.lower()

        metric_patterns = {
            "revenue": r"\b(revenue|revenues|sales)\b",
            "gross profit": r"\bgross profit\b",
            "operating income": r"\boperating income\b",
            "net income": r"\bnet income\b",
            "total assets": r"\btotal assets\b",
            "operating cash flow": (
                r"\b(operating cash flow|cash flow from operating "
                r"activities|net cash provided by operating activities)\b"
            ),
            "total liabilities": r"\btotal liabilities\b",
            "total equity": r"\b(total equity|stockholders'? equity)\b",
        }

        for metric, pattern in metric_patterns.items():

            if re.search(pattern, query):
                return metric

        return None

    def is_exact_metric_unit(self, unit, metric):
        """
        Identify lines containing the requested financial metric.

        Avoid selecting segment-level rows when the user
        asks for a consolidated metric.
        """

        if not metric:
            return False

        text = unit.lower()

        if metric == "revenue":

            return bool(
                re.search(r"\btotal revenues?\b", text)
                and not re.search(
                    r"\b(segment|automotive|energy generation)\b",
                    text
                )
            )

        if metric == "gross profit":

            return bool(
                re.search(r"\btotal gross profit\b", text)
            )

        if metric == "operating income":

            return bool(
                re.search(r"\boperating income\b", text)
                and "segment" not in text
            )

        if metric == "net income":

            return bool(
                re.search(r"\bnet income\b", text)
                and "excluding" not in text
            )

        if metric == "total assets":

            return bool(
                re.search(r"\btotal assets\b", text)
                and "segment" not in text
            )

        if metric == "operating cash flow":

            return bool(
                re.search(
                    r"\bnet cash provided by operating activities\b",
                    text
                )
            )

        if metric == "total liabilities":

            return bool(
                re.search(r"\btotal liabilities\b", text)
                and "segment" not in text
            )

        if metric == "total equity":

            return bool(
                re.search(
                    r"\btotal (?:stockholders'? )?equity\b",
                    text
                )
            )

        return False

    def compress_document(
        self,
        query,
        document,
        max_units=5
    ):

        units = self.split_into_units(
            document.page_content
        )

        if not units:
            return document

        # Identify the requested financial metric.
        metric = self.identify_metric(query)

        # Separate headers, exact metric rows and other content.
        context_units = []
        exact_metric_units = []
        data_units = []

        for unit in units:

            if self.is_context_unit(unit):
                context_units.append(unit)

            elif self.is_exact_metric_unit(unit, metric):
                exact_metric_units.append(unit)

            else:
                data_units.append(unit)

        # Score remaining data units against the user query.
        selected_units = set()

        if data_units:

            pairs = [
                [query, unit]
                for unit in data_units
            ]

            scores = self.model.predict(pairs)

            ranked_units = sorted(
                zip(data_units, scores),
                key=lambda item: float(item[1]),
                reverse=True
            )

            selected_units = {
                unit
                for unit, score in ranked_units[:max_units]
            }

        # Preserve original ordering.
        # Always retain exact metric rows and useful headers.
        compressed_units = [
            unit
            for unit in units
            if (
                self.is_context_unit(unit)
                or unit in exact_metric_units
                or unit in selected_units
            )
        ]

        compressed_text = "\n".join(
            compressed_units
        )

        # Safety: don't return an empty compressed document.
        if not compressed_text.strip():
            return document

        # Preserve all original metadata.
        return Document(
            page_content=compressed_text,
            metadata={
                **document.metadata,
                "compressed": True,
                "original_length": len(
                    document.page_content
                ),
                "compressed_length": len(
                    compressed_text
                ),
                "matched_metric": metric,
                "exact_metric_found": bool(
                    exact_metric_units
                )
            }
        )

    def compress_documents(
        self,
        query,
        documents,
        max_units=5
    ):

        compressed_documents = []

        for document in documents:

            compressed_document = self.compress_document(
                query,
                document,
                max_units=max_units
            )

            compressed_documents.append(
                compressed_document
            )

        return compressed_documents


if __name__ == "__main__":

    query = "What was Tesla's revenue in Q1 2026?"

    # 1. Initialize hybrid retrieval.
    hybrid_retriever = HybridRetriever()
    hybrid_retriever.initialize()

    # 2. Retrieve candidate documents.
    retrieved_documents = hybrid_retriever.search(
        query,
        k=10
    )

    # 3. Rerank candidates.
    reranker = FinancialReranker()

    reranked_results = reranker.rerank(
        query,
        retrieved_documents,
        top_k=5
    )

    # 4. Compress reranked documents.
    compressor = ContextualCompressor(
        reranker.model
    )

    compressed_documents = compressor.compress_documents(
        query,
        [
            document
            for document, score in reranked_results
        ],
        max_units=5
    )

    # 5. Display results.
    print("\nCOMPRESSED CONTEXT")
    print("=" * 60)

    for rank, document in enumerate(
        compressed_documents,
        start=1
    ):

        print(
            f"\nRank {rank} | "
            f"Company: {document.metadata.get('company')} | "
            f"Report ID: {document.metadata.get('report_id')} | "
            f"Page: {document.metadata.get('page_number')}"
        )

        print(
            "Matched metric:",
            document.metadata.get("matched_metric")
        )

        print(
            "Exact metric found:",
            document.metadata.get("exact_metric_found")
        )

        print(
            "Original length:",
            document.metadata.get("original_length")
        )

        print(
            "Compressed length:",
            document.metadata.get("compressed_length")
        )

        print("Content:")
        print(document.page_content)

