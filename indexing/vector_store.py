
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma

from indexing.embedding_model import EmbeddingModel


class VectorStore:

    def __init__(self):

        # Directory containing report-specific chunk files.
        self.processed_directory = Path("data/processed")

        # Persistent ChromaDB directory.
        self.persist_directory = "data/chroma_db"

        # Keep the existing collection.
        self.collection_name = "financial_reports_v2"

        self.embedding_model = EmbeddingModel()

        self.vector_store = None

    def initialize(self):
        """
        Connect to the existing persistent Chroma collection.
        """

        if self.vector_store is None:

            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_model.model,
                persist_directory=self.persist_directory
            )

        return self.vector_store

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

    def prepare_documents(self, chunks, fallback_report_id):
        """
        Convert chunk dictionaries into LangChain Documents
        and generate globally unique Chroma IDs.
        """

        documents = []
        ids = []

        for chunk in chunks:

            metadata = dict(chunk.get("metadata", {}))

            # Flatten nested report metadata.
            report_metadata = metadata.pop(
                "report_metadata", {}
            )

            if isinstance(report_metadata, dict):
                metadata.update(report_metadata)

            # Identify the report.
            report_id = (
                chunk.get("report_id")
                or metadata.get("report_id")
                or fallback_report_id
            )

            report_id = str(report_id)

            metadata["report_id"] = report_id

            # Keep only Chroma-compatible metadata values.
            clean_metadata = {}

            for key, value in metadata.items():

                if value is None:
                    continue

                if isinstance(value, (str, int, float, bool)):
                    clean_metadata[key] = value

            document = Document(
                page_content=chunk["text"],
                metadata=clean_metadata
            )

            documents.append(document)

            # Unique ID across different reports.
            vector_id = f"{report_id}_{chunk['chunk_id']}"

            ids.append(vector_id)

        return documents, ids

    def load_chunks(self):
        """
        Load chunks from all reports and prepare Documents.
        Used for initial indexing.
        """

        report_files = self.get_report_files()

        documents = []
        ids = []

        for chunks_path in report_files:

            with open(
                chunks_path,
                "r",
                encoding="utf-8"
            ) as file:

                chunks = json.load(file)

            report_documents, report_ids = self.prepare_documents(
                chunks=chunks,
                fallback_report_id=chunks_path.parent.name
            )

            documents.extend(report_documents)
            ids.extend(report_ids)

        print("Loaded report files:", len(report_files))
        print("Total chunks loaded:", len(documents))

        return documents, ids

    def create_vector_store(self):
        """
        Create or update the persistent Chroma collection.

        This method is used for initial indexing or
        indexing all reports again while skipping existing IDs.
        """

        documents, ids = self.load_chunks()

        self.initialize()

        # Get IDs already present in ChromaDB.
        existing_ids = set(
            self.vector_store.get()["ids"]
        )

        new_documents = []
        new_ids = []

        for document, doc_id in zip(documents, ids):

            if doc_id not in existing_ids:

                new_documents.append(document)
                new_ids.append(doc_id)

        if new_documents:

            self.vector_store.add_documents(
                documents=new_documents,
                ids=new_ids
            )

        print("\nCHROMADB INDEXING RESULTS")
        print("=" * 50)

        print("Total chunks:", len(documents))
        print("New chunks indexed:", len(new_documents))

        print(
            "Already existing:",
            len(documents) - len(new_documents)
        )

        print(
            "Total vectors in collection:",
            self.vector_store._collection.count()
        )

        print("Collection:", self.collection_name)
        print("Database location:", self.persist_directory)

        return self.vector_store

    def add_report(self, chunks_path):
        """
        Add chunks from one newly processed report.

        Existing report vectors are not reprocessed.
        Only missing vector IDs are added.
        """

        chunks_path = Path(chunks_path)

        if not chunks_path.exists():
            raise FileNotFoundError(
                f"Chunk file not found: {chunks_path}"
            )

        with open(
            chunks_path,
            "r",
            encoding="utf-8"
        ) as file:

            chunks = json.load(file)

        report_id = chunks_path.parent.name

        documents, ids = self.prepare_documents(
            chunks=chunks,
            fallback_report_id=report_id
        )

        self.initialize()

        # Get existing IDs from ChromaDB.
        existing_ids = set(
            self.vector_store.get()["ids"]
        )

        new_documents = []
        new_ids = []

        for document, vector_id in zip(documents, ids):

            if vector_id not in existing_ids:

                new_documents.append(document)
                new_ids.append(vector_id)

        if new_documents:

            self.vector_store.add_documents(
                documents=new_documents,
                ids=new_ids
            )

        print("\nNEW REPORT INDEXING RESULTS")
        print("=" * 50)

        print("Report ID:", report_id)
        print("Total chunks:", len(documents))
        print("New vectors indexed:", len(new_documents))

        print(
            "Existing vectors skipped:",
            len(documents) - len(new_documents)
        )

        print(
            "Total vectors in collection:",
            self.vector_store._collection.count()
        )

        return report_id

    def similarity_search(
        self,
        query,
        k=3,
        report_id=None
    ):
        """
        Search all reports or restrict retrieval to one report.
        """

        if self.vector_store is None:
            raise ValueError(
                "Vector store is not initialized. "
                "Run initialize() or create_vector_store() first."
            )

        search_kwargs = {"k": k}

        if report_id is not None:

            search_kwargs["filter"] = {
                "report_id": report_id
            }

        results = self.vector_store.similarity_search(
            query,
            **search_kwargs
        )

        return results

    def get_report_ids(self):
        """
        Return the report IDs currently indexed in ChromaDB.
        """

        if self.vector_store is None:
            raise ValueError(
                "Vector store is not initialized."
            )

        data = self.vector_store.get(
            include=["metadatas"]
        )

        report_ids = sorted({
            metadata["report_id"]
            for metadata in data["metadatas"]
            if metadata and metadata.get("report_id")
        })

        return report_ids


if __name__ == "__main__":

    store = VectorStore()

    store.create_vector_store()

    print("\nINDEXED REPORTS")
    print("=" * 50)

    for report_id in store.get_report_ids():
        print(report_id)

    # Test a general search across all indexed reports.
    results = store.similarity_search(
        query="What was the total revenue?",
        k=3
    )

    print("\nSEARCH RESULTS")
    print("=" * 70)

    for index, document in enumerate(results, start=1):

        print(f"\nRESULT {index}")
        print("Report ID:", document.metadata.get("report_id"))
        print("Company:", document.metadata.get("company"))
        print("Period:", document.metadata.get("period_end"))
        print("PDF page:", document.metadata.get("page_number"))
        print("Content:", document.page_content[:500])


