
from retrieval.bm25_retriever import BM25Retriever
from indexing.vector_store import VectorStore
from retrieval.query_expansion import QueryExpander


class HybridRetriever:

    def __init__(self):

        self.bm25_retriever = BM25Retriever()
        self.vector_store_manager = VectorStore()
        self.query_expander = QueryExpander()

        self.vector_db = None
        self.rrf_k = 60
        self.initialized = False

    # --------------------------------------------------
    # 1. Initialize retrievers
    # --------------------------------------------------

    def initialize(self):

        if self.initialized:
            return

        # Initialize BM25.
        self.bm25_retriever.load_chunks()
        self.bm25_retriever.build_index()

        # Initialize ChromaDB.
        self.vector_db = (
            self.vector_store_manager.create_vector_store()
        )

        self.initialized = True

        print("\nHybrid retriever initialized successfully!")

    # --------------------------------------------------
    # 2. Refresh after uploading a new PDF
    # --------------------------------------------------

    def refresh(self, chunks_path=None):

        if not self.initialized:
            self.initialize()

        # Add new report to ChromaDB.
        if chunks_path is not None:

            report_id = self.vector_store_manager.add_report(
                chunks_path
            )

            print(
                f"\nChromaDB updated for report: {report_id}"
            )

        # Reload all chunks into BM25.
        self.bm25_retriever.load_chunks()

        # Rebuild BM25 index.
        self.bm25_retriever.build_index()

        # Synchronize vector DB reference.
        self.vector_db = (
            self.vector_store_manager.vector_store
        )

        print("\nHybrid retriever refreshed successfully!")

        return True

    # --------------------------------------------------
    # 3. Get reporting periods from uploaded reports
    # --------------------------------------------------

    def get_available_periods(self):

        available_periods = []
        seen_periods = set()

        for document in self.bm25_retriever.documents:

            metadata = document.metadata

            period_end = metadata.get("period_end")

            # Fallback if metadata is nested.
            if not period_end:

                report_metadata = metadata.get(
                    "report_metadata",
                    {}
                )

                period_end = report_metadata.get(
                    "period_end"
                )

            company = metadata.get("company")
            ticker = metadata.get("ticker")

            if not period_end:
                continue

            # Keep separate entries for different companies.
            period_key = (
                company,
                ticker,
                period_end
            )

            if period_key not in seen_periods:

                available_periods.append({
                    "period_end": period_end,
                    "company": company,
                    "ticker": ticker
                })

                seen_periods.add(period_key)

        return available_periods

    # --------------------------------------------------
    # 4. Generate unique document key
    # --------------------------------------------------

    def get_document_key(self, document):

        metadata = document.metadata

        return (
            metadata.get("report_id"),
            metadata.get("chunk_id"),
            document.page_content,
            metadata.get("source"),
            metadata.get("page_number"),
            metadata.get("table_id"),
            metadata.get("content_type"),
            metadata.get("section")
        )

    # --------------------------------------------------
    # 5. Reciprocal Rank Fusion
    # --------------------------------------------------

    def reciprocal_rank_fusion(self, all_results):

        fused_scores = {}
        documents = {}

        for results in all_results:

            for rank, (document, score) in enumerate(
                results,
                start=1
            ):

                key = self.get_document_key(document)

                fused_scores[key] = (
                    fused_scores.get(key, 0)
                    + 1 / (self.rrf_k + rank)
                )

                documents[key] = document

        ranked_results = sorted(
            fused_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        return [
            (documents[key], score)
            for key, score in ranked_results
        ]

    # --------------------------------------------------
    # 6. Hybrid search
    # --------------------------------------------------

    def search(
        self,
        query,
        k=10,
        report_id=None
    ):

        if not self.initialized:

            raise ValueError(
                "Hybrid retriever is not initialized. "
                "Call initialize() first."
            )

        # Step 1: Get reporting periods from loaded documents.
        available_periods = self.get_available_periods()

        # Step 2: Expand the original query.
        expanded_queries = (
            self.query_expander.expand_query(
                query,
                available_periods=available_periods
            )
        )

        print("\nAVAILABLE REPORTING PERIODS")

        for item in available_periods:
            print(
                item.get("company"),
                item.get("ticker"),
                item.get("period_end")
            )

        print("\nQUERIES USED FOR RETRIEVAL")

        for expanded_query in expanded_queries:
            print("-", expanded_query)

        if report_id:
            print("\nFiltering report:", report_id)

        else:
            print("\nSearching across all reports.")

        # Step 3: Retrieve candidates for each query.
        all_results = []

        for expanded_query in expanded_queries:

            print(
                f"\nRetrieving for query: {expanded_query}"
            )

            # BM25 retrieval.
            bm25_results = self.bm25_retriever.search(
                expanded_query,
                k=20,
                report_id=report_id
            )

            # Vector retrieval.
            if report_id is not None:

                vector_results = (
                    self.vector_db.similarity_search_with_score(
                        expanded_query,
                        k=20,
                        filter={
                            "report_id": report_id
                        }
                    )
                )

            else:

                vector_results = (
                    self.vector_db.similarity_search_with_score(
                        expanded_query,
                        k=20
                    )
                )

            print(
                "BM25 documents:",
                len(bm25_results)
            )

            print(
                "Vector documents:",
                len(vector_results)
            )

            all_results.append(bm25_results)
            all_results.append(vector_results)

        # Step 4: Combine rankings using RRF.
        hybrid_results = self.reciprocal_rank_fusion(
            all_results
        )

        # Step 5: Select top results.
        final_results = hybrid_results[:k]

        # Step 6: Print retrieved documents for debugging.
        print(
            "\n========== HYBRID RETRIEVAL RESULTS =========="
        )

        for index, (document, score) in enumerate(
            final_results,
            start=1
        ):

            metadata = document.metadata

            print(f"\nDocument {index}")
            print("RRF Score:", round(score, 6))
            print("Company:", metadata.get("company"))
            print("Report ID:", metadata.get("report_id"))
            print("Page:", metadata.get("page_number"))
            print("Content type:", metadata.get("content_type"))
            print("Section:", metadata.get("section"))
            print("Content:")
            print(document.page_content[:700])

        print("\n==============================================")

        return final_results


# --------------------------------------------------
# 7. Test hybrid retrieval
# --------------------------------------------------

if __name__ == "__main__":

    retriever = HybridRetriever()
    retriever.initialize()

    test_questions = [
        "What was Tesla's revenue in Q1?",
        "What was Tesla's revenue in Q1 2026?",
        "What was Tesla's gross profit?",
        "What were Tesla's total assets?"
    ]

    for query in test_questions:

        print("\n" + "=" * 70)
        print("QUESTION:", query)

        results = retriever.search(
            query,
            k=10
        )

        print("\nFINAL RESULTS")
        print("=" * 70)

        for rank, (document, score) in enumerate(
            results,
            start=1
        ):

            print(
                f"\nRank: {rank} | "
                f"Page: {document.metadata.get('page_number')} | "
                f"RRF Score: {score:.6f}"
            )

            print(document.page_content[:500])



