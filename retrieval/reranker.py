
from model_cache import get_reranker_model


class FinancialReranker:

    def __init__(self):

        # Reuse the cached Cross-Encoder model.
        self.model = get_reranker_model()

    def rerank(
        self,
        query,
        retrieved_documents,
        top_k=5,
        report_id=None
    ):
        """
        Rerank retrieved documents using Cross-Encoder scores.

        Parameters:
            query: User question.
            retrieved_documents: List of (Document, retrieval_score).
            top_k: Number of documents to return.
            report_id: Optional report filter.

        Returns:
            List of (Document, reranker_score) tuples.
        """

        # Filter documents by report ID if provided.
        if report_id is not None:

            retrieved_documents = [
                (document, score)
                for document, score in retrieved_documents
                if document.metadata.get("report_id") == report_id
            ]

        if not retrieved_documents:
            return []

        # Prepare query-document pairs.
        pairs = [
            [query, document.page_content]
            for document, _ in retrieved_documents
        ]

        # Calculate Cross-Encoder scores.
        reranker_scores = self.model.predict(pairs)

        # Combine documents with their new scores.
        reranked_results = []

        for index, (document, _) in enumerate(
            retrieved_documents
        ):

            score = float(reranker_scores[index])

            reranked_results.append(
                (document, score)
            )

        # Sort by descending relevance score.
        reranked_results.sort(
            key=lambda item: item[1],
            reverse=True
        )

        return reranked_results[:top_k]

