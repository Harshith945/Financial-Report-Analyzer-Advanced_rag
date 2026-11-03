
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import FinancialReranker
from retrieval.contextual_compression import ContextualCompressor

from generation.prompts import financial_prompt
from generation.guardrails import (
    validate_question,
    check_retrieved_context,
    no_answer_response,
)

load_dotenv()


class FinancialRAG:

    def __init__(self):

        # 1. Initialize hybrid retriever.
        self.retriever = HybridRetriever()
        self.retriever.initialize()

        # 2. Initialize reranker.
        self.reranker = FinancialReranker()

        # 3. Initialize contextual compressor.
        self.compressor = ContextualCompressor(
            self.reranker.model
        )

        # 4. Initialize Groq LLM.
        api_key = os.getenv("GROQ_API_KEY")
        model = os.getenv("GROQ_MODEL")

        if not api_key:
            raise ValueError("GROQ_API_KEY is missing from .env")

        if not model:
            raise ValueError("GROQ_MODEL is missing from .env")

        self.llm = ChatGroq(
            groq_api_key=api_key,
            model=model,
            temperature=0
        )

        # 5. Create prompt and LLM chain.
        self.chain = financial_prompt | self.llm

    # --------------------------------------------------
    # 1. Refresh after uploading a new PDF
    # --------------------------------------------------

    def refresh_after_upload(self, chunks_path):

        self.retriever.refresh(
            chunks_path=chunks_path
        )

        print("Knowledge base refreshed successfully!")

        return True

    # --------------------------------------------------
    # 2. Identify the requested financial metric
    # --------------------------------------------------

    def identify_requested_metric(self, question):

        question = question.lower()

        metric_patterns = {
            "revenue": [
                "total revenues",
                "total revenue",
                "revenue",
                "revenues"
            ],
            "gross profit": [
                "gross profit"
            ],
            "operating income": [
                "operating income"
            ],
            "net income": [
                "net income"
            ],
            "total assets": [
                "total assets"
            ],
            "operating cash flow": [
                "operating cash flow",
                "cash flow from operations",
                "cash generated from operations"
            ]
        }

        for metric, patterns in metric_patterns.items():

            if any(pattern in question for pattern in patterns):
                return metric

        return None

    # --------------------------------------------------
    # 3. Prioritize exact financial evidence
    # --------------------------------------------------

    def prioritize_sources(self, question, documents):

        metric = self.identify_requested_metric(question)

        if not metric:
            return documents

        metric_patterns = {
            "revenue": [
                "total revenues:",
                "total revenue:"
            ],
            "gross profit": [
                "total gross profit:",
                "gross profit:"
            ],
            "operating income": [
                "total operating income:",
                "operating income:"
            ],
            "net income": [
                "net income:"
            ],
            "total assets": [
                "total assets:"
            ],
            "operating cash flow": [
                "net cash provided by operating activities",
                "net cash from operating activities"
            ]
        }

        patterns = metric_patterns.get(metric, [])

        def source_priority(document):

            content = document.page_content.lower()

            # Prefer exact consolidated metric rows.
            exact_match = any(
                pattern in content
                for pattern in patterns
            )

            # Prefer documents with the requested period.
            question_lower = question.lower()

            requested_period = (
                "2026" in question_lower
                and "2026" in content
            )

            return (
                int(exact_match),
                int(requested_period)
            )

        prioritized_documents = sorted(
            documents,
            key=source_priority,
            reverse=True
        )

        return prioritized_documents

    # --------------------------------------------------
    # 4. Prepare context with source metadata
    # --------------------------------------------------

    def prepare_context(self, documents):

        context_parts = []

        for index, document in enumerate(documents, start=1):

            metadata = document.metadata or {}

            company = metadata.get("company", "Not specified")
            report_type = metadata.get("report_type", "Not specified")
            fiscal_year = metadata.get("fiscal_year", "Not specified")
            fiscal_quarter = metadata.get("fiscal_quarter", "Not specified")
            period_end = metadata.get("period_end", "Not specified")
            source = metadata.get("source", "Not specified")
            page_number = metadata.get("page_number", "Not specified")

            formatted_context = f"""
SOURCE CHUNK {index}

Company: {company}
Report type: {report_type}
Fiscal year: {fiscal_year}
Fiscal quarter: {fiscal_quarter}
Period end: {period_end}
Source PDF: {source}
PDF page number: {page_number}

Financial content:
{document.page_content}
"""

            context_parts.append(formatted_context)

        return "\n\n".join(context_parts)

    # --------------------------------------------------
    # 5. Ask questions
    # --------------------------------------------------

    def ask(self, question):

        # Step 1: Validate user input.
        if not validate_question(question):
            return "Please enter a valid question."

        # Step 2: Retrieve relevant chunks.
        retrieved_results = self.retriever.search(
            question,
            k=10
        )

        if not retrieved_results:
            return no_answer_response()

        # Step 3: Rerank retrieved documents.
        reranked_results = self.reranker.rerank(
            question,
            retrieved_results,
            top_k=5
        )

        if not reranked_results:
            return no_answer_response()

        # Step 4: Check context relevance.
        if not check_retrieved_context(reranked_results):
            return no_answer_response()

        # Step 5: Extract reranked documents.
        reranked_documents = [
            document
            for document, score in reranked_results
        ]

        # Step 6: Compress retrieved context.
        compressed_documents = self.compressor.compress_documents(
            question,
            reranked_documents,
            max_units=5
        )

        if not compressed_documents:
            return no_answer_response()

        # Step 7: Prioritize exact financial evidence.
        prioritized_documents = self.prioritize_sources(
            question,
            compressed_documents
        )

        # Step 8: Prepare context.
        context = self.prepare_context(
            prioritized_documents
        )

        # Step 9: Generate answer.
        response = self.chain.invoke({
            "context": context,
            "question": question
        })

        # Step 10: Return generated answer directly.
        return response.content


# --------------------------------------------------
# 6. Test RAG pipeline
# --------------------------------------------------

if __name__ == "__main__":

    rag = FinancialRAG()

    test_questions = [
        "What were Tesla's total assets?",
        "What was Tesla's revenue in Q1 2026?",
        "What was Tesla's gross profit?",
        "What was Tesla's operating cash flow?"
    ]

    for question in test_questions:

        print("\n" + "=" * 60)
        print("Question:", question)

        answer = rag.ask(question)

        print("\nAnswer:")
        print(answer)





