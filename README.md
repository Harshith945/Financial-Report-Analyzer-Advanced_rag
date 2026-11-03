
# 📊 Financial Report Analyzer

An AI-powered application that allows users to upload financial reports in PDF format and ask questions about their contents using natural language.

The application uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from financial documents and generate answers grounded in the uploaded reports, including reporting periods and PDF page references.

## 🚀 Project Overview

Financial reports contain large amounts of structured and unstructured information. Finding specific details such as revenue, net income, total assets, and cash flow can be time-consuming when manually searching through lengthy documents.

The Financial Report Analyzer simplifies this process by combining document processing, hybrid retrieval, reranking, and LLM-based answer generation.

Users can upload multiple financial reports and ask questions across the uploaded documents.

## ✨ Key Features

- **Multiple PDF Uploads:** Upload multiple financial reports.
- **Document Processing:** Extract text and tables from PDFs using Docling.
- **Semantic Chunking:** Split document content into meaningful chunks.
- **Vector Search:** Retrieve semantically relevant information using embeddings and ChromaDB.
- **Keyword Search:** Retrieve relevant content using BM25.
- **Hybrid Retrieval:** Combine vector and keyword search using Reciprocal Rank Fusion (RRF).
- **Cross-Encoder Reranking:** Reorder retrieved documents based on query relevance.
- **Contextual Compression:** Retain the most relevant information before sending context to the LLM.
- **LLM-Based Answers:** Generate responses using Groq-hosted language models.
- **Source References:** Include relevant reporting periods and PDF page references in answers.
- **Duplicate Detection:** Use SHA-256 file hashing to identify previously processed PDFs.
- **Interactive Interface:** Ask questions through a Streamlit chat interface.

## 🏗️ Architecture

```text
                  User
                   |
                   v
             Streamlit UI
                   |
          Upload Financial PDFs
                   |
                   v
          Document Processing
                Docling
                   |
            Extract Text
              and Tables
                   |
                   v
             Text Chunking
                   |
             Embeddings
                   |
          +--------+--------+
          |                 |
          v                 v
      ChromaDB            BM25
    Vector Search      Keyword Search
          |                 |
          +--------+--------+
                   |
                   v
          Hybrid Retrieval
                   |
          Reciprocal Rank
             Fusion
                   |
                   v
          Cross-Encoder
             Reranking
                   |
                   v
          Contextual
          Compression
                   |
                   v
             Groq LLM
                   |
                   v
           Final Answer
                   |
          Source References
                   |
                   v
             Streamlit UI
```

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| Document Processing | Docling, PyMuPDF, pdfplumber |
| LLM Framework | LangChain |
| LLM Provider | Groq |
| Embedding Model | Sentence Transformers – all-MiniLM-L6-v2 |
| Vector Database | ChromaDB |
| Keyword Retrieval | BM25 |
| Hybrid Retrieval | Reciprocal Rank Fusion |
| Reranking | Cross-Encoder |
| Data Validation | Pydantic |

## 🔄 Workflow

1. **Upload:** The user uploads one or more financial PDF reports through the Streamlit interface.
2. **Extraction:** Docling extracts document text and tables.
3. **Metadata Extraction:** Relevant report information, such as company, ticker, report type, fiscal year, quarter, and reporting period, is identified.
4. **Chunking:** Extracted content is divided into smaller meaningful chunks.
5. **Embedding:** Sentence Transformer embeddings are generated for the chunks.
6. **Indexing:** The chunks are stored in ChromaDB, while BM25 builds a keyword-search index.
7. **Retrieval:** A user question is processed using both semantic and keyword retrieval.
8. **Fusion:** Reciprocal Rank Fusion combines the retrieval results.
9. **Reranking:** A Cross-Encoder reranks the retrieved documents based on relevance.
10. **Contextual Compression:** Relevant information is selected to create focused context.
11. **Generation:** The Groq LLM generates an answer using the retrieved context.
12. **Response:** The application displays the answer with relevant reporting period and PDF page references.

## 📂 Project Structure

```text
financial_report_analyzer/
│
├── app.py
├── model_cache.py
├── requirements.txt
├── .env
├── .gitignore
│
├── ingestion/
│   ├── financial_ingestion.py
│   ├── metadata_extractor.py
│   └── docling_parser.py
│
├── chunking/
│   └── text_chunker.py
│
├── indexing/
│   ├── embedding_model.py
│   └── vector_store.py
│
├── retrieval/
│   ├── bm25_retriever.py
│   ├── hybrid_retriever.py
│   ├── reranker.py
│   ├── contextual_compression.py
│   └── query_expansion.py
│
├── generation/
│   ├── rag_chain.py
│   ├── prompts.py
│   ├── guardrails.py
│   └── conversation_handler.py
│
└── data/
    ├── raw_pdfs/
    ├── extracted/
    ├── processed/
    └── chroma_db/
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Harshith945/financial-report-analyzer.git
cd financial-report-analyzer
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Never commit your API key or `.env` file to GitHub.

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 💬 Example Questions

- What was the total revenue for Q1 2026?
- What was the company's net income?
- What were the total assets?
- How much cash was generated from operating activities?
- What was the gross profit for the reporting period?

## 🔐 Responsible Answer Generation

The application is designed to generate answers using retrieved financial document content rather than relying on unsupported assumptions.

It also includes prompt-level instructions to avoid inventing financial figures and to identify when information is unavailable in the retrieved documents.

## 🎯 Project Objective

The objective of this project is to explore and implement advanced RAG techniques for document intelligence, particularly for financial reports.

It demonstrates how document processing, hybrid retrieval, reranking, contextual compression, and LLMs can work together to build a practical question-answering application.

## 👨‍💻 Author

**Harshith Kumar**

ECE Graduate | AI & GenAI Enthusiast

Skills: Python | LangChain | RAG | LLMs | ChromaDB | Streamlit
