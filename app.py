import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

from generation.rag_chain import FinancialRAG
from generation.conversation_handler import handle_conversation

from ingestion.financial_ingestion import FinancialDocumentIngestion
from chunking.text_chunker import FinancialTextChunker


# ================================================
# STREAMLIT CONFIGURATION
# ================================================

st.set_page_config(
    page_title="Financial Report Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ================================================
# CUSTOM UI STYLING
# ================================================

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #0a0f1c;
        --surface: #111a2e;
        --surface-2: #16223a;
        --border: #24324d;
        --text: #e8edf6;
        --muted: #8b9ab3;
        --accent: #3b82f6;
        --accent-2: #8b5cf6;
        --success: #22c55e;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(1200px 500px at 85% -10%, rgba(139, 92, 246, 0.14), transparent 60%),
            radial-gradient(900px 500px at -10% 0%, rgba(59, 130, 246, 0.14), transparent 60%),
            var(--bg);
        color: var(--text);
    }

    /* Hide default Streamlit chrome */
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }

    .block-container {
        padding-top: 2rem;
        max-width: 980px;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #0b1220 100%);
        border-right: 1px solid var(--border);
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 4px;
    }

    .brand-logo {
        width: 42px;
        height: 42px;
        border-radius: 12px;
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        box-shadow: 0 6px 18px rgba(59, 130, 246, 0.35);
    }

    .brand-name {
        font-weight: 700;
        font-size: 1.05rem;
        color: #fff;
        line-height: 1.2;
    }

    .brand-tag {
        font-size: 0.75rem;
        color: var(--muted);
    }

    .section-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
        margin: 6px 0 8px 0;
    }

    .sidebar-note {
        color: var(--muted);
        font-size: 0.82rem;
        line-height: 1.55;
    }

    .report-item {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 9px 12px;
        margin-bottom: 8px;
    }

    .report-item .company {
        font-weight: 600;
        font-size: 0.88rem;
        color: #fff;
    }

    .report-item .period {
        font-size: 0.75rem;
        color: var(--muted);
        margin-top: 2px;
    }

    [data-testid="stFileUploader"] section {
        background: var(--surface);
        border: 1.5px dashed #334768;
        border-radius: 12px;
    }

    [data-testid="stFileUploader"] section:hover {
        border-color: var(--accent);
    }

    /* ---------- Hero ---------- */
    .hero {
        padding: 6px 0 18px 0;
    }

    .hero-badge {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        color: #93c5fd;
        background: rgba(59, 130, 246, 0.12);
        border: 1px solid rgba(59, 130, 246, 0.3);
        padding: 4px 12px;
        border-radius: 999px;
        margin-bottom: 12px;
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.15;
        background: linear-gradient(90deg, #ffffff 0%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }

    .subtitle {
        color: var(--muted);
        font-size: 1.02rem;
        max-width: 640px;
        line-height: 1.6;
    }

    /* ---------- Stat cards ---------- */
    .stat-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 16px 18px;
    }

    .stat-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #fff;
        line-height: 1.1;
    }

    .stat-label {
        font-size: 0.78rem;
        color: var(--muted);
        margin-top: 4px;
    }

    /* ---------- Welcome / empty state ---------- */
    .welcome-card {
        background: linear-gradient(135deg, rgba(59,130,246,0.10), rgba(139,92,246,0.10));
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 24px 26px;
        margin: 20px 0 22px 0;
    }

    .welcome-card h3 {
        color: #fff;
        margin: 0 0 8px 0;
        font-size: 1.2rem;
    }

    .welcome-card p {
        color: #aab6cc;
        line-height: 1.65;
        margin: 0;
    }

    .steps {
        display: flex;
        gap: 14px;
        margin-top: 18px;
        flex-wrap: wrap;
    }

    .step {
        flex: 1;
        min-width: 180px;
        background: rgba(10, 15, 28, 0.55);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 12px 14px;
        font-size: 0.85rem;
        color: #c3cde0;
    }

    .step b {
        display: block;
        color: #fff;
        margin-bottom: 2px;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid var(--border);
        background: var(--surface);
        color: var(--text);
        transition: all 0.15s ease;
    }

    .stButton > button:hover {
        border-color: var(--accent);
        color: #fff;
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.18);
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
        border: none;
        color: #fff;
    }

    .stButton > button[kind="primary"]:hover {
        filter: brightness(1.1);
    }

    .stDownloadButton > button {
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid var(--border);
        background: var(--surface);
        color: var(--text);
    }

    /* ---------- Chat ---------- */
    [data-testid="stChatMessage"] {
        background-color: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 14px;
    }

    [data-testid="stChatInput"] {
        border-radius: 14px;
    }

    [data-testid="stChatInput"] textarea {
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stBottom"] > div {
        background: transparent;
    }

    /* Force readable text inside chat bubbles and main area */
    [data-testid="stChatMessage"],
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div,
    [data-testid="stChatMessage"] td,
    [data-testid="stChatMessage"] th,
    [data-testid="stChatMessageContent"],
    [data-testid="stChatMessageContent"] * {
        color: #e8edf6 !important;
    }

    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3,
    [data-testid="stChatMessage"] h4 {
        color: #ffffff !important;
    }

    [data-testid="stChatMessage"] code {
        color: #93c5fd !important;
        background: #0a0f1c;
    }

    [data-testid="stChatMessage"] table {
        border-color: var(--border);
    }

    /* Chat input readability */
    [data-testid="stChatInput"] textarea {
        color: #e8edf6 !important;
        background: var(--surface) !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--muted) !important;
    }

    /* Headings and general text in main area */
    .stMarkdown h5, .stMarkdown h4, .stMarkdown h3 {
        color: #f1f5f9 !important;
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] small {
        color: #cbd5e1;
    }

    hr { border-color: var(--border); }
</style>
""",
    unsafe_allow_html=True,
)


# ================================================
# DIRECTORIES
# ================================================

RAW_DIRECTORY = Path("data/raw_pdfs")
PROCESSED_DIRECTORY = Path("data/processed")
EXTRACTED_DIRECTORY = Path("data/extracted")

RAW_DIRECTORY.mkdir(parents=True, exist_ok=True)


# ================================================
# SESSION STATE
# ================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_name" not in st.session_state:
    st.session_state.user_name = None

if "uploaded_hashes" not in st.session_state:
    st.session_state.uploaded_hashes = set()


# ================================================
# LOAD RAG SYSTEM
# ================================================

@st.cache_resource
def load_rag():
    return FinancialRAG()


try:
    rag = load_rag()

except Exception as error:
    st.error(f"Could not initialize the RAG system: {error}")
    st.stop()


# ================================================
# HELPERS
# ================================================

def calculate_file_hash(file_bytes):
    """Generate SHA-256 hash to identify duplicate PDFs."""
    return hashlib.sha256(file_bytes).hexdigest()


def report_already_processed(file_hash):
    """Check extraction JSON files for an existing report with the same hash."""

    for extraction_file in EXTRACTED_DIRECTORY.glob(
        "*/financial_extraction.json"
    ):
        try:
            with open(extraction_file, "r", encoding="utf-8") as file:
                data = json.load(file)

            if data.get("file_hash") == file_hash:
                return True

        except (OSError, ValueError):
            continue

    return False


def list_indexed_reports():
    """Return basic metadata for every report already extracted."""

    reports = []

    for extraction_file in sorted(
        EXTRACTED_DIRECTORY.glob("*/financial_extraction.json")
    ):
        try:
            with open(extraction_file, "r", encoding="utf-8") as file:
                data = json.load(file)

            metadata = data.get("report_metadata", {}) or {}

            reports.append({
                "company": metadata.get("company") or "Unknown company",
                "period": metadata.get("period_end") or "Period not identified",
            })

        except (OSError, ValueError):
            continue

    return reports


def stream_text(text, delay=0.012):
    """Yield text word by word for a typing effect."""

    for word in text.split(" "):
        yield word + " "
        time.sleep(delay)


def build_transcript(messages):
    """Create a markdown transcript of the conversation."""

    lines = [
        "# Financial Report Analyzer - Chat Export",
        f"_Exported {datetime.now().strftime('%d %b %Y, %H:%M')}_",
        "",
    ]

    for message in messages:
        speaker = "You" if message["role"] == "user" else "Assistant"
        lines.append(f"**{speaker}:**\n\n{message['content']}\n")

    return "\n".join(lines)


# ================================================
# PROCESS UPLOADED PDF
# ================================================

def process_uploaded_pdf(uploaded_file, rag):

    file_bytes = uploaded_file.getvalue()
    file_hash = calculate_file_hash(file_bytes)

    # Check duplicate in current session
    if file_hash in st.session_state.uploaded_hashes:
        return (
            "duplicate",
            "This PDF was already uploaded in this session.",
        )

    # Check duplicate across previous sessions
    if report_already_processed(file_hash):

        st.session_state.uploaded_hashes.add(file_hash)

        return (
            "duplicate",
            "This PDF has already been processed.",
        )

    # Save uploaded PDF
    safe_filename = Path(uploaded_file.name).name

    pdf_path = RAW_DIRECTORY / safe_filename

    if pdf_path.exists():
        pdf_path = RAW_DIRECTORY / f"{file_hash[:12]}_{safe_filename}"

    with open(pdf_path, "wb") as file:
        file.write(file_bytes)

    # Step 1: Extract document content
    ingestion = FinancialDocumentIngestion(pdf_path)

    extracted_data = ingestion.process_document()

    extracted_data["file_hash"] = file_hash

    saved_extraction = ingestion.save_results(extracted_data)

    # Step 2: Create chunks
    chunker = FinancialTextChunker(extraction_path=saved_extraction)

    chunks = chunker.run()

    if not chunks:
        raise ValueError("No text chunks were generated from this PDF.")

    # Step 3: Get chunk file path
    report_id = extracted_data["report_id"]

    chunks_path = PROCESSED_DIRECTORY / report_id / "text_chunks.json"

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunk file was not created: {chunks_path}")

    # Step 4: Refresh RAG system
    rag.refresh_after_upload(chunks_path)

    st.session_state.uploaded_hashes.add(file_hash)

    company = (
        extracted_data.get("report_metadata", {}).get("company")
        or "Unknown company"
    )

    period = (
        extracted_data.get("report_metadata", {}).get("period_end")
        or "Period not identified"
    )

    return (
        "success",
        f"**{company}** · Period end: {period} · {len(chunks)} chunks indexed",
    )


# ================================================
# SIDEBAR
# ================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            <div class="brand-logo">📊</div>
            <div>
                <div class="brand-name">Financial Analyzer</div>
                <div class="brand-tag">AI-powered document Q&amp;A</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ---- Upload ----
    st.markdown(
        '<div class="section-label">Upload reports</div>',
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="You can upload multiple financial reports at once.",
    )

    if not uploaded_files:
        st.markdown(
            '<p class="sidebar-note">Drag and drop annual or quarterly '
            'report PDFs here. They will be indexed for question '
            'answering.</p>',
            unsafe_allow_html=True,
        )

    if uploaded_files:

        st.caption(f"{len(uploaded_files)} file(s) ready")

        if st.button(
            "⚡ Process Reports",
            type="primary",
            use_container_width=True,
        ):

            total_files = len(uploaded_files)
            progress_bar = st.progress(0, text="Starting...")

            for index, uploaded_file in enumerate(uploaded_files):

                progress_bar.progress(
                    index / total_files,
                    text=f"Processing {index + 1} of {total_files}",
                )

                with st.status(
                    uploaded_file.name,
                    expanded=False,
                ) as status_box:

                    try:
                        status, message = process_uploaded_pdf(
                            uploaded_file,
                            rag,
                        )

                        if status == "success":
                            status_box.update(
                                label=f"✅ {uploaded_file.name}",
                                state="complete",
                            )
                            st.markdown(message)

                        else:
                            status_box.update(
                                label=f"ℹ️ {uploaded_file.name}",
                                state="complete",
                            )
                            st.markdown(message)

                    except Exception as error:
                        status_box.update(
                            label=f"❌ {uploaded_file.name}",
                            state="error",
                        )
                        st.markdown(f"Could not process file: {error}")

            progress_bar.progress(1.0, text="All files handled")
            st.toast("Document processing completed", icon="✅")

    st.divider()

    # ---- Conversation controls ----
    st.markdown(
        '<div class="section-label">Conversation</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.messages:
        st.download_button(
            "⬇️ Export chat",
            data=build_transcript(st.session_state.messages),
            file_name="financial_chat.md",
            mime="text/markdown",
            use_container_width=True,
        )

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.user_name = None
        st.rerun()

    st.divider()

    st.markdown(
        '<p class="sidebar-note">'
        '<b>Format:</b> PDF<br>'
        '<b>Retrieval:</b> Hybrid search + reranking<br>'
        '<b>Generation:</b> Groq LLM'
        '</p>',
        unsafe_allow_html=True,
    )


# ================================================
# MAIN CONTENT: HERO
# ================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">✨ Grounded answers with page references</div>
        <div class="main-title">Financial Report Analyzer</div>
        <div class="subtitle">
            Ask questions about your financial reports in plain language and
            get answers backed by the document itself.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ================================================
# WELCOME SCREEN
# ================================================

if not st.session_state.messages:

    st.markdown(
        """
        <div class="welcome-card">
            <h3>👋 Welcome</h3>
            <p>
                Upload one or more financial PDFs, then start asking.
                Answers include the relevant reporting period and PDF
                page references so you can verify every number.
            </p>
            <div class="steps">
                <div class="step"><b>1 · Upload</b>Add PDFs from the sidebar</div>
                <div class="step"><b>2 · Process</b>Click Process Reports</div>
                <div class="step"><b>3 · Ask</b>Chat in natural language</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### 💡 Try asking")

    example_questions = [
        ("💰", "What was the total revenue?"),
        ("📈", "What was the net income?"),
        ("🏦", "What were the total assets?"),
        ("💵", "How much cash was generated from operating activities?"),
    ]

    question_cols = st.columns(2)

    for index, (icon, question) in enumerate(example_questions):

        with question_cols[index % 2]:

            if st.button(
                f"{icon}  {question}",
                key=f"example_question_{index}",
                use_container_width=True,
            ):
                st.session_state.pending_question = question
                st.rerun()


# ================================================
# DISPLAY CHAT HISTORY
# ================================================

AVATARS = {"user": "🧑‍💼", "assistant": "📊"}

for message in st.session_state.messages:

    with st.chat_message(
        message["role"],
        avatar=AVATARS.get(message["role"]),
    ):
        st.markdown(message["content"])


# ================================================
# USER QUESTION
# ================================================

user_input = st.chat_input("Ask anything about your financial reports...")

# Handle suggested questions
if not user_input and st.session_state.get("pending_question"):
    user_input = st.session_state.pending_question
    del st.session_state.pending_question


if user_input:

    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
    })

    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(user_input)

    response, updated_name = handle_conversation(
        user_input,
        st.session_state.user_name,
    )

    st.session_state.user_name = updated_name

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):

        if response is None:

            with st.spinner("Searching financial reports..."):

                try:
                    response = rag.ask(user_input)

                except Exception as error:
                    response = (
                        "An error occurred while analyzing the reports: "
                        f"{error}"
                    )

        st.write_stream(stream_text(response))

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
    })

    st.rerun()

