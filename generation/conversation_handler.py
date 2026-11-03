
import re


APP_INFORMATION = """
I am a Financial Report Analyzer assistant.

I help users analyze uploaded financial reports and answer
questions based on the information available in those documents.
"""


def extract_name(message):

    patterns = [
        r"\bmy name is ([A-Za-z]+)\b",
        r"\bi am ([A-Za-z]+)\b",
        r"\bi'm ([A-Za-z]+)\b"
    ]

    for pattern in patterns:

        match = re.search(pattern, message, re.IGNORECASE)

        if match:
            return match.group(1).capitalize()

    return None


def handle_conversation(message, user_name=None):

    message = message.strip()

    message_lower = message.lower().strip()

    # Normalize punctuation for intent matching
    normalized_message = re.sub(
        r"[^\w\s]",
        "",
        message_lower
    ).strip()

    # --------------------------------------------------
    # 1. Detect user's name
    # --------------------------------------------------

    detected_name = extract_name(message)

    if detected_name:

        user_name = detected_name

        return (
            f"Hello, {user_name}! Nice to meet you. "
            "How can I help you with your financial report today?",
            user_name
        )

    # --------------------------------------------------
    # 2. Handle greetings
    # --------------------------------------------------

    greetings = [
        "hi",
        "hii",
        "hiii",
        "hiiii",
        "hello",
        "hey",
        "heyy",
        "good morning",
        "good afternoon",
        "good evening"
    ]

    if normalized_message in greetings:

        if user_name:

            return (
                f"Hello, {user_name}! How can I help you "
                "with your financial report today?",
                user_name
            )

        return (
            "Hello! Welcome to Financial Report Analyzer. "
            "How can I help you today?",
            user_name
        )

    # --------------------------------------------------
    # 3. Handle application-related questions
    # --------------------------------------------------

    app_questions = [
        "what is this application",
        "what is this app",
        "about this application",
        "about this app",
        "what can you do",
        "who are you",
        "what will you do",
        "what is your purpose"
    ]

    if normalized_message in app_questions:

        return APP_INFORMATION, user_name

    # --------------------------------------------------
    # 4. Handle user's name questions
    # --------------------------------------------------

    name_questions = [
        "what is my name",
        "do you know my name",
        "who am i"
    ]

    if normalized_message in name_questions:

        if user_name:

            return (
                f"Your name is {user_name}.",
                user_name
            )

        return (
            "You haven't told me your name yet.",
            user_name
        )

    # --------------------------------------------------
    # 5. Pass all other questions to the RAG pipeline
    # --------------------------------------------------

    return None, user_name
