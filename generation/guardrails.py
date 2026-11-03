
import re


NO_ANSWER = (
    "I couldn't find that information in the uploaded "
    "financial report."
)


# --------------------------------------------------
# 1. Validate user input
# --------------------------------------------------

def validate_question(question):
    """Check whether the user entered a valid question."""

    return bool(question and question.strip())


# --------------------------------------------------
# 2. Validate retrieved context
# --------------------------------------------------

def check_retrieved_context(reranked_documents):
    """Check whether retrieval returned documents."""

    return bool(reranked_documents)


# --------------------------------------------------
# 3. Clean text before extracting numbers
# --------------------------------------------------

def clean_text_for_number_check(text):
    """
    Remove dates, years, quarter labels, page references,
    table references and clearly formatted footnote markers.
    """

    # Remove dates: March 31, 2026
    text = re.sub(
        r"\b(?:January|February|March|April|May|June|July|"
        r"August|September|October|November|December)"
        r"\s+\d{1,2},?\s+\d{4}\b",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Remove dates: 03/31/2026 or 2026-03-31
    text = re.sub(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
        r"\d{4}-\d{2}-\d{2})\b",
        " ",
        text
    )

    # Remove years
    text = re.sub(r"\b(?:19|20)\d{2}\b", " ", text)

    # Remove quarter labels
    text = re.sub(r"\bQ[1-4]\b", " ", text, flags=re.IGNORECASE)

    # Remove page references
    text = re.sub(
        r"\b(?:PDF\s+)?pages?\s+(?:number\s*)?:?\s*\d+\b",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Remove table and note references
    text = re.sub(
        r"\bFinancial\s+Table\s+\d+\b",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\bNote\s+\d+\b",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Remove explicitly bracketed footnote markers: [1], (1), [12]
    # Do not remove ordinary standalone numbers.
    text = re.sub(
        r"(?<![\d,.])(?:\[\s*\d{1,2}\s*\]|\(\s*\d{1,2}\s*\))",
        " ",
        text
    )

    return text


# --------------------------------------------------
# 4. Extract financial numerical values
# --------------------------------------------------

def extract_numbers(text):
    """
    Extract numerical values from financial text.

    Supports:
        22,387
        22387
        4,720.50
        -450
        (1,250)
        21.08%
        2.04x
    """

    text = clean_text_for_number_check(text)

    pattern = re.compile(
        r"(?<![A-Za-z0-9])"
        r"(?:"
        r"\(\s*\d[\d,]*(?:\.\d+)?\s*\)"
        r"|"
        r"[-+]?\d[\d,]*(?:\.\d+)?"
        r")"
        r"\s*(?:%|[xX])?"
        r"(?![A-Za-z0-9])"
    )

    matches = pattern.findall(text)

    numbers = set()

    for value in matches:

        value = re.sub(r"\s+", "", value)

        # Parentheses represent negative financial values
        if value.startswith("(") and value.endswith(")"):
            value = "-" + value[1:-1]

        suffix = ""

        if value.endswith("%") or value.lower().endswith("x"):
            suffix = value[-1].lower()
            value = value[:-1]

        value = value.replace(",", "")

        try:
            number = float(value)

            if number.is_integer():
                normalized = str(int(number))
            else:
                normalized = str(number)

            numbers.add(normalized + suffix)

        except ValueError:
            continue

    return numbers


# --------------------------------------------------
# 5. Validate generated answer
# --------------------------------------------------

def validate_answer(answer, documents):
    """
    Check whether numerical values in the generated answer
    are supported by retrieved context.

    This validates numerical presence, not semantic accuracy.
    """

    if not answer or not answer.strip():
        return False

    if not documents:
        return False

    context = " ".join(
        document.page_content
        for document in documents
    )

    answer_numbers = extract_numbers(answer)
    context_numbers = extract_numbers(context)

    unsupported_numbers = answer_numbers - context_numbers

    if unsupported_numbers:
        print(
            "Unsupported numerical values:",
            unsupported_numbers
        )
        return False

    return True


# --------------------------------------------------
# 6. Fallback response
# --------------------------------------------------

def no_answer_response():
    return NO_ANSWER


if __name__ == "__main__":

    test_values = [
        "Revenue was $22,387.",
        "Gross profit was $4,720 and margin was 21.08%.",
        "Net cash used in investing activities was $(5,023).",
        "Revenue was $22,387 [1].",
        "Revenue was $22,387 (1).",
    ]

    for value in test_values:
        print(value)
        print(extract_numbers(value))
        print()



