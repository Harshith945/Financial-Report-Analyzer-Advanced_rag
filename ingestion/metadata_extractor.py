import re
from datetime import datetime


class FinancialMetadataExtractor:

    def extract_metadata(self, extracted_data):

        text_elements = extracted_data["text_elements"]

        # Collect text from the first three pages
        first_pages = [
            item["text"]
            for item in text_elements
            if item["page_number"] is not None
            and item["page_number"] <= 3
        ]

        text = "\n".join(first_pages)

        metadata = {
            "company": None,
            "ticker": None,
            "report_type": None,
            "fiscal_year": None,
            "fiscal_quarter": None,
            "period_end": None
        }

        # Identify company name
        company_match = re.search(
            r"([A-Z][A-Za-z0-9.,& ]+?)\s*"
            r"\n\(Exact name of registrant",
            text
        )

        if company_match:
            metadata["company"] = company_match.group(1).strip()

        # Identify report type
        if re.search(
            r"QUARTERLY REPORT PURSUANT TO",
            text,
            re.IGNORECASE
        ):
            metadata["report_type"] = "10-Q"

        elif re.search(
            r"ANNUAL REPORT PURSUANT TO",
            text,
            re.IGNORECASE
        ):
            metadata["report_type"] = "10-K"

        # Identify reporting period
        period_match = re.search(
            r"quarterly period ended\s+"
            r"([A-Za-z]+ \d{1,2}, \d{4})",
            text,
            re.IGNORECASE
        )

        if period_match:

            period_date = datetime.strptime(
                period_match.group(1),
                "%B %d, %Y"
            )

            metadata["period_end"] = (
                period_date.strftime("%Y-%m-%d")
            )

            metadata["fiscal_year"] = period_date.year

            month = period_date.month

            metadata["fiscal_quarter"] = (
                f"Q{((month - 1) // 3) + 1}"
            )

        # Identify ticker from a controlled mapping
        ticker_mapping = {
            "Tesla, Inc.": "TSLA"
        }

        metadata["ticker"] = ticker_mapping.get(
            metadata["company"]
        )

        return metadata


if __name__ == "__main__":

    import json
    import glob

    files = glob.glob(
        "data/extracted/*/financial_extraction.json"
    )

    if not files:
        raise FileNotFoundError("No extracted reports found.")

    with open(files[-1], "r", encoding="utf-8") as file:
        extracted_data = json.load(file)

    extractor = FinancialMetadataExtractor()

    metadata = extractor.extract_metadata(extracted_data)

    print("\nEXTRACTED REPORT METADATA")
    print("------------------------")

    for key, value in metadata.items():
        print(f"{key}: {value}")