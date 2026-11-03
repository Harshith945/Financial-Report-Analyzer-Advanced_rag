
import re
from difflib import get_close_matches


class QueryExpander:

    FINANCIAL_TERMS = [
        "revenue",
        "sales",
        "profit",
        "gross",
        "net",
        "income",
        "expense",
        "expenses",
        "operating",
        "asset",
        "assets",
        "liability",
        "liabilities",
        "cash",
        "balance",
        "equity",
        "debt",
        "growth",
        "margin",
        "ratio"
    ]

    QUARTER_WORDS = {
        "first quarter": "Q1",
        "1st quarter": "Q1",
        "quarter one": "Q1",
        "second quarter": "Q2",
        "2nd quarter": "Q2",
        "quarter two": "Q2",
        "third quarter": "Q3",
        "3rd quarter": "Q3",
        "quarter three": "Q3",
        "fourth quarter": "Q4",
        "4th quarter": "Q4",
        "quarter four": "Q4"
    }

    QUARTER_END_DATES = {
        1: "March 31",
        2: "June 30",
        3: "September 30",
        4: "December 31"
    }

    def correct_spelling(self, query):

        words = re.findall(r"\b[a-zA-Z]+\b", query)

        for word in words:

            word_lower = word.lower()

            matches = get_close_matches(
                word_lower,
                self.FINANCIAL_TERMS,
                n=1,
                cutoff=0.80
            )

            if matches and word_lower != matches[0]:

                query = re.sub(
                    rf"\b{re.escape(word)}\b",
                    matches[0],
                    query,
                    count=1,
                    flags=re.IGNORECASE
                )

        return query

    def normalize_quarter(self, query):

        for phrase, quarter in self.QUARTER_WORDS.items():

            query = re.sub(
                rf"\b{phrase}\b",
                quarter,
                query,
                flags=re.IGNORECASE
            )

        return query

    def identify_metric(self, query):

        query_lower = query.lower()

        # Check more specific metrics before generic terms.
        if re.search(r"\b(operating cash flow|cash flow from operating activities|net cash provided by operating activities|net cash flows from operating activities)\b", query_lower):
            return "Net cash provided by operating activities"

        elif "revenue" in query_lower or "sales" in query_lower:
            return "Total revenues"

        elif "gross profit" in query_lower:
            return "Gross profit"

        elif "net income" in query_lower or "net profit" in query_lower:
            return "Net income"

        elif "profit" in query_lower:
            return "Gross profit"

        elif "expense" in query_lower:
            return "Operating expenses"

        elif "asset" in query_lower:
            return "Total assets"

        elif "liabilit" in query_lower:
            return "Total liabilities"

        elif "cash" in query_lower:
            return "Cash and cash equivalents"

        return None

    def identify_quarter(self, query):

        quarter_match = re.search(
            r"\bQ([1-4])\b",
            query,
            re.IGNORECASE
        )

        if not quarter_match:
            return None

        return int(quarter_match.group(1))

    def identify_period(self, query):

        quarter_match = re.search(
            r"\bQ([1-4])\s*(?:of\s*)?(20\d{2})\b",
            query,
            re.IGNORECASE
        )

        if not quarter_match:
            return None

        quarter = int(quarter_match.group(1))
        year = quarter_match.group(2)

        return (
            f"Three Months Ended "
            f"{self.QUARTER_END_DATES[quarter]} {year}"
        )

    def get_company_from_query(self, query, available_periods):

        query_lower = query.lower()

        companies = set()

        for item in available_periods:

            company = item.get("company")
            ticker = item.get("ticker")

            if company:
                companies.add(company)

            if ticker:
                companies.add(ticker)

        for company in companies:

            if company and company.lower() in query_lower:
                return company

        return None

    def find_latest_period(
        self,
        quarter,
        available_periods,
        query
    ):

        matching_periods = []

        requested_company = self.get_company_from_query(
            query,
            available_periods
        )

        expected_month = quarter * 3

        for item in available_periods:

            if not isinstance(item, dict):
                continue

            company = item.get("company")
            ticker = item.get("ticker")
            period_end = item.get("period_end")

            if not period_end:
                continue

            # Filter by company only when the user specifies one.
            if requested_company:

                if (
                    company != requested_company
                    and ticker != requested_company
                ):
                    continue

            date_match = re.search(
                r"(20\d{2})-(\d{2})-(\d{2})",
                str(period_end)
            )

            if not date_match:
                continue

            year, month, day = map(
                int,
                date_match.groups()
            )

            if month != expected_month:
                continue

            matching_periods.append({
                "year": year,
                "company": company,
                "ticker": ticker,
                "period_end": period_end
            })

        if not matching_periods:
            return None

        # Select the latest matching reporting period.
        latest_period = max(
            matching_periods,
            key=lambda item: item["year"]
        )

        return latest_period

    def expand_query(
        self,
        query,
        available_periods=None
    ):

        query = query.strip()

        if not query:
            return []

        if available_periods is None:
            available_periods = []

        expanded_queries = [query]

        # Step 1: Correct common financial spelling mistakes.
        corrected_query = self.correct_spelling(query)

        # Step 2: Normalize quarter expressions.
        corrected_query = self.normalize_quarter(
            corrected_query
        )

        if corrected_query.lower() != query.lower():
            expanded_queries.append(corrected_query)

        # Step 3: Identify financial metric.
        metric = self.identify_metric(corrected_query)

        # Step 4: Identify explicitly requested reporting period.
        period = self.identify_period(corrected_query)

        if metric:

            expanded_queries.append(metric)

            # User explicitly provided the year.
            if period:

                expanded_queries.append(
                    f"{metric} {period}"
                )

            else:

                # User mentioned a quarter without a year.
                quarter = self.identify_quarter(
                    corrected_query
                )

                if quarter:

                    latest_period = self.find_latest_period(
                        quarter,
                        available_periods,
                        corrected_query
                    )

                    if latest_period:

                        year = latest_period["year"]

                        period_text = (
                            f"Three Months Ended "
                            f"{self.QUARTER_END_DATES[quarter]} "
                            f"{year}"
                        )

                        expanded_queries.append(
                            f"{metric} {period_text}"
                        )

        # Remove duplicate queries while preserving order.
        return list(dict.fromkeys(expanded_queries))


if __name__ == "__main__":

    expander = QueryExpander()

    available_periods = [
        {
            "company": "Tesla, Inc.",
            "ticker": "TSLA",
            "period_end": "2026-03-31"
        },
        {
            "company": "Tesla, Inc.",
            "ticker": "TSLA",
            "period_end": "2025-03-31"
        }
    ]

    test_queries = [
        "What was Tesla's revenue in Q1?",
        "What was Tesla's revenue in Q1 2026?",
        "What was Tesla's revenue in first quarter?",
        "What was Tesla's gross profit in Q1?",
        "What were Tesla's total assets?",
        "What was Tesla's operating cash flow?"
    ]

    for query in test_queries:

        print("\nOriginal query:", query)

        print("Expanded queries:")

        for item in expander.expand_query(
            query,
            available_periods
        ):
            print("-", item)

