
import json
import pandas as pd

from pathlib import Path


class FinancialDataProcessor:

    def __init__(self):

        self.file_path = Path(
            "data/extracted/financial_extraction.json"
        )

        with open(self.file_path, "r", encoding="utf-8") as file:
            self.data = json.load(file)

    def get_table_by_page(self, page_number):

        for table in self.data["tables"]:

            if table["page_number"] == page_number:

                return pd.DataFrame(
                    table["rows"],
                    columns=table["columns"]
                )

        return None

    def clean_numeric_value(self, value):

        if not isinstance(value, str):
            return value

        value = value.strip().replace("$", "").replace(",", "")

        if value in ["", "-", "—"]:
            return None

        if value.startswith("(") and value.endswith(")"):
            value = "-" + value[1:-1]

        try:
            return float(value)
        except ValueError:
            return value

    def get_value(self, page_number, metric_name, column_index=1):

        dataframe = self.get_table_by_page(page_number)

        if dataframe is None:
            raise ValueError(f"Table not found on page {page_number}")

        matches = dataframe[
            dataframe.iloc[:, 0].astype(str).str.strip() == metric_name
        ]

        if matches.empty:
            raise ValueError(f"Metric not found: {metric_name}")

        value = matches.iloc[0, column_index]

        return self.clean_numeric_value(value)

    def calculate_revenue_growth(self):

        current_revenue = self.get_value(
            5, "Total revenues", 1
        )

        previous_revenue = self.get_value(
            5, "Total revenues", 2
        )

        return (
            (current_revenue - previous_revenue)
            / previous_revenue
        ) * 100

    def calculate_profit_margins(self):

        revenue = self.get_value(5, "Total revenues", 1)

        gross_profit = self.get_value(5, "Gross profit", 1)

        operating_income = self.get_value(
            5, "Income from operations", 1
        )

        net_income = self.get_value(5, "Net income", 1)

        return {
            "gross_profit_margin": gross_profit / revenue * 100,
            "operating_profit_margin": operating_income / revenue * 100,
            "net_profit_margin": net_income / revenue * 100
        }

    def calculate_current_ratio(self):

        current_assets = self.get_value(
            4, "Total current assets", 1
        )

        current_liabilities = self.get_value(
            4, "Total current liabilities", 1
        )

        return current_assets / current_liabilities

    def calculate_all_metrics(self):

        return {
            "revenue_growth": self.calculate_revenue_growth(),
            **self.calculate_profit_margins(),
            "current_ratio": self.calculate_current_ratio()
        }


if __name__ == "__main__":

    processor = FinancialDataProcessor()

    metrics = processor.calculate_all_metrics()

    print("\nFINANCIAL ANALYSIS RESULTS")
    print("=" * 50)

    for metric, value in metrics.items():

        if metric == "current_ratio":
            print(f"{metric}: {value:.2f}x")
        else:
            print(f"{metric}: {value:.2f}%")

