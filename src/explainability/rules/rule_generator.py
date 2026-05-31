"""
rule_generator.py

Generate business-readable credit rules.
"""

import pandas as pd


class RuleGenerator:

    def __init__(self):
        pass

    def generate_rule(
            self,
            customer_data,
            risk_band
    ):

        rules = []

        income = customer_data.get(
            "AMT_INCOME_TOTAL",
            0
        )

        credit = customer_data.get(
            "AMT_CREDIT",
            0
        )

        annuity = customer_data.get(
            "AMT_ANNUITY",
            0
        )

        age = customer_data.get(
            "AGE_YEARS",
            0
        )

        if income < 100000:

            rules.append(
                "Income is relatively low."
            )

        if credit > 800000:

            rules.append(
                "Requested credit amount is high."
            )

        if annuity > 50000:

            rules.append(
                "Loan annuity burden is high."
            )

        if age < 25:

            rules.append(
                "Applicant is relatively young."
            )

        if risk_band == "High":

            decision = (
                "Recommend manual review."
            )

        elif risk_band == "Medium":

            decision = (
                "Recommend additional verification."
            )

        else:

            decision = (
                "Suitable for automatic approval."
            )

        return {

            "risk_band":
                risk_band,

            "rules":
                rules,

            "decision":
                decision
        }

    def generate_dataframe(
            self,
            customer_data,
            risk_band
    ):

        result = self.generate_rule(
            customer_data,
            risk_band
        )

        return pd.DataFrame({

            "Rule":
                result["rules"]

        })


if __name__ == "__main__":

    sample = {

        "AMT_INCOME_TOTAL":
            80000,

        "AMT_CREDIT":
            1000000,

        "AMT_ANNUITY":
            70000,

        "AGE_YEARS":
            23
    }

    generator = RuleGenerator()

    result = generator.generate_rule(
        sample,
        "High"
    )

    print(result)
