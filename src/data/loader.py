"""
loader.py

Load all Home Credit dataset files.
"""

from pathlib import Path
import pandas as pd


class DataLoader:

    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)

    def load_application_train(self):
        return pd.read_csv(
            self.data_dir / "application_train.csv"
        )

    def load_application_test(self):
        return pd.read_csv(
            self.data_dir / "application_test.csv"
        )

    def load_bureau(self):
        return pd.read_csv(
            self.data_dir / "bureau.csv"
        )

    def load_bureau_balance(self):
        return pd.read_csv(
            self.data_dir / "bureau_balance.csv"
        )

    def load_previous_application(self):
        return pd.read_csv(
            self.data_dir / "previous_application.csv"
        )

    def load_pos_cash_balance(self):
        return pd.read_csv(
            self.data_dir / "POS_CASH_balance.csv"
        )

    def load_installments(self):
        return pd.read_csv(
            self.data_dir / "installments_payments.csv"
        )

    def load_credit_card_balance(self):
        return pd.read_csv(
            self.data_dir / "credit_card_balance.csv"
        )

    def load_all(self):

        return {
            "application_train":
                self.load_application_train(),

            "application_test":
                self.load_application_test(),

            "bureau":
                self.load_bureau(),

            "bureau_balance":
                self.load_bureau_balance(),

            "previous_application":
                self.load_previous_application(),

            "pos_cash_balance":
                self.load_pos_cash_balance(),

            "installments":
                self.load_installments(),

            "credit_card_balance":
                self.load_credit_card_balance()
        }


if __name__ == "__main__":

    loader = DataLoader()

    train = loader.load_application_train()

    print(train.shape)
