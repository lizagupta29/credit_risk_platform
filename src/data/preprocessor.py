"""
preprocessor.py

Data cleaning and feature engineering.
"""

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


class DataPreprocessor:

    def __init__(self):

        self.label_encoders = {}

        self.imputer = SimpleImputer(
            strategy="median"
        )

    def remove_high_missing_columns(
            self,
            df,
            threshold=0.60
    ):

        missing_ratio = (
            df.isnull()
            .mean()
        )

        cols_to_keep = (
            missing_ratio[
                missing_ratio < threshold
            ]
            .index
        )

        return df[cols_to_keep]

    def create_features(
            self,
            df
    ):

        if (
            "AMT_CREDIT" in df.columns and
            "AMT_INCOME_TOTAL" in df.columns
        ):

            df[
                "CREDIT_INCOME_RATIO"
            ] = (
                df["AMT_CREDIT"] /
                (
                    df["AMT_INCOME_TOTAL"]
                    + 1
                )
            )

        if (
            "AMT_ANNUITY" in df.columns and
            "AMT_INCOME_TOTAL" in df.columns
        ):

            df[
                "ANNUITY_INCOME_RATIO"
            ] = (
                df["AMT_ANNUITY"] /
                (
                    df["AMT_INCOME_TOTAL"]
                    + 1
                )
            )

        if "DAYS_BIRTH" in df.columns:

            df["AGE_YEARS"] = (
                abs(
                    df["DAYS_BIRTH"]
                ) / 365
            )

        if "DAYS_EMPLOYED" in df.columns:

            df[
                "EMPLOYMENT_YEARS"
            ] = (
                abs(
                    df["DAYS_EMPLOYED"]
                ) / 365
            )

        return df

    def encode_categorical(
            self,
            df
    ):

        cat_cols = (
            df.select_dtypes(
                include=["object"]
            )
            .columns
        )

        for col in cat_cols:

            le = LabelEncoder()

            df[col] = (
                df[col]
                .fillna("Unknown")
                .astype(str)
            )

            df[col] = (
                le.fit_transform(
                    df[col]
                )
            )

            self.label_encoders[col] = le

        return df

    def impute_missing(
            self,
            df
    ):

        df[df.columns] = (
            self.imputer.fit_transform(
                df
            )
        )

        return df

    def remove_identifier_columns(
            self,
            df
    ):

        cols_to_drop = []

        for col in df.columns:

            if col.startswith(
                    "SK_ID"
            ):
                cols_to_drop.append(
                    col
                )

        return df.drop(
            columns=cols_to_drop,
            errors="ignore"
        )

    def preprocess(
            self,
            df
    ):

        df = df.copy()

        df = (
            self.remove_high_missing_columns(
                df
            )
        )

        df = (
            self.create_features(
                df
            )
        )

        df = (
            self.remove_identifier_columns(
                df
            )
        )

        df = (
            self.encode_categorical(
                df
            )
        )

        df = (
            self.impute_missing(
                df
            )
        )

        return df


if __name__ == "__main__":

    sample = pd.DataFrame({

        "Gender": [
            "M",
            "F",
            np.nan
        ],

        "Income": [
            10000,
            np.nan,
            20000
        ]
    })

    processor = (
        DataPreprocessor()
    )

    processed = (
        processor.preprocess(
            sample
        )
    )

    print(processed)
