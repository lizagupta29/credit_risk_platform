"""
helpers.py
"""

import os
import pandas as pd


def create_directory(
        path
):

    os.makedirs(
        path,
        exist_ok=True
    )


def save_dataframe(
        df,
        filepath
):

    df.to_csv(
        filepath,
        index=False
    )


def load_dataframe(
        filepath
):

    return pd.read_csv(
        filepath
    )


def percentage(
        numerator,
        denominator
):

    if denominator == 0:
        return 0

    return round(
        (numerator / denominator) * 100,
        2
    )


def risk_band(
        probability
):

    if probability < 0.30:
        return "Low"

    elif probability < 0.70:
        return "Medium"

    return "High"
