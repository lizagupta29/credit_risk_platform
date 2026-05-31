import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")

DATA_PATH = "../data/application_train.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset Shape: {df.shape}")
    return df


def basic_info(df):

    print("\n========== HEAD ==========")
    print(df.head())

    print("\n========== INFO ==========")
    print(df.info())

    print("\n========== DESCRIBE ==========")
    print(df.describe())


def missing_values(df):

    missing = pd.DataFrame({
        "Missing Count": df.isnull().sum(),
        "Missing %": round(
            df.isnull().mean() * 100,
            2
        )
    })

    missing = missing.sort_values(
        by="Missing %",
        ascending=False
    )

    print("\nTop Missing Columns")
    print(missing.head(20))

    return missing


def target_distribution(df):

    plt.figure(figsize=(6, 4))

    sns.countplot(
        x="TARGET",
        data=df
    )

    plt.title("Default Distribution")

    plt.savefig(
        "../documents/default_distribution.png"
    )

    plt.show()


def income_distribution(df):

    plt.figure(figsize=(8, 5))

    sns.histplot(
        df["AMT_INCOME_TOTAL"],
        bins=50,
        kde=True
    )

    plt.title("Income Distribution")

    plt.savefig(
        "../documents/income_distribution.png"
    )

    plt.show()


def age_distribution(df):

    age_years = abs(df["DAYS_BIRTH"]) / 365

    plt.figure(figsize=(8, 5))

    sns.histplot(
        age_years,
        bins=40
    )

    plt.title("Age Distribution")

    plt.savefig(
        "../documents/age_distribution.png"
    )

    plt.show()


def default_rate_by_gender(df):

    result = df.groupby(
        "CODE_GENDER"
    )["TARGET"].mean()

    result.plot(
        kind="bar",
        figsize=(6, 4)
    )

    plt.title(
        "Default Rate by Gender"
    )

    plt.ylabel(
        "Default Rate"
    )

    plt.savefig(
        "../documents/default_gender.png"
    )

    plt.show()


def default_rate_by_education(df):

    result = df.groupby(
        "NAME_EDUCATION_TYPE"
    )["TARGET"].mean()

    result.sort_values().plot(
        kind="barh",
        figsize=(10, 5)
    )

    plt.title(
        "Default Rate by Education"
    )

    plt.savefig(
        "../documents/default_education.png"
    )

    plt.show()


def correlation_analysis(df):

    numeric_df = df.select_dtypes(
        include=np.number
    )

    corr = numeric_df.corr()

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        corr,
        cmap="coolwarm"
    )

    plt.title(
        "Correlation Heatmap"
    )

    plt.savefig(
        "../documents/correlation_heatmap.png"
    )

    plt.show()


def business_insights(df):

    print("\n===== BUSINESS INSIGHTS =====")

    default_rate = round(
        df["TARGET"].mean() * 100,
        2
    )

    print(
        f"Overall Default Rate: {default_rate}%"
    )

    highest_edu = (
        df.groupby(
            "NAME_EDUCATION_TYPE"
        )["TARGET"]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    print(
        "\nHighest Default Education Group:"
    )
    print(
        highest_edu.head(1)
    )

    highest_gender = (
        df.groupby(
            "CODE_GENDER"
        )["TARGET"]
        .mean()
    )

    print(
        "\nDefault Rate by Gender:"
    )

    print(
        highest_gender
    )


def main():

    df = load_data()

    basic_info(df)

    missing_values(df)

    target_distribution(df)

    income_distribution(df)

    age_distribution(df)

    default_rate_by_gender(df)

    default_rate_by_education(df)

    correlation_analysis(df)

    business_insights(df)


if __name__ == "__main__":
    main()
