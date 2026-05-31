import streamlit as st
import pandas as pd
import joblib
import sqlite3
import plotly.express as px

from src.ml.predict import CreditRiskPredictor
from src.explainability.shap_explainer import SHAPExplainer
from src.rules.rule_generator import RuleGenerator
from src.talk_to_data.nl_to_sql import NLtoSQL
from src.talk_to_data.query_runner import QueryRunner


st.set_page_config(
    page_title="Credit Risk Intelligence Platform",
    layout="wide"
)

st.title("🏦 AI-Powered Credit Risk Intelligence Platform")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

page = st.sidebar.selectbox(
    "Select Module",
    [
        "EDA Dashboard",
        "Risk Prediction",
        "Explainability",
        "Business Rules",
        "Talk-to-Data"
    ]
)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(
        "data/application_train.csv"
    )

df = load_data()

# --------------------------------------------------
# EDA
# --------------------------------------------------

if page == "EDA Dashboard":

    st.header("Dataset Overview")

    st.write(df.head())

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Rows",
            df.shape[0]
        )

    with col2:
        st.metric(
            "Columns",
            df.shape[1]
        )

    st.subheader("Default Distribution")

    fig = px.histogram(
        df,
        x="TARGET"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    if "AMT_INCOME_TOTAL" in df.columns:

        st.subheader(
            "Income Distribution"
        )

        fig = px.histogram(
            df,
            x="AMT_INCOME_TOTAL"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    if "CODE_GENDER" in df.columns:

        st.subheader(
            "Default Rate by Gender"
        )

        gender_df = (
            df.groupby("CODE_GENDER")
            ["TARGET"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            gender_df,
            x="CODE_GENDER",
            y="TARGET"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# --------------------------------------------------
# Risk Prediction
# --------------------------------------------------

elif page == "Risk Prediction":

    st.header("Loan Default Prediction")

    income = st.number_input(
        "Income",
        value=200000
    )

    credit = st.number_input(
        "Credit Amount",
        value=500000
    )

    annuity = st.number_input(
        "Annuity",
        value=25000
    )

    age = st.number_input(
        "Age",
        value=35
    )

    employment = st.number_input(
        "Employment Years",
        value=5
    )

    if st.button("Predict Risk"):

        predictor = CreditRiskPredictor()

        customer = {

            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": annuity,
            "AGE_YEARS": age,
            "EMPLOYMENT_YEARS": employment
        }

        result = predictor.predict(
            customer
        )

        st.success(
            f"Risk Band: {result['risk_band']}"
        )

        st.metric(
            "Risk Score",
            result["risk_score"]
        )

        st.metric(
            "Default Probability",
            result["default_probability"]
        )

# --------------------------------------------------
# Explainability
# --------------------------------------------------

elif page == "Explainability":

    st.header("SHAP Explainability")

    try:

        shap_img = (
            "documents/shap_summary.png"
        )

        st.image(
            shap_img,
            caption="SHAP Summary Plot"
        )

    except Exception:

        st.warning(
            "Run SHAP generation first."
        )

# --------------------------------------------------
# Rules
# --------------------------------------------------

elif page == "Business Rules":

    st.header(
        "Business Readable Rules"
    )

    income = st.number_input(
        "Income",
        value=80000
    )

    credit = st.number_input(
        "Credit Amount",
        value=1000000
    )

    annuity = st.number_input(
        "Annuity",
        value=70000
    )

    age = st.number_input(
        "Age",
        value=23
    )

    risk_band = st.selectbox(
        "Risk Band",
        [
            "Low",
            "Medium",
            "High"
        ]
    )

    if st.button("Generate Rules"):

        generator = RuleGenerator()

        customer = {

            "AMT_INCOME_TOTAL":
                income,

            "AMT_CREDIT":
                credit,

            "AMT_ANNUITY":
                annuity,

            "AGE_YEARS":
                age
        }

        result = (
            generator.generate_rule(
                customer,
                risk_band
            )
        )

        st.subheader(
            "Decision"
        )

        st.info(
            result["decision"]
        )

        st.subheader(
            "Generated Rules"
        )

        for rule in result["rules"]:

            st.write(
                f"• {rule}"
            )

# --------------------------------------------------
# Talk to Data
# --------------------------------------------------

elif page == "Talk-to-Data":

    st.header(
        "Natural Language to SQL"
    )

    question = st.text_input(
        "Ask a question"
    )

    if st.button("Submit"):

        if question:

            try:

                agent = NLtoSQL()

                sql_query = (
                    agent.generate_sql(
                        question
                    )
                )

                st.subheader(
                    "Generated SQL"
                )

                st.code(
                    sql_query,
                    language="sql"
                )

                runner = QueryRunner()

                result = (
                    runner.run_query(
                        sql_query
                    )
                )

                st.subheader(
                    "Query Result"
                )

                st.dataframe(
                    result
                )

                insight = (
                    agent.generate_insight(
                        result
                    )
                )

                st.subheader(
                    "Business Insight"
                )

                st.write(
                    insight
                )

            except Exception as e:

                st.error(str(e))
