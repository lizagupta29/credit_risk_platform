"""
predict.py

Generate risk prediction for a customer.
"""

import joblib
import pandas as pd


MODEL_PATH = "models/saved_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"


class CreditRiskPredictor:

    def __init__(self):

        self.model = joblib.load(
            MODEL_PATH
        )

        self.preprocessor = joblib.load(
            PREPROCESSOR_PATH
        )

    def get_risk_band(
            self,
            probability
    ):

        if probability < 0.30:
            return "Low"

        elif probability < 0.70:
            return "Medium"

        return "High"

    def predict(
            self,
            customer_data
    ):

        df = pd.DataFrame(
            [customer_data]
        )

        df = self.preprocessor.preprocess(
            df
        )

        probability = (
            self.model
            .predict_proba(df)[0][1]
        )

        risk_band = (
            self.get_risk_band(
                probability
            )
        )

        return {
            "default_probability":
                round(probability, 4),

            "risk_score":
                round(probability * 100, 2),

            "risk_band":
                risk_band
        }


if __name__ == "__main__":

    sample_customer = {

        "AMT_INCOME_TOTAL": 200000,
        "AMT_CREDIT": 500000,
        "AMT_ANNUITY": 25000,
        "DAYS_BIRTH": -12000,
        "DAYS_EMPLOYED": -2500
    }

    predictor = CreditRiskPredictor()

    result = predictor.predict(
        sample_customer
    )

    print(result)
