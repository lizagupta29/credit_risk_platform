"""
shap_explainer.py

Generate SHAP explanations for LightGBM predictions.
"""

import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt


MODEL_PATH = "models/saved_model.pkl"


class SHAPExplainer:

    def __init__(self):

        self.model = joblib.load(
            MODEL_PATH
        )

        self.explainer = shap.TreeExplainer(
            self.model
        )

    def explain_instance(
            self,
            input_df
    ):

        shap_values = (
            self.explainer.shap_values(
                input_df
            )
        )

        return shap_values

    def summary_plot(
            self,
            X
    ):

        shap_values = (
            self.explainer.shap_values(
                X
            )
        )

        shap.summary_plot(
            shap_values,
            X,
            show=False
        )

        plt.savefig(
            "documents/shap_summary.png",
            bbox_inches="tight"
        )

        plt.close()

    def waterfall_plot(
            self,
            X
    ):

        shap_values = (
            self.explainer(
                X
            )
        )

        shap.plots.waterfall(
            shap_values[0],
            show=False
        )

        plt.savefig(
            "documents/shap_waterfall.png",
            bbox_inches="tight"
        )

        plt.close()

    def feature_importance(
            self,
            X
    ):

        shap_values = (
            self.explainer.shap_values(
                X
            )
        )

        importance = pd.DataFrame({

            "feature":
                X.columns,

            "importance":
                abs(
                    shap_values
                ).mean(axis=0)

        })

        importance = (
            importance
            .sort_values(
                by="importance",
                ascending=False
            )
        )

        return importance


if __name__ == "__main__":

    sample = pd.DataFrame({

        "AMT_INCOME_TOTAL":
            [200000],

        "AMT_CREDIT":
            [500000],

        "AMT_ANNUITY":
            [25000],

        "AGE_YEARS":
            [35]

    })

    explainer = SHAPExplainer()

    print(
        explainer.feature_importance(
            sample
        )
    )
