"""
config.py
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:

    # Project Paths
    DATA_DIR = "data"
    MODEL_DIR = "models"
    DOCUMENT_DIR = "documents"

    # Database
    DB_PATH = "credit_risk.db"

    # Gemini
    GEMINI_API_KEY = os.getenv(
        "GEMINI_API_KEY"
    )

    GEMINI_MODEL = "gemini-2.5-flash"

    # Model
    TARGET_COLUMN = "TARGET"

    TEST_SIZE = 0.2

    RANDOM_STATE = 42

    # LightGBM
    N_ESTIMATORS = 500

    LEARNING_RATE = 0.05

    MAX_DEPTH = 8

    NUM_LEAVES = 31

    # Risk Thresholds
    LOW_RISK_THRESHOLD = 0.30

    HIGH_RISK_THRESHOLD = 0.70
