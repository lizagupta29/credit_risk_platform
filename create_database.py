import sqlite3
import pandas as pd

DATABASE_NAME = "credit_risk.db"
CSV_PATH = "data/application_train.csv"

df = pd.read_csv(CSV_PATH)

conn = sqlite3.connect(DATABASE_NAME)

df.to_sql(
    "application_train",
    conn,
    if_exists="replace",
    index=False
)

conn.close()

print("Database created successfully!")
