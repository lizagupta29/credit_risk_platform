"""
query_runner.py
"""

import sqlite3
import pandas as pd


class QueryRunner:

    def __init__(
        self,
        db_path="credit_risk.db"
    ):
        self.db_path = db_path

    def run_query(
        self,
        sql_query
    ):

        try:

            conn = sqlite3.connect(
                self.db_path
            )

            df = pd.read_sql_query(
                sql_query,
                conn
            )

            conn.close()

            return df

        except Exception as e:

            return pd.DataFrame({
                "error": [str(e)]
            })


if __name__ == "__main__":

    runner = QueryRunner()

    query = """
    SELECT COUNT(*) as total_rows
    FROM application_train
    """

    result = runner.run_query(
        query
    )

    print(result)
