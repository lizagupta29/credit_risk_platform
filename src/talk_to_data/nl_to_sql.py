"""
nl_to_sql.py
"""

import os
import google.generativeai as genai

from dotenv import load_dotenv

from src.talk_to_data.prompt_templates import (
    SYSTEM_PROMPT,
    INSIGHT_PROMPT
)

load_dotenv()

genai.configure(
    api_key=os.getenv(
        "GEMINI_API_KEY"
    )
)


class NLtoSQL:

    def __init__(self):

        self.model = genai.GenerativeModel(
            "gemini-2.5-flash"
        )

    def generate_sql(
        self,
        user_question
    ):

        prompt = f"""
        {SYSTEM_PROMPT}

        User Question:
        {user_question}
        """

        response = self.model.generate_content(
            prompt
        )

        sql = response.text.strip()

        sql = (
            sql.replace(
                "```sql",
                ""
            )
            .replace(
                "```",
                ""
            )
            .strip()
        )

        return sql

    def generate_insight(
        self,
        result_df
    ):

        prompt = (
            INSIGHT_PROMPT.format(
                result=result_df.to_string()
            )
        )

        response = self.model.generate_content(
            prompt
        )

        return response.text


if __name__ == "__main__":

    agent = NLtoSQL()

    question = (
        "Which education group "
        "has highest default rate?"
    )

    sql = agent.generate_sql(
        question
    )

    print(sql)
