"""
docker_utils.py
"""

import os


def check_environment():

    required_vars = [
        "GEMINI_API_KEY"
    ]

    missing = []

    for var in required_vars:

        if not os.getenv(var):

            missing.append(var)

    return missing


def validate_startup():

    missing = check_environment()

    if missing:

        raise EnvironmentError(
            f"Missing env vars: {missing}"
        )

    print(
        "Environment validation passed."
    )


if __name__ == "__main__":

    validate_startup()
