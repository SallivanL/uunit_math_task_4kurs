import os
from datetime import datetime


LOG_FILE = "results/log.md"


def log(message="", **data):
    os.makedirs("results", exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    line = f"### {timestamp}\n\n"

    if message:
        line += f"{message}\n\n"

    if data:
        for key, value in data.items():
            line += f"- **{key}:** {value}\n"

        line += "\n"

    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(line)