import os

from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
# The model the app under test calls. Override with RAGAS_EXAMPLE_APP_MODEL.
MODEL = os.getenv("RAGAS_EXAMPLE_APP_MODEL", "gpt-6-luna")


SYSTEM_PROMPT = """
You are a helpful assistant. I will provide a movie review and you will classify it as either positive or negative.
Please respond with "positive" or "negative" only.
"""


def run_prompt(prompt: str):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    response = (
        response.choices[0].message.content.strip()
        if response.choices[0].message.content
        else ""
    )
    return response


if __name__ == "__main__":
    prompt = "The movie was fantastic and I loved every moment of it!"
    print(run_prompt(prompt))
