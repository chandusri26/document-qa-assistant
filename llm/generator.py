import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class LLMGenerator:

    def __init__(self):

        self.api_key = os.getenv(
            "OPENROUTER_API_KEY"
        )

        self.model = os.getenv(
            "OPENROUTER_MODEL",
            "openrouter/free"
        )

        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY was not found. "
                "Check your .env file."
            )

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key
        )

    def generate(self, prompt):

        response = self.client.chat.completions.create(
            model=self.model,

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content