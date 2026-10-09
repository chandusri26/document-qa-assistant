import os
from dotenv import load_dotenv


load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if api_key:
    print("API key loaded successfully.")
    print(f"Key starts with: {api_key[:10]}...")
else:
    print("API key was not found.")