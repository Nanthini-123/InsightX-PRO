# modules/chatbot.py
import os
import requests
import pandas as pd

class DataChatbot:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set. Run: export OPENROUTER_API_KEY=your_key")

    def ask(self, question: str) -> str:
        # Lightweight dataset summary
        summary = f"Dataset has {self.df.shape[0]} rows and {self.df.shape[1]} columns. " \
                  f"Columns: {', '.join(self.df.columns[:10])}"

        prompt = f"""
        You are an AI data assistant. 
        Dataset Summary: {summary}
        Question: {question}
        Answer in plain English, give insights, and suggest chart types if relevant.
        """

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "google/gemini-2.0-flash-001",   # or another available model
            "messages": [{"role": "user", "content": prompt}],
        }

        response = requests.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            return f"Error: {response.text}"

        return response.json()["choices"][0]["message"]["content"]