# modules/gemini_chat.py
import requests
import os

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

def ask_gemini(prompt, model="google/gemini-2.0-flash-001", temperature=0.7, top_p=0.9):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p
    }
    try:
        response = requests.post(BASE_URL, headers=headers, json=data, timeout=120)
        response.raise_for_status()
        res_json = response.json()
        # Gemini 2.0 response
        if 'choices' in res_json:
            return res_json['choices'][0]['message']['content']
        elif 'completion' in res_json:
            return res_json['completion']
        elif 'output_text' in res_json:
            return res_json['output_text']
        else:
            return str(res_json)
    except requests.exceptions.Timeout:
        return "Request timed out. Try again with a shorter prompt."
    except requests.exceptions.RequestException as e:
        return f"Request error: {e}"
    except Exception as e:
        return f"Error: {e}"