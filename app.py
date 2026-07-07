import os
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# ✅ Load environment variables from .env
load_dotenv()

API_URL = "https://router.huggingface.co/v1/chat/completions"
API_KEY = os.getenv("HF_API_KEY")

if not API_KEY:
    raise ValueError("HF_API_KEY not found. Please set it in your .env file.")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

RK_CV = """..."""  # (same as your original)

SYSTEM_STYLE = """..."""  # (same as your original)


def classify_query(user_input: str) -> str:
    code_keywords = ["code", "script", "function", "python", "javascript", "react", "flask"]
    strategic_keywords = ["strategy", "career", "advice", "growth", "project", "leadership"]
    lower_input = user_input.lower()

    if any(k in lower_input for k in code_keywords):
        return "code"
    elif any(k in lower_input for k in strategic_keywords):
        return "strategic"
    else:
        return "explanation"


def determine_response_length(user_input: str) -> str:
    word_count = len(user_input.split())
    if word_count > 12:
        return "long"
    elif word_count > 3:
        return "medium"
    else:
        return "short"


length_params = {
    "short": {
        "max_tokens": 100,
        "temperature": 0.5,
        "top_p": 0.8
    },
    "medium": {
        "max_tokens": 300,
        "temperature": 0.7,
        "top_p": 0.9
    },
    "long": {
        "max_tokens": 600,
        "temperature": 1.0,
        "top_p": 1.0        # ← FIXED: was 1.2, now valid 1.0
    }
}

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message")

    if not user_input:
        return jsonify({"reply": "Please provide a message."}), 400

    response_type = classify_query(user_input)
    response_length = determine_response_length(user_input)
    params = length_params[response_length]

    # Optional: tailor system prompt based on response_type
    system_content = RK_CV + "\n\n" + SYSTEM_STYLE
    if response_type == "code":
        system_content += "\n\nThe user is asking a technical coding question. Provide code examples where helpful."

    payload = {
        "model": "meta-llama/Meta-Llama-3-70B-Instruct",
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_input}
        ],
        "max_tokens": params["max_tokens"],
        "temperature": params["temperature"],
        "top_p": params["top_p"]
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return jsonify({"reply": data["choices"][0]["message"]["content"]})

    except requests.exceptions.HTTPError as e:
        return jsonify({"reply": f"API Error {e.response.status_code}: {e.response.text}"}), 502
    except requests.exceptions.ConnectionError:
        return jsonify({"reply": "❌🌐 Connection lost — Please check your internet connection and try again."}), 503
    except requests.exceptions.Timeout:
        return jsonify({"reply": "⏱️ Request timed out. Please try again."}), 504
    except Exception as e:
        return jsonify({"reply": f"Unexpected error: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
