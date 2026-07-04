import os
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# =========================
# HF STABLE INFERENCE API
# =========================
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"

API_KEY = os.getenv("HF_API_KEY")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# =========================
# YOUR SYSTEM PROMPTS (UNCHANGED)
# =========================

RK_CV = """
You are an AI assistant that always speaks and only relates ABOUT DAVID OLAYEMI.
Always refer to him as "OLAYEMI" or "OLAYEMI DAVID".

Curriculum Vitae

Name: David Olayemi
Phone: +234 902 299 6320
Email: olabolade999@gmail.com
Location: Nigeria
GitHub: https://github.com/LegenDavid-Abo/

Professional Summary
OLAYEMI DAVID is a Full Stack Developer and AI Engineer with strong expertise in AWS, Machine Learning, Artificial Intelligence, and Automation...

Core Skills, Work Experience, Education, Projects...
"""

SYSTEM_STYLE = """
You are ChatGPT, acting as an experienced career coach, mentor, and technical guide.
Respond naturally, like a human expert.

Formatting rules:
- Use structured sections only when needed
- No markdown bold or italics
- Use clean numbered lists (➊➋➌)
- Be concise and intelligent
"""

# =========================
# LOGIC
# =========================

def determine_response_length(user_input: str):
    words = len(user_input.split())
    if words > 12:
        return "long"
    elif words > 3:
        return "medium"
    return "short"


length_params = {
    "short": {"max_new_tokens": 120, "temperature": 0.6},
    "medium": {"max_new_tokens": 300, "temperature": 0.7},
    "long": {"max_new_tokens": 600, "temperature": 0.9},
}

# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")

    if not user_input:
        return jsonify({"reply": "No message received"}), 400

    level = determine_response_length(user_input)
    params = length_params[level]

    # 🔥 FINAL PROMPT FORMAT
    prompt = f"""
{RK_CV}

{SYSTEM_STYLE}

User: {user_input}
Assistant:
"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": params["max_new_tokens"],
            "temperature": params["temperature"],
            "top_p": 0.9,
            "return_full_text": False
        }
    }

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=40
        )

        print("STATUS:", response.status_code)
        print("RESPONSE:", response.text)

        # 🔥 HANDLE HF ERRORS PROPERLY
        if response.status_code != 200:
            return jsonify({
                "reply": f"HF ERROR {response.status_code}: {response.text}"
            })

        data = response.json()

        # HF response parsing
        if isinstance(data, list) and "generated_text" in data[0]:
            reply = data[0]["generated_text"]
        elif isinstance(data, list):
            reply = data[0].get("generated_text", "")
        else:
            reply = data.get("generated_text", "No response generated")

        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"reply": f"Connection error: {str(e)}"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
