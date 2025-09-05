import os
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# ✅ Load environment variables from .env
load_dotenv()

API_URL = "https://router.huggingface.co/v1/chat/completions"
API_KEY = os.getenv("HF_API_KEY")  # ✅ Load from .env, not hardcoded

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

RK_CV = """
You are an AI assistant that only talks about David Olayemi.
Refer to him as "OLAYEMI" or "OLAYEMI DAVID".

Curriculum Vitae

Name: David Olayemi
Phone: +234 902 299 6320
Email: olabolade999@gmail.com
Location: Kogi, Nigeria

Professional Summary, Core Skills, Work Experience, Education, Certifications, Interests.
Always give structured, complete, and professional answers.
"""

# 🔹 Minimized SYSTEM_STYLE
SYSTEM_STYLE = """
You are ChatGPT, an experienced career coach and technical mentor.
Speak naturally and clearly.
Always provide complete, structured answers using these sections:
👤 Professional Summary
📞 Contact Information
⚙️ Core Skills
💼 Work Experience
📚 Education
🏆 Certifications
🎯 Projects
🌱 Interests
🛠️ Tools & Technologies
Number lists: ➊ ➋ ➌ ➍ ➎ ...
Do not cut off mid-sentence.
Keep answers coherent, professional, and tailored to RK’s expertise.
"""

# 🔹 Determine response type
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

# 🔹 Determine response length
def determine_response_length(user_input: str) -> str:
    word_count = len(user_input.split())
    if word_count > 12:
        return "long"
    elif word_count > 3:
        return "medium"
    else:
        return "short"

# 🔹 Map response parameters
length_params = {
    "short": {"max_tokens": 100, "temperature": 0.3, "top_p": 0.8},
    "medium": {"max_tokens": 300, "temperature": 0.5, "top_p": 0.9},
    "long": {"max_tokens": 2000, "temperature": 0.7, "top_p":0.95 }
}

# Initialize Flask
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# 🔹 Helper function to handle continuation
def get_full_response(payload):
    full_text = ""
    while True:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        chunk = data["choices"][0]["message"]["content"]
        full_text += chunk

        # Check if response indicates it's finished or truncated
        if "finish_reason" in data["choices"][0] and data["choices"][0]["finish_reason"] == "length":
            # If truncated due to length, continue
            payload["messages"].append({"role": "user", "content": "Please continue from where you left off."})
        else:
            break
    return full_text

# 🔹 API route for chat
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message")
    response_type = classify_query(user_input)
    response_length = determine_response_length(user_input)
    params = length_params[response_length]

    payload = {
        "model": "meta-llama/Meta-Llama-3-70B-Instruct",
        "messages": [
            {"role": "system", "content": RK_CV + "\n\n" + SYSTEM_STYLE},
            {"role": "user", "content": user_input}
        ],
        "max_tokens": params["max_tokens"],
        "temperature": params["temperature"],
        "top_p": params["top_p"]
    }

    try:
        full_reply = get_full_response(payload)
        return jsonify({"reply": full_reply})
    except Exception:
        return jsonify({"reply": "❌🌐 Connection lost — Please check your internet connection and try again."})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
