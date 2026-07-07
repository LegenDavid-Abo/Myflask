import os
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
API_URL = "https://api.groq.com/openai/v1/chat/completions"

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")

headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

# ==========================================
# OLAYEMI AI — System Prompt
# ==========================================

OLAYEMI_SYSTEM = """You are OLAYEMI AI, a personal AI assistant that knows everything about DAVID OLAYEMI.

RULES:
1. ALWAYS refer to him as "OLAYEMI" or "OLAYEMI DAVID"
2. ONLY answer questions about OLAYEMI DAVID — redirect off-topic questions back to him
3. NEVER mention you are an AI model, Groq, Llama, or any other company — you are OLAYEMI AI
4. If asked "who are you", say: "I am OLAYEMI AI, the personal assistant of OLAYEMI DAVID."
5. Speak warmly, clearly, and in complete sentences
6. Use bullet points (➊ ➋ ➌) for lists
7. Never cut off mid-sentence

ABOUT OLAYEMI DAVID:
• Full Stack Developer & AI Engineer from Nigeria (Kogi State)
• Contact: +234 902 299 6320 | olabolade999@gmail.com | GitHub: github.com/LegenDavid-Abo
• Expert in Python, AWS, Machine Learning, AI, Automation, Flask, REST APIs, Computer Vision
• Worked at MTN Nigeria (2021-2022), Emirates Tech Solutions Dubai (2023), TechNova Solutions Germany (2024-Present)
• B.Sc Computer Science, Federal University Lokoja (2021-2024)
• Awards: Best Leading Innovator, 2nd Place ML/AI Competition
• Projects: Portillo Chatbot, AI Face Swap, Bird Detection App, Resume Screener, Fraud Detection, Smart Attendance, Sentiment Analysis, AWS ML Pipeline
"""

# Simple cache
response_cache = {}

def get_cached_response(question):
    return response_cache.get(question.lower().strip())

def cache_response(question, answer):
    response_cache[question.lower().strip()] = answer

# ==========================================
# Flask App
# ==========================================

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"reply": "Please send a message."}), 400

    # Check cache
    cached = get_cached_response(user_input)
    if cached:
        return jsonify({"reply": cached})

    payload = {
        "model": "llama-3.1-70b-versatile",
        "messages": [
            {"role": "system", "content": OLAYEMI_SYSTEM},
            {"role": "user", "content": user_input}
        ],
        "max_tokens": 2048,
        "temperature": 0.5
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        reply = data["choices"][0]["message"]["content"].strip()
        cache_response(user_input, reply)
        return jsonify({"reply": reply})
    except requests.exceptions.HTTPError as e:
        return jsonify({"reply": f"🔴 API Error {e.response.status_code}: {e.response.text}"})
    except Exception as e:
        return jsonify({"reply": f"🔴 Error: {str(e)}"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
