import os
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# ✅ Correct Hugging Face endpoint (IMPORTANT FIX)
API_URL = "https://router.huggingface.co/v1/chat/completions"

API_KEY = os.getenv("HF_API_KEY")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# =========================
# YOUR ORIGINAL SYSTEM DATA
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

OLAYEMI DAVID is a Full Stack Developer and AI Engineer with strong expertise in AWS, Machine Learning, Artificial Intelligence, and Automation. He has hands-on experience building production-ready web applications, intelligent systems, and cloud-deployed machine learning solutions. He has worked across Nigeria, the Middle East, and Europe, delivering scalable and efficient solutions,He also lives in kogi state. He is recognized for leadership, innovation, and excellence in ML and AI-driven projects.

Core Skills
Hard Skills:
Full Stack Development, Python, AWS Cloud Services, Machine Learning, Artificial Intelligence, Automation, Flask, REST APIs, Computer Vision, Chatbot Development, Data Analysis, Model Deployment, Git, GitHub

Soft Skills:
Leadership, Strategic Thinking, Problem Solving, Team Collaboration, Communication, Project Ownership

Work Experience
Full Stack Developer - MTN Nigeria (2021–2022)
Software Engineer - Emirates Tech Solutions (2023)
AI & Automation Specialist - TechNova Solutions (2024–Present)

Education
B.Sc. Computer Science - Federal University Lokoja (2021–2024)

Projects
- Portillo Chatbot
- AI Face Swapping System
- Bird Detection System
- Fraud Detection Model
- Smart Attendance System
- Sentiment Analysis Engine
- AWS ML Deployment Pipeline
- Business Automation Bots

Achievements
Second Place – Machine Learning Competition
Best Leading Innovator Award

Certifications
Python Programming Certification
Machine Learning Certification

Languages
English – Fluent

Interests
AI, ML, Automation, Cloud Systems, Computer Vision
"""

SYSTEM_STYLE = """
You are ChatGPT, acting as an experienced career coach, mentor, and technical guide.
Your voice must be natural, human-like, and full of reasoning, wisdom, and understanding.
Always respond in a way that feels conversational and insightful, not mechanical.

Formatting rules:
- For short answers: plain sentences only
- For structured answers: use sections with icons only:
  👤 Professional Summary
  📞 Contact Information
  ⚙️ Core Skills
  💼 Work Experience
  📚 Education
  🏆 Certifications
  🎯 Projects
  🌱 Interests
- No markdown bold or italics
- Use numbered lists ➊➋➌ if needed

Behavior rules:
- Think like a senior mentor
- Be clear, structured, and helpful
"""

# =========================
# LOGIC FUNCTIONS
# =========================

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
    "short": {"max_tokens": 120, "temperature": 0.6},
    "medium": {"max_tokens": 300, "temperature": 0.7},
    "long": {"max_tokens": 600, "temperature": 0.9}
}

# =========================
# ROUTES
# =========================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")

    if not user_input:
        return jsonify({"reply": "No message received"}), 400

    response_length = determine_response_length(user_input)
    params = length_params[response_length]

    # 🔥 FIXED PROMPT FORMAT (HF-compatible)
    prompt = f"""
{RK_CV}

{SYSTEM_STYLE}

User: {user_input}
Assistant:
"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": params["max_tokens"],
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
            timeout=30
        )

        print("STATUS:", response.status_code)
        print("RESPONSE:", response.text)

        response.raise_for_status()
        data = response.json()

        # HF response handling
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
