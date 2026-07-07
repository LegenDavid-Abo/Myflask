import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ✅ Load environment variables from .env
load_dotenv()

# ==========================================
# Google Gemini Setup
# ==========================================

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")

client = genai.Client(api_key=API_KEY)

# ==========================================
# OLAYEMI AI — System Prompt (Ultra-Condensed)
# ==========================================

OLAYEMI_SYSTEM = """
You are OLAYEMI AI, a personal AI assistant that knows everything about DAVID OLAYEMI.

RULES:
1. ALWAYS refer to him as "OLAYEMI" or "OLAYEMI DAVID" — never "David" alone
2. ONLY answer questions about OLAYEMI DAVID — redirect off-topic questions back to him
3. NEVER mention you are an AI model, Gemini, Google, or any other company — you are OLAYEMI AI
4. If asked "who are you", say: "I am OLAYEMI AI, the personal assistant of OLAYEMI DAVID. I know everything about his skills, experience, projects, and achievements."
5. Speak warmly, clearly, and in complete sentences
6. Use bullet points (➊ ➋ ➌) for lists
7. Never cut off mid-sentence — finish every thought completely

ABOUT OLAYEMI DAVID:
• Full Stack Developer & AI Engineer from Nigeria (Kogi State)
• Contact: +234 902 299 6320 | olabolade999@gmail.com | GitHub: github.com/LegenDavid-Abo
• Expert in Python, AWS, Machine Learning, AI, Automation, Flask, REST APIs, Computer Vision, Chatbots
• Worked at MTN Nigeria (2021-2022), Emirates Tech Solutions Dubai (2023), TechNova Solutions Germany (2024-Present)
• B.Sc Computer Science, Federal University Lokoja (2021-2024), 2nd Place ML/AI Competition
• Awards: Best Leading Innovator, 2nd Place ML/AI & Automation Competition
• Certifications: Python, Machine Learning & AI
• Projects: Portillo Chatbot, AI Face Swap, Bird Detection App, Resume Screener, Fraud Detection, Smart Attendance, Sentiment Analysis, AWS ML Pipeline
"""


# ==========================================
# Query Classification
# ==========================================

def classify_query(user_input: str) -> str:
    code_keywords = [
        "code", "script", "function", "python", "javascript",
        "react", "flask", "html", "css", "api", "database", "sql",
        "programming", "developer", "debug", "error", "bug"
    ]
    strategic_keywords = [
        "strategy", "career", "advice", "growth", "project",
        "leadership", "job", "interview", "salary", "promotion",
        "future", "plan", "goal", "improve", "skill"
    ]

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


# ==========================================
# Gemini Config
# ==========================================

def get_gemini_config(length: str):
    """Build Gemini config with HIGH token limits to prevent cutoffs."""
    
    base_config = {
        "short": {
            "max_output_tokens": 512,
            "temperature": 0.3,
            "top_p": 0.9,
        },
        "medium": {
            "max_output_tokens": 1024,
            "temperature": 0.4,
            "top_p": 0.95,
        },
        "long": {
            "max_output_tokens": 4096,
            "temperature": 0.5,
            "top_p": 0.95,
        }
    }

    return types.GenerateContentConfig(**base_config[length])


# ==========================================
# Initialize Flask App
# ==========================================

app = Flask(__name__)


# ==========================================
# Routes
# ==========================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message", "").strip()

    if not user_input:
        return jsonify({"reply": "Please send a message."}), 400

    # Classify query
    response_type = classify_query(user_input)
    response_length = determine_response_length(user_input)

    # Add type-specific guidance
    extra_guidance = ""
    if response_type == "code":
        extra_guidance = "The user wants code. Provide clean, working code with brief explanations."
    elif response_type == "strategic":
        extra_guidance = "The user wants career advice. Be practical and encouraging, use OLAYEMI's experience as examples."
    else:
        extra_guidance = "Give a complete, thorough answer. Do not stop mid-sentence."

    # Build prompt — system first, then user question
    full_prompt = f"""{OLAYEMI_SYSTEM}

{extra_guidance}

USER QUESTION: {user_input}

YOUR ANSWER (complete and full, no cutoff):
"""

    # Get config
    config = get_gemini_config(response_length)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
            config=config
        )

        reply_text = response.text.strip()

        # Safety: if still cut off, add a note
        if reply_text and reply_text[-1] not in ".!?\"'":
            reply_text += "."

        return jsonify({
            "reply": reply_text
        })

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"❌ Gemini Error:\n{error_detail}")
        return jsonify({
            "reply": f"🔴 Error: {str(e)}"
        })


# ==========================================
# Run Application
# ==========================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
