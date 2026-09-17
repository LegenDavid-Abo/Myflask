import os
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# ==========================================
# Load Environment Variables
# ==========================================

load_dotenv()

# ==========================================
# Groq API Setup
# ==========================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

API_URL = "https://api.groq.com/openai/v1/chat/completions"

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Please set it in your environment variables."
    )

headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

# ==========================================
# OLAYEMI AI — System Prompt
# ==========================================

OLAYEMI_SYSTEM = """You are OLAYEMI AI, a personal AI assistant that knows everything about DAVID OLAYEMI.

RULES:
1. ALWAYS refer to him as "OLAYEMI" or "OLAYEMI DAVID" — never "David" alone.
2. ONLY answer questions about OLAYEMI DAVID — redirect off-topic questions back to him.
3. NEVER mention you are an AI model, Groq, Llama, or any other company — you are OLAYEMI AI.
4. If asked "who are you", say:
"I am OLAYEMI AI, the personal assistant of OLAYEMI DAVID. I know everything about his skills, experience, projects, and achievements."
5. Use Markdown formatting freely:
   - **bold** for emphasis
   - `code` for technical terms
   - Bullet points with - or numbers
6. Speak clearly and in complete sentences.
7. Never cut off mid-sentence — finish every thought completely.
8. For lists, use this format:
   ➊ ➋ ➌ ➍ ➎ ➏ ➐ ➑ ➒ ➓

ABOUT OLAYEMI DAVID:

• Full Stack Developer & AI Engineer from Nigeria (Kogi State)

• Contact:
  +234 902 299 6320
  olabolade999@gmail.com
  GitHub: github.com/LegenDavid-Abo

• Expert in:
  Python
  AWS
  Machine Learning
  Artificial Intelligence
  Automation
  Flask
  REST APIs
  Computer Vision
  Chatbots

• Worked at:
  MTN Nigeria (2021-2022)
  Emirates Tech Solutions Dubai (2023)
  TechNova Solutions Germany (2024-Present)

• Education:
  B.Sc Computer Science
  Federal University Lokoja (2021-2024)

• Achievement:
  2nd Place ML/AI Competition

• Awards:
  Best Leading Innovator
  2nd Place ML/AI & Automation Competition

• Certifications:
  Python Programming
  Machine Learning & AI

• Projects:
  Portillo Chatbot
  AI Face Swap
  Bird Detection App
  Resume Screener
  Fraud Detection
  Smart Attendance
  Sentiment Analysis
  AWS ML Pipeline
"""

# ==========================================
# Simple In-Memory Cache
# ==========================================

response_cache = {}


def get_cached_response(question):
    """Return cached response if the question exists."""
    return response_cache.get(question.lower().strip())


def cache_response(question, answer):
    """Store response in cache."""
    response_cache[question.lower().strip()] = answer


# ==========================================
# Flask Application
# ==========================================

app = Flask(__name__)


# ==========================================
# Home Page
# ==========================================

@app.route("/")
def index():
    return render_template("index.html")


# ==========================================
# Health Check
# Useful for Render / Uptime Monitoring
# ==========================================

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "OLAYEMI AI"
    }), 200


# ==========================================
# Chat Endpoint
# ==========================================

@app.route("/chat", methods=["POST"])
def chat():

    # Safely get JSON request
    data = request.get_json(silent=True) or {}

    user_input = data.get("message", "").strip()

    # Validate input
    if not user_input:
        return jsonify({
            "reply": "Please send a message."
        }), 400

    # ======================================
    # Check Cache
    # ======================================

    cached = get_cached_response(user_input)

    if cached:
        return jsonify({
            "reply": cached
        })

    # ======================================
    # Groq API Payload
    # ======================================

    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {
                "role": "system",
                "content": OLAYEMI_SYSTEM
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        "max_tokens": 2048,
        "temperature": 0.5
    }

    # ======================================
    # Send Request to Groq
    # ======================================

    try:

        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )

        # Raise error for HTTP failures
        response.raise_for_status()

        # Parse JSON response
        data = response.json()

        # ==================================
        # Extract AI Response
        # ==================================

        reply = data["choices"][0]["message"]["content"].strip()

        # ==================================
        # Cache Response
        # ==================================

        cache_response(
            user_input,
            reply
        )

        # ==================================
        # Return Response
        # ==================================

        return jsonify({
            "reply": reply
        })

    # ======================================
    # HTTP Error
    # ======================================

    except requests.exceptions.HTTPError as e:

        status_code = (
            e.response.status_code
            if e.response is not None
            else 500
        )

        error_text = (
            e.response.text
            if e.response is not None
            else str(e)
        )

        error_msg = (
            f"API Error {status_code}: {error_text}"
        )

        print(f"❌ {error_msg}")

        return jsonify({
            "reply": f"🔴 {error_msg}"
        }), 500

    # ======================================
    # Connection Error
    # ======================================

    except requests.exceptions.ConnectionError:

        print("❌ Connection to Groq API failed.")

        return jsonify({
            "reply": "🔴 Connection lost. Please try again."
        }), 503

    # ======================================
    # Timeout
    # ======================================

    except requests.exceptions.Timeout:

        print("❌ Groq API request timed out.")

        return jsonify({
            "reply": "🔴 Request timed out. Please try again."
        }), 504

    # ======================================
    # Invalid JSON / API Response
    # ======================================

    except (KeyError, IndexError, TypeError, ValueError) as e:

        print(f"❌ Invalid API response: {str(e)}")

        return jsonify({
            "reply": "🔴 The AI service returned an unexpected response."
        }), 500

    # ======================================
    # Any Other Error
    # ======================================

    except Exception as e:

        import traceback

        print(
            f"❌ Unexpected error:\n{traceback.format_exc()}"
        )

        return jsonify({
            "reply": f"🔴 Error: {str(e)}"
        }), 500


# ==========================================
# Run Application
# ==========================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
