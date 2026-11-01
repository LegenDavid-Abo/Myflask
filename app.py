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

Here is the complete information from the CV, structured according to your format:

---

**RULES:**
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

---

## ABOUT OLAYEMI DAVID:

• **Client Support Specialist · Trading Operations · Technical Support** from Nigeria (Lagos, Nigeria)

• **Contact:**
  - +234 902 299 6320
  - olabolade999@gmail.com
  - GitHub: github.com/LegenDavid-Abo
  - LinkedIn: linkedin.com/in/david-olayemi

• **Expert in:**
  - Python
  - JavaScript
  - TypeScript
  - SQL
  - AWS (`EC2`, `Lambda`, `S3`, `CloudWatch`)
  - Machine Learning & AI
  - Automation
  - Flask
  - REST & GraphQL APIs
  - React.js
  - Node.js
  - PostgreSQL, MySQL, MongoDB, Redis
  - Computer Vision
  - Chatbots & NLP
  - Trading Systems & Risk Management
  - Discord Moderation & Support Bots
  - Zendesk & Intercom Workflows
  - Cloud Infrastructure

• **Worked at:**
  ➊ **Trading Bot Developer & Strategy Operator** — Independent (Mar 2024 – Present)
  ➋ **Senior Full Stack Engineer** — NovaEdge Technologies, Remote (Feb 2024 – Feb 2026)
  ➌ **Client Support Specialist / Discord Moderator** — FundingPips, Remote (Mar 2023 – Dec 2023)
  ➍ **Software Engineer** — CloudSync Digital Solutions, Remote (Feb 2023 – Dec 2023)

• **Education:**
  - **B.Sc Computer Science**
  - University of Lokoja, Nigeria (2021–2024)
  - **First Class Honours** — GPA: 4.65/5.0
  - **Capstone Project:** AI-powered plagiarism detection (Best Final Year Project, Dept. of Computer Science)

• **Achievements:**
  - Best Final Year Project Award (AI-powered plagiarism detection)
  - Reduced client operational costs by **$120,000/year** through AI-powered automation at NovaEdge Technologies
  - **Zero production outages** across 18+ months of AWS infrastructure management; maintained **99.95% uptime SLA**
  - Recognized at FundingPips for consistently delivering **sub-60-second response times** in high-volume Discord support
  - Published technical article: *"Scaling Node.js Microservices with Docker & AWS"* — 5,000+ views on Dev.to
  - **50+ public repositories** with **200+ commits** on GitHub; consistent open-source contributions (2022–2024)

• **Awards & Recognition:**
  - Best Final Year Project (Dept. of Computer Science, University of Lokoja)
  - Recognized for exceptional speed and accuracy in client issue resolution at FundingPips

• **Certifications:**
  - Full Stack Web Development — freeCodeCamp (2022)
  - Backend API Development — Codecademy (2022)

• **Key Projects:**
  ➊ **Automated Trading Bot with Live Dashboard** — `Python` · `AWS EC2` · Risk Management · Live Trade Execution
     - Live dashboard: http://16.170.35.248:8000/
  ➋ **AI Customer Support Assistant** — `Python` · `Flask` · `spaCy` · `AWS EC2` · Zendesk · WhatsApp API
     - Live demo: https://olayemi-portfolio-ai.onrender.com
  ➌ **AI-powered Plagiarism Detection System** (Capstone Project)

• **Languages:**
  - **English** — Professional Working Proficiency
  - **Yoruba** — Native Speaker

---

**Core Competencies Breakdown:**

**CLIENT SUPPORT & TRADING:**
- High-volume live chat & ticket resolution
- SLA-driven response under 60 seconds
- Clear, accurate technical writing
- Escalation & incident management
- Pattern spotting & feedback loops
- Automated strategy execution
- Risk management & position sizing
- Live trade tracking & monitoring
- Prop firm & brokerage environments
- Account verification & access control

**PLATFORM & TOOLS:**
- Intercom help desk & ticket workflows
- Discord moderation & support bots
- Zendesk & help desk workflows
- AWS cloud infrastructure
- API integrations & troubleshooting

**TECHNICAL SKILLS:**
- `Python`, `JavaScript`, `TypeScript`, `SQL`
- `REST` & `GraphQL` API design
- `PostgreSQL`, `MySQL`, `MongoDB`, `Redis`
- `React.js`, `Node.js`, `Flask`
- `AWS` (`EC2`, `Lambda`, `S3`, `CloudWatch`)

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
