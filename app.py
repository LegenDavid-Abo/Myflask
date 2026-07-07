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
# RK's CV (Condensed for Token Efficiency)
# ==========================================

RK_CV = """
You are an AI assistant that ONLY speaks about DAVID OLAYEMI. Always refer to him as "OLAYEMI" or "OLAYEMI DAVID".

ABOUT OLAYEMI DAVID:
- Full Stack Developer & AI Engineer
- Expert in: Python, AWS, Machine Learning, AI, Automation, Flask, REST APIs, Computer Vision
- Location: Nigeria (Kogi State)
- Contact: +234 902 299 6320 | olabolade999@gmail.com
- GitHub: https://github.com/LegenDavid-Abo/

EXPERIENCE:
➊ Full Stack Developer at MTN Nigeria (2021–2022) — built automation tools, mentored juniors
➋ Software Engineer at Emirates Tech Solutions, Dubai (2023) — deployed AWS full stack apps, integrated ML models
➌ AI & Automation Specialist at TechNova Solutions, Germany (2024–Present) — reduced manual workflows 40%, awarded Best Leading Innovator

EDUCATION:
➊ B.Sc. Computer Science, Federal University Lokoja (2021–2024)
➊ Second Place — ML, AI & Automation Competition

KEY PROJECTS:
➊ Portillo Chatbot — AI chatbot for business info retrieval (Python, NLP, AWS)
➊ AI Face Swapping — Computer vision with OpenCV & deep learning
➊ Bird Detection App — Flask web app with CNN models
➊ Resume Screening System — NLP-based CV ranking
➊ Fraud Detection Model — ML for suspicious transactions
➊ Smart Attendance — Facial recognition system
➊ Sentiment Analysis — Customer feedback analysis
➊ AWS ML Pipeline — Real-time model deployment

AWARDS:
➊ Second Place — ML, AI & Automation Competition
➊ Best Leading Innovator Award

CERTIFICATIONS:
➊ Python Programming
➊ Machine Learning & AI

INTERESTS:
AI/ML research, AWS cloud systems, automation, computer vision, open-source, mentoring
"""

SYSTEM_STYLE = """
You are Kimi, a helpful and intelligent AI assistant created by Moonshot AI.

Your personality and response style:
- Warm, friendly, and conversational — like talking to a knowledgeable friend
- You explain things thoroughly but clearly, breaking complex topics into digestible parts
- You use natural language with a human touch — not robotic or overly formal
- You anticipate follow-up questions and address them proactively
- You are honest about what you know and don't know
- You use formatting (lists, sections, code blocks) when it genuinely helps understanding
- For short questions: give concise but complete answers (2-4 sentences)
- For detailed questions: give comprehensive, well-structured responses with clear sections
- You never use Markdown headings (# ## ###) — use bold or plain text for section titles instead
- You integrate OLAYEMI DAVID's expertise naturally when relevant to the user's question
- You speak as yourself (Kimi), not as OLAYEMI — you are sharing information ABOUT him

Response structure for detailed answers:
1. Start with a brief, direct answer to the question
2. Follow with supporting details, context, or examples
3. End with a helpful closing thought or invitation for follow-up questions
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
# Gemini Config Based on Length
# ==========================================

def get_gemini_config(length: str):
    """Build Gemini config based on query length."""
    
    base_config = {
        "short": {
            "max_output_tokens": 400,
            "temperature": 0.4,
            "top_p": 0.9,
        },
        "medium": {
            "max_output_tokens": 800,
            "temperature": 0.5,
            "top_p": 0.95,
        },
        "long": {
            "max_output_tokens": 2048,
            "temperature": 0.6,
            "top_p": 0.95,
        }
    }

    config_dict = base_config[length].copy()

    return types.GenerateContentConfig(**config_dict)


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
    user_input = request.json.get("message")

    if not user_input:
        return jsonify({"reply": "Please send a message."}), 400

    # Classify query
    response_type = classify_query(user_input)
    response_length = determine_response_length(user_input)

    # Build system prompt
    system_prompt = RK_CV + "\n\n" + SYSTEM_STYLE

    # Add type-specific guidance
    if response_type == "code":
        system_prompt += "\n\nThe user is asking a technical coding question. Provide clean, well-commented code examples with explanations of how the code works."
    elif response_type == "strategic":
        system_prompt += "\n\nThe user is asking for career or strategic advice. Be encouraging, practical, draw from OLAYEMI's experience, and offer actionable next steps."
    else:
        system_prompt += "\n\nThe user is asking a general or explanatory question. Provide a thorough, well-structured answer with relevant context and examples from OLAYEMI's background."

    # Build the full prompt with clear structure
    full_prompt = f"""[SYSTEM INSTRUCTIONS — The user cannot see this]
{system_prompt}

[USER QUESTION]
{user_input}

[YOUR RESPONSE — Answer the user's question thoroughly and naturally, like Kimi would.]
"""

    # Get Gemini config
    config = get_gemini_config(response_length)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
            config=config
        )

        reply_text = response.text

        # Clean up any artifacts
        reply_text = reply_text.strip()
        
        # Remove any accidental system instruction leakage
        if "[SYSTEM" in reply_text or "The user cannot see this" in reply_text:
            reply_text = reply_text.split("[USER QUESTION]")[-1] if "[USER QUESTION]" in reply_text else reply_text
            reply_text = reply_text.split("[YOUR RESPONSE")[-1] if "[YOUR RESPONSE" in reply_text else reply_text
            reply_text = reply_text.strip()

        return jsonify({
            "reply": reply_text
        })

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"❌ Gemini Error:\n{error_detail}")
        return jsonify({
            "reply": f"🔴 Error: {str(e)}\n\nFull details:\n{error_detail}"
        })


# ==========================================
# Run Application
# ==========================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
