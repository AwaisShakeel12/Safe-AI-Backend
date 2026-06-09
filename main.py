import os
import json
import smtplib
from email.mime.text import MIMEText
from typing import TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from langgraph.graph import StateGraph, END
from groq import Groq
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# ==========================================================
# CONFIG
# ==========================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")



# ==========================================================
# FASTAPI
# ==========================================================

app = FastAPI(title="AI Safety Backend")

# ==========================================================
# CLIENTS
# ==========================================================

groq_client = Groq(api_key=GROQ_API_KEY)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)

# ==========================================================
# KEYWORDS
# ==========================================================

DANGER_KEYWORDS = [
    "help",
    "help me",
    "save me",
    "police",
    "call police",
    "emergency",
    "leave me alone",
    "don't touch me",
    "stay away",
    "stop",
    "bachao",
    "madad",
    "meri madad karo",
    "chor",
    "dakait",
    "kidnap",
    "knife",
    "gun",
    "ambulance",
]

# ==========================================================
# LANGGRAPH STATE
# ==========================================================

class SafetyState(TypedDict):
    transcript: str
    latitude: float
    longitude: float
    family_email: str

    risk_level: str
    reason: str
    action: str

# ==========================================================
# EMAIL TOOL
# ==========================================================

def send_email_alert(
    recipient_email,
    transcript,
    risk_level,
    reason,
    latitude,
    longitude
):
    maps_link = (
        f"https://maps.google.com/?q={latitude},{longitude}"
    )

    body = f"""
AI SAFETY ALERT

Risk Level:
{risk_level}

Reason:
{reason}

Transcript:
{transcript}

Location:
{maps_link}
"""

    msg = MIMEText(body)

    msg["Subject"] = f"🚨 AI Safety Alert - {risk_level}"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = recipient_email

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()

        server.login(
            EMAIL_ADDRESS,
            EMAIL_PASSWORD
        )

        server.send_message(msg)
        server.quit()

        print("EMAIL SENT")

    except Exception as e:
        print("EMAIL ERROR:", e)

# ==========================================================
# GEMINI NODE
# ==========================================================

def analyze_node(state: SafetyState):

    transcript = state["transcript"]

    prompt = f"""
You are an AI safety analyst.

Analyze the following transcript for signs of danger.

Transcript:
{transcript}

Return ONLY a JSON object.

DO NOT include explanations.
DO NOT include markdown.
DO NOT include ```json blocks.

Valid risk levels:
LOW
MEDIUM
HIGH

Return exactly this format:

{{
    "risk_level":"HIGH",
    "reason":"short explanation",
    "action":"email"
}}

If there is no danger:

{{
    "risk_level":"LOW",
    "reason":"No immediate threat detected",
    "action":"none"
}}
"""

    response = llm.invoke(prompt)

    text = response.content.strip()

    print("RAW GEMINI RESPONSE:")
    print(text)

    try:

        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

        start = text.find("{")
        end = text.rfind("}") + 1

        text = text[start:end]

        data = json.loads(text)

        state["risk_level"] = data.get(
            "risk_level",
            "LOW"
        )

        state["reason"] = data.get(
            "reason",
            ""
        )

        state["action"] = data.get(
            "action",
            "none"
        )

    except Exception as e:

        print("JSON PARSE ERROR:", e)
        print("GEMINI TEXT:", text)

        state["risk_level"] = "LOW"
        state["reason"] = "Could not parse Gemini response"
        state["action"] = "none"

    return state

# ==========================================================
# EMAIL NODE
# ==========================================================

def email_node(state: SafetyState):

    if (
        state["risk_level"] == "HIGH"
        and state["action"] == "email"
    ):

        send_email_alert(
            recipient_email=state["family_email"],
            transcript=state["transcript"],
            risk_level=state["risk_level"],
            reason=state["reason"],
            latitude=state["latitude"],
            longitude=state["longitude"]
        )

    return state

# ==========================================================
# LANGGRAPH
# ==========================================================

builder = StateGraph(SafetyState)

builder.add_node(
    "analyze",
    analyze_node
)

builder.add_node(
    "email",
    email_node
)

builder.set_entry_point("analyze")

builder.add_edge(
    "analyze",
    "email"
)

builder.add_edge(
    "email",
    END
)

graph = builder.compile()

# ==========================================================
# HEALTH
# ==========================================================

@app.get("/health")
def health():
    return {
        "status": "running"
    }

# ==========================================================
# TEST EMAIL
# ==========================================================

@app.get("/test-alert")
def test_alert():

    send_email_alert(
        recipient_email="YOUR_EMAIL@gmail.com",
        transcript="Help me. Leave me alone.",
        risk_level="HIGH",
        reason="Manual test",
        latitude=31.435,
        longitude=72.982
    )

    return {
        "message": "test email sent"
    }

# ==========================================================
# MAIN ENDPOINT
# ==========================================================



@app.post("/analyze-audio")
async def analyze_audio(
    audio_file: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    family_email: str = Form(...)
):

    # -----------------------------------
    # READ AUDIO DIRECTLY FROM MEMORY
    # -----------------------------------

    audio_bytes = await audio_file.read()

    # -----------------------------------
    # GROQ TRANSCRIPTION
    # -----------------------------------

    transcription = groq_client.audio.transcriptions.create(
        file=(
            audio_file.filename,
            audio_bytes
        ),
        model="whisper-large-v3",
        temperature=0
    )

    transcript = transcription.text

    # -----------------------------------
    # QUICK FILTER
    # -----------------------------------

    transcript_lower = transcript.lower()

    should_run_ai = False

    for keyword in DANGER_KEYWORDS:
        if keyword in transcript_lower:
            should_run_ai = True
            break

    if not should_run_ai:

        return {
        "transcript": transcript,
        "risk_level": "LOW",
        "reason": "No danger keywords found",
        "action": "none",
        "email_sent": False
    }

    # -----------------------------------
    # LANGGRAPH
    # -----------------------------------

    result = graph.invoke({
        "transcript": transcript,
        "latitude": latitude,
        "longitude": longitude,
        "family_email": family_email
    })

    return {
        "transcript": transcript,
        "risk_level": result["risk_level"],
        "reason": result["reason"],
        "action": result["action"],
        "email_sent": (
            result["risk_level"] == "HIGH"
        )
    }
