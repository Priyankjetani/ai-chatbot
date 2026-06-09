from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import anthropic
from google import genai
from google.genai import types
import openai
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# ── API Clients ──
claude_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
openai.api_key = os.environ.get("OPENAI_API_KEY")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data     = request.get_json()
        messages = data.get("messages", [])
        model    = data.get("model", "")
        system   = data.get("system", "You are a helpful assistant.")

        if not messages:
            return jsonify({"error": "No messages provided"}), 400

        # ── Route to correct AI ──
        if "claude" in model:
            reply = call_claude(messages, system, model)

        elif "gemini" in model:
            reply = call_gemini(messages, system, model)

        elif "gpt" in model:
            reply = call_openai(messages, system, model)

        else:
            return jsonify({"error": f"Unknown model: {model}"}), 400

        return jsonify({"reply": reply, "status": "ok"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Claude (Anthropic) ──
def call_claude(messages, system, model):
    response = claude_client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        messages=messages
    )
    return response.content[0].text


# ── Gemini (Google) — new google-genai package ──
def call_gemini(messages, system, model):
    # Build contents from history
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=1024,
        )
    )
    return response.text


# ── ChatGPT (OpenAI) ──
def call_openai(messages, system, model):
    formatted = [{"role": "system", "content": system}]
    for msg in messages:
        formatted.append({"role": msg["role"], "content": msg["content"]})

    response = openai.chat.completions.create(
        model=model,
        messages=formatted,
        max_tokens=1024
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    app.run(debug=True, port=5000)