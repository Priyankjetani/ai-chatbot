from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import anthropic
import google.generativeai as genai
import openai
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# ── API Clients ──
claude_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
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


# ── Gemini (Google) — FREE tier ──
def call_gemini(messages, system, model):
    gemini_model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=system
    )

    # Convert messages to Gemini format
    gemini_history = []
    for msg in messages[:-1]:  # all except last
        gemini_history.append({
            "role": "user" if msg["role"] == "user" else "model",
            "parts": [msg["content"]]
        })

    chat = gemini_model.start_chat(history=gemini_history)
    response = chat.send_message(messages[-1]["content"])
    return response.text


# def call_gemini(messages, system, model):
#     gemini_model = genai.GenerativeModel(
#         model_name=model,
#         system_instruction=system
#     )

#     # Convert messages to Gemini format
#     gemini_history = []
#     for msg in messages[:-1]:  # all except last
#         gemini_history.append({
#             "role": "user" if msg["role"] == "user" else "model",
#             "parts": [msg["content"]]
#         })

#     chat = gemini_model.start_chat(history=gemini_history)
#     response = chat.send_message(messages[-1]["content"])
#     return response.text





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