from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import anthropic
import os

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app)  # Allow frontend to call this API

# Init Anthropic client (reads ANTHROPIC_API_KEY from env)
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are Nova, a friendly and helpful AI assistant.
Be concise, clear, and conversational.
If asked about code, provide clean examples."""

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()

        # Validate input
        if not data or "messages" not in data:
            return jsonify({"error": "No messages provided"}), 400

        messages = data["messages"]  # List of {role, content}

        # Call Claude API
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages
        )

        reply = response.content[0].text

        return jsonify({
            "reply": reply,
            "status": "ok"
        })

    except anthropic.APIConnectionError:
        return jsonify({"error": "Cannot connect to Anthropic API"}), 503

    except anthropic.RateLimitError:
        return jsonify({"error": "Rate limit reached. Try again later."}), 429

    except anthropic.APIStatusError as e:
        return jsonify({"error": str(e)}), e.status_code

    except Exception as e:
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)