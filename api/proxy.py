from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

PLGRID_API_KEY = "plg-rLA29o1RwkSVP8zqBQDoSlz4PZdjhem0yP-rpc3v6Mg"
# PLGRID_MODEL = "CYFRAGOVPL/Llama-PLLuM-70B-chat-250801"
PLGRID_MODEL = "google/gemma-4-31B"

PLGRID_ENDPOINT = "https://llmlab.plgrid.pl/api/v1/chat/completions"

@app.route("/v1/chat/completions", methods=["POST"])
def chat():

    incoming = request.json

    temperature = incoming.get("temperature", 0.7)
    temperature = 0.7 #min(float(temperature), 1.0)

    # Convert request to PLGrid format
    payload = {
        "model": PLGRID_MODEL,
        "messages": incoming["messages"],
        "max_tokens": incoming.get("max_tokens", 200),
        "temperature": temperature,
        "top_p": incoming.get("top_p", 1),
        "stream": False
    }

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {PLGRID_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        PLGRID_ENDPOINT,
        json=payload,
        headers=headers
    )

    print("TEXT:", response.text[:500])

    # Parse JSON safely
    try:
        plgrid_response = response.json()
    except Exception as e:
        return jsonify({
            "error": "invalid_json",
            "detail": str(e),
            "raw": response.text[:500]
        }), 500

    # Propagate upstream errors properly
    if response.status_code != 200:
        return jsonify(plgrid_response), response.status_code

    # Extract content safely
    try:
        content = plgrid_response["choices"][0]["message"]["content"]
    except Exception as e:
        return jsonify({
            "error": "bad_response_format",
            "detail": str(e),
            "response": plgrid_response
        }), 500

    # Return MINIMAL OpenAI-compatible schema
    fixed_response = {
        "id": plgrid_response.get("id", "plgrid"),
        "object": "chat.completion",
        "created": plgrid_response.get("created", 0),
        "model": PLGRID_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }
        ]
    }

    return jsonify(fixed_response)

@app.route("/v1/models", methods=["GET"])
def models():

    return jsonify({
        "object": "list",
        "data": [
            {
                "id": PLGRID_MODEL,
                "object": "model",
                "owned_by": "plgrid"
            }
        ]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)