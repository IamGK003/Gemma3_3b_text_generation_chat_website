import os
import re
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from llama_cpp import Llama

app = Flask(__name__)

LOGS_DIR = os.path.join(os.path.dirname(__file__), 'chat_logs')
os.makedirs(LOGS_DIR, exist_ok=True)


MODEL_PATH = "model/Llama-3.2-1B-Instruct-Q4_K_M.gguf"

print("Loading model...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=4,
    verbose=False
)
print("Model ready!")

# Session states
chat_history = []
current_session_filename = None


def escape_rtf(text):
    """Escapes special characters and unicode symbols for RTF formatting."""
    res = []
    for char in text:
        code = ord(char)
        if char == '\\':
            res.append(r'\\')
        elif char == '{':
            res.append(r'\{')
        elif char == '}':
            res.append(r'\}')
        elif char == '\n':
            res.append(r'\line ')
        elif code > 127:
            # Unicode character escaping in RTF
            res.append(f'\\u{code}?')
        else:
            res.append(char)
    return "".join(res)


def initialize_rtf_file(filepath):
    """Creates a new RTF document header with proper font and color tables."""
    rtf_header = (
        "{\\rtf1\\ansi\\ansicpg1252\\deff0"
        "{\\fonttbl{\\f0\\fnil\\fcharset0 Arial;}{\\f1\\fmodern\\fcharset0 Courier New;}}"
        "{\\colortbl ;\\red0\\green102\\blue204;\\red0\\green153\\blue76;\\red128\\red128\\blue128;}\n"
        "{\\b\\fs28 Llama Chat Session Log}\\line\\line\n"
    )
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(rtf_header)


def append_rtf_turn(filepath, user_query, model_response):
    """Appends turn content in Rich Text Format with separators."""
    escaped_user = escape_rtf(user_query)
    escaped_model = escape_rtf(model_response)

    # Format in RTF syntax
    rtf_entry = (
        "\\par\\cf1\\b --- USER ---\\b0\\cf0\\line\n"
        f"{escaped_user}\\line\\line\n"
        "\\cf2\\b --- MODEL ---\\b0\\cf0\\line\n"
        f"{escaped_model}\\line\\line\n"
        "\\cf3 --------------------------------------------------\\cf0\\line\n"
    )

    # Insert before closing brace of RTF
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(rtf_entry)


@app.route("/")
def index():
    return render_template("index.html")


# 1. START A NEW CHAT
@app.route("/api/new_chat", methods=["POST"])
def new_chat():
    global chat_history, current_session_filename
    chat_history = []
    filename = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.rtf"
    current_session_filename = filename
    filepath = os.path.join(LOGS_DIR, filename)
    initialize_rtf_file(filepath)
    return jsonify({"filename": filename})


# 2. CHAT WITH MODEL
@app.route("/api/chat", methods=["POST"])
def chat():
    global current_session_filename
    if not current_session_filename:
        new_chat()

    data = request.json or {}
    user_prompt = data.get("prompt", "").strip()

    if not user_prompt:
        return jsonify({"error": "Empty prompt"}), 400

    chat_history.append({"role": "user", "content": user_prompt})

    response = llm.create_chat_completion(
        messages=chat_history,
        max_tokens=1024,
        temperature=0.7
    )

    model_reply = response["choices"][0]["message"]["content"]
    chat_history.append({"role": "assistant", "content": model_reply})

    filepath = os.path.join(LOGS_DIR, current_session_filename)
    append_rtf_turn(filepath, user_prompt, model_reply)

    return jsonify({"response": model_reply, "filename": current_session_filename})


# 3. LIST ALL SAVED RTF FILES
@app.route("/api/logs", methods=["GET"])
def list_logs():
    files = [f for f in os.listdir(LOGS_DIR) if f.endswith('.rtf')]
    files.sort(reverse=True)
    return jsonify({"files": files})


# 4. LOAD & PARSE RTF FILE TO SHOW IN CHAT INTERFACE
@app.route("/api/logs/<filename>", methods=["GET"])
def load_rtf_log(filename):
    filepath = os.path.join(LOGS_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Convert RTF unicode escapes (\uXXXX?) back to normal symbols
    def rtf_decode(match):
        code = int(match.group(1))
        return chr(code if code >= 0 else code + 65536)

    text_content = re.sub(r'\\u(-?\d+)\?', rtf_decode, content)
    text_content = text_content.replace(r'\line', '\n').replace(r'\par', '\n')

    # Parse out turns based on user/model separators
    blocks = text_content.split("--------------------------------------------------")
    parsed_turns = []

    for block in blocks:
        user_match = re.search(r"--- USER ---\s*(.*?)(?=--- MODEL ---|\Z)", block, re.DOTALL)
        model_match = re.search(r"--- MODEL ---\s*(.*)", block, re.DOTALL)

        u_text = user_match.group(1).strip() if user_match else ""
        m_text = model_match.group(1).strip() if model_match else ""

        # Cleanup remaining RTF syntax tags
        u_text = re.sub(r'\\[a-z0-9]+\s?', '', u_text)
        m_text = re.sub(r'\\[a-z0-9]+\s?', '', m_text)

        if u_text or m_text:
            parsed_turns.append({"user": u_text, "model": m_text})

    return jsonify({"turns": parsed_turns})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)