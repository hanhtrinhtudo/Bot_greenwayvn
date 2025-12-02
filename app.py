# app.py
import time
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS

from config import ADMIN_SECRET
from db_utils import (
    save_message,
    get_recent_history,
    get_last_user_message,
    upsert_tvv_user,
    list_tvv_users,
    get_db_conn,
)
from intent_engine import handle_chat, looks_like_repeat_request, log_conversation

app = Flask(__name__)
CORS(app)

# ============ /openai-chat ============
@app.route("/openai-chat", methods=["POST"])
def openai_chat():
    start_time = time.time()
    try:
        if request.is_json:
            body = request.get_json(force=True) or {}
        else:
            body = request.form.to_dict() or {}

        user_message = (body.get("message") or "").strip()
        mode = (body.get("mode") or "").strip().lower() if isinstance(body, dict) else ""
        session_id = body.get("session_id") or ""
        channel = body.get("channel") or "web"
        user_id = body.get("user_id") or ""

        if not session_id:
            session_id = f"web-{request.remote_addr}-{int(time.time())}"

        used_history_message = ""
        message_for_ai = user_message

        if looks_like_repeat_request(user_message) and session_id:
            last_q = get_last_user_message(session_id)
            if last_q:
                used_history_message = last_q
                message_for_ai = last_q

        save_message(session_id, "user", user_message)

        history = get_recent_history(session_id, limit=10)

        reply_text, meta = handle_chat(
            message_for_ai,
            mode or None,
            session_id=session_id,
            return_meta=True,
            history=history,
        )

        save_message(session_id, "assistant", reply_text)

        latency_ms = int((time.time() - start_time) * 1000)

        log_payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "channel": channel,
            "session_id": session_id,
            "user_id": user_id,
            "user_message": user_message,
            "message_for_ai": message_for_ai,
            "used_history_message": used_history_message,
            "bot_reply": reply_text,
            "intent": meta.get("intent", ""),
            "mode_detected": meta.get("mode_detected"),
            "health_tags": meta.get("health_tags", []),
            "selected_combos": meta.get("selected_combos", []),
            "selected_products": meta.get("selected_products", []),
            "latency_ms": latency_ms,
        }
        log_conversation(log_payload)

        return jsonify({"reply": reply_text})

    except Exception as e:
        print("❌ ERROR /openai-chat:", e)
        return jsonify({"reply": "Xin lỗi, hiện tại hệ thống đang gặp lỗi. Anh/chị vui lòng thử lại sau nhé."}), 500


# ============ AUTH /auth/register ============
@app.route("/auth/register", methods=["POST"])
def auth_register():
    # giống logic cũ, chỉ đổi import upsert_tvv_user & log_conversation từ module
    ...


# ============ ADMIN /admin/users ============
def require_admin_secret():
    if not ADMIN_SECRET:
        return False, "ADMIN_SECRET chưa được cấu hình trên server."
    header_secret = request.headers.get("X-Admin-Secret") or ""
    if header_secret != ADMIN_SECRET:
        return False, "Sai ADMIN_SECRET."
    return True, ""


@app.route("/admin/users", methods=["GET"])
def admin_list_users():
    ...


@app.route("/debug-db", methods=["GET"])
def debug_db():
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT NOW()")
            now = cur.fetchone()[0]
        conn.close()
        return f"DB OK, time = {now}", 200
    except Exception as e:
        return f"DB ERROR: {e}", 500


@app.route("/", methods=["GET"])
def home():
    return "🔥 Greenway / Welllab Chatbot Gateway đang chạy ngon lành!", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
