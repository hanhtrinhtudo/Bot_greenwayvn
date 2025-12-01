import os
import json
import time
import unicodedata
from datetime import datetime
import psycopg2
from psycopg2.extras import DictCursor

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

# ===== OpenAI SDK (Responses API) =====
try:
    from openai import OpenAI
except ImportError:
    raise Exception("Chưa cài openai SDK. Chạy: pip install openai")

# ===== Load ENV =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DATABASE_URL   = os.getenv("DATABASE_URL", "")

if not OPENAI_API_KEY:
    raise Exception("Thiếu biến môi trường OPENAI_API_KEY")

HOTLINE = os.getenv("HOTLINE", "09xx.xxx.xxx")
FANPAGE_URL = os.getenv("FANPAGE_URL", "https://facebook.com/ten-fanpage")
ZALO_OA_URL = os.getenv("ZALO_OA_URL", "https://zalo.me/ten-oa")
WEBSITE_URL = os.getenv("WEBSITE_URL", "https://greenwayglobal.vn")

LOG_WEBHOOK_URL = os.getenv("LOG_WEBHOOK_URL", "")  # 👈 Webhook Apps Script

# ===== Init App =====
app = Flask(__name__)
CORS(app)  # Cho phép web / Conversational Agents gọi API không bị CORS

client = OpenAI(api_key=OPENAI_API_KEY)

# ====== DB HELPER ======
def get_db_conn():
    # Render khuyến nghị dùng 1 connection / process
    # nên có thể cache connection ở global nếu muốn tối ưu hơn
    return psycopg2.connect(DATABASE_URL, cursor_factory=DictCursor)

def get_recent_history(session_id: str, limit: int = 8):
    """Lấy lịch sử gần nhất của 1 phiên chat (user + assistant)."""
    if not session_id:
        return []

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT role, content
                FROM chat_logs
                WHERE session_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (session_id, limit),
            )
            rows = cur.fetchall()
        # đảo ngược lại theo thứ tự cũ
        rows = list(reversed(rows))
        return [{"role": r["role"], "content": r["content"]} for r in rows]
    finally:
        conn.close()

def save_message(session_id: str, role: str, content: str):
    """Lưu 1 message vào DB."""
    if not session_id or not content:
        return

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_logs (session_id, role, content)
                VALUES (%s, %s, %s)
                """,
                (session_id, role, content),
            )
        conn.commit()
    finally:
        conn.close()

# >>> MỚI: hàm nhận diện câu “trả lời lại câu hỏi trên”
def strip_accents(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text

def is_retry_phrase(text: str) -> bool:
    """
    Nhận diện các câu kiểu:
    - 'trả lời lại câu hỏi trên'
    - 'trả lời lại câu vừa rồi'
    - 'trả lời lại câu hỏi trước'
    """
    t = strip_accents((text or "").strip())
    if not t:
        return False

    patterns = [
        "tra loi lai cau hoi tren",
        "tra loi lai cau hoi vua roi",
        "tra loi lai cau vua roi",
        "tra loi lai cau truoc",
        "tra loi lai cau hoi truoc",
        "tra loi lai cau hoi nay",
        "tra loi lai cau hoi luc nay",
    ]
    return any(p in t for p in patterns)

def get_last_user_question_for_retry(session_id: str) -> str | None:
    """
    Lấy câu hỏi user gần nhất (role='user') nhưng KHÔNG phải các câu 'trả lời lại...'
    dùng cho tình huống retry.
    """
    history = get_recent_history(session_id, limit=20)
    # Duyệt từ cuối lên đầu để lấy câu gần nhất
    for msg in reversed(history):
        if msg.get("role") == "user" and not is_retry_phrase(msg.get("content", "")):
            return msg.get("content")
    return None
    
# =====================================================================
#   LOAD DỮ LIỆU JSON
# =====================================================================
def load_json_file(path, default=None):
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Không đọc được file {path}: {e}")
        return default


PRODUCTS_DATA = load_json_file("products.json", {"products": []})
COMBOS_DATA = load_json_file("combos.json", {"combos": []})
HEALTH_TAGS_CONFIG = load_json_file("health_tags_config.json", {})
COMBOS_META = load_json_file("combos_meta.json", {})
MULTI_ISSUE_RULES = load_json_file("multi_issue_rules.json", {"rules": []})

PRODUCTS = PRODUCTS_DATA.get("products", [])
COMBOS = COMBOS_DATA.get("combos", [])

def strip_accents(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text


def extract_tags_from_text(text: str):
    """Dựa trên HEALTH_TAGS_CONFIG, map câu hỏi sang health_tags."""
    text_norm = strip_accents(text)
    found = set()

    for tag, cfg in HEALTH_TAGS_CONFIG.items():
        for syn in cfg.get("synonyms", []):
            syn_norm = strip_accents(syn)
            if syn_norm and syn_norm in text_norm:
                found.add(tag)
                break
    return list(found)


def apply_multi_issue_rules(text: str):
    """Thử match các rule nhiều vấn đề trong multi_issue_rules."""
    text_norm = strip_accents(text)
    best_rule = None
    best_count = 0

    for rule in MULTI_ISSUE_RULES.get("rules", []):
        match_phrases = rule.get("match_phrases", [])
        count = 0
        for phrase in match_phrases:
            if strip_accents(phrase) in text_norm:
                count += 1
        if count > best_count and count > 0:
            best_count = count
            best_rule = rule

    return best_rule


def score_combo_for_tags(combo, requested_tags):
    requested_tags = set(requested_tags)
    combo_tags = set(combo.get("health_tags", []))
    intersection = requested_tags & combo_tags
    score = 0

    # Mỗi tag trùng +3 điểm
    score += 3 * len(intersection)

    # Ưu tiên combo core/support
    meta = COMBOS_META.get(combo.get("id", ""), {})
    role = meta.get("role", "core")
    if role == "core":
        score += 2
    elif role == "support":
        score += 1

    # Thêm weight theo tỉ lệ phủ
    if combo_tags and requested_tags:
        overlap_ratio = len(intersection) / len(requested_tags)
        score += overlap_ratio

    return score, list(intersection)


def select_combos_for_tags(requested_tags, user_text):
    """Chọn 1–3 combo phù hợp nhất với tập requested_tags."""
    if not requested_tags and user_text:
        requested_tags = extract_tags_from_text(user_text)

    requested_tags = list(set(requested_tags))
    if not requested_tags:
        return [], []

    # Ưu tiên rule nhiều ý nếu match
    rule = apply_multi_issue_rules(user_text or "")
    if rule:
        candidate_ids = set(rule.get("recommended_combos", []))
        candidates = [c for c in COMBOS if c.get("id") in candidate_ids]
    else:
        candidates = COMBOS

    scored = []
    for combo in candidates:
        s, matched = score_combo_for_tags(combo, requested_tags)
        if s > 0:
            scored.append((s, combo, matched))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:3]

    selected_combos = [item[1] for item in top]
    covered_tags = set()
    for _, _, matched in top:
        covered_tags.update(matched)

    return selected_combos, list(covered_tags)


def search_products_by_tags(requested_tags, limit=5):
    requested_tags = set(requested_tags)
    if not requested_tags:
        return []

    results = []
    for p in PRODUCTS:
        tags = set(p.get("health_tags") or [])
        group = p.get("group")  # group: gan, tieu_hoa, than, tim_mach...
        if group:
            tags.add(group)
        if tags & requested_tags:
            results.append(p)

    return results[:limit]


def call_openai_responses(prompt_text: str) -> str:
    """Gọi Responses API giống style dự án cũ của anh."""
    try:
        res = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt_text,
        )
        reply_text = getattr(res, "output_text", "") or ""
        reply_text = reply_text.strip()
        if not reply_text:
            reply_text = "Hiện tại em không nhận được kết quả từ hệ thống OpenAI."
        return reply_text
    except Exception as e:
        print("❌ ERROR OpenAI Responses:", e)
        return (
            "Xin lỗi, hiện tại hệ thống AI đang gặp lỗi, anh/chị vui lòng thử lại sau "
            "hoặc liên hệ hotline để tuyến trên hỗ trợ trực tiếp."
        )


def llm_answer_for_combos(user_question, requested_tags, combos, covered_tags):
    if not combos:
        return (
            "Hiện em chưa tìm thấy combo phù hợp trong dữ liệu cho trường hợp này. "
            f"Anh/chị vui lòng liên hệ hotline {HOTLINE} để tuyến trên tư vấn chi tiết hơn ạ."
        )

    combos_json = json.dumps(combos, ensure_ascii=False, indent=2)
    tags_text = ", ".join(requested_tags)

    prompt = f"""
Bạn là trợ lý tư vấn cho công ty thực phẩm chức năng Greenway/Welllab.
Bạn chỉ được dùng đúng dữ liệu combo và sản phẩm trong JSON bên dưới, không được bịa thêm sản phẩm hay công dụng.

Dưới đây là câu hỏi và dữ liệu:

- Câu hỏi của khách / tư vấn viên: "{user_question}"
- Các tags/vấn đề sức khỏe hệ thống trích xuất được: {tags_text}

Dữ liệu các combo đã được hệ thống chọn (JSON):

{combos_json}

YÊU CẦU TRẢ LỜI (bằng tiếng Việt, dễ hiểu, rõ ràng):

1. Mở đầu 1–3 câu: tóm tắt các vấn đề/nhu cầu chính và định hướng xử lý (theo combo) cho khách.
2. Với từng combo:
   - Nêu rõ combo này hỗ trợ những vấn đề nào trong các vấn đề khách đang gặp.
   - Liệt kê từng sản phẩm trong combo:
     + Tên sản phẩm
     + Lợi ích chính / tác dụng hỗ trợ
     + Thời gian dùng gợi ý (nếu có trong dữ liệu)
     + Cách dùng tóm tắt (dựa trên dose_text/usage_text nếu có)
     + Giá (price_text)
     + Link sản phẩm (product_url)
3. Nếu vấn đề có vẻ nặng/nhạy cảm (ung thư, tim mạch nặng, suy thận, v.v.) hãy khuyến nghị khách nên thăm khám và tái khám định kỳ.
4. Cuối câu trả lời, luôn nhắc: "Sản phẩm không phải là thuốc và không có tác dụng thay thế thuốc chữa bệnh.".
5. Viết giọng điệu gần gũi, lịch sự, hướng dẫn như đang nói chuyện với tư vấn viên/khách hàng thật.
"""
    return call_openai_responses(prompt)


def llm_answer_for_products(user_question, requested_tags, products):
    if not products:
        return (
            "Hiện em chưa tìm thấy sản phẩm phù hợp trong dữ liệu cho trường hợp này. "
            f"Anh/chị vui lòng liên hệ hotline {HOTLINE} để được tư vấn rõ hơn ạ."
        )

    products_json = json.dumps(products, ensure_ascii=False, indent=2)
    tags_text = ", ".join(requested_tags)

    prompt = f"""
Bạn là trợ lý tư vấn cho công ty thực phẩm chức năng Greenway/Welllab.
Bạn chỉ được dùng đúng dữ liệu sản phẩm trong JSON bên dưới, không được bịa thêm sản phẩm hay công dụng.

- Câu hỏi: "{user_question}"
- Các tags/vấn đề sức khỏe: {tags_text}

Dữ liệu các sản phẩm đã được hệ thống chọn (JSON):

{products_json}

YÊU CẦU TRẢ LỜI:

1. Mở đầu 1–2 câu: giới thiệu đây là các sản phẩm hỗ trợ phù hợp với vấn đề mà khách đang gặp.
2. Với từng sản phẩm:
   - Tên sản phẩm
   - Vấn đề chính mà sản phẩm hỗ trợ (dựa trên group/health_tags)
   - Lợi ích chính (dựa trên benefits_text hoặc mô tả)
   - Cách dùng tóm tắt (usage_text hoặc dose_text nếu có)
   - Giá (price_text)
   - Link sản phẩm (product_url)
3. Cuối cùng nhắc: sản phẩm không phải là thuốc và không có tác dụng thay thế thuốc chữa bệnh.
4. Viết ngắn gọn, rõ ràng, dễ dùng cho tư vấn viên khi chát với khách.
"""
    return call_openai_responses(prompt)


def handle_buy_and_payment_info():
    return (
        "Để mua hàng, anh/chị có thể chọn một trong các cách sau:\n\n"
        "1️⃣ Đặt hàng trực tiếp trên website:\n"
        f"   • {WEBSITE_URL}\n\n"
        "2️⃣ Nhắn tin qua Zalo OA của công ty để được tư vấn và chốt đơn:\n"
        f"   • {ZALO_OA_URL}\n\n"
        "3️⃣ Gọi hotline để được hỗ trợ nhanh:\n"
        f"   • {HOTLINE}\n\n"
        "Về thanh toán, hiện công ty hỗ trợ:\n"
        "- Thanh toán khi nhận hàng (COD)\n"
        "- Chuyển khoản ngân hàng theo hướng dẫn từ tư vấn viên hoặc trên website."
    )


def handle_escalate_to_hotline():
    return (
        "Câu hỏi này thuộc nhóm chính sách/kế hoạch kinh doanh chuyên sâu nên cần tuyến trên hỗ trợ trực tiếp ạ.\n\n"
        "Anh/chị vui lòng để lại:\n"
        "- Họ tên\n"
        "- Số điện thoại\n"
        "- Mã TVV (nếu có)\n\n"
        f"Hoặc gọi thẳng hotline: {HOTLINE}\n"
        "Tuyến trên sẽ liên hệ và tư vấn chi tiết cho anh/chị sớm nhất có thể."
    )


def handle_channel_navigation():
    return (
        "Anh/chị có thể theo dõi thông tin, chương trình ưu đãi và kiến thức sức khỏe tại các kênh sau:\n\n"
        f"📘 Fanpage: {FANPAGE_URL}\n"
        f"💬 Zalo OA: {ZALO_OA_URL}\n"
        f"🌐 Website: {WEBSITE_URL}\n\n"
        "Nếu cần hỗ trợ gấp, anh/chị gọi trực tiếp hotline giúp em nhé."
    )


def detect_mode(user_message: str) -> str:
    """Đoán xem user đang hỏi về combo / sản phẩm / mua hàng / kênh / kinh doanh."""
    text_norm = strip_accents(user_message)

    # Hỏi kinh doanh, chính sách, hoa hồng
    business_keywords = [
        "chinh sach", "hoa hong", "tuyen dung", "len cap",
        "leader", "doanh so", "muc tieu thang"
    ]
    if any(k in text_norm for k in business_keywords):
        return "business"

    # Hỏi mua hàng / thanh toán
    buy_keywords = [
        "mua", "dat hang", "thanh toan", "ship", "giao hang", "dat mua"
    ]
    if any(k in text_norm for k in buy_keywords):
        return "buy"

    # Hỏi kênh, fanpage, zalo
    channel_keywords = [
        "fanpage", "zalo", "kenh", "website", "trang web"
    ]
    if any(k in text_norm for k in channel_keywords):
        return "channel"

    # Nhắc đến combo / sản phẩm
    if "combo" in text_norm:
        return "combo"
    if "san pham" in text_norm or "sản phẩm" in user_message.lower():
        return "product"

    return "auto"

# =====================================================================
#   LOG CONVERSATION → GOOGLE SHEETS
# =====================================================================
def log_conversation(payload: dict):
    if not LOG_WEBHOOK_URL:
        return
    try:
        requests.post(LOG_WEBHOOK_URL, json=payload, timeout=2)
    except Exception as e:
        print("[WARN] Log error:", e)

def handle_chat(user_message: str, mode: str | None = None, return_meta: bool = False):
    text = (user_message or "").strip()
    if not text:
        reply = "Em chưa nhận được câu hỏi của anh/chị."
        if return_meta:
            meta = {
                "mode_detected": "",
                "health_tags": [],
                "selected_combos": [],
                "selected_products": [],
            }
            return reply, meta
        return reply

    detected_mode = detect_mode(text) if not mode else mode.lower().strip()
    mode = detected_mode

    # meta mặc định
    requested_tags = extract_tags_from_text(text)
    meta = {
        "mode_detected": mode,
        "health_tags": requested_tags,
        "selected_combos": [],
        "selected_products": [],
    }
    
    # 👇 THÊM DÒNG NÀY
    print("[DEBUG] handle_chat mode =", mode, "| text =", text)
    
    # Các mode đơn giản
    if mode == "buy":
        reply = handle_buy_and_payment_info()
        if return_meta:
            return reply, meta
        return reply

    if mode == "channel":
        reply = handle_channel_navigation()
        if return_meta:
            return reply, meta
        return reply

    if mode == "business":
        reply = handle_escalate_to_hotline()
        if return_meta:
            return reply, meta
        return reply

    # Các mode về sức khỏe: combo / product / auto
    want_combo = "combo" in strip_accents(text) or mode == "combo"
    want_product = "san pham" in strip_accents(text) or "sản phẩm" in text.lower() or mode == "product"

    if want_combo and not want_product:
        combos, covered_tags = select_combos_for_tags(requested_tags, text)
        meta["selected_combos"] = [c.get("id") for c in combos]
        reply = llm_answer_for_combos(text, requested_tags, combos, covered_tags)
        if return_meta:
            return reply, meta
        return reply

    if want_product and not want_combo:
        products = search_products_by_tags(requested_tags)
        meta["selected_products"] = [p.get("id") for p in products]
        reply = llm_answer_for_products(text, requested_tags, products)
        if return_meta:
            return reply, meta
        return reply

    # AUTO: ưu tiên combo, nếu không có thì show sản phẩm
    combos, covered_tags = select_combos_for_tags(requested_tags, text)
    if combos:
        meta["selected_combos"] = [c.get("id") for c in combos]
        reply = llm_answer_for_combos(text, requested_tags, combos, covered_tags)
        if return_meta:
            return reply, meta
        return reply

    products = search_products_by_tags(requested_tags)
    if products:
        meta["selected_products"] = [p.get("id") for p in products]
        reply = llm_answer_for_products(text, requested_tags, products)
        if return_meta:
            return reply, meta
        return reply

    # Không match gì
    reply = (
        "Hiện em chưa tìm thấy combo hay sản phẩm nào phù hợp trong dữ liệu cho trường hợp này. "
        f"Anh/chị có thể nói rõ hơn tình trạng sức khỏe, hoặc liên hệ hotline {HOTLINE} để tuyến trên hỗ trợ kỹ hơn ạ."
    )
    if return_meta:
        return reply, meta
    return reply

@app.route("/openai-chat", methods=["POST"])
def openai_chat():
    data = request.get_json(silent=True) or {}
    start_time = time.time()
    try:
        body = request.get_json(force=True)
        user_message = (body.get("message") or "").strip()
        mode = (body.get("mode") or "").strip().lower() if isinstance(body, dict) else ""
        session_id = body.get("session_id") or ""
        channel = body.get("channel") or "web"
        user_id = body.get("user_id") or ""
        
        # >>> MỚI: lưu câu hỏi của user vào DB NGAY LẬP TỨC
        try:
            if session_id and user_message:
                save_message(session_id, "user", user_message)
        except Exception as db_err:
            print("[WARN] DB log user error:", db_err)

        # >>> MỚI: xử lý case 'trả lời lại câu hỏi trên'
        effective_message = user_message
        retry_used = False
        if session_id and is_retry_phrase(user_message):
            last_q = get_last_user_question_for_retry(session_id)
            if last_q:
                print("[DEBUG] Retry phrase detected, dùng lại câu hỏi:", last_q)
                effective_message = last_q
                retry_used = True

        # Gọi handler với effective_message
        reply_text, meta = handle_chat(user_message, mode or None, return_meta=True)

        latency_ms = int((time.time() - start_time) * 1000)

        # >>> MỚI: lưu trả lời của Bot vào DB
        try:
            if session_id and reply_text:
                save_message(session_id, "assistant", reply_text)
        except Exception as db_err:
            print("[WARN] DB log assistant error:", db_err)

        log_payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "channel": channel,
            "session_id": session_id,
            "user_id": user_id,
            "user_message": user_message,
            "effective_message": effective_message,  # 👈 xem Bot đã dùng câu nào để xử lý
            "retry_used": retry_used,
            "bot_reply": reply_text,
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
        return jsonify({
            "reply": "Xin lỗi, hiện tại hệ thống đang gặp lỗi. Anh/chị vui lòng thử lại sau nhé."
        }), 500


@app.route("/", methods=["GET"])
def home():
    return "🔥 Greenway / Welllab Chatbot Gateway đang chạy ngon lành!", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
