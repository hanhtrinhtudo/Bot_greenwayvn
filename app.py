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
DATABASE_URL = os.getenv("DATABASE_URL", "")

if not OPENAI_API_KEY:
    raise Exception("Thiếu biến môi trường OPENAI_API_KEY")

HOTLINE = os.getenv("HOTLINE", "09xx.xxx.xxx")
FANPAGE_URL = os.getenv("FANPAGE_URL", "https://facebook.com/ten-fanpage")
ZALO_OA_URL = os.getenv("ZALO_OA_URL", "https://zalo.me/ten-oa")
WEBSITE_URL = os.getenv("WEBSITE_URL", "https://greenwayglobal.vn")

LOG_WEBHOOK_URL = os.getenv("LOG_WEBHOOK_URL", "")  # Webhook Apps Script
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")  # dùng chung cho /admin/*

# ===== Init App =====
app = Flask(__name__)
CORS(app)  # Cho phép web / Conversational Agents gọi API không bị CORS

client = OpenAI(api_key=OPENAI_API_KEY)

# =====================================================================
#   DB – QUẢN LÝ TVV (HỒ SƠ TƯ VẤN VIÊN)
# =====================================================================
def upsert_tvv_user(tvv_code: str, full_name: str, phone: str, email: str, company_name: str):
    """
    Tạo mới hoặc cập nhật hồ sơ TVV theo tvv_code.
    """
    if not tvv_code or not full_name:
        raise ValueError("Thiếu tvv_code hoặc full_name")

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO tvv_users (tvv_code, full_name, phone, email, company_name)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (tvv_code)
                DO UPDATE SET
                  full_name    = EXCLUDED.full_name,
                  phone        = EXCLUDED.phone,
                  email        = EXCLUDED.email,
                  company_name = EXCLUDED.company_name,
                  updated_at   = NOW()
                """,
                (tvv_code, full_name, phone, email, company_name),
            )
        conn.commit()
    finally:
        conn.close()


def list_tvv_users(q: str = "", limit: int = 200):
    """
    Lấy danh sách TVV cho trang admin (có search q).
    """
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            if q:
                pattern = f"%{q}%"
                cur.execute(
                    """
                    SELECT tvv_code, full_name, phone, email, company_name, created_at, updated_at
                    FROM tvv_users
                    WHERE
                      tvv_code ILIKE %s OR
                      full_name ILIKE %s OR
                      phone ILIKE %s OR
                      email ILIKE %s OR
                      company_name ILIKE %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (pattern, pattern, pattern, pattern, pattern, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT tvv_code, full_name, phone, email, company_name, created_at, updated_at
                    FROM tvv_users
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
            rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()

# =====================================================================
#   DB HELPER – KẾT NỐI & LỊCH SỬ HỘI THOẠI
# =====================================================================
def get_db_conn():
    """
    Mở connection tới PostgreSQL (Render cung cấp DATABASE_URL).
    """
    if not DATABASE_URL:
        raise Exception("Thiếu biến môi trường DATABASE_URL")
    return psycopg2.connect(DATABASE_URL, cursor_factory=DictCursor)


def get_recent_history(session_id: str, limit: int = 8):
    """
    Lấy lịch sử gần nhất của 1 phiên chat (user + assistant).
    Kết quả: list [{role, content}], đã sort từ cũ -> mới.
    """
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
        rows = list(reversed(rows))  # đảo lại theo thứ tự cũ
        return [{"role": r["role"], "content": r["content"]} for r in rows]
    finally:
        conn.close()


def save_message(session_id: str, role: str, content: str):
    """
    Lưu 1 message vào DB (nếu có session_id & content).
    """
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


def get_last_user_message(session_id: str):
    """
    Lấy câu hỏi gần nhất của USER trong 1 session.
    Dùng cho các câu kiểu: 'trả lời lại câu hỏi trên'.
    """
    if not session_id:
        return None

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT content
                FROM chat_logs
                WHERE session_id = %s AND role = 'user'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (session_id,),
            )
            row = cur.fetchone()
            return row["content"] if row else None
    finally:
        conn.close()

# =====================================================================
#   TIỆN ÍCH XỬ LÝ TEXT
# =====================================================================
def strip_accents(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text


def looks_like_repeat_request(text: str) -> bool:
    """
    Nhận diện câu kiểu: 'trả lời lại câu hỏi trên / vừa nãy'.
    """
    if not text:
        return False
    t = strip_accents(text)
    t = " ".join(t.split())

    patterns = [
        "tra loi lai cau hoi",
        "tra loi lai cau tren",
        "tra loi lai cau vua nay",
        "tra loi lai cau truoc",
        "hoi lai cau hoi truoc",
        "hoi lai cau truoc",
    ]
    return any(p in t for p in patterns)


def looks_like_followup(text: str) -> bool:
    """
    Nhận diện câu follow-up dựa trên câu trả lời trước:
    'combo trên uống bao lâu', 'sản phẩm đó giá bao nhiêu', ...
    """
    if not text:
        return False
    t = strip_accents(text)
    t = " ".join(t.split())

    # Nhắc 'combo / sản phẩm / gói' + 'trên / đó / vừa nãy / trước'
    core_phrases = [
        "combo tren",
        "combo truoc",
        "combo vua nay",
        "combo do",
        "san pham tren",
        "san pham truoc",
        "san pham vua nay",
        "san pham do",
        "goi tren",
        "goi truoc",
        "goi vua nay",
        "goi do",
    ]
    if any(p in t for p in core_phrases):
        return True

    # Câu hỏi về thời gian uống / liều / giá mà thường là follow-up
    if "bao lau" in t and ("uong" in t or "dung" in t):
        return True
    if "gia bao nhieu" in t or "gia the nao" in t:
        return True
    if "moi lan uong" in t or "ngay uong" in t or "cach uong" in t or "cach dung" in t:
        return True

    return False

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

# =====================================================================
#   TAG & SELECTION
# =====================================================================
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

def search_products_by_groups(groups, limit=5):
    """
    Chọn sản phẩm theo group (tieu_hoa, gan, than, ...),
    dùng khi health_tags không match nhưng AI đã gợi ý nhóm.
    """
    group_set = {g for g in (groups or []) if g}
    if not group_set:
        return []

    results = []
    for p in PRODUCTS:
        g = p.get("group")
        if g and g in group_set:
            results.append(p)

    return results[:limit]

# =====================================================================
#   OPENAI RESPONSES
# =====================================================================
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


def safe_parse_json(text: str, default=None):
    """Cố gắng bóc JSON từ câu trả lời của model."""
    if default is None:
        default = {}
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        # Thử bóc từ { ... }
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start:end+1])
        except Exception:
            return default
    return default


def ai_classify_intent(
    user_message: str, history_messages: list[dict] | None = None
) -> dict:
    """
    Phân loại ý định của người dùng:
    - greeting: chào hỏi
    - smalltalk: nói chuyện linh tinh, hỏi thăm, câu đời thường
    - health_question: hỏi về vấn đề sức khỏe chung (chưa rõ combo/sản phẩm)
    - product_question: hỏi về 1 sản phẩm cụ thể
    - combo_question: hỏi gợi ý combo
    - business_policy: chính sách / hoa hồng / tuyển dụng
    - buy_payment: cách mua hàng, thanh toán, giao hàng
    - channel_info: hỏi link fanpage, zalo, website
    - other: không rõ / chủ đề khác
    """
    history_messages = history_messages or []
    # Ghép lịch sử thành text
    history_text_lines = []
    for m in history_messages[-6:]:  # lấy tối đa 6 câu gần nhất
        role = m.get("role", "user")
        content = (m.get("content") or "").replace("\n", " ").strip()
        if not content:
            continue
        prefix = "KHÁCH" if role == "user" else "BOT"
        history_text_lines.append(f"{prefix}: {content}")
    history_text = "\n".join(history_text_lines)

    prompt = f"""
Bạn là module PHÂN LOẠI Ý ĐỊNH cho chatbot tư vấn sức khỏe & sản phẩm Greenway / Welllab.

Nhiệm vụ:
- Chỉ phân loại ý định, KHÔNG tự tư vấn sức khỏe.
- Dựa vào lịch sử hội thoại (nếu có) và câu mới nhất của người dùng.

Các loại intent hợp lệ:
- "greeting"       : chào hỏi, hỏi thăm kiểu "chào em", "hello", "dạo này sao rồi"...
- "smalltalk"      : nói chuyện đời thường, hỏi linh tinh, đùa vui, không yêu cầu tư vấn sản phẩm/chính sách.
- "health_question": hỏi về triệu chứng, tình trạng sức khỏe chung (có hoặc không nhắc combo/sản phẩm).
- "product_question": hỏi về MỘT sản phẩm cụ thể, tên, cách dùng, tác dụng, giá, link...
- "combo_question" : hỏi gợi ý combo / bộ sản phẩm cho vấn đề sức khỏe.
- "business_policy": hỏi về chính sách, hoa hồng, tuyển dụng, thăng cấp, KPI, doanh số...
- "buy_payment"    : hỏi về cách mua hàng, giao hàng, thanh toán.
- "channel_info"   : hỏi xin link fanpage, Zalo OA, website, kênh liên hệ.
- "other"          : mọi trường hợp khác không nằm trong các nhóm trên.

Hãy trả về JSON **duy nhất**, không giải thích thêm, dạng:

{{
  "intent": "greeting | smalltalk | health_question | product_question | combo_question | business_policy | buy_payment | channel_info | other",
  "reason": "giải thích rất ngắn, tiếng Việt"
}}

----- LỊCH SỬ HỘI THOẠI (nếu có) -----
{history_text}

----- CÂU MỚI NHẤT CỦA NGƯỜI DÙNG -----
"{user_message}"
"""

    raw = call_openai_responses(prompt)
    data = safe_parse_json(raw, default={"intent": "other", "reason": ""})
    intent = data.get("intent") or "other"
    data["intent"] = intent
    return data

def ai_analyze_symptom(user_message: str, history_messages: list[dict] | None = None) -> dict:
    """
    Phân tích triệu chứng / tình huống sức khỏe ở mức 'chuyên gia'.

    Trả về JSON dạng:
    {
      "main_issue": "tiêu hoá / đại tràng / gan mật / ...",
      "body_system": "digestive | liver | immune | cardio | other",
      "symptom_keywords": ["đi ngoài nhiều lần", "đau bụng", ...],
      "severity": "mild | moderate | severe",
      "recommended_groups": ["tieu_hoa", "dai_trang"],
      "suggested_tags": ["tieu_hoa", "dai_trang"]
    }
    """
    history_messages = history_messages or []
    history_text_lines = []
    for m in history_messages[-6:]:
        role = m.get("role", "user")
        content = (m.get("content") or "").replace("\n", " ").strip()
        if not content:
            continue
        prefix = "KHÁCH" if role == "user" else "BOT"
        history_text_lines.append(f"{prefix}: {content}")
    history_text = "\n".join(history_text_lines)

    prompt = f"""
Bạn là module PHÂN TÍCH TRIỆU CHỨNG cho trợ lý sức khỏe Greenway/Welllab.

Nhiệm vụ:
- ĐỌC và HIỂU mô tả triệu chứng của người dùng (TVV/Leader hoặc khách).
- SUY LUẬN xem vấn đề chính thuộc nhóm nào, mức độ ra sao.
- Gợi ý các nhóm sản phẩm NÊN ƯU TIÊN (theo group trong dữ liệu: tieu_hoa, gan, than, tim_mach, mien_dich, xuong_khop,...).
- Đề xuất thêm các health_tags liên quan (nếu có).

Đầu ra là JSON DUY NHẤT, KHÔNG giải thích thêm, có dạng:

{{
  "main_issue": "<mô tả ngắn vấn đề chính>",
  "body_system": "digestive | liver | immune | cardio | neuro | other",
  "symptom_keywords": ["..."],
  "severity": "mild | moderate | severe",
  "recommended_groups": ["tieu_hoa", "dai_trang", "men_vi_sinh"],
  "suggested_tags": ["tieu_hoa", "dai_trang"]
}}

----- LỊCH SỬ HỘI THOẠI GẦN ĐÂY (nếu có) -----
{history_text}

----- CÂU MÔ TẢ TRIỆU CHỨNG / VẤN ĐỀ MỚI NHẤT -----
"{user_message}"
"""
    raw = call_openai_responses(prompt)
    data = safe_parse_json(
        raw,
        default={
            "main_issue": "",
            "body_system": "other",
            "symptom_keywords": [],
            "severity": "mild",
            "recommended_groups": [],
            "suggested_tags": [],
        },
    )
    # Đảm bảo các field tối thiểu tồn tại
    data.setdefault("main_issue", "")
    data.setdefault("body_system", "other")
    data.setdefault("symptom_keywords", [])
    data.setdefault("severity", "mild")
    data.setdefault("recommended_groups", [])
    data.setdefault("suggested_tags", [])
    return data


# =====================================================================
#   LLM PROMPTS
# =====================================================================
def llm_answer_for_combos(user_question, requested_tags, combos, covered_tags,
                          extra_instruction: str = ""):
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

Hướng dẫn bổ sung từ hệ thống (có thể để trống):
{extra_instruction}

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


def llm_answer_for_products(user_question, requested_tags, products,
                            extra_instruction: str = ""):
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

Hướng dẫn bổ sung từ hệ thống (có thể để trống):
{extra_instruction}

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


def llm_answer_with_history(latest_question: str, history: list) -> str:
    """
    Dùng khi câu hỏi là follow-up: tận dụng transcript hội thoại gần đây.
    """
    if not history:
        # fallback cho chắc
        return call_openai_responses(
            f"Khách hỏi: {latest_question}\nHãy tư vấn như trợ lý Greenway/Welllab."
        )

    lines = []
    # Lấy khoảng 10 message gần nhất để tránh prompt quá dài
    for msg in history[-10:]:
        role = msg.get("role")
        prefix = "Khách" if role == "user" else "Trợ lý"
        content = msg.get("content", "")
        lines.append(f"{prefix}: {content}")
    convo = "\n".join(lines)

    prompt = f"""
Bạn là trợ lý tư vấn sức khỏe & sản phẩm cho Greenway/Welllab.

Dưới đây là đoạn hội thoại gần đây giữa khách và trợ lý (bạn):

{convo}

Câu hỏi mới nhất của khách là: "{latest_question}"

NHIỆM VỤ:

1. Hiểu 'combo trên', 'combo đó', 'sản phẩm trên', 'sản phẩm đó', 'gói trên'...
   là đang nói về combo/sản phẩm mà bạn vừa tư vấn trước đó trong đoạn hội thoại.
2. Trả lời ngắn gọn, rõ ràng, dựa trên thông tin đã được tư vấn ở trên
   (liều dùng, thời gian uống, số viên mỗi ngày, giá, cách dùng...).
3. Nếu trong đoạn hội thoại chưa có đủ thông tin để trả lời, hãy nói rõ:
   'Trong phần tư vấn phía trên em chưa ghi rõ phần này, anh/chị cho em xin lại câu hỏi đầy đủ hơn...'
4. Cuối cùng vẫn nhắc: Sản phẩm không phải là thuốc và không có tác dụng thay thế thuốc chữa bệnh (nếu câu trả lời liên quan đến sản phẩm).

Bắt đầu trả lời bằng tiếng Việt, giọng tư vấn viên thân thiện, chuyên nghiệp.
"""
    return call_openai_responses(prompt)

# =====================================================================
#   HANDLER CHO CÁC MODE ĐẶC BIỆT
# =====================================================================
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

# =====================================================================
#   MODE DETECTION
# =====================================================================
def detect_mode(user_message: str) -> str:
    """Đoán xem user đang hỏi về combo / sản phẩm / mua hàng / kênh / kinh doanh."""
    text_norm = strip_accents(user_message)

    # Hỏi kinh doanh, chính sách, hoa hồng
    business_keywords = [
        "chinh sach",
        "hoa hong",
        "tuyen dung",
        "len cap",
        "leader",
        "doanh so",
        "muc tieu thang",
    ]
    if any(k in text_norm for k in business_keywords):
        return "business"

    # Hỏi mua hàng / thanh toán
    buy_keywords = [
        "mua",
        "dat hang",
        "thanh toan",
        "ship",
        "giao hang",
        "dat mua",
    ]
    if any(k in text_norm for k in buy_keywords):
        return "buy"

    # Hỏi kênh, fanpage, zalo
    channel_keywords = [
        "fanpage",
        "zalo",
        "kenh",
        "website",
        "trang web",
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

# =====================================================================
#   CORE CHAT LOGIC
# =====================================================================
def handle_chat(
    user_message: str,
    mode: str | None = None,
    session_id: str | None = None,
    return_meta: bool = False,
    history: list | None = None,
):
    text = (user_message or "").strip()
    history = history or []

    if not text:
        reply = "Em chưa nhận được câu hỏi của anh/chị."
        if return_meta:
            meta = {
                "intent": "",
                "mode_detected": "",
                "health_tags": [],
                "selected_combos": [],
                "selected_products": [],
            }
            return reply, meta
        return reply

    # Dùng history được truyền từ /openai-chat cho AI phân loại intent
    history_messages = history

    # 1) Gọi AI phân loại ý định
    intent_info = ai_classify_intent(text, history_messages)
    intent = intent_info.get("intent", "other")
    print("[INTENT]", intent, "|", intent_info.get("reason", ""))

    # 2) PHÂN TÍCH TRIỆU CHỨNG Ở TẦNG CHUYÊN GIA
    analysis = {}
    ai_tags = []
    ai_groups = []
    expert_extra_note = ""

    if intent in ("health_question", "combo_question", "product_question", "other"):
        analysis = ai_analyze_symptom(text, history_messages)
        ai_tags = analysis.get("suggested_tags") or []
        ai_groups = analysis.get("recommended_groups") or []

        expert_extra_note = (
            "TÓM TẮT PHÂN TÍCH CHUYÊN GIA (không cần in nguyên văn, chỉ dùng để định hướng tư vấn):\n"
            f"- Vấn đề chính: {analysis.get('main_issue', '')}\n"
            f"- Hệ cơ quan: {analysis.get('body_system', '')}\n"
            f"- Mức độ gợi ý: {analysis.get('severity', '')}\n"
            "Hãy giải thích cho người dùng theo hướng chuyên gia sức khỏe, dễ hiểu, "
            "trình bày rõ: vấn đề chính là gì, hướng hỗ trợ ưu tiên ra sao, "
            "sau đó mới đi vào combo/sản phẩm cụ thể.\n"
        )
    else:
        analysis = {
            "main_issue": "",
            "body_system": "other",
            "symptom_keywords": [],
            "severity": "mild",
            "recommended_groups": [],
            "suggested_tags": [],
        }

    # ================== ROUTING THEO INTENT TỰ NHIÊN ==================
    # 1. Chào hỏi
    if intent == "greeting":
        reply = (
            "Dạ em chào anh/chị ạ 😊\n"
            "Anh/chị cứ chia sẻ giúp em vấn đề sức khỏe hoặc nhu cầu về sản phẩm, "
            "em sẽ gợi ý combo/sản phẩm phù hợp ạ."
        )
        if return_meta:
            meta = {
                "intent": intent,
                "mode_detected": "greeting",
                "health_tags": [],
                "selected_combos": [],
                "selected_products": [],
                "ai_main_issue": analysis.get("main_issue", ""),
                "ai_body_system": analysis.get("body_system", ""),
                "ai_severity": analysis.get("severity", ""),
                "ai_groups": ai_groups,
            }
            return reply, meta
        return reply

    # 2. Nói chuyện đời thường / hỏi vu vơ
    if intent == "smalltalk":
        smalltalk_reply = call_openai_responses(
            f"""
    Bạn là trợ lý sức khỏe Greenway/Welllab.
    Người dùng đang CHỈ NÓI CHUYỆN ĐỜI THƯỜNG, không yêu cầu tư vấn cụ thể.

    Hãy trả lời thân thiện, ngắn gọn (2-4 câu), có thể đùa nhẹ, 
    sau đó khéo léo gợi ý rằng nếu họ cần tư vấn về sức khỏe / sản phẩm / combo thì bạn luôn sẵn sàng.

    Câu của người dùng: "{text}"
    """
        )
        if return_meta:
            meta = {
                "intent": intent,
                "mode_detected": "smalltalk",
                "health_tags": [],
                "selected_combos": [],
                "selected_products": [],
                "ai_main_issue": analysis.get("main_issue", ""),
                "ai_body_system": analysis.get("body_system", ""),
                "ai_severity": analysis.get("severity", ""),
                "ai_groups": ai_groups,
            }
            return smalltalk_reply, meta
        return smalltalk_reply

    # 3. Chính sách / kinh doanh
    if intent == "business_policy":
        reply = handle_escalate_to_hotline()
        if return_meta:
            meta = {
                "intent": intent,
                "mode_detected": "business",
                "health_tags": [],
                "selected_combos": [],
                "selected_products": [],
                "ai_main_issue": analysis.get("main_issue", ""),
                "ai_body_system": analysis.get("body_system", ""),
                "ai_severity": analysis.get("severity", ""),
                "ai_groups": ai_groups,
            }
            return reply, meta
        return reply

    # 4. Cách mua hàng / thanh toán
    if intent == "buy_payment":
        reply = handle_buy_and_payment_info()
        if return_meta:
            meta = {
                "intent": intent,
                "mode_detected": "buy",
                "health_tags": [],
                "selected_combos": [],
                "selected_products": [],
                "ai_main_issue": analysis.get("main_issue", ""),
                "ai_body_system": analysis.get("body_system", ""),
                "ai_severity": analysis.get("severity", ""),
                "ai_groups": ai_groups,
            }
            return reply, meta
        return reply

    # 5. Hỏi kênh liên hệ
    if intent == "channel_info":
        reply = handle_channel_navigation()
        if return_meta:
            meta = {
                "intent": intent,
                "mode_detected": "channel",
                "health_tags": [],
                "selected_combos": [],
                "selected_products": [],
                "ai_main_issue": analysis.get("main_issue", ""),
                "ai_body_system": analysis.get("body_system", ""),
                "ai_severity": analysis.get("severity", ""),
                "ai_groups": ai_groups,
            }
            return reply, meta
        return reply

    # 6. Tuning mode cho các câu sức khỏe (ưu tiên intent AI)
    if intent == "combo_question":
        mode = "combo"
    elif intent == "product_question":
        mode = "product"
    elif intent == "health_question":
        if not mode:
            mode = "auto"

    # 7. Follow-up kiểu "combo trên uống bao lâu" → dùng lịch sử
    if history and looks_like_followup(text):
        reply = llm_answer_with_history(text, history)
        if return_meta:
            meta = {
                "intent": intent,
                "mode_detected": "followup",
                "health_tags": [],
                "selected_combos": [],
                "selected_products": [],
                "ai_main_issue": analysis.get("main_issue", ""),
                "ai_body_system": analysis.get("body_system", ""),
                "ai_severity": analysis.get("severity", ""),
                "ai_groups": ai_groups,
            }
            return reply, meta
        return reply

    # 8. Mode + tags + extra_instruction cho LLM
    detected_mode = detect_mode(text) if not mode else mode.lower().strip()
    mode = detected_mode

    requested_tags = extract_tags_from_text(text)
    requested_tags = list(set((requested_tags or []) + (ai_tags or [])))
    extra_instruction = expert_extra_note  # riêng tầng chuyên gia A1

    meta = {
        "intent": intent,
        "mode_detected": mode,
        "health_tags": requested_tags,
        "selected_combos": [],
        "selected_products": [],
        "ai_main_issue": analysis.get("main_issue", ""),
        "ai_body_system": analysis.get("body_system", ""),
        "ai_severity": analysis.get("severity", ""),
        "ai_groups": ai_groups,
    }

    print("[DEBUG] handle_chat mode =", mode, "| text =", text)
    print("[DEBUG] requested_tags =", requested_tags, "| ai_groups =", ai_groups)

    # 9. Các mode đơn giản
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

    # 10. Các mode về sức khỏe: combo / product / auto
    want_combo = "combo" in strip_accents(text) or mode == "combo"
    want_product = (
        "san pham" in strip_accents(text)
        or "sản phẩm" in text.lower()
        or mode == "product"
    )

    # 10.1. Ưu tiên combo nếu người dùng hỏi combo
    if want_combo and not want_product:
        combos, covered_tags = select_combos_for_tags(requested_tags, text)
        meta["selected_combos"] = [c.get("id") for c in combos]

        if combos:
            reply = llm_answer_for_combos(text, requested_tags, combos, covered_tags, extra_instruction)
            if return_meta:
                return reply, meta
            return reply

        # Không có combo → fallback sang sản phẩm (tags + group chuyên gia)
        products = search_products_by_tags(requested_tags)
        if (not products) and ai_groups:
            products = search_products_by_groups(ai_groups)
        meta["selected_products"] = [p.get("id") for p in products]

        if products:
            reply = llm_answer_for_products(text, requested_tags, products, extra_instruction)
            if return_meta:
                return reply, meta
            return reply

    # 10.2. Người dùng hỏi sản phẩm
    if want_product and not want_combo:
        products = search_products_by_tags(requested_tags)
        if (not products) and ai_groups:
            products = search_products_by_groups(ai_groups)
        meta["selected_products"] = [p.get("id") for p in products]
        reply = llm_answer_for_products(text, requested_tags, products, extra_instruction)
        if return_meta:
            return reply, meta
        return reply

    # 10.3. AUTO: ưu tiên combo, nếu không có thì show sản phẩm
    combos, covered_tags = select_combos_for_tags(requested_tags, text)
    if combos:
        meta["selected_combos"] = [c.get("id") for c in combos]
        reply = llm_answer_for_combos(text, requested_tags, combos, covered_tags, extra_instruction)
        if return_meta:
            return reply, meta
        return reply

    products = search_products_by_tags(requested_tags)
    if (not products) and ai_groups:
        products = search_products_by_groups(ai_groups)
    if products:
        meta["selected_products"] = [p.get("id") for p in products]
        reply = llm_answer_for_products(text, requested_tags, products, extra_instruction)
        if return_meta:
            return reply, meta
        return reply

    # 11. Không match được gì
    reply = (
        "Hiện em chưa tìm thấy combo hay sản phẩm nào phù hợp trong dữ liệu cho trường hợp này. "
        f"Anh/chị có thể nói rõ hơn tình trạng sức khỏe, hoặc liên hệ hotline {HOTLINE} để tuyến trên hỗ trợ kỹ hơn ạ."
    )
    if return_meta:
        return reply, meta
    return reply

# =====================================================================
#   API /openai-chat – LOG DB + NHỚ CÂU CŨ + NGỮ CẢNH
# =====================================================================
@app.route("/openai-chat", methods=["POST"])
def openai_chat():
    start_time = time.time()
    try:
        body = request.get_json(force=True) or {}

        user_message = (body.get("message") or "").strip()
        mode = (
            (body.get("mode") or "").strip().lower()
            if isinstance(body, dict)
            else ""
        )
        session_id = body.get("session_id") or ""
        channel = body.get("channel") or "web"
        user_id = body.get("user_id") or ""

        # Nếu client không gửi session_id, tự sinh tạm (ít nhất cho web demo)
        if not session_id:
            session_id = f"web-{request.remote_addr}-{int(time.time())}"

        used_history_message = ""
        message_for_ai = user_message

        # 1) Trước khi lưu DB, kiểm tra xem có phải 'trả lời lại câu hỏi trên' không
        if looks_like_repeat_request(user_message) and session_id:
            last_q = get_last_user_message(session_id)
            if last_q:
                used_history_message = last_q
                message_for_ai = last_q
                print(
                    "[DEBUG] Repeat request detected. Using last user question:",
                    last_q,
                )

        # 2) Lưu message gốc của user vào DB
        try:
            save_message(session_id, "user", user_message)
        except Exception as e:
            print("[DB ERROR] Cannot save user message:", e)

        # 3) Lấy history sau khi đã lưu, để follow-up hiểu được cả câu vừa hỏi
        history = []
        try:
            history = get_recent_history(session_id, limit=10)
        except Exception as e:
            print("[DB ERROR] Cannot get history:", e)

        # 4) Xử lý chat – dùng message_for_ai (đã xử lý 'trả lời lại câu hỏi trên')
        reply_text, meta = handle_chat(
            message_for_ai,
            mode or None,
            session_id=session_id,
            return_meta=True,
            history=history,
        )

        # 5) Lưu bot reply vào DB
        try:
            save_message(session_id, "assistant", reply_text)
        except Exception as e:
            print("[DB ERROR] Cannot save bot reply:", e)

        latency_ms = int((time.time() - start_time) * 1000)

        # 6) Gửi log sang Google Sheets (webhook Apps Script)
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
            "ai_main_issue": meta.get("ai_main_issue", ""),
            "ai_body_system": meta.get("ai_body_system", ""),
            "ai_severity": meta.get("ai_severity", ""),
            "ai_groups": meta.get("ai_groups", []),
            "latency_ms": latency_ms,
        }

        log_conversation(log_payload)

        return jsonify({"reply": reply_text})

    except Exception as e:
        print("❌ ERROR /openai-chat:", e)
        return jsonify(
            {
                "reply": "Xin lỗi, hiện tại hệ thống đang gặp lỗi. Anh/chị vui lòng thử lại sau nhé."
            }
        ), 500

# =====================================================================
#   AUTH – ĐĂNG KÝ TVV TỪ TRANG INDEX
# =====================================================================
@app.route("/auth/register", methods=["POST"])
def auth_register():
    """
    Trang index.html gửi thông tin:
    {
      "full_name": "...",
      "phone": "...",
      "email": "...",
      "company_name": "...",
      "tvv_code": "..."   # có thể bỏ trống, server tự dùng phone làm mã
    }
    Trả về: { "tvv_code": "...", "message": "..." }
    """
    try:
        body = request.get_json(force=True) or {}
        full_name = (body.get("full_name") or "").strip()
        phone = (body.get("phone") or "").strip()
        email = (body.get("email") or "").strip()
        company_name = (body.get("company_name") or "").strip()
        tvv_code = (body.get("tvv_code") or "").strip()

        if not full_name or not phone:
            return jsonify(
                {"error": "Họ tên và số điện thoại là bắt buộc."}
            ), 400

        # Nếu không nhập mã TVV, dùng luôn số điện thoại làm mã
        if not tvv_code:
            tvv_code = phone

        upsert_tvv_user(
            tvv_code=tvv_code,
            full_name=full_name,
            phone=phone,
            email=email,
            company_name=company_name,
        )

        # Log sang Google Sheets nếu ông chủ muốn theo dõi đăng ký
        try:
            log_conversation(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "channel": "web_register",
                    "session_id": "",
                    "user_id": tvv_code,
                    "user_message": f"REGISTER: {full_name} / {phone} / {email}",
                    "message_for_ai": "",
                    "used_history_message": "",
                    "bot_reply": "",
                    "intent": "register_tvv",
                    "mode_detected": "",
                    "health_tags": [],
                    "selected_combos": [],
                    "selected_products": [],
                    "latency_ms": 0,
                }
            )
        except Exception as e:
            print("[WARN] log register error:", e)

        return jsonify(
            {
                "tvv_code": tvv_code,
                "message": "Đăng ký thành công. Leader sẽ kích hoạt gói sử dụng cho tài khoản này.",
            }
        )

    except Exception as e:
        print("❌ ERROR /auth/register:", e)
        return jsonify({"error": "Lỗi hệ thống khi đăng ký TVV."}), 500
# =====================================================================
#   ADMIN – XEM DANH SÁCH TVV (HỒ SƠ TƯ VẤN VIÊN)
# =====================================================================
def require_admin_secret():
    if not ADMIN_SECRET:
        return False, "ADMIN_SECRET chưa được cấu hình trên server."
    header_secret = request.headers.get("X-Admin-Secret") or ""
    if header_secret != ADMIN_SECRET:
        return False, "Sai ADMIN_SECRET."
    return True, ""


@app.route("/admin/users", methods=["GET"])
def admin_list_users():
    ok, msg = require_admin_secret()
    if not ok:
        status = 500 if "chưa được cấu hình" in msg else 401
        return jsonify({"error": msg}), status

    q = (request.args.get("q") or "").strip()
    try:
        limit = int(request.args.get("limit") or "200")
    except ValueError:
        limit = 200

    try:
        items = list_tvv_users(q=q, limit=limit)
        return jsonify({"items": items})
    except Exception as e:
        print("❌ ERROR /admin/users:", e)
        return jsonify({"error": "Không lấy được danh sách TVV."}), 500


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

# =====================================================================
#   HEALTHCHECK
# =====================================================================
@app.route("/", methods=["GET"])
def home():
    return "🔥 Greenway / Welllab Chatbot Gateway đang chạy ngon lành!", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
