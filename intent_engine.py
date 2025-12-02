# intent_engine.py

import json
import unicodedata
from typing import List, Dict, Any

import requests

from config import (
    client,
    HOTLINE,
    FANPAGE_URL,
    ZALO_OA_URL,
    WEBSITE_URL,
    LOG_WEBHOOK_URL,
)
from health_engine import (
    extract_tags_from_text,
    select_combos_for_tags,
    search_products_by_tags,
)
from db_utils import get_last_user_message_from_history

# =====================================================================
#   TEXT & MODE UTILS
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


def is_meta_for_customer(text_norm: str) -> bool:
    """
    Nhận diện câu kiểu: anh/chị đang hỏi giùm khách.
    """
    meta_kw = [
        "anh hoi cho khach",
        "em hoi cho khach",
        "hoi cho khach",
        "hoi giup khach",
        "hoi giup ban",
        "tu van vien",
        "tvv",
    ]
    return any(k in text_norm for k in meta_kw)


def is_duration_followup(text_norm: str) -> bool:
    """
    Nhận diện câu hỏi về thời gian dùng / liệu trình (có thể không nhắc 'combo trên').
    """
    duration_kw = [
        "bao lau",
        "bao lâu",
        "may ngay",
        "mấy ngày",
        "may thang",
        "mấy tháng",
        "dung trong bao",
        "uống trong bao",
        "lieu trinh",
        "liệu trình",
        "thoi gian dung",
        "thời gian dùng",
    ]
    return any(strip_accents(k) in text_norm for k in duration_kw)


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
#   OPENAI RESPONSES & INTENT CLASSIFIER
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
                return json.loads(text[start : end + 1])
        except Exception:
            return default
    return default


def ai_classify_intent(
    user_message: str, history_messages: List[Dict[str, Any]] | None = None
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
    - channel_info: hỏi link fanpage, zalo, website, kênh liên hệ
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


# =====================================================================
#   LLM ANSWERS (COMBO / PRODUCT / HISTORY)
# =====================================================================

def llm_answer_for_combos(
    user_question: str,
    requested_tags: List[str],
    combos: List[dict],
    covered_tags: List[str],
    extra_instruction: str = "",
) -> str:
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

Hướng dẫn bổ sung từ hệ thống (nếu có, có thể để trống):
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


def llm_answer_for_products(
    user_question: str,
    requested_tags: List[str],
    products: List[dict],
    extra_instruction: str = "",
) -> str:
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

Hướng dẫn bổ sung từ hệ thống (nếu có, có thể để trống):
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
3. Cuối cùng nhắc: "Sản phẩm không phải là thuốc và không có tác dụng thay thế thuốc chữa bệnh."
4. Viết ngắn gọn, rõ ràng, dễ dùng cho tư vấn viên khi chát với khách.
"""
    return call_openai_responses(prompt)


def llm_answer_with_history(latest_question: str, history: List[Dict[str, Any]]) -> str:
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
#   HANDLER MẶC ĐỊNH & LOG
# =====================================================================

def handle_buy_and_payment_info() -> str:
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


def handle_escalate_to_hotline() -> str:
    return (
        "Câu hỏi này thuộc nhóm chính sách/kế hoạch kinh doanh chuyên sâu nên cần tuyến trên hỗ trợ trực tiếp ạ.\n\n"
        "Anh/chị vui lòng để lại:\n"
        "- Họ tên\n"
        "- Số điện thoại\n"
        "- Mã TVV (nếu có)\n\n"
        f"Hoặc gọi thẳng hotline: {HOTLINE}\n"
        "Tuyến trên sẽ liên hệ và tư vấn chi tiết cho anh/chị sớm nhất có thể."
    )


def handle_channel_navigation() -> str:
    return (
        "Anh/chị có thể theo dõi thông tin, chương trình ưu đãi và kiến thức sức khỏe tại các kênh sau:\n\n"
        f"📘 Fanpage: {FANPAGE_URL}\n"
        f"💬 Zalo OA: {ZALO_OA_URL}\n"
        f"🌐 Website: {WEBSITE_URL}\n\n"
        "Nếu cần hỗ trợ gấp, anh/chị gọi trực tiếp hotline giúp em nhé."
    )


def log_conversation(payload: dict):
    if not LOG_WEBHOOK_URL:
        return
    try:
        requests.post(LOG_WEBHOOK_URL, json=payload, timeout=2)
    except Exception as e:
        print("[WARN] Log error:", e)


# =====================================================================
#   CORE handle_chat
# =====================================================================

def handle_chat(
    user_message: str,
    mode: str | None = None,
    session_id: str | None = None,
    return_meta: bool = False,
    history: List[Dict[str, Any]] | None = None,
):
    """
    Core xử lý 1 lượt chat:
    - Kết hợp AI intent + rule-based mode
    - Hỗ trợ follow-up & duration follow-up
    - Ưu tiên combo -> sản phẩm -> fallback
    """
    text = (user_message or "").strip()
    history = history or []

    if not text:
        reply = "Em chưa nhận được câu hỏi của anh/chị."
        meta = {
            "intent": "",
            "mode_detected": "",
            "health_tags": [],
            "selected_combos": [],
            "selected_products": [],
        }
        return (reply, meta) if return_meta else reply

    text_norm = strip_accents(text)

    # Ưu tiên rule cho case "hỏi cho khách"
    if is_meta_for_customer(text_norm):
        reply = (
            "À, em hiểu là anh/chị đang hỏi để tư vấn cho khách ạ 👌\n"
            "Anh/chị cho em biết thêm: tuổi, giới tính và vấn đề sức khỏe chính của khách, "
            "em sẽ gợi ý combo/sản phẩm cho anh/chị dễ tư vấn nhé."
        )
        meta = {
            "intent": "meta_for_customer",
            "mode_detected": "meta_for_customer",
            "health_tags": [],
            "selected_combos": [],
            "selected_products": [],
        }
        return (reply, meta) if return_meta else reply

    # Dùng history được truyền từ /openai-chat cho AI phân loại intent
    history_messages = history

    # Gọi AI phân loại ý định
    intent_info = ai_classify_intent(text, history_messages)
    intent = intent_info.get("intent", "other")
    print("[INTENT]", intent, "|", intent_info.get("reason", ""))

    # ================== ROUTING THEO INTENT TỰ NHIÊN ==================
    # 1. Chào hỏi
    if intent == "greeting":
        reply = (
            "Dạ em chào anh/chị ạ 😊\n"
            "Anh/chị cứ chia sẻ giúp em vấn đề sức khỏe hoặc nhu cầu về sản phẩm, "
            "em sẽ gợi ý combo/sản phẩm phù hợp ạ."
        )
        meta = {
            "intent": intent,
            "mode_detected": "greeting",
            "health_tags": [],
            "selected_combos": [],
            "selected_products": [],
        }
        return (reply, meta) if return_meta else reply

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
        meta = {
            "intent": intent,
            "mode_detected": "smalltalk",
            "health_tags": [],
            "selected_combos": [],
            "selected_products": [],
        }
        return (smalltalk_reply, meta) if return_meta else smalltalk_reply

    # 3. Chính sách / kinh doanh
    if intent == "business_policy":
        reply = handle_escalate_to_hotline()
        meta = {
            "intent": intent,
            "mode_detected": "business",
            "health_tags": [],
            "selected_combos": [],
            "selected_products": [],
        }
        return (reply, meta) if return_meta else reply

    # 4. Cách mua hàng / thanh toán
    if intent == "buy_payment":
        reply = handle_buy_and_payment_info()
        meta = {
            "intent": intent,
            "mode_detected": "buy",
            "health_tags": [],
            "selected_combos": [],
            "selected_products": [],
        }
        return (reply, meta) if return_meta else reply

    # 5. Hỏi kênh liên hệ
    if intent == "channel_info":
        reply = handle_channel_navigation()
        meta = {
            "intent": intent,
            "mode_detected": "channel",
            "health_tags": [],
            "selected_combos": [],
            "selected_products": [],
        }
        return (reply, meta) if return_meta else reply

    # 6. Tuning mode cho các câu sức khỏe
    #    (giữ nguyên pipeline cũ, nhưng ưu tiên intent AI)
    if intent == "combo_question":
        mode = "combo"
    elif intent == "product_question":
        mode = "product"
    elif intent == "health_question":
        # để auto cho pipeline combo/product tự chọn
        if not mode:
            mode = "auto"

    # 7. Nếu là câu follow-up kiểu "combo trên / sản phẩm đó..."
    if history and looks_like_followup(text):
        reply = llm_answer_with_history(text, history)
        meta = {
            "intent": intent,
            "mode_detected": "followup",
            "health_tags": [],
            "selected_combos": [],
            "selected_products": [],
        }
        return (reply, meta) if return_meta else reply

    # 8. Rule cho duration follow-up (không nhắc combo trên nhưng hỏi bao lâu/ liệu trình)
    extra_instruction = ""
    if history and is_duration_followup(text_norm):
        base_question = get_last_user_message_from_history(history)
        if base_question:
            # Lấy tags từ câu hỏi sức khỏe trước đó
            requested_tags_base = extract_tags_from_text(base_question)
            # Dùng cả câu hỏi trước + câu hỏi hiện tại
            text = (
                f"Câu hỏi trước của khách/tư vấn viên: \"{base_question}\".\n"
                f"Hỏi tiếp: \"{user_message}\"."
            )
            text_norm = strip_accents(text)
            extra_instruction = (
                "Người dùng đang hỏi tiếp về THỜI GIAN DÙNG / LIỆU TRÌNH. "
                "Trong câu trả lời, hãy nhấn mạnh rõ:\n"
                "- Nên dùng trong bao lâu thì phù hợp (theo dữ liệu hiện có).\n"
                "- Nếu dữ liệu không ghi rõ, đưa ra gợi ý chung chung nhưng vẫn an toàn.\n"
            )
            # override tags theo câu trước
            requested_tags = requested_tags_base
        else:
            requested_tags = extract_tags_from_text(text)
    else:
        requested_tags = extract_tags_from_text(text)

    detected_mode = detect_mode(text) if not mode else mode.lower().strip()
    mode = detected_mode

    meta = {
        "intent": intent,
        "mode_detected": mode,
        "health_tags": requested_tags,
        "selected_combos": [],
        "selected_products": [],
    }

    print("[DEBUG] handle_chat mode =", mode, "| text =", text)

    # Các mode đơn giản
    if mode == "buy":
        reply = handle_buy_and_payment_info()
        return (reply, meta) if return_meta else reply

    if mode == "channel":
        reply = handle_channel_navigation()
        return (reply, meta) if return_meta else reply

    if mode == "business":
        reply = handle_escalate_to_hotline()
        return (reply, meta) if return_meta else reply

    # Các mode về sức khỏe: combo / product / auto
    want_combo = "combo" in strip_accents(text) or mode == "combo"
    want_product = (
        "san pham" in strip_accents(text)
        or "sản phẩm" in text.lower()
        or mode == "product"
    )

    if want_combo and not want_product:
        combos, covered_tags = select_combos_for_tags(requested_tags, text)
        meta["selected_combos"] = [c.get("id") for c in combos]
        reply = llm_answer_for_combos(
            text, requested_tags, combos, covered_tags, extra_instruction
        )
        return (reply, meta) if return_meta else reply

    if want_product and not want_combo:
        products = search_products_by_tags(requested_tags)
        meta["selected_products"] = [p.get("id") for p in products]
        reply = llm_answer_for_products(
            text, requested_tags, products, extra_instruction
        )
        return (reply, meta) if return_meta else reply

    # AUTO: ưu tiên combo, nếu không có thì show sản phẩm
    combos, covered_tags = select_combos_for_tags(requested_tags, text)
    if combos:
        meta["selected_combos"] = [c.get("id") for c in combos]
        reply = llm_answer_for_combos(
            text, requested_tags, combos, covered_tags, extra_instruction
        )
        return (reply, meta) if return_meta else reply

    products = search_products_by_tags(requested_tags)
    if products:
        meta["selected_products"] = [p.get("id") for p in products]
        reply = llm_answer_for_products(
            text, requested_tags, products, extra_instruction
        )
        return (reply, meta) if return_meta else reply

    # Không match gì
    reply = (
        "Hiện em chưa tìm thấy combo hay sản phẩm nào phù hợp trong dữ liệu cho trường hợp này. "
        f"Anh/chị có thể nói rõ hơn tình trạng sức khỏe, hoặc liên hệ hotline {HOTLINE} để tuyến trên hỗ trợ kỹ hơn ạ."
    )
    return (reply, meta) if return_meta else reply
