import os
import json
import unicodedata
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# ============== OpenAI ==============
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

HOTLINE = os.getenv("HOTLINE", "09xx.xxx.xxx")
FANPAGE_URL = os.getenv("FANPAGE_URL", "https://facebook.com/ten-fanpage")
ZALO_OA_URL = os.getenv("ZALO_OA_URL", "https://zalo.me/ten-oa")
WEBSITE_URL = os.getenv("WEBSITE_URL", "https://greenwayglobal.vn")

app = Flask(__name__)

client = None
if OPENAI_API_KEY and OpenAI is not None:
    client = OpenAI(api_key=OPENAI_API_KEY)


# ============== Load dữ liệu ==============

def load_json_file(path, default=None):
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Cannot read {path}: {e}")
        return default


PRODUCTS_DATA = load_json_file("products.json", {"products": []})
COMBOS_DATA = load_json_file("combos.json", {"combos": []})
HEALTH_TAGS_CONFIG = load_json_file("health_tags_config.json", {})
COMBOS_META = load_json_file("combos_meta.json", {})
MULTI_ISSUE_RULES = load_json_file("multi_issue_rules.json", {"rules": []})

PRODUCTS = PRODUCTS_DATA.get("products", [])
COMBOS = COMBOS_DATA.get("combos", [])


# ============== Tiền xử lý & tag ==============

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


# ============== Scoring combo theo tags ==============

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


# ============== Tìm sản phẩm theo tags ==============

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


# ============== Gọi OpenAI để viết câu trả lời ==============

def llm_answer_for_combos(user_question, requested_tags, combos, covered_tags):
    if not combos:
        return (
            "Hiện em chưa tìm thấy combo phù hợp trong dữ liệu cho trường hợp này. "
            f"Anh/chị vui lòng liên hệ hotline {HOTLINE} để tuyến trên tư vấn chi tiết hơn ạ."
        )

    if not client or not OPENAI_API_KEY:
        return fallback_text_combos(user_question, combos, requested_tags, covered_tags)

    try:
        combos_json = json.dumps(combos, ensure_ascii=False, indent=2)
        tags_text = ", ".join(requested_tags)

        system_prompt = (
            "Bạn là trợ lý tư vấn cho công ty thực phẩm chức năng. "
            "Bạn chỉ được dùng đúng dữ liệu combo và sản phẩm ở dạng JSON, "
            "không được bịa thêm sản phẩm hay công dụng. "
            "Luôn trình bày dễ hiểu, chia thành các mục rõ ràng, ưu tiên dạng gạch đầu dòng. "
            "Luôn nhắc: 'Sản phẩm không phải là thuốc và không có tác dụng thay thế thuốc chữa bệnh.'"
        )

        user_prompt = f"""
Khách hỏi: "{user_question}"

Các vấn đề sức khỏe/mục tiêu hệ thống trích xuất được (tags): {tags_text}
Các combo được chọn (dữ liệu JSON):

{combos_json}

Yêu cầu:
1. Tóm tắt 1–3 dòng: khách đang gặp những vấn đề/nhu cầu nào và hướng xử lý tổng quan.
2. Với từng combo:
   - Nêu rõ combo này đang hỗ trợ các vấn đề nào trong những vấn đề khách nêu.
   - Liệt kê các sản phẩm trong combo (tên, lợi ích chính, giá, cách dùng tóm tắt, link).
3. Nếu vẫn còn vấn đề nhạy cảm hoặc quá nặng, hãy khuyến nghị khách tái khám, làm xét nghiệm, và liên hệ hotline để được tư vấn kỹ hơn.
4. Kết thúc bằng lưu ý: sản phẩm không phải là thuốc...
"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[ERROR] OpenAI combo answer error: {e}")
        return fallback_text_combos(user_question, combos, requested_tags, covered_tags)


def llm_answer_for_products(user_question, requested_tags, products):
    if not products:
        return (
            "Hiện em chưa tìm thấy sản phẩm phù hợp trong dữ liệu cho trường hợp này. "
            f"Anh/chị vui lòng liên hệ hotline {HOTLINE} để được tư vấn rõ hơn ạ."
        )

    if not client or not OPENAI_API_KEY:
        return fallback_text_products(user_question, requested_tags, products)

    try:
        products_json = json.dumps(products, ensure_ascii=False, indent=2)
        tags_text = ", ".join(requested_tags)

        system_prompt = (
            "Bạn là trợ lý tư vấn cho công ty thực phẩm chức năng. "
            "Bạn chỉ được dùng đúng dữ liệu sản phẩm ở dạng JSON, "
            "không được bịa thêm sản phẩm hay công dụng. "
            "Trình bày câu trả lời ngắn gọn, rõ ràng, dễ hiểu cho tư vấn viên."
        )

        user_prompt = f"""
Khách hỏi: "{user_question}"

Các vấn đề sức khỏe/mục tiêu hệ thống trích xuất được (tags): {tags_text}
Các sản phẩm được chọn (dữ liệu JSON):

{products_json}

Yêu cầu:
1. Mở đầu 1–2 câu: đây là các sản phẩm hỗ trợ cho vấn đề mà khách đang gặp phải.
2. Với từng sản phẩm, trình bày:
   - Tên sản phẩm
   - Nhóm/vấn đề chính mà sản phẩm hỗ trợ
   - Lợi ích chính (dựa trên benefits_text nếu có, hoặc mô tả)
   - Cách dùng (usage_text/dose_text nếu có)
   - Giá (price_text)
   - Link sản phẩm (product_url)
3. Nhắc: sản phẩm không phải là thuốc...
"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[ERROR] OpenAI products answer error: {e}")
        return fallback_text_products(user_question, requested_tags, products)


# ============== Fallback nếu không dùng được OpenAI ==============

def fallback_text_combos(user_question, combos, requested_tags, covered_tags):
    lines = []
    if requested_tags:
        lines.append(
            "Em ghi nhận các vấn đề/mục tiêu chính của anh/chị là: "
            + ", ".join(requested_tags)
        )
    lines.append("Dưới đây là một số combo phù hợp từ dữ liệu hiện có:")

    for combo in combos:
        lines.append(f"\n👉 {combo.get('name', 'Combo chưa đặt tên')}")
        if combo.get("header_text"):
            lines.append(f"- Mục tiêu chính: {combo['header_text']}")
        if combo.get("duration_text"):
            lines.append(f"- Thời gian dùng khuyến nghị: {combo['duration_text']}")
        tags = combo.get("health_tags", [])
        if tags:
            lines.append(f"- Nhóm vấn đề hỗ trợ: {', '.join(tags)}")

        products = combo.get("products", [])
        if products:
            lines.append("- Các sản phẩm trong combo:")
            for p in products:
                line_p = f"   • {p.get('name', 'Sản phẩm')}"
                price_text = p.get("price_text")
                if price_text:
                    line_p += f" – {price_text}"
                lines.append(line_p)
                dose_text = p.get("dose_text")
                if dose_text:
                    lines.append(f"     Cách dùng: {dose_text}")
                url = p.get("product_url")
                if url:
                    lines.append(f"     Link: {url}")

    lines.append(
        "\nLưu ý: Sản phẩm không phải là thuốc và không có tác dụng thay thế thuốc chữa bệnh. "
        "Nếu anh/chị có bệnh lý nền hoặc đang dùng thuốc, nên hỏi ý kiến bác sĩ và tuyến trên."
    )
    return "\n".join(lines)


def fallback_text_products(user_question, requested_tags, products):
    lines = []
    if requested_tags:
        lines.append(
            "Các vấn đề/mục tiêu chính hệ thống nhận diện được: "
            + ", ".join(requested_tags)
        )
    lines.append("Một số sản phẩm hỗ trợ trong dữ liệu hiện có:")

    for p in products:
        lines.append(f"\n👉 {p.get('name', 'Sản phẩm')}")
        group = p.get("group")
        if group:
            lines.append(f"- Nhóm vấn đề chính: {group}")
        price_text = p.get("price_text")
        if price_text:
            lines.append(f"- Giá tham khảo: {price_text}")
        ingredients_text = p.get("ingredients_text")
        if ingredients_text:
            lines.append(f"- Thành phần chính: {ingredients_text}")
        benefits_text = p.get("benefits_text")
        if benefits_text:
            lines.append(f"- Lợi ích: {benefits_text}")
        usage_text = p.get("usage_text")
        if usage_text:
            lines.append(f"- Cách dùng: {usage_text}")
        url = p.get("product_url")
        if url:
            lines.append(f"- Link sản phẩm: {url}")

    lines.append(
        "\nLưu ý: Sản phẩm không phải là thuốc và không có tác dụng thay thế thuốc chữa bệnh."
    )
    return "\n".join(lines)


# ============== Handlers cho các TAG trong Dialogflow CX ==============

def handle_get_combo_by_condition(params):
    user_question = (
        params.get("user_text")
        or params.get("condition")
        or params.get("health_issue")
        or ""
    )

    requested_tags = params.get("tags") or []
    if isinstance(requested_tags, str):
        requested_tags = [requested_tags]

    extracted = extract_tags_from_text(user_question)
    requested_tags = list(set(requested_tags) | set(extracted))

    combos, covered_tags = select_combos_for_tags(requested_tags, user_question)
    reply = llm_answer_for_combos(user_question, requested_tags, combos, covered_tags)
    return reply


def handle_get_products_by_condition(params):
    user_question = (
        params.get("user_text")
        or params.get("condition")
        or params.get("health_issue")
        or ""
    )

    requested_tags = params.get("tags") or []
    if isinstance(requested_tags, str):
        requested_tags = [requested_tags]

    extracted = extract_tags_from_text(user_question)
    requested_tags = list(set(requested_tags) | set(extracted))

    products = search_products_by_tags(requested_tags)
    reply = llm_answer_for_products(user_question, requested_tags, products)
    return reply


def handle_buy_and_payment_info(params):
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


def handle_escalate_to_hotline(params):
    return (
        "Câu hỏi này thuộc nhóm chính sách/kế hoạch kinh doanh chuyên sâu nên cần tuyến trên hỗ trợ trực tiếp ạ.\n\n"
        "Anh/chị vui lòng để lại:\n"
        "- Họ tên\n"
        "- Số điện thoại\n"
        "- Mã TVV (nếu có)\n\n"
        f"Hoặc gọi thẳng hotline: {HOTLINE}\n"
        "Tuyến trên sẽ liên hệ và tư vấn chi tiết cho anh/chị sớm nhất có thể."
    )


def handle_channel_navigation(params):
    return (
        "Anh/chị có thể theo dõi thông tin, chương trình ưu đãi và kiến thức sức khỏe tại các kênh sau:\n\n"
        f"📘 Fanpage: {FANPAGE_URL}\n"
        f"💬 Zalo OA: {ZALO_OA_URL}\n"
        f"🌐 Website: {WEBSITE_URL}\n\n"
        "Nếu cần hỗ trợ gấp, anh/chị gọi trực tiếp hotline giúp em nhé."
    )


# ============== Webhook cho Dialogflow CX ==============

@app.route("/dfcx-webhook", methods=["POST"])
def dfcx_webhook():
    body = request.get_json(force=True, silent=True) or {}
    print("[DEBUG] Webhook request:", json.dumps(body, ensure_ascii=False))

    tag = body.get("fulfillmentInfo", {}).get("tag", "")
    session_info = body.get("sessionInfo", {})
    params = session_info.get("parameters", {}) or {}

    reply_text = "Em chưa hiểu rõ yêu cầu, anh/chị nói rõ hơn giúp em với ạ."

    if tag == "GET_COMBO_BY_CONDITION":
        reply_text = handle_get_combo_by_condition(params)
    elif tag == "GET_PRODUCTS_BY_CONDITION":
        reply_text = handle_get_products_by_condition(params)
    elif tag == "BUY_AND_PAYMENT_INFO":
        reply_text = handle_buy_and_payment_info(params)
    elif tag == "ESCALATE_TO_HOTLINE":
        reply_text = handle_escalate_to_hotline(params)
    elif tag == "CHANNEL_NAVIGATION_INFO":
        reply_text = handle_channel_navigation(params)

    response = {
        "fulfillment_response": {
            "messages": [
                {
                    "text": {"text": [reply_text]}
                }
            ]
        },
        "sessionInfo": {
            "parameters": params
        }
    }

    return jsonify(response)


@app.route("/", methods=["GET"])
def health_check():
    return "DFCX Webhook is running", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
