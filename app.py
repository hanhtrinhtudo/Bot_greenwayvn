from __future__ import annotations
import os
import json
import time
import unicodedata
import traceback
import random
from datetime import datetime, timedelta

import psycopg2
from psycopg2.extras import DictCursor

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

# ===== Dialogflow CX (tuỳ chọn) =====
try:
    from google.oauth2 import service_account
    from google.auth.transport.requests import AuthorizedSession
except ImportError:
    service_account = None
    AuthorizedSession = None


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
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")        # dùng chung cho /admin/*

DFCX_ENABLED = os.getenv("DFCX_ENABLED", "0") == "1"
DFCX_PROJECT_ID = os.getenv("DFCX_PROJECT_ID", "")
DFCX_LOCATION   = os.getenv("DFCX_LOCATION", "global")
DFCX_AGENT_ID   = os.getenv("DFCX_AGENT_ID", "")
DFCX_LANGUAGE_CODE = os.getenv("DFCX_LANGUAGE_CODE", "vi")
DFCX_SERVICE_ACCOUNT_JSON = os.getenv("DFCX_SERVICE_ACCOUNT_JSON", "")  # JSON string

# ===== Init App =====
app = Flask(__name__)
CORS(app)  # Cho phép web / Conversational Agents gọi API không bị CORS

client = OpenAI(api_key=OPENAI_API_KEY)

def send_sms_viettel(phone: str, message: str):
    """
    Gửi SMS OTP qua Viettel (API mẫu).
    Khi tích hợp thật, thay URL + auth theo tài liệu Viettel cung cấp.
    """
    try:
        url = os.getenv("VIETTEL_SMS_URL", "")
        username = os.getenv("VIETTEL_USERNAME", "")
        password = os.getenv("VIETTEL_PASSWORD", "")

        payload = {
            "from": "GWGLOBAL",       # brandname nếu có
            "to": phone,
            "text": message,
        }

        res = requests.post(
            url,
            json=payload,
            auth=(username, password),
            timeout=5
        )
        print("[SMS RESPONSE]", res.status_code, res.text)
        return True
    except Exception as e:
        print("❌ SMS ERROR:", e)
        return False


# =====================================================================
#   DB – KẾT NỐI
# =====================================================================
def get_db_conn():
    """
    Mở connection tới PostgreSQL (Render cung cấp DATABASE_URL).
    Có bọc try/except ở ngoài các hàm sử dụng.
    """
    if not DATABASE_URL:
        raise Exception("Thiếu biến môi trường DATABASE_URL")
    return psycopg2.connect(DATABASE_URL, cursor_factory=DictCursor)

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

def get_tenant_id_by_tvv_code(tvv_code: str) -> Optional[int]:
    """
    Trả về tenant_id theo tvv_code, hoặc None nếu không tìm thấy.
    """
    if not tvv_code:
        return None

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT tenant_id FROM tvv_users WHERE tvv_code = %s LIMIT 1",
                (tvv_code,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return row["tenant_id"]
    finally:
        conn.close()

# =====================================================================
#   HELPER: LẤY USER + TENANT TỪ SESSION TOKEN
# =====================================================================
def get_user_and_tenant_from_session(token: str):
    """
    Session token dạng: token-<phone>-<timestamp>
    Trả về (user_dict, tenant_dict) hoặc (None, None) nếu không hợp lệ.
    """
    if not token or not token.startswith("token-"):
        return None, None

    try:
        parts = token.split("-")
        if len(parts) < 3:
            return None, None
        phone = parts[1]
    except Exception:
        return None, None

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            # Lấy user
            cur.execute("SELECT * FROM tvv_users WHERE phone = %s LIMIT 1", (phone,))
            u = cur.fetchone()
            if not u:
                return None, None

            user = dict(u)
            tenant_id = user.get("tenant_id")
            if not tenant_id:
                return user, None

            # Lấy tenant
            cur.execute("SELECT * FROM tenants WHERE id = %s LIMIT 1", (tenant_id,))
            t = cur.fetchone()
            tenant = dict(t) if t else None

        return user, tenant
    finally:
        conn.close()

# ===== BILLING CONFIG =====
BILLING_ENABLED = os.getenv("BILLING_ENABLED", "1") == "1"
SMART_COST_PER_MESSAGE_CENTS = int(os.getenv("SMART_COST_PER_MESSAGE_CENTS", "5"))
LOW_BALANCE_THRESHOLD_CENTS = int(os.getenv("LOW_BALANCE_THRESHOLD_CENTS", "100"))

LOW_BALANCE_NOTICE_TEXT = (
    "🔔 Tài khoản của anh/chị sắp hết số dư dùng cho trợ lý thông minh.\n\n"
    "Hiện tại hệ thống vẫn đang hoạt động ở chế độ thông minh, nhưng số dư còn khá ít. "
    "Anh/chị nên nạp thêm để tránh trường hợp đang tư vấn cho khách mà Bot bị chuyển về chế độ cơ bản.\n\n"
    "Nếu cần hướng dẫn nạp tiền, anh/chị chỉ cần nhắn: \"Hướng dẫn em cách nạp tiền\" là được ạ."
)

NO_BALANCE_NOTICE_TEXT = (
    "⛔ Tài khoản của anh/chị đã hết số dư dùng cho trợ lý thông minh.\n\n"
    "Từ bây giờ, Bot sẽ tự động trả lời ở chế độ cơ bản: vẫn hỗ trợ được những nội dung đã được cài sẵn, "
    "nhưng sẽ tạm tắt phần phân tích sâu để không phát sinh thêm chi phí.\n\n"
    "Khi anh/chị nạp thêm tiền, trợ lý thông minh sẽ tự động hoạt động trở lại mà không cần cài đặt gì thêm."
)


def get_tenant_balance_cents(tenant_id: int) -> int:
    if not tenant_id:
        return 0
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT balance_cents FROM tenant_billing WHERE tenant_id = %s",
                (tenant_id,),
            )
            row = cur.fetchone()
            if not row:
                # tạo mới record nếu chưa có
                cur.execute(
                    """
                    INSERT INTO tenant_billing (tenant_id, balance_cents, updated_at)
                    VALUES (%s, 0, NOW())
                    """,
                    (tenant_id,),
                )
                conn.commit()
                return 0
            return row["balance_cents"] or 0
    finally:
        conn.close()


def charge_tenant_for_smart_request(tenant_id: int, messages: int = 1) -> dict:
    """
    Trừ tiền cho 1 lần dùng "trợ lý thông minh".
    """
    if not tenant_id or not BILLING_ENABLED:
        return {
            "old_balance_cents": 0,
            "new_balance_cents": 0,
            "became_zero": False,
            "is_low": False,
        }

    cost_cents = SMART_COST_PER_MESSAGE_CENTS * max(messages, 1)

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            # Lấy số dư hiện tại, lock hàng
            cur.execute(
                """
                SELECT balance_cents
                FROM tenant_billing
                WHERE tenant_id = %s
                FOR UPDATE
                """,
                (tenant_id,),
            )
            row = cur.fetchone()
            if not row:
                old_balance = 0
                cur.execute(
                    """
                    INSERT INTO tenant_billing (tenant_id, balance_cents, updated_at)
                    VALUES (%s, 0, NOW())
                    """,
                    (tenant_id,),
                )
            else:
                old_balance = row["balance_cents"] or 0

            new_balance = old_balance - cost_cents
            if new_balance < 0:
                new_balance = 0

            cur.execute(
                """
                UPDATE tenant_billing
                SET balance_cents = %s, updated_at = NOW()
                WHERE tenant_id = %s
                """,
                (new_balance, tenant_id),
            )

            cur.execute(
                """
                INSERT INTO billing_usage (tenant_id, usage_date, messages, cost_cents, created_at)
                VALUES (%s, CURRENT_DATE, %s, %s, NOW())
                """,
                (tenant_id, messages, cost_cents),
            )
        conn.commit()
    finally:
        conn.close()

    return {
        "old_balance_cents": old_balance,
        "new_balance_cents": new_balance,
        "became_zero": old_balance > 0 and new_balance == 0,
        "is_low": new_balance > 0 and new_balance <= LOW_BALANCE_THRESHOLD_CENTS,
    }

# =====================================================================
#   HANDLER: HƯỚNG DẪN NẠP TIỀN
# =====================================================================
def handle_topup_instruction(brand: BrandSettings | None = None):
    b = brand or BrandSettings()
    return (
        "Để nạp tiền vào tài khoản sử dụng trợ lý thông minh của "
        f"{b.brand_name}, anh/chị có thể làm như sau:\n\n"
        "1️⃣ Liên hệ tuyến trên hoặc quản trị viên để được cấp thông tin thanh toán (số tài khoản / ví điện tử).\n"
        "2️⃣ Chuyển khoản với nội dung: họ tên + số điện thoại hoặc mã tài khoản (TVV code).\n"
        "3️⃣ Sau khi nhận được tiền, quản trị viên sẽ nạp số dư tương ứng vào hệ thống. "
        "Anh/chị có thể vào trang \"Tài khoản\" để kiểm tra số dư hiện tại.\n\n"
        "💡 Lưu ý:\n"
        "- Số dư càng cao thì anh/chị sử dụng trợ lý thông minh càng lâu "
        "(hệ thống chỉ trừ tiền khi dùng AI phân tích sâu).\n"
        "- Khi số dư về 0, Bot tự chuyển sang chế độ cơ bản miễn phí, không phát sinh thêm chi phí.\n\n"
        f"Nếu anh/chị cần thông tin thanh toán cụ thể, vui lòng liên hệ trực tiếp hotline {b.hotline} để được hỗ trợ chi tiết ạ."
    )

def looks_like_topup_help(text: str) -> bool:
    t = strip_accents(text or "")
    t = " ".join(t.split())
    patterns = [
        "huong dan nap tien",
        "nap tien nhu the nao",
        "nap them tien",
        "nap them so du",
        "nap them tien vao tai khoan",
        "cach nap tien",
        "nap tai khoan",
    ]
    return any(p in t for p in patterns)

# =====================================================================
#   HELPER: TỔNG HỢP USAGE (THEO NGÀY) CHO 1 TENANT
# =====================================================================
def get_tenant_usage_timeseries(tenant_id: int, days: int = 30):
    """
    Trả về danh sách usage theo ngày trong N ngày gần đây:
    [
      { "date": "2025-12-01", "messages": 10, "cost_cents": 50 },
      ...
    ]
    """
    if not tenant_id:
        return []

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  usage_date,
                  SUM(messages) AS messages,
                  SUM(cost_cents) AS cost_cents
                FROM billing_usage
                WHERE tenant_id = %s
                  AND usage_date >= CURRENT_DATE - %s * INTERVAL '1 day'
                GROUP BY usage_date
                ORDER BY usage_date
                """,
                (tenant_id, days),
            )
            rows = cur.fetchall()

        result = []
        for r in rows:
            result.append(
                {
                    "date": r["usage_date"].isoformat(),
                    "messages": int(r["messages"] or 0),
                    "cost_cents": int(r["cost_cents"] or 0),
                }
            )
        return result
    finally:
        conn.close()


# =====================================================================
#   DB HELPER – LỊCH SỬ HỘI THOẠI
# =====================================================================
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
def get_brand_settings_for_tenant(tenant_id: int | None) -> BrandSettings:
    """
    Lấy thông tin brand cho 1 tenant từ bảng tenant_settings.
    Nếu chưa có bản ghi thì trả về default (ENV).
    """
    if not tenant_id:
        return BrandSettings()

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM tenant_settings WHERE tenant_id = %s LIMIT 1",
                (tenant_id,),
            )
            row = cur.fetchone()
        return BrandSettings.from_db(dict(row) if row else None)
    finally:
        conn.close()

# =====================================================================
#   DIALOGFLOW CX – DETECT INTENT
# =====================================================================
def get_dfcx_authed_session():
    """
    Tạo session đã auth từ service account JSON trong DFCX_SERVICE_ACCOUNT_JSON.
    Trả về AuthorizedSession hoặc None nếu lỗi.
    """
    if not DFCX_ENABLED:
        return None

    if not service_account or not AuthorizedSession:
        print("[DFCX] Chưa cài google-auth, bỏ qua CX.")
        return None

    if not DFCX_SERVICE_ACCOUNT_JSON:
        print("[DFCX] Thiếu DFCX_SERVICE_ACCOUNT_JSON.")
        return None

    try:
        info = json.loads(DFCX_SERVICE_ACCOUNT_JSON)
        creds = service_account.Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        return AuthorizedSession(creds)
    except Exception as e:
        print("❌ DFCX auth error:", e)
        print(traceback.format_exc())
        return None


def call_dialogflow_cx(session_id: str, text: str, language_code: str | None = None):
    """
    Gọi Dialogflow CX DetectIntent.
    Trả về (reply_text, debug_info).
    Nếu thất bại → (None, None).
    """
    if not DFCX_ENABLED:
        return None, None

    if not (DFCX_PROJECT_ID and DFCX_LOCATION and DFCX_AGENT_ID):
        print("[DFCX] Thiếu cấu hình PROJECT/LOCATION/AGENT.")
        return None, None

    language_code = language_code or DFCX_LANGUAGE_CODE

    authed_session = get_dfcx_authed_session()
    if not authed_session:
        return None, None

    base_url = f"https://{DFCX_LOCATION}-dialogflow.googleapis.com"
    session_path = (
        f"projects/{DFCX_PROJECT_ID}/locations/{DFCX_LOCATION}"
        f"/agents/{DFCX_AGENT_ID}/sessions/{session_id}"
    )
    url = f"{base_url}/v3/{session_path}:detectIntent"

    payload = {
        "queryInput": {
            "text": {"text": text},
            "languageCode": language_code,
        }
    }

    try:
        res = authed_session.post(url, json=payload, timeout=6)
        if res.status_code != 200:
            print("[DFCX] HTTP", res.status_code, res.text)
            return None, None

        data = res.json()
        q = data.get("queryResult", {})

        # Lấy text đầu tiên trong responseMessages
        reply_text = ""
        for msg in q.get("responseMessages", []):
            text_obj = msg.get("text")
            if text_obj and text_obj.get("text"):
                reply_text = text_obj["text"][0]
                break

        if not reply_text:
            reply_text = q.get("responseMessages", [{}])[0].get("payload", {}).get("text", "")

        cx_intent = q.get("intent", {}).get("displayName")
        cx_conf   = q.get("intentDetectionConfidence")

        debug_info = {
            "cx_intent": cx_intent,
            "cx_confidence": cx_conf,
        }

        return reply_text.strip(), debug_info

    except Exception as e:
        print("❌ DFCX detectIntent error:", e)
        print(traceback.format_exc())
        return None, None
# =====================================================================
#   RULE: QUYẾT ĐỊNH ROUTE SANG DIALOGFLOW CX
# =====================================================================
def should_route_to_cx(intent: str, user_message: str) -> bool:
    """
    Quy tắc đơn giản:
    - Chỉ route nếu DFCX_ENABLED = 1.
    - Ưu tiên các câu mang tính 'quy trình, hướng dẫn, thao tác hệ thống, nạp tiền,...'
    - Sau này nếu anh muốn, ta chỉ cần sửa rule này mà không đụng phần khác.
    """
    if not DFCX_ENABLED:
        return False

    t = strip_accents(user_message)

    # Các keyword hay dùng cho flow kịch bản (ví dụ: hướng dẫn nạp tiền, quy trình,...)
    keywords = [
        "huong dan nap tien",
        "nap tien",
        "huong dan su dung bot",
        "cach su dung bot",
        "kich hoat tai khoan",
        "dang ky su dung",
        "quy trinh lam viec",
        "quy trinh tu van",
        "khoa dao tao",
        "chuong trinh dao tao",
        "huong dan he thong",
        "flow ",
    ]
    if any(k in t for k in keywords):
        return True

    # Nếu intent là business_policy hoặc buy_payment có thể cũng cho CX xử lý
    if intent in ("business_policy", "buy_payment"):
        # tuỳ anh, tạm thời cho qua CX để chạy kịch bản nếu có
        return True

    return False

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
    """
    Đọc file JSON an toàn – lỗi gì cũng trả default.
    """
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Không đọc được file {path}: {e}")
        print(traceback.format_exc())
        return default


PRODUCTS_DATA = load_json_file("products.json", {"products": []})
COMBOS_DATA = load_json_file("combos.json", {"combos": []})
HEALTH_TAGS_CONFIG = load_json_file("health_tags_config.json", {})
COMBOS_META = load_json_file("combos_meta.json", {})
MULTI_ISSUE_RULES = load_json_file("multi_issue_rules.json", {"rules": []})

PRODUCTS = PRODUCTS_DATA.get("products", [])
COMBOS = COMBOS_DATA.get("combos", [])

from dataclasses import dataclass
from typing import Optional

@dataclass
class BrandSettings:
    """
    Cấu hình thương hiệu & kênh liên hệ cho từng tenant.
    Nếu không có trong DB thì dùng default từ ENV.
    """
    brand_name: str = "Greenway / Welllab"
    hotline: str = HOTLINE
    fanpage_url: str = FANPAGE_URL
    zalo_oa_url: str = ZALO_OA_URL
    website_url: str = WEBSITE_URL
    primary_color: str = "#16a34a"
    secondary_color: str = "#22c55e"
    ai_disclaimer: str = "Sản phẩm không phải là thuốc và không có tác dụng thay thế thuốc chữa bệnh."

    @classmethod
    def from_db(cls, row: dict | None) -> "BrandSettings":
        """
        Tạo BrandSettings từ bản ghi tenant_settings (nếu có).
        Nếu thiếu trường nào thì fallback về ENV / default.
        """
        if not row:
            return cls()

        return cls(
            brand_name=row.get("brand_name") or "Greenway / Welllab",
            hotline=row.get("hotline") or HOTLINE,
            fanpage_url=row.get("fanpage_url") or FANPAGE_URL,
            zalo_oa_url=row.get("zalo_oa_url") or ZALO_OA_URL,
            website_url=row.get("website_url") or WEBSITE_URL,
            primary_color=row.get("primary_color") or "#16a34a",
            secondary_color=row.get("secondary_color") or "#22c55e",
            ai_disclaimer=row.get("ai_disclaimer") or
                          "Sản phẩm không phải là thuốc và không có tác dụng thay thế thuốc chữa bệnh.",
        )

@dataclass
class AISettings:
    use_openai: bool = True
    use_dfcx: bool = DFCX_ENABLED
    openai_model: str = "gpt-4.1-mini"
    assistant_style_prompt: str = ""
    product_disclaimer: str = "Sản phẩm không phải là thuốc và không có tác dụng thay thế thuốc chữa bệnh."
    dfcx_project_id: str = DFCX_PROJECT_ID
    dfcx_location: str = DFCX_LOCATION
    dfcx_agent_id: str = DFCX_AGENT_ID
    dfcx_language_code: str = DFCX_LANGUAGE_CODE

@dataclass
class CatalogSettings:
    # Catalog mặc định sẽ là dữ liệu JSON global (products.json, combos.json,...)
    products: List[Dict[str, Any]] = field(default_factory=list)
    combos: List[Dict[str, Any]] = field(default_factory=list)
    health_tags_config: Dict[str, Any] = field(default_factory=dict)
    combos_meta: Dict[str, Any] = field(default_factory=dict)
    multi_issue_rules: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TenantConfig:
    tenant_id: Optional[int]
    brand: BrandSettings = field(default_factory=BrandSettings)
    ai: AISettings = field(default_factory=AISettings)
    catalogs: CatalogSettings = field(default_factory=CatalogSettings)

def _json_or_default(value, default):
    if value is None:
        return default
    return value


def load_tenant_config(tenant_id: Optional[int]) -> TenantConfig:
    """
    Đọc toàn bộ cấu hình cho 1 tenant từ DB.
    - Nếu tenant_id = None hoặc không tìm thấy config → dùng default:
      ENV (hotline/url) + JSON global (products.json, combos.json...).
    """
    cfg = TenantConfig(tenant_id=tenant_id)

    # Gán default catalog = JSON global hiện có
    cfg.catalogs.products = PRODUCTS
    cfg.catalogs.combos = COMBOS
    cfg.catalogs.health_tags_config = HEALTH_TAGS_CONFIG
    cfg.catalogs.combos_meta = COMBOS_META
    cfg.catalogs.multi_issue_rules = MULTI_ISSUE_RULES

    if not tenant_id:
        return cfg

    conn = None
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            # ---------- 1) tenant_settings ----------
            cur.execute(
                """
                SELECT brand_name, hotline, fanpage_url, zalo_oa_url, website_url,
                       logo_url, primary_color, secondary_color
                FROM tenant_settings
                WHERE tenant_id = %s
                LIMIT 1
                """,
                (tenant_id,),
            )
            row = cur.fetchone()
            if row:
                cfg.brand = BrandSettings(
                    brand_name=row["brand_name"] or cfg.brand.brand_name,
                    hotline=row["hotline"] or cfg.brand.hotline,
                    fanpage_url=row["fanpage_url"] or cfg.brand.fanpage_url,
                    zalo_oa_url=row["zalo_oa_url"] or cfg.brand.zalo_oa_url,
                    website_url=row["website_url"] or cfg.brand.website_url,
                    logo_url=row["logo_url"] or cfg.brand.logo_url,
                    primary_color=row["primary_color"] or cfg.brand.primary_color,
                    secondary_color=row["secondary_color"] or cfg.brand.secondary_color,
                )

            # ---------- 2) tenant_ai_settings ----------
            cur.execute(
                """
                SELECT use_openai, use_dfcx, openai_model,
                       assistant_style_prompt, product_disclaimer,
                       dfcx_project_id, dfcx_location, dfcx_agent_id, dfcx_language_code
                FROM tenant_ai_settings
                WHERE tenant_id = %s
                LIMIT 1
                """,
                (tenant_id,),
            )
            row = cur.fetchone()
            if row:
                cfg.ai = AISettings(
                    use_openai=row["use_openai"] if row["use_openai"] is not None else cfg.ai.use_openai,
                    use_dfcx=row["use_dfcx"] if row["use_dfcx"] is not None else cfg.ai.use_dfcx,
                    openai_model=row["openai_model"] or cfg.ai.openai_model,
                    assistant_style_prompt=row["assistant_style_prompt"] or cfg.ai.assistant_style_prompt,
                    product_disclaimer=row["product_disclaimer"] or cfg.ai.product_disclaimer,
                    dfcx_project_id=row["dfcx_project_id"] or cfg.ai.dfcx_project_id,
                    dfcx_location=row["dfcx_location"] or cfg.ai.dfcx_location,
                    dfcx_agent_id=row["dfcx_agent_id"] or cfg.ai.dfcx_agent_id,
                    dfcx_language_code=row["dfcx_language_code"] or cfg.ai.dfcx_language_code,
                )

            # ---------- 3) tenant_catalogs ----------
            cur.execute(
                """
                SELECT
                  products_json,
                  combos_json,
                  health_tags_config_json,
                  combos_meta_json,
                  multi_issue_rules_json
                FROM tenant_catalogs
                WHERE tenant_id = %s
                LIMIT 1
                """,
                (tenant_id,),
            )
            row = cur.fetchone()
            if row:
                products = _json_or_default(row["products_json"], cfg.catalogs.products)
                combos = _json_or_default(row["combos_json"], cfg.catalogs.combos)
                tags_cfg = _json_or_default(row["health_tags_config_json"], cfg.catalogs.health_tags_config)
                combos_meta = _json_or_default(row["combos_meta_json"], cfg.catalogs.combos_meta)
                multi_rules = _json_or_default(row["multi_issue_rules_json"], cfg.catalogs.multi_issue_rules)

                # Chuẩn hóa format
                if isinstance(products, dict) and "products" in products:
                    products = products["products"]
                if isinstance(combos, dict) and "combos" in combos:
                    combos = combos["combos"]

                cfg.catalogs = CatalogSettings(
                    products=list(products) if isinstance(products, list) else cfg.catalogs.products,
                    combos=list(combos) if isinstance(combos, list) else cfg.catalogs.combos,
                    health_tags_config=dict(tags_cfg) if isinstance(tags_cfg, dict) else cfg.catalogs.health_tags_config,
                    combos_meta=dict(combos_meta) if isinstance(combos_meta, dict) else cfg.catalogs.combos_meta,
                    multi_issue_rules=dict(multi_rules) if isinstance(multi_rules, dict) else cfg.catalogs.multi_issue_rules,
                )

        return cfg

    except Exception as e:
        print("❌ ERROR load_tenant_config:", e)
        print(traceback.format_exc())
        return cfg
    finally:
        if conn:
            conn.close()

# =====================================================================
#   TAG & SELECTION
# =====================================================================
def extract_tags_from_text(text: str, health_tags_config: dict | None = None):
    """
    Dựa trên HEALTH_TAGS_CONFIG (có thể lấy theo tenant), map câu hỏi sang health_tags.
    """
    text_norm = strip_accents(text)
    found = set()

    cfg_source = health_tags_config or HEALTH_TAGS_CONFIG

    for tag, cfg in (cfg_source or {}).items():
        for syn in cfg.get("synonyms", []):
            syn_norm = strip_accents(syn)
            if syn_norm and syn_norm in text_norm:
                found.add(tag)
                break
    return list(found)


def apply_multi_issue_rules(text: str, multi_issue_rules: dict | None = None):
    text_norm = strip_accents(text)
    best_rule = None
    best_count = 0

    source = multi_issue_rules or MULTI_ISSUE_RULES
    rules = source.get("rules", []) if isinstance(source, dict) else []

    for rule in rules:
        match_phrases = rule.get("match_phrases", [])
        count = 0
        for phrase in match_phrases:
            if strip_accents(phrase) in text_norm:
                count += 1
        if count > best_count and count > 0:
            best_count = count
            best_rule = rule

    return best_rule


def score_combo_for_tags(combo, requested_tags, combos_meta: dict | None = None):
    requested_tags = set(requested_tags)
    combo_tags = set(combo.get("health_tags", []))
    intersection = requested_tags & combo_tags
    score = 0

    score += 3 * len(intersection)

    meta_source = combos_meta or COMBOS_META
    meta = meta_source.get(combo.get("id", ""), {}) if meta_source else {}
    role = meta.get("role", "core")
    if role == "core":
        score += 2
    elif role == "support":
        score += 1

    if combo_tags and requested_tags:
        overlap_ratio = len(intersection) / len(requested_tags)
        score += overlap_ratio

    return score, list(intersection)


def select_combos_for_tags(requested_tags, user_text, catalogs: CatalogSettings | None = None):
    """
    Chọn 1–3 combo phù hợp nhất theo requested_tags, dùng catalog theo tenant nếu có.
    """
    cats = catalogs or CatalogSettings(
        products=PRODUCTS,
        combos=COMBOS,
        health_tags_config=HEALTH_TAGS_CONFIG,
        combos_meta=COMBOS_META,
        multi_issue_rules=MULTI_ISSUE_RULES,
    )

    if not requested_tags and user_text:
        requested_tags = extract_tags_from_text(user_text, cats.health_tags_config)

    requested_tags = list(set(requested_tags))
    if not requested_tags:
        return [], []

    rule = apply_multi_issue_rules(user_text or "", cats.multi_issue_rules)
    if rule:
        candidate_ids = set(rule.get("recommended_combos", []))
        candidates = [c for c in cats.combos if c.get("id") in candidate_ids]
    else:
        candidates = cats.combos

    scored = []
    for combo in candidates:
        s, matched = score_combo_for_tags(combo, requested_tags, cats.combos_meta)
        if s > 0:
            scored.append((s, combo, matched))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:3]

    selected_combos = [item[1] for item in top]
    covered_tags = set()
    for _, _, matched in top:
        covered_tags.update(matched)

    return selected_combos, list(covered_tags)


def search_products_by_tags(requested_tags, limit=5, catalogs: CatalogSettings | None = None):
    requested_tags = set(requested_tags)
    if not requested_tags:
        return []

    cats = catalogs or CatalogSettings(
        products=PRODUCTS,
        combos=COMBOS,
        health_tags_config=HEALTH_TAGS_CONFIG,
        combos_meta=COMBOS_META,
        multi_issue_rules=MULTI_ISSUE_RULES,
    )

    results = []
    for p in cats.products:
        tags = set(p.get("health_tags") or [])
        group = p.get("group")
        if group:
            tags.add(group)
        if tags & requested_tags:
            results.append(p)

    return results[:limit]

def search_products_by_groups(groups, limit=5, catalogs: CatalogSettings | None = None):
    group_set = {g for g in (groups or []) if g}
    if not group_set:
        return []

    cats = catalogs or CatalogSettings(
        products=PRODUCTS,
        combos=COMBOS,
        health_tags_config=HEALTH_TAGS_CONFIG,
        combos_meta=COMBOS_META,
        multi_issue_rules=MULTI_ISSUE_RULES,
    )

    results = []
    for p in cats.products:
        g = p.get("group")
        if g and g in group_set:
            results.append(p)

    return results[:limit]

# =====================================================================
#   OPENAI RESPONSES
# =====================================================================
def call_openai_responses(prompt_text: str, model: str | None = None) -> str:
    """
    Gọi Responses API an toàn:
    - Có retry, không để exception văng ra ngoài.
    - Có thể chỉ định model riêng, nếu không truyền thì dùng default.
    """
    if not prompt_text:
        return "Em chưa nhận được nội dung để xử lý."

    model_name = model or "gpt-4.1-mini"

    for attempt in range(2):  # tối đa 2 lần thử
        try:
            res = client.responses.create(
                model=model_name,
                input=prompt_text,
            )
            reply_text = getattr(res, "output_text", "") or ""
            reply_text = str(reply_text).strip()
            if not reply_text:
                reply_text = "Hiện tại em không nhận được kết quả từ hệ thống OpenAI."
            return reply_text
        except Exception as e:
            print(f"❌ ERROR OpenAI Responses (attempt {attempt+1}):", e)
            print(traceback.format_exc())
            time.sleep(0.3)

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
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start:end + 1])
        except Exception:
            return default
    return default

# =====================================================================
#   AI INTENT & PHÂN TÍCH TRIỆU CHỨNG
# =====================================================================
def ai_classify_intent(
    user_message: str, history_messages: list[dict] | None = None
) -> dict:
    """
    Phân loại ý định của người dùng ở tầng "ngữ nghĩa", không chỉ dựa vào từ khoá.

    Các intent hợp lệ:
    - greeting         : chào hỏi đơn thuần
    - smalltalk        : nói chuyện linh tinh, đời sống, đùa vui
    - conversation_flow: câu MỞ ĐẦU / DẪN NHẬP / ĐỊNH HƯỚNG CHỦ ĐỀ nhưng CHƯA hỏi nội dung
    - health_question  : hỏi về triệu chứng, tình trạng sức khoẻ
    - product_question : hỏi về MỘT sản phẩm cụ thể (tên, cách dùng, giá, link...)
    - combo_question   : hỏi gợi ý combo / bộ sản phẩm
    - business_policy  : hỏi về chính sách, hoa hồng, tuyển dụng, KPI...
    - buy_payment      : hỏi cách mua hàng, giao hàng, thanh toán
    - channel_info     : hỏi link fanpage, Zalo OA, website, kênh liên hệ
    - other            : mọi trường hợp khác
    """
    history_messages = history_messages or []

    # Ghép lịch sử thành text ngắn gọn cho model (nếu có)
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
Bạn là MODULE PHÂN LOẠI Ý ĐỊNH cho trợ lý sức khỏe & sản phẩm Greenway / Welllab.

NHIỆM VỤ:
- Hiểu NGỮ CẢNH hội thoại và câu nói mới nhất của người dùng.
- Chỉ phân loại intent, KHÔNG tự tư vấn sức khỏe hay sản phẩm.
- Đặc biệt phân biệt rõ các câu CHỈ ĐỊNH HƯỚNG (mở đầu, dọn đường) với câu HỎI THẬT.

ĐỊNH NGHĨA CÁC INTENT:

1. "greeting"
   - Câu chào đơn giản: "chào em", "hello", "hi", "chào buổi sáng"...

2. "smalltalk"
   - Nói chuyện đời thường: hỏi thăm, đùa vui, tâm sự, nhưng không yêu cầu tư vấn
     sản phẩm/chính sách rõ ràng.
   - Ví dụ: "Hôm nay trời nóng ghê", "Dạo này bận không em?"...

3. "conversation_flow"
   - Câu MỞ ĐẦU, DẪN NHẬP, ĐỊNH HƯỚNG CHỦ ĐỀ nhưng CHƯA hỏi nội dung cụ thể.
   - Người dùng báo trước là HỌ SẮP HỎI về sản phẩm/chính sách/vấn đề gì đó.
   - Ví dụ:
     * "Anh muốn hỏi về sản phẩm và chính sách."
     * "Cho em hỏi xíu về chế độ hoa hồng."
     * "Giờ chị muốn hỏi về mấy sản phẩm cho mẹ chị."
     * "Em đang có một số câu hỏi về sức khỏe."
   - Điểm quan trọng: câu này CHƯA ĐỦ THÔNG TIN để tư vấn combo/sản phẩm cụ thể.

4. "health_question"
   - Hỏi về TRIỆU CHỨNG, VẤN ĐỀ SỨC KHỎE cụ thể: đau chỗ nào, bệnh gì, đang điều trị gì...
   - Có thể kèm câu hỏi dùng sản phẩm/combo, nhưng trọng tâm là mô tả tình trạng.

5. "product_question"
   - Hỏi về MỘT sản phẩm cụ thể (đã nêu tên, mã, hoặc mô tả rõ ràng).
   - Quan tâm đến: công dụng, cách dùng, giá, thành phần, có dùng chung được không...
   - "product_question": hỏi về MỘT sản phẩm cụ thể, 
  HOẶC hỏi chung về sản phẩm/giá như: 
  "có sản phẩm nào giá mềm không", 
  "công ty có sản phẩm dành cho người thu nhập thấp không"...


6. "combo_question"
   - Hỏi GỢI Ý COMBO/BỘ SẢN PHẨM cho một vấn đề sức khỏe.
   - Ví dụ: "Bị đau dạ dày thì nên dùng combo nào?", "Cho chị combo xương khớp tốt nhất."

7. "business_policy"
   - Hỏi về chính sách, hoa hồng, tuyển dụng, thăng cấp, KPI, thưởng, quyền lợi khi làm cộng tác viên/TVV/leader...

8. "buy_payment"
   - Hỏi về mua hàng, giao hàng, thanh toán, đổi trả.
   - Ví dụ: "Mua ở đâu?", "Ship thế nào?", "Có COD không?", "Thanh toán ra sao?"

9. "channel_info"
   - Hỏi link, kênh liên hệ: fanpage, Zalo OA, website, hotline, nhóm Telegram...

10. "other"
   - Không thuộc các nhóm trên.

LUẬT QUAN TRỌNG:
- Nếu câu nói vừa là chào hỏi, vừa báo trước chủ đề (ví dụ: "Chào em, hôm nay anh muốn hỏi
  về sản phẩm cho bố anh"), thì:
  * Nếu đã có VẤN ĐỀ SỨC KHỎE CỤ THỂ → ưu tiên "health_question" / "combo_question" / "product_question".
  * Nếu mới chỉ nói kiểu "muốn hỏi về sản phẩm/chính sách" mà CHƯA có vấn đề cụ thể
    → chọn "conversation_flow".
- CHỈ chọn "health_question" / "combo_question" / "product_question" khi nội dung đủ cụ thể
  để bắt đầu tư vấn chuyên môn.
- Nếu lưỡng lự giữa "smalltalk" và "conversation_flow":
  * Nếu câu giống như "cho em hỏi cái này với", "em định hỏi chị chuyện này" → "conversation_flow".
  * Nếu chỉ là tán gẫu, chia sẻ cảm xúc → "smalltalk".

Hãy trả về JSON DUY NHẤT, không giải thích thêm, có dạng:

{{
  "intent": "greeting | smalltalk | conversation_flow | health_question | product_question | combo_question | business_policy | buy_payment | channel_info | other",
  "reason": "giải thích rất ngắn, tiếng Việt tại sao chọn intent này"
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
- Gợi ý các nhóm sản phẩm NÊN ƯU TIÊN (theo group trong dữ liệu).
- Đề xuất thêm các health_tags liên quan (nếu có).

Đầu ra là JSON DUY NHẤT, KHÔNG giải thích thêm, có dạng:

{{
  "main_issue": "<mô tả ngắn vấn đề chính>",
  "body_system": "digestive | liver | immune | cardio | neuro | other",
  "symptom_keywords": ["..."],
  "severity": "mild | moderate | severe",
  "recommended_groups": ["tieu_hoa", "dai_trang"],
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
    data.setdefault("main_issue", "")
    data.setdefault("body_system", "other")
    data.setdefault("symptom_keywords", [])
    data.setdefault("severity", "mild")
    data.setdefault("recommended_groups", [])
    data.setdefault("suggested_tags", [])
    return data


def build_expert_note(analysis: dict) -> str:
    """
    Tạo note tóm tắt phân tích chuyên gia để nhúng vào prompt tư vấn.
    Người dùng không nhìn thấy nguyên văn, chỉ dùng để định hướng LLM.
    """
    if not analysis:
        return ""

    main_issue = analysis.get("main_issue", "")
    body_system = analysis.get("body_system", "")
    severity = analysis.get("severity", "")
    sym_keywords = analysis.get("symptom_keywords") or []
    sym_text = ", ".join(sym_keywords) if sym_keywords else ""

    note = (
        "TÓM TẮT PHÂN TÍCH CHUYÊN GIA (để định hướng tư vấn, KHÔNG in nguyên văn cho khách):\n"
        f"- Vấn đề chính: {main_issue}\n"
        f"- Hệ cơ quan liên quan: {body_system}\n"
        f"- Mức độ gợi ý: {severity}\n"
    )
    if sym_text:
        note += f"- Từ khoá triệu chứng: {sym_text}\n"

    note += (
        "Hãy giải thích cho người dùng theo hướng chuyên gia sức khỏe, dễ hiểu, "
        "trình bày rõ: vấn đề chính là gì, hướng hỗ trợ ưu tiên ra sao, "
        "sau đó mới đi vào combo/sản phẩm cụ thể.\n"
    )
    return note

# =====================================================================
#   LLM PROMPTS
# =====================================================================
def llm_answer_for_combos(
    user_question: str,
    requested_tags,
    combos,
    covered_tags,
    extra_instruction: str = "",
    assistant_style_prompt: str = "",
    product_disclaimer: str | None = None,
    model: str | None = None,
):

    if not combos:
        return (
            "Hiện em chưa tìm thấy combo phù hợp trong dữ liệu cho trường hợp này. "
            f"Anh/chị vui lòng liên hệ hotline {HOTLINE} để tuyến trên tư vấn chi tiết hơn ạ."
        )

    combos_json = json.dumps(combos, ensure_ascii=False, indent=2)
    tags_text = ", ".join(requested_tags)

    style_block = ""
    if assistant_style_prompt:
        style_block = (
            "PHONG CÁCH TRỢ LÝ RIÊNG CHO CÔNG TY (hãy luôn tuân thủ):\n"
            f"{assistant_style_prompt}\n\n"
        )

    disclaimer_text = product_disclaimer or "Sản phẩm không phải là thuốc và không có tác dụng thay thế thuốc chữa bệnh."

    expert_block = extra_instruction or ""

    prompt = f"""
{style_block}
Bạn là trợ lý tư vấn cho công ty thực phẩm chức năng Greenway/Welllab.
Bạn chỉ được dùng đúng dữ liệu combo và sản phẩm trong JSON bên dưới, không được bịa thêm sản phẩm hay công dụng.

Dưới đây là câu hỏi và dữ liệu:

- Câu hỏi của khách / tư vấn viên: "{user_question}"
- Các tags/vấn đề sức khỏe hệ thống trích xuất được: {tags_text}

{expert_block}

Dữ liệu các combo đã được hệ thống chọn (JSON):

{combos_json}

YÊU CẦU RẤT QUAN TRỌNG:

1. Đọc kỹ câu hỏi, nếu người dùng hỏi NHIỀU Ý (ví dụ: nên dùng combo hay sản phẩm lẻ, loại nào tốt hơn, dùng bao lâu, giá thế nào,...)
   thì trước khi trả lời hãy tự xác định và LIỆT KÊ NGẮN GỌN các ý chính họ đang hỏi, dạng:
   - Ý 1: ...
   - Ý 2: ...
   - Ý 3: ...

2. Sau đó TRẢ LỜI TUẦN TỰ TỪNG Ý, không được bỏ sót ý nào.
   Nếu trong câu hỏi có lựa chọn A/B (ví dụ: "dùng sản phẩm hay combo thì tốt hơn", "nếu là sản phẩm thì sản phẩm gì, nếu là combo thì combo nào"):
   - Hãy đưa ra KHUYẾN NGHỊ CHÍNH (ví dụ ưu tiên combo vì ...).
   - Đồng thời nêu luôn PHƯƠNG ÁN THAY THẾ (ví dụ nếu khách chỉ đủ khả năng dùng sản phẩm lẻ thì chọn sản phẩm nào, dùng thế nào).

3. Phần tư vấn chính:
   - Mở đầu 1–3 câu: tóm tắt các vấn đề/nhu cầu chính và logic chuyên môn (tại sao ưu tiên xử lý nhóm cơ quan nào trước).
   - Với từng combo:
     + Nêu rõ combo này hỗ trợ những vấn đề nào trong các vấn đề khách đang gặp.
     + Liệt kê từng sản phẩm trong combo:
       * Tên sản phẩm
       * Lợi ích chính / tác dụng hỗ trợ
       * Thời gian dùng gợi ý (nếu có trong dữ liệu)
       * Cách dùng tóm tắt (dựa trên dose_text/usage_text nếu có)
       * Giá (price_text)
       * Link sản phẩm (product_url)

4. Nếu vấn đề có vẻ nặng/nhạy cảm (ung thư, tim mạch nặng, suy thận, v.v.) hãy khuyến nghị khách nên thăm khám và tái khám định kỳ.

5. Cuối câu trả lời, luôn nhắc: "{disclaimer_text}".

6. Viết giọng điệu gần gũi, lịch sự, như đang nói chuyện với tư vấn viên/khách hàng thật.
"""
    return call_openai_responses(prompt, model=model)


def llm_general_product_chat(user_question: str, assistant_style_prompt: str = "", model: str | None = None) -> str:
    style_block = ""
    if assistant_style_prompt:
        style_block = (
            "PHONG CÁCH TRỢ LÝ RIÊNG CHO CÔNG TY (hãy luôn tuân thủ):\n"
            f"{assistant_style_prompt}\n\n"
        )

    prompt = f"""
{style_block}
Bạn là trợ lý AI của một công ty thực phẩm bảo vệ sức khỏe.

Người dùng đang hỏi CHUNG CHUNG về sản phẩm hoặc phân khúc giá, 
ví dụ như: "có sản phẩm dành cho người thu nhập thấp không", 
nhưng chưa nói rõ tình trạng sức khỏe hay nhu cầu cụ thể.

YÊU CẦU TRẢ LỜI (TIẾNG VIỆT, NGẮN GỌN, DỄ HIỂU):

1. Khẳng định nhẹ nhàng:
   - Công ty có nhiều dòng sản phẩm với nhiều mức giá khác nhau,
     có thể sắp xếp được giải pháp phù hợp với khả năng tài chính.

2. Giải thích nguyên tắc:
   - Quan trọng nhất vẫn là chọn đúng giải pháp cho tình trạng sức khỏe,
     sau đó tối ưu theo ngân sách (ưu tiên combo nếu có điều kiện,
     còn nếu kinh phí hạn chế thì chọn 1–2 sản phẩm trọng tâm).

3. Gợi ý rõ ràng cho bước tiếp theo:
   - Hỏi lại người dùng về: tình trạng sức khỏe đang quan tâm
     (hoặc vấn đề chính) và khoảng ngân sách dự kiến,
     để tư vấn cụ thể combo/sản phẩm phù hợp.

4. Không bịa tên thuốc, không hứa hẹn quá mức, 
   không cần liệt kê tên sản phẩm cụ thể ở đây.

Câu hỏi của người dùng: "{user_question}"
"""
    return call_openai_responses(prompt, model=model)

def llm_answer_for_products(
    user_question: str,
    requested_tags,
    products,
    extra_instruction: str = "",
    assistant_style_prompt: str = "",
    product_disclaimer: str | None = None,
    model: str | None = None,
):

    if not products:
        return (
            "Hiện em chưa tìm thấy sản phẩm phù hợp trong dữ liệu cho trường hợp này. "
            f"Anh/chị vui lòng liên hệ hotline {HOTLINE} để được tư vấn rõ hơn ạ."
        )

    products_json = json.dumps(products, ensure_ascii=False, indent=2)
    tags_text = ", ".join(requested_tags)

    style_block = ""
    if assistant_style_prompt:
        style_block = (
            "PHONG CÁCH TRỢ LÝ RIÊNG CHO CÔNG TY (hãy luôn tuân thủ):\n"
            f"{assistant_style_prompt}\n\n"
        )

    disclaimer_text = product_disclaimer or "Sản phẩm không phải là thuốc và không có tác dụng thay thế thuốc chữa bệnh."

    expert_block = extra_instruction or ""

    prompt = f"""
{style_block}
Bạn là trợ lý tư vấn cho công ty thực phẩm chức năng Greenway/Welllab.
Bạn chỉ được dùng đúng dữ liệu sản phẩm trong JSON bên dưới, không được bịa thêm sản phẩm hay công dụng.

- Câu hỏi: "{user_question}"
- Các tags/vấn đề sức khỏe: {tags_text}

{expert_block}

Dữ liệu các sản phẩm đã được hệ thống chọn (JSON):

{products_json}

YÊU CẦU RẤT QUAN TRỌNG:

1. Đọc kỹ câu hỏi, nếu người dùng hỏi NHIỀU Ý (ví dụ: hỏi về công dụng, cách dùng, thời gian dùng, giá, so sánh giữa các sản phẩm...)
   thì hãy LIỆT KÊ NGẮN GỌN lại các ý chính, dạng:
   - Ý 1: ...
   - Ý 2: ...
   - Ý 3: ...

2. Sau đó trả lời lần lượt theo từng ý, không được bỏ sót ý nào.
   Nếu câu hỏi có dạng lựa chọn A/B:
   - Nêu rõ sản phẩm nào NÊN ƯU TIÊN và vì sao.
   - Đưa thêm phương án dự phòng nếu khách không dùng được sản phẩm ưu tiên.

3. Phần tư vấn chi tiết:
   - Mở đầu 1–2 câu: giới thiệu đây là các sản phẩm hỗ trợ phù hợp với vấn đề mà khách đang gặp.
   - Với từng sản phẩm:
     * Tên sản phẩm
     * Vấn đề chính mà sản phẩm hỗ trợ (dựa trên group/health_tags)
     * Lợi ích chính (dựa trên benefits_text hoặc mô tả)
     * Cách dùng tóm tắt (usage_text hoặc dose_text nếu có)
     * Giá (price_text)
     * Link sản phẩm (product_url)

4. Cuối cùng nhắc: "{disclaimer_text}"

5. Viết ngắn gọn, rõ ràng, dễ dùng cho tư vấn viên khi chát với khách.
"""
    return call_openai_responses(prompt, model=model)



def llm_answer_with_history(
    latest_question: str,
    history: list,
    assistant_style_prompt: str = "",
    product_disclaimer: str | None = None,
    model: str | None = None,
) -> str:

    """
    Dùng khi câu hỏi là follow-up: tận dụng transcript hội thoại gần đây.
    """
    if not history:
        return call_openai_responses(
            f"Khách hỏi: {latest_question}\nHãy tư vấn như trợ lý Greenway/Welllab."
        )

    lines = []
    for msg in history[-10:]:
        role = msg.get("role")
        prefix = "Khách" if role == "user" else "Trợ lý"
        content = msg.get("content", "")
        lines.append(f"{prefix}: {content}")
    convo = "\n".join(lines)
    style_block = ""
    if assistant_style_prompt:
        style_block = (
            "PHONG CÁCH TRỢ LÝ RIÊNG CHO CÔNG TY (hãy luôn tuân thủ):\n"
            f"{assistant_style_prompt}\n\n"
        )

    disclaimer_text = product_disclaimer or "Sản phẩm không phải là thuốc và không có tác dụng thay thế thuốc chữa bệnh."

    prompt = f"""
{style_block}
Bạn là trợ lý tư vấn sức khỏe & sản phẩm cho Greenway/Welllab.

Dưới đây là đoạn hội thoại gần đây giữa khách và trợ lý (bạn):

{convo}

Câu hỏi mới nhất của khách là: "{latest_question}"

YÊU CẦU:

1. Hiểu 'combo trên', 'combo đó', 'sản phẩm trên', 'sản phẩm đó', 'gói trên'...
   là đang nói về combo/sản phẩm mà bạn vừa tư vấn trước đó trong đoạn hội thoại.

2. Đọc kỹ câu hỏi mới. Nếu khách hỏi NHIỀU Ý (ví dụ: vừa hỏi lại liều dùng, vừa hỏi giá, vừa hỏi thời gian dùng...),
   hãy LIỆT KÊ NGẮN GỌN các ý chính rồi trả lời tuần tự từng ý, không được bỏ sót.

3. Trả lời ngắn gọn, rõ ràng, dựa trên thông tin đã được tư vấn ở trên.
   Nếu trong đoạn hội thoại chưa có đủ thông tin để trả lời một ý nào đó, hãy nói rõ:
   "Trong phần tư vấn phía trên em chưa ghi rõ phần này, anh/chị cho em xin lại câu hỏi đầy đủ hơn..."

4. Nếu câu trả lời liên quan đến sản phẩm, cuối cùng vẫn nhắc:
   "{disclaimer_text}"

Viết bằng tiếng Việt, giọng tư vấn viên thân thiện, chuyên nghiệp.
"""
    return call_openai_responses(prompt, model=model)


# =====================================================================
#   HANDLER CHO CÁC MODE ĐẶC BIỆT
# =====================================================================
def handle_buy_and_payment_info(brand: BrandSettings | None = None):
    website = WEBSITE_URL
    zalo = ZALO_OA_URL
    hotline = HOTLINE

    if brand:
        if getattr(brand, "website_url", None):
            website = brand.website_url
        if getattr(brand, "zalo_oa_url", None):
            zalo = brand.zalo_oa_url
        if getattr(brand, "hotline", None):
            hotline = brand.hotline

    return (
        "Để mua hàng, anh/chị có thể chọn một trong các cách sau:\n\n"
        "1️⃣ Đặt hàng trực tiếp trên website:\n"
        f"   • {website}\n\n"
        "2️⃣ Nhắn tin qua Zalo OA của công ty để được tư vấn và chốt đơn:\n"
        f"   • {zalo}\n\n"
        "3️⃣ Gọi hotline để được hỗ trợ nhanh:\n"
        f"   • {hotline}\n\n"
        "Về thanh toán, hiện công ty hỗ trợ:\n"
        "- Thanh toán khi nhận hàng (COD)\n"
        "- Chuyển khoản ngân hàng theo hướng dẫn từ tư vấn viên hoặc trên website."
    )


def handle_escalate_to_hotline(brand: BrandSettings | None = None):
    hotline = HOTLINE
    if brand and getattr(brand, "hotline", None):
        hotline = brand.hotline

    return (
        "Câu hỏi này thuộc nhóm chính sách/kế hoạch kinh doanh chuyên sâu nên cần tuyến trên hỗ trợ trực tiếp ạ.\n\n"
        "Anh/chị vui lòng để lại:\n"
        "- Họ tên\n"
        "- Số điện thoại\n"
        "- Mã TVV (nếu có)\n\n"
        f"Hoặc gọi thẳng hotline: {hotline}\n"
        "Tuyến trên sẽ liên hệ và tư vấn chi tiết cho anh/chị sớm nhất có thể."
    )

def handle_channel_navigation(brand: BrandSettings | None = None):
    fanpage = FANPAGE_URL
    zalo = ZALO_OA_URL
    website = WEBSITE_URL
    hotline = HOTLINE

    if brand:
        if getattr(brand, "fanpage_url", None):
            fanpage = brand.fanpage_url
        if getattr(brand, "zalo_oa_url", None):
            zalo = brand.zalo_oa_url
        if getattr(brand, "website_url", None):
            website = brand.website_url
        if getattr(brand, "hotline", None):
            hotline = brand.hotline

    return (
        "Anh/chị có thể theo dõi thông tin, chương trình ưu đãi và kiến thức sức khỏe tại các kênh sau:\n\n"
        f"📘 Fanpage: {fanpage}\n"
        f"💬 Zalo OA: {zalo}\n"
        f"🌐 Website: {website}\n\n"
        f"Nếu cần hỗ trợ gấp, anh/chị gọi trực tiếp hotline {hotline} giúp em nhé."
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
    tenant_cfg: TenantConfig | None = None,
    catalogs = tenant_cfg.catalogs if tenant_cfg else CatalogSettings(
    products=PRODUCTS,
    combos=COMBOS,
    health_tags_config=HEALTH_TAGS_CONFIG,
    combos_meta=COMBOS_META,
    multi_issue_rules=MULTI_ISSUE_RULES,
)):

    text = (user_message or "").strip()
    history = history or []
    brand = tenant_cfg.brand if tenant_cfg else None

    # Cấu hình AI cho tenant
    ai_settings = tenant_cfg.ai if tenant_cfg else None
    assistant_style_prompt = ai_settings.assistant_style_prompt if ai_settings else ""
    product_disclaimer = (
        ai_settings.product_disclaimer
        if (ai_settings and ai_settings.product_disclaimer)
        else "Sản phẩm không phải là thuốc và không có tác dụng thay thế thuốc chữa bệnh."
    )
    model_name = ai_settings.openai_model if ai_settings and ai_settings.openai_model else "gpt-4.1-mini"
    use_openai = ai_settings.use_openai if ai_settings is not None else True
    use_dfcx = ai_settings.use_dfcx if ai_settings is not None else DFCX_ENABLED

    # Catalog theo tenant (đã làm ở bước trước)
    catalogs = tenant_cfg.catalogs if tenant_cfg else CatalogSettings(
        products=PRODUCTS,
        combos=COMBOS,
        health_tags_config=HEALTH_TAGS_CONFIG,
        combos_meta=COMBOS_META,
        multi_issue_rules=MULTI_ISSUE_RULES,
    )

    if not text:
        reply = "Em chưa nhận được câu hỏi của anh/chị."
        if return_meta:
            meta = {
                "intent": "",
                "mode_detected": "",
                "health_tags": [],
                "selected_combos": [],
                "selected_products": [],
                "ai_main_issue": "",
                "ai_body_system": "",
                "ai_severity": "",
                "ai_groups": [],
                "ai_tags": [],
            }
            return reply, meta
        return reply

    # ================== PHÂN LOẠI Ý ĐỊNH & PHÂN TÍCH CHUYÊN GIA ==================
    history_messages = history
    if use_openai:
        # 1) Ý định (intent)
        intent_info = ai_classify_intent(text, history_messages)
        intent = intent_info.get("intent", "other")
        print("[INTENT]", intent, "|", intent_info.get("reason", ""))

        # 2) Phân tích triệu chứng
        if intent in ("health_question", "combo_question", "product_question", "other"):
            try:
                analysis = ai_analyze_symptom(text, history_messages)
            except Exception as e:
                print("❌ ERROR ai_analyze_symptom:", e)
                print(traceback.format_exc())
                # giữ analysis default
    else:
        # Nếu tắt OpenAI: không gọi model phân loại.
        # Ta fallback ý định đơn giản bằng từ khóa cho 1 số case rõ ràng.
        t_norm = strip_accents(text)
        if any(k in t_norm for k in ["chinh sach", "hoa hong", "tuyen dung", "leader"]):
            intent = "business_policy"
        elif any(k in t_norm for k in ["mua", "dat hang", "thanh toan", "giao hang", "ship"]):
            intent = "buy_payment"
        elif any(k in t_norm for k in ["fanpage", "zalo", "website", "trang web", "kenh"]):
            intent = "channel_info"
        elif any(k in t_norm for k in ["chao", "hello", "hi", "xin chao"]):
            intent = "greeting"
        else:
            intent = "other"
        print("[INTENT-BASIC]", intent, "| use_openai = False")
    # ƯU TIÊN: HƯỚNG DẪN NẠP TIỀN (KHÔNG CẦN GỌI OPENAI)
    if looks_like_topup_help(text):
        reply = handle_topup_instruction(brand)
        if return_meta:
            meta = {
                "intent": "topup_help",
                "mode_detected": "topup_help",
                "health_tags": [],
                "selected_combos": [],
                "selected_products": [],
                "ai_main_issue": "",
                "ai_body_system": "",
                "ai_severity": "",
                "ai_groups": [],
                "ai_tags": [],
            }
            return reply, meta
        return reply


    # === ROUTING SANG DIALOGFLOW CX (NẾU PHÙ HỢP) ===
    if should_route_to_cx(intent, text, ai_settings=ai_settings):
        cx_session_id = session_id or f"cx-{int(time.time())}"
        cx_reply, cx_debug = call_dialogflow_cx(
            cx_session_id,
            text,
            DFCX_LANGUAGE_CODE,
        )

        if cx_reply:
            # Nếu CX trả lời được → dùng luôn, không gọi OpenAI để tiết kiệm chi phí.
            meta = {
                "intent": intent,
                "mode_detected": "dialogflow_cx",
                "health_tags": [],
                "selected_combos": [],
                "selected_products": [],
                "ai_main_issue": "",
                "ai_body_system": "",
                "ai_severity": "",
                "ai_groups": [],
                "ai_tags": [],
            }

            if cx_debug:
                meta["cx_intent"] = cx_debug.get("cx_intent")
                meta["cx_confidence"] = cx_debug.get("cx_confidence")

            if return_meta:
                return cx_reply, meta
            return cx_reply
    # Nếu CX lỗi hoặc không trả lời được → tiếp tục flow bình thường (OpenAI)

    # === 0. Xử lý ý định "conversation_flow" (mở đầu – định hướng – chưa hỏi rõ) ===
    if intent == "conversation_flow":
        reply = (
            "Dạ em hiểu anh/chị đang muốn trao đổi về sản phẩm hoặc chính sách ạ. "
            "Anh/chị nói rõ giúp em nội dung cụ thể mà anh/chị quan tâm, "
            "để em tư vấn sát nhất và chính xác hơn nha. 😊"
        )

        if return_meta:
            meta = {
                "intent": intent,
                "mode_detected": "conversation_flow",
                "health_tags": [],
                "selected_combos": [],
                "selected_products": [],
                "ai_main_issue": "",
                "ai_body_system": "",
                "ai_severity": "",
                "ai_groups": [],
            }
            return reply, meta

        return reply

    # 2) Phân tích triệu chứng ở tầng 'chuyên gia'
    analysis = {
        "main_issue": "",
        "body_system": "other",
        "symptom_keywords": [],
        "severity": "mild",
        "recommended_groups": [],
        "suggested_tags": [],
    }
    ai_tags: list[str] = []
    ai_groups: list[str] = []

    if intent in ("health_question", "combo_question", "product_question", "other"):
        try:
            analysis = ai_analyze_symptom(text, history_messages)
        except Exception as e:
            print("❌ ERROR ai_analyze_symptom:", e)
            print(traceback.format_exc())
            # giữ analysis default

    ai_tags = analysis.get("suggested_tags") or []
    ai_groups = analysis.get("recommended_groups") or []
    expert_extra_note = build_expert_note(analysis)

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
                "ai_tags": ai_tags,
            }
            return reply, meta
        return reply

    # 2. Nói chuyện đời thường / hỏi vu vơ
    if intent == "smalltalk":
        style_block = ""
        if assistant_style_prompt:
            style_block = (
                "PHONG CÁCH TRẢ LỜI RIÊNG CHO TRỢ LÝ CỦA CÔNG TY NÀY "
                "(hãy tuân thủ tuyệt đối):\n"
                f"{assistant_style_prompt}\n\n"
            )

        smalltalk_prompt = f"""
{style_block}
Bạn là trợ lý sức khỏe cho một công ty thực phẩm bảo vệ sức khỏe.

Người dùng đang CHỈ NÓI CHUYỆN ĐỜI THƯỜNG, không yêu cầu tư vấn cụ thể.

Hãy trả lời thân thiện, ngắn gọn (2-4 câu), có thể đùa nhẹ,
sau đó khéo léo gợi ý rằng nếu họ cần tư vấn về sức khỏe / sản phẩm / combo thì bạn luôn sẵn sàng.

Câu của người dùng: "{text}"
"""

        smalltalk_reply = call_openai_responses(smalltalk_prompt, model=model_name)
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
                "ai_tags": ai_tags,
            }
            return smalltalk_reply, meta
        return smalltalk_reply

    # 3. Chính sách / kinh doanh
    if intent == "business_policy":
        reply = handle_escalate_to_hotline(brand)
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
                "ai_tags": ai_tags,
            }
            return reply, meta
        return reply

    # 4. Cách mua hàng / thanh toán
    if intent == "buy_payment":
        reply = handle_buy_and_payment_info(brand)

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
                "ai_tags": ai_tags,
            }
            return reply, meta
        return reply

    # 5. Hỏi kênh liên hệ
    if intent == "channel_info":
        reply = handle_channel_navigation(brand)

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
                "ai_tags": ai_tags,
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
        reply = llm_answer_with_history(
            text,
            history,
            assistant_style_prompt=assistant_style_prompt,
            product_disclaimer=product_disclaimer,
            model=model_name,
        )

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
                "ai_tags": ai_tags,
            }
            return reply, meta
        return reply

    # ================== MODE + TAGS + EXPERT NOTE ==================
    detected_mode = detect_mode(text) if not mode else mode.lower().strip()
    mode = detected_mode

    # Tags từ từ điển + tags do AI gợi ý
    requested_tags = extract_tags_from_text(text, catalogs.health_tags_config) or []
    requested_tags = list({*requested_tags, *ai_tags})


    # Expert note nhúng vào prompt (không cho khách thấy nguyên văn)
    question_for_llm = text
    if expert_extra_note:
        question_for_llm = (
            expert_extra_note.strip()
            + "\n\nCÂU HỎI GỐC CỦA NGƯỜI DÙNG:\n"
            + text
        )

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
        "ai_tags": ai_tags,
    }

    print("[DEBUG] handle_chat mode =", mode, "| text =", text)
    print("[DEBUG] requested_tags =", requested_tags, "| ai_groups =", ai_groups)

    # 8.5. Câu hỏi CHUNG về sản phẩm / phân khúc giá
    # Không có tag sức khỏe, không có nhóm chuyên gia → chỉ nên tư vấn định hướng
    if intent in ("product_question", "other") and not requested_tags and not ai_groups:
        reply = llm_general_product_chat(
            text,
            assistant_style_prompt=assistant_style_prompt,
            model=model_name,
        )

        if return_meta:
            return reply, meta
        return reply


    # 9. Các mode đơn giản
    if mode == "buy":
        reply = handle_buy_and_payment_info(brand)
        if return_meta:
            return reply, meta
        return reply

    if mode == "channel":
        reply = handle_channel_navigation(brand)
        if return_meta:
            return reply, meta
        return reply

    if mode == "business":
        reply = handle_escalate_to_hotline(brand)
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
        combos, covered_tags = select_combos_for_tags(requested_tags, text, catalogs)
        meta["selected_combos"] = [c.get("id") for c in combos]

        if combos:
            reply = llm_answer_for_combos(
                question_for_llm,
                requested_tags,
                combos,
                covered_tags,
                extra_instruction=expert_extra_note,
                assistant_style_prompt=assistant_style_prompt,
                product_disclaimer=product_disclaimer,
                model=model_name,
            )
            if return_meta:
                return reply, meta
            return reply

        # Không có combo → fallback sang sản phẩm (tags + group chuyên gia)
        products = search_products_by_tags(requested_tags, catalogs=catalogs)
        if (not products) and ai_groups:
            products = search_products_by_groups(ai_groups, catalogs=catalogs)
        meta["selected_products"] = [p.get("id") for p in products]

        if products:
            reply = llm_answer_for_products(
                question_for_llm,
                requested_tags,
                products,
                extra_instruction=expert_extra_note,
                assistant_style_prompt=assistant_style_prompt,
                product_disclaimer=product_disclaimer,
                model=model_name,
            )
            if return_meta:
                return reply, meta
            return reply

    # 10.2. Người dùng hỏi sản phẩm
    if want_product and not want_combo:
        products = search_products_by_tags(requested_tags, catalogs=catalogs)
        if (not products) and ai_groups:
            products = search_products_by_groups(ai_groups, catalogs=catalogs)
        meta["selected_products"] = [p.get("id") for p in products]
        reply = llm_answer_for_products(
            question_for_llm,
            requested_tags,
            products,
            extra_instruction=expert_extra_note,
            assistant_style_prompt=assistant_style_prompt,
            product_disclaimer=product_disclaimer,
            model=model_name,
        )
        if return_meta:
            return reply, meta
        return reply

    # 10.3. AUTO: ưu tiên combo, nếu không có thì show sản phẩm
        combos, covered_tags = select_combos_for_tags(requested_tags, text, catalogs)
    if combos:
        meta["selected_combos"] = [c.get("id") for c in combos]
        reply = llm_answer_for_combos(
            question_for_llm, requested_tags, combos, covered_tags
        )
        if return_meta:
            return reply, meta
        return reply

    products = search_products_by_tags(requested_tags, catalogs=catalogs)
    if (not products) and ai_groups:
         products = search_products_by_groups(ai_groups, catalogs=catalogs)
    if products:
        meta["selected_products"] = [p.get("id") for p in products]
        reply = llm_answer_for_products(
            question_for_llm, requested_tags, products
        )
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
#   DIALOGFLOW CX WEBHOOK – PHÂN LUỒNG DF CX ↔ OPENAI
# =====================================================================
@app.route("/dfcx-webhook", methods=["POST"])
def dfcx_webhook():
    """
    Webhook cho Dialogflow CX.

    Ý tưởng:
    - CX match intent + gán fulfillmentInfo.tag.
    - Webhook đọc tag để quyết định:
      + Một số tag flow cứng: trả lời trực tiếp (mua hàng, kênh, chính sách...).
      + Các tag tư vấn sức khỏe/combo/sản phẩm: đẩy vào handle_chat() để OpenAI xử lý.

    👉 Anh có thể đặt tag trong CX trùng với các giá trị dưới đây:
       - "BUSINESS_POLICY"   → chính sách/hoa hồng → handle_escalate_to_hotline
       - "BUY_PAYMENT"       → mua hàng/thanh toán → handle_buy_and_payment_info
       - "CHANNEL_INFO"      → hỏi kênh liên hệ    → handle_channel_navigation
       - "HEALTH_COMBO"      → tư vấn combo        → handle_chat(..., mode="combo")
       - "HEALTH_PRODUCT"    → tư vấn sản phẩm     → handle_chat(..., mode="product")
       - Các tag khác        → mặc định: handle_chat auto
    """
    start_time = time.time()
    try:
        body = request.get_json(force=True) or {}
        print("[DFCX] Raw body:", json.dumps(body, ensure_ascii=False))

        # Lấy text người dùng
        text = (body.get("text") or body.get("queryText") or "").strip()

        # Lấy session & parameters từ CX
        session_info = body.get("sessionInfo") or {}
        session_id = session_info.get("session") or ""
        params = session_info.get("parameters") or {}

        # Có thể lấy user_id từ tham số trong CX (nếu anh truyền)
        user_id = params.get("tvv_code") or ""

        tenant_id = get_tenant_id_by_tvv_code(user_id) if user_id else None
        tenant_cfg = load_tenant_config(tenant_id)
        brand = tenant_cfg.brand if tenant_cfg else None



        # Lấy tag do CX gán cho fulfillment
        fulfillment_info = body.get("fulfillmentInfo") or {}
        tag = (fulfillment_info.get("tag") or "").strip()
        print(f"[DFCX] tag = {tag}, session_id = {session_id}, text = {text}")

        if not text:
            reply_text = "Em chưa nhận được câu hỏi rõ ràng từ anh/chị ạ."
            return jsonify(
                {
                    "fulfillment_response": {
                        "messages": [
                            {"text": {"text": [reply_text]}}
                        ]
                    },
                    "sessionInfo": {
                        "session": session_id,
                        "parameters": params,
                    },
                }
            )

        # Nếu session_id rỗng, tạo tạm (ít nhất để log)
        if not session_id:
            session_id = f"dfcx-{request.remote_addr}-{int(time.time())}"

        # Lưu câu của user vào DB
        try:
            save_message(session_id, "user", text)
        except Exception as e:
            print("[DFCX] DB ERROR save user:", e)
            print(traceback.format_exc())

        # Lấy lịch sử để handle follow-up cho path dùng OpenAI
        history = []
        try:
            history = get_recent_history(session_id, limit=10)
        except Exception as e:
            print("[DFCX] DB ERROR get history:", e)
            print(traceback.format_exc())

        # ========== ROUTER THEO TAG CỦA DIALOGFLOW CX ==========
        reply_text = ""
        meta = {
            "intent": "",
            "mode_detected": "",
            "health_tags": [],
            "selected_combos": [],
            "selected_products": [],
            "ai_main_issue": "",
            "ai_body_system": "",
            "ai_severity": "",
            "ai_groups": [],
            "ai_tags": [],
        }

        tag_upper = tag.upper()

        # 1. Flow cứng – không cần OpenAI
        if tag_upper in ("BUSINESS_POLICY", "DF_BUSINESS_POLICY"):
            reply_text = handle_escalate_to_hotline(brand)
            meta["intent"] = "business_policy"
            meta["mode_detected"] = "business"

        elif tag_upper in ("BUY_PAYMENT", "DF_BUY_PAYMENT"):
            reply_text = handle_buy_and_payment_info(brand)
            meta["intent"] = "buy_payment"
            meta["mode_detected"] = "buy"

        elif tag_upper in ("CHANNEL_INFO", "DF_CHANNEL_INFO"):
            reply_text = handle_channel_navigation(brand)
            meta["intent"] = "channel_info"
            meta["mode_detected"] = "channel"

        # 2. Ý định tư vấn combo / sản phẩm – cho OpenAI xử lý sâu
        elif tag_upper in ("HEALTH_COMBO", "DF_HEALTH_COMBO"):
            reply_text, meta = handle_chat(
                text,
                mode="combo",
                session_id=session_id,
                return_meta=True,
                history=history,
                tenant_cfg=tenant_cfg,
            )

        elif tag_upper in ("HEALTH_PRODUCT", "DF_HEALTH_PRODUCT"):
            reply_text, meta = handle_chat(
                text,
                mode="product",
                session_id=session_id,
                return_meta=True,
                history=history,
                tenant_cfg=tenant_cfg,
            )

        # 3. Các tag khác hoặc không có tag – mặc định dùng handle_chat auto
        else:
            reply_text, meta = handle_chat(
                text,
                mode=None,
                session_id=session_id,
                return_meta=True,
                history=history,
                tenant_cfg=tenant_cfg,
            )

        # Lưu trả lời bot
        try:
            save_message(session_id, "assistant", reply_text)
        except Exception as e:
            print("[DFCX] DB ERROR save bot:", e)
            print(traceback.format_exc())

        latency_ms = int((time.time() - start_time) * 1000)

        # Log sang Google Sheets để anh theo dõi cả traffic từ CX
        try:
            log_payload = {
                "timestamp": datetime.utcnow().isoformat(),
                "channel": "dialogflow_cx",
                "session_id": session_id,
                "user_id": user_id,
                "user_message": text,
                "message_for_ai": text,
                "used_history_message": "",
                "bot_reply": reply_text,
                "intent": meta.get("intent", ""),
                "mode_detected": meta.get("mode_detected", ""),
                "health_tags": meta.get("health_tags", []),
                "selected_combos": meta.get("selected_combos", []),
                "selected_products": meta.get("selected_products", []),
                "analysis_main_issue": meta.get("ai_main_issue", ""),
                "analysis_body_system": meta.get("ai_body_system", ""),
                "analysis_severity": meta.get("ai_severity", ""),
                "analysis_groups": meta.get("ai_groups", []),
                "analysis_tags": meta.get("ai_tags", []),
                "latency_ms": latency_ms,
            }
            log_conversation(log_payload)
        except Exception as e:
            print("[DFCX] log_conversation error:", e)
            print(traceback.format_exc())

        # Trả kết quả theo format của Dialogflow CX
        return jsonify(
            {
                "fulfillment_response": {
                    "messages": [
                        {"text": {"text": [reply_text]}}
                    ]
                },
                "sessionInfo": {
                    "session": session_id,
                    "parameters": params,
                },
            }
        )

    except Exception as e:
        print("❌ ERROR /dfcx-webhook:", e)
        print(traceback.format_exc())
        reply_text = "Xin lỗi, hiện hệ thống đang gặp lỗi. Anh/chị vui lòng thử lại sau giúp em ạ."
        return jsonify(
            {
                "fulfillment_response": {
                    "messages": [
                        {"text": {"text": [reply_text]}}
                    ]
                }
            }
        ), 500

# =====================================================================
#   API /openai-chat – LOG DB + NHỚ CÂU CŨ + NGỮ CẢNH
# =====================================================================
# =====================================================================
#   API /openai-chat – GATEWAY: SESSION + BILLING + LOG + AI
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
        channel = body.get("channel") or "web"

        # Lấy session token từ header (ưu tiên) hoặc từ body (fallback)
        session_token = (request.headers.get("X-Session-Token") or "").strip()
        if not session_token:
            session_token = (body.get("session_token") or "").strip()

        # Xác định user + tenant từ session
        user_obj, tenant_obj = get_user_and_tenant_from_session(session_token)
        user_id = ""
        tenant_id = None
        if user_obj:
            user_id = user_obj.get("tvv_code") or user_obj.get("phone") or ""
            tenant_id = user_obj.get("tenant_id")
            # Load cấu hình tenant (brand, AI, catalogs...)
                    # Load cấu hình tenant (brand + AI + catalogs)
        tenant_cfg = load_tenant_config(tenant_id) if tenant_id else load_tenant_config(None)

        # Nếu client không gửi session_id (ID phiên chat), tự sinh
        session_id = body.get("session_id") or ""
        if not session_id:
            # Gắn thêm user_id cho dễ trace
            sess_suffix = user_id if user_id else request.remote_addr
            session_id = f"web-{sess_suffix}-{int(time.time())}"

        used_history_message = ""
        message_for_ai = user_message

        # ================== BILLING: KIỂM TRA SỐ DƯ ==================
        tenant_balance_cents = 0
        has_credit = True
        billing_info = None

        if BILLING_ENABLED and tenant_id:
            try:
                tenant_balance_cents = get_tenant_balance_cents(tenant_id)
                has_credit = tenant_balance_cents > 0
            except Exception as e:
                print("[BILLING] Lỗi lấy số dư:", e)
                print(traceback.format_exc())
                # lỗi lấy số dư thì cho chạy nhưng không trừ (tránh chặn user vì bug)

        # ========== CASE 1: HẾT TIỀN → CHẾ ĐỘ CƠ BẢN ==========
        if BILLING_ENABLED and tenant_id and not has_credit:
            # Vẫn lưu message user
            try:
                save_message(session_id, "user", user_message)
            except Exception as e:
                print("[DB ERROR] Cannot save user message:", e)
                print(traceback.format_exc())

            # Trả lời chế độ basic (không gọi OpenAI / CX)
            reply_text = (
                "Hiện tại tài khoản của anh/chị đã hết số dư cho chế độ trợ lý thông minh.\n\n"
                "Bot vẫn có thể hỗ trợ anh/chị ở chế độ cơ bản miễn phí với những nội dung đã được cài đặt sẵn "
                "(ví dụ: hướng dẫn nạp tiền, các câu hỏi thường gặp). "
                "Để kích hoạt lại chế độ thông minh (phân tích sâu, trả lời theo ngữ cảnh), "
                "anh/chị vui lòng nạp thêm tiền vào tài khoản.\n\n"
                "Anh/chị có thể nhắn: \"Hướng dẫn nạp tiền\" để xem chi tiết.\n\n"
                + NO_BALANCE_NOTICE_TEXT
            )

            meta = {
                "intent": "no_credit",
                "mode_detected": "basic_fallback",
                "health_tags": [],
                "selected_combos": [],
                "selected_products": [],
                "ai_main_issue": "",
                "ai_body_system": "",
                "ai_severity": "",
                "ai_groups": [],
                "ai_tags": [],
            }

            try:
                save_message(session_id, "assistant", reply_text)
            except Exception as e:
                print("[DB ERROR] Cannot save bot reply:", e)
                print(traceback.format_exc())

            latency_ms = int((time.time() - start_time) * 1000)

            try:
                log_payload = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "channel": channel,
                    "session_id": session_id,
                    "user_id": user_id,
                    "user_message": user_message,
                    "message_for_ai": "",
                    "used_history_message": "",
                    "bot_reply": reply_text,
                    "intent": meta.get("intent", ""),
                    "mode_detected": meta.get("mode_detected"),
                    "health_tags": meta.get("health_tags", []),
                    "selected_combos": meta.get("selected_combos", []),
                    "selected_products": meta.get("selected_products", []),
                    "analysis_main_issue": meta.get("ai_main_issue", ""),
                    "analysis_body_system": meta.get("ai_body_system", ""),
                    "analysis_severity": meta.get("ai_severity", ""),
                    "analysis_groups": meta.get("ai_groups", []),
                    "analysis_tags": meta.get("ai_tags", []),
                    "latency_ms": latency_ms,
                    "old_balance_cents": tenant_balance_cents,
                    "new_balance_cents": tenant_balance_cents,
                }
                log_conversation(log_payload)
            except Exception as e:
                print("[WARN] log_conversation error:", e)
                print(traceback.format_exc())

            return jsonify({"reply": reply_text})

        # ========== CASE 2: CÒN TIỀN HOẶC BILLING TẮT → DÙNG TRỢ LÝ THÔNG MINH ==========
        # 1) Xử lý "trả lời lại câu hỏi trên"
        if looks_like_repeat_request(user_message) and session_id:
            last_q = get_last_user_message(session_id)
            if last_q:
                used_history_message = last_q
                message_for_ai = last_q
                print("[DEBUG] Repeat request detected, dùng lại câu hỏi:", last_q)

        # 2) Lưu message user
        try:
            save_message(session_id, "user", user_message)
        except Exception as e:
            print("[DB ERROR] Cannot save user message:", e)
            print(traceback.format_exc())

        # 3) Lấy history
        history = []
        try:
            history = get_recent_history(session_id, limit=10)
        except Exception as e:
            print("[DB ERROR] Cannot get history:", e)
            print(traceback.format_exc())

        # 4) Gọi core handle_chat (có dùng OpenAI bên trong)
            reply_text, meta = handle_chat(
            message_for_ai,
            mode or None,
            session_id=session_id,
            return_meta=True,
            history=history,
            tenant_cfg=tenant_cfg,
        )


        # 5) Lưu bot reply
        try:
            save_message(session_id, "assistant", reply_text)
        except Exception as e:
            print("[DB ERROR] Cannot save bot reply:", e)
            print(traceback.format_exc())

        # 6) TRỪ TIỀN (nếu có tenant + billing bật)
        extra_notice = ""
        if BILLING_ENABLED and tenant_id:
            try:
                billing_info = charge_tenant_for_smart_request(tenant_id, messages=1)
                old_bal = billing_info["old_balance_cents"]
                new_bal = billing_info["new_balance_cents"]

                if billing_info["became_zero"]:
                    extra_notice = "\n\n" + NO_BALANCE_NOTICE_TEXT
                elif billing_info["is_low"]:
                    extra_notice = "\n\n" + LOW_BALANCE_NOTICE_TEXT

                if extra_notice:
                    reply_text = reply_text.rstrip() + "\n\n" + extra_notice
            except Exception as e:
                print("[BILLING] Lỗi trừ tiền:", e)
                print(traceback.format_exc())

        latency_ms = int((time.time() - start_time) * 1000)

        # 7) Log sang Google Sheets
        try:
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
                "analysis_main_issue": meta.get("ai_main_issue", ""),
                "analysis_body_system": meta.get("ai_body_system", ""),
                "analysis_severity": meta.get("ai_severity", ""),
                "analysis_groups": meta.get("ai_groups", []),
                "analysis_tags": meta.get("ai_tags", []),
                "latency_ms": latency_ms,
            }
            if billing_info:
                log_payload["old_balance_cents"] = billing_info["old_balance_cents"]
                log_payload["new_balance_cents"] = billing_info["new_balance_cents"]
            log_conversation(log_payload)
        except Exception as e:
            print("[WARN] log_conversation error:", e)
            print(traceback.format_exc())

        return jsonify({"reply": reply_text})

    except Exception as e:
        print("❌ ERROR /openai-chat:", e)
        print(traceback.format_exc())
        return jsonify(
            {"reply": "Xin lỗi, hiện tại hệ thống đang gặp lỗi. Anh/chị vui lòng thử lại sau nhé."}
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

        # Log sang Google Sheets nếu cần theo dõi đăng ký
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
                    "analysis_main_issue": "",
                    "analysis_body_system": "",
                    "analysis_severity": "",
                    "analysis_groups": [],
                    "analysis_tags": [],
                    "latency_ms": 0,
                }
            )
        except Exception as e:
            print("[WARN] log register error:", e)
            print(traceback.format_exc())

        return jsonify(
            {
                "tvv_code": tvv_code,
                "message": "Đăng ký thành công. Leader sẽ kích hoạt gói sử dụng cho tài khoản này.",
            }
        )

    except Exception as e:
        print("❌ ERROR /auth/register:", e)
        print(traceback.format_exc())
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

# =====================================================================
#   API /admin/tenants – DANH SÁCH TENANT + BALANCE
# =====================================================================
@app.route("/admin/tenants", methods=["GET"])
def admin_list_tenants():
    ok, msg = require_admin_secret()
    if not ok:
        status = 500 if "chưa được cấu hình" in msg else 401
        return jsonify({"error": msg}), status

    q = (request.args.get("q") or "").strip()

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            if q:
                pattern = f"%{q}%"
                cur.execute(
                    """
                    SELECT
                      t.id,
                      t.name,
                      t.contact_phone,
                      t.contact_email,
                      t.status,
                      COALESCE(b.balance_cents, 0) AS balance_cents,
                      t.created_at,
                      t.updated_at
                    FROM tenants t
                    LEFT JOIN tenant_billing b ON b.tenant_id = t.id
                    WHERE
                      t.name ILIKE %s
                      OR t.contact_phone ILIKE %s
                      OR t.contact_email ILIKE %s
                    ORDER BY t.created_at DESC
                    LIMIT 200
                    """,
                    (pattern, pattern, pattern),
                )
            else:
                cur.execute(
                    """
                    SELECT
                      t.id,
                      t.name,
                      t.contact_phone,
                      t.contact_email,
                      t.status,
                      COALESCE(b.balance_cents, 0) AS balance_cents,
                      t.created_at,
                      t.updated_at
                    FROM tenants t
                    LEFT JOIN tenant_billing b ON b.tenant_id = t.id
                    ORDER BY t.created_at DESC
                    LIMIT 200
                    """
                )
            rows = cur.fetchall()

        items = []
        for r in rows:
            balance_cents = int(r["balance_cents"] or 0)
            if BILLING_ENABLED:
                plan_mode = "smart" if balance_cents > 0 else "basic"
            else:
                plan_mode = "smart"  # nếu billing tắt, coi như luôn smart

            items.append(
                {
                    "id": r["id"],
                    "name": r["name"],
                    "contact_phone": r["contact_phone"],
                    "contact_email": r["contact_email"],
                    "status": r["status"],
                    "balance_cents": balance_cents,
                    "plan_mode": plan_mode,
                    "created_at": r["created_at"].isoformat(),
                    "updated_at": r["updated_at"].isoformat(),
                }
            )

        return jsonify({"items": items})
    except Exception as e:
        print("❌ ERROR /admin/tenants:", e)
        print(traceback.format_exc())
        return jsonify({"error": "Không lấy được danh sách tenants."}), 500
    finally:
        conn.close()

# =====================================================================
#   API /admin/tenants/topup – ADMIN NẠP TIỀN CHO TENANT
# =====================================================================
@app.route("/admin/tenants/topup", methods=["POST"])
def admin_tenant_topup():
    ok, msg = require_admin_secret()
    if not ok:
        status = 500 if "chưa được cấu hình" in msg else 401
        return jsonify({"error": msg}), status

    try:
        body = request.get_json(force=True) or {}
        tenant_id = int(body.get("tenant_id") or 0)
        amount_cents = int(body.get("amount_cents") or 0)
        note = (body.get("note") or "").strip()

        if tenant_id <= 0:
            return jsonify({"error": "tenant_id không hợp lệ."}), 400
        if amount_cents <= 0:
            return jsonify({"error": "amount_cents phải > 0."}), 400

        # kiểm tra tenant tồn tại
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, name FROM tenants WHERE id = %s LIMIT 1",
                    (tenant_id,),
                )
                row = cur.fetchone()
            if not row:
                return jsonify({"error": "Tenant không tồn tại."}), 404
            tenant_name = row["name"]
        finally:
            conn.close()

        result = topup_tenant_balance(tenant_id, amount_cents, note=note)

        return jsonify(
            {
                "tenant_id": tenant_id,
                "tenant_name": tenant_name,
                "amount_cents": amount_cents,
                "old_balance_cents": result["old_balance_cents"],
                "new_balance_cents": result["new_balance_cents"],
                "message": "Nạp tiền thành công.",
            }
        )
    except Exception as e:
        print("❌ ERROR /admin/tenants/topup:", e)
        print(traceback.format_exc())
        return jsonify({"error": "Lỗi hệ thống khi nạp tiền."}), 500


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
        print(traceback.format_exc())
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
        print("❌ DB ERROR:", e)
        print(traceback.format_exc())
        return f"DB ERROR: {e}", 500

@app.route("/auth/request-otp", methods=["POST"])
def request_otp():
    try:
        body = request.get_json(force=True) or {}
        phone = (body.get("phone") or "").strip()

        if not phone:
            return jsonify({"error": "Vui lòng nhập số điện thoại."}), 400

        # Tạo mã OTP
        otp = str(random.randint(100000, 999999))

        expires = datetime.utcnow() + timedelta(minutes=5)

        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO otp_codes (phone, code, purpose, expires_at)
                VALUES (%s, %s, 'login', %s)
            """, (phone, otp, expires))
        conn.commit()
        conn.close()

        # Gửi SMS
        msg = f"Mã OTP đăng nhập của bạn là: {otp}. Hiệu lực 5 phút."
        send_sms_viettel(phone, msg)

        return jsonify({"success": True, "message": "OTP đã được gửi."})

    except Exception as e:
        print("❌ ERROR /auth/request-otp:", e)
        return jsonify({"error": "Không thể gửi OTP lúc này."}), 500

@app.route("/auth/verify-otp", methods=["POST"])
def verify_otp():
    try:
        body = request.get_json(force=True) or {}
        phone = (body.get("phone") or "").strip()
        code = (body.get("code") or "").strip()

        if not phone or not code:
            return jsonify({"error": "Thiếu số điện thoại hoặc OTP."}), 400

        conn = get_db_conn()
        with conn.cursor() as cur:

            # Lấy OTP mới nhất
            cur.execute("""
                SELECT id, code, expires_at, is_used 
                FROM otp_codes
                WHERE phone = %s AND purpose = 'login'
                ORDER BY created_at DESC
                LIMIT 1
            """, (phone,))
            row = cur.fetchone()

            if not row:
                return jsonify({"error": "OTP không hợp lệ."}), 400

            if row["is_used"]:
                return jsonify({"error": "OTP đã sử dụng."}), 400

            if row["code"] != code:
                return jsonify({"error": "OTP không chính xác."}), 400

            if datetime.utcnow() > row["expires_at"]:
                return jsonify({"error": "OTP đã hết hạn."}), 400

            # OTP hợp lệ → đánh dấu đã dùng
            cur.execute("UPDATE otp_codes SET is_used = TRUE WHERE id = %s", (row["id"],))
            conn.commit()

            # Lấy hoặc tạo user
            cur.execute("SELECT * FROM tvv_users WHERE phone = %s LIMIT 1", (phone,))
            user = cur.fetchone()

            if not user:
                # Nếu chưa có user → tạo mới + tạo tenant mới
                cur.execute("""
                    INSERT INTO tenants (name, contact_phone)
                    VALUES (%s, %s)
                    RETURNING id
                """, (f"Khách hàng {phone}", phone))
                tenant_id = cur.fetchone()["id"]

                cur.execute("""
                    INSERT INTO tenant_billing (tenant_id, balance_cents)
                    VALUES (%s, 0)
                """, (tenant_id,))

                cur.execute("""
                    INSERT INTO tvv_users (tvv_code, full_name, phone, tenant_id)
                    VALUES (%s, %s, %s, %s)
                    RETURNING *
                """, (phone, f"User {phone}", phone, tenant_id))
                user = cur.fetchone()

                conn.commit()

            conn.close()

            # Tạo session token (simple)
            session_token = f"token-{phone}-{int(time.time())}"

            return jsonify({
                "success": True,
                "session_token": session_token,
                "user": {
                    "tvv_code": user["tvv_code"],
                    "full_name": user["full_name"],
                    "phone": user["phone"],
                    "tenant_id": user["tenant_id"],
                }
            })

    except Exception as e:
        print("❌ ERROR /auth/verify-otp:", e)
        return jsonify({"error": "Lỗi xác thực OTP."}), 500

# =====================================================================
#   API /me – THÔNG TIN CÁ NHÂN + BILLING CỦA USER HIỆN TẠI
# =====================================================================
@app.route("/me", methods=["GET"])
def me():
    # Lấy session từ header
    session_token = (request.headers.get("X-Session-Token") or "").strip()
    if not session_token:
        return jsonify({"error": "Thiếu X-Session-Token."}), 401

    user_obj, tenant_obj = get_user_and_tenant_from_session(session_token)
    if not user_obj:
        return jsonify({"error": "Session không hợp lệ hoặc user không tồn tại."}), 401

    tenant_id = user_obj.get("tenant_id")
    balance_cents = 0
    low_balance = False
    plan_mode = "basic"  # basic | smart

    if tenant_id:
        try:
            balance_cents = get_tenant_balance_cents(tenant_id)
            if BILLING_ENABLED:
                if balance_cents > 0:
                    plan_mode = "smart"
                else:
                    plan_mode = "basic"

                low_balance = (
                    balance_cents > 0
                    and balance_cents <= LOW_BALANCE_THRESHOLD_CENTS
                )
        except Exception as e:
            print("[/me] Lỗi lấy balance:", e)
            print(traceback.format_exc())

    # Usage hôm nay
    usage_today = {"messages": 0, "cost_cents": 0}
    if tenant_id:
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                      COALESCE(SUM(messages), 0) AS messages,
                      COALESCE(SUM(cost_cents), 0) AS cost_cents
                    FROM billing_usage
                    WHERE tenant_id = %s
                      AND usage_date = CURRENT_DATE
                    """,
                    (tenant_id,),
                )
                row = cur.fetchone()
                if row:
                    usage_today["messages"] = int(row["messages"] or 0)
                    usage_today["cost_cents"] = int(row["cost_cents"] or 0)
        finally:
            conn.close()

    # Usage 30 ngày gần đây (chỉ tổng, chi tiết dùng /billing/usage)
    usage_30d = {"messages": 0, "cost_cents": 0}
    if tenant_id:
        timeseries = get_tenant_usage_timeseries(tenant_id, days=30)
        total_msg = sum(item["messages"] for item in timeseries)
        total_cost = sum(item["cost_cents"] for item in timeseries)
        usage_30d["messages"] = total_msg
        usage_30d["cost_cents"] = total_cost

    # Chuẩn bị tenant info
    tenant_data = None
    if tenant_obj:
        tenant_data = {
            "id": tenant_obj.get("id"),
            "name": tenant_obj.get("name"),
            "status": tenant_obj.get("status"),
            "contact_phone": tenant_obj.get("contact_phone"),
            "contact_email": tenant_obj.get("contact_email"),
        }

    user_data = {
        "tvv_code": user_obj.get("tvv_code"),
        "full_name": user_obj.get("full_name"),
        "phone": user_obj.get("phone"),
        "email": user_obj.get("email"),
        "company_name": user_obj.get("company_name"),
        "tenant_id": tenant_id,
    }

    billing_data = {
        "enabled": BILLING_ENABLED,
        "balance_cents": balance_cents,
        "plan_mode": plan_mode,  # basic | smart
        "low_balance": low_balance,
        "low_balance_threshold_cents": LOW_BALANCE_THRESHOLD_CENTS,
    }

    return jsonify(
        {
            "user": user_data,
            "tenant": tenant_data,
            "billing": billing_data,
            "usage_today": usage_today,
            "usage_30d": usage_30d,
        }
    )

# =====================================================================
#   API /billing/usage – USAGE THEO NGÀY CHO USER HIỆN TẠI
# =====================================================================
@app.route("/billing/usage", methods=["GET"])
def billing_usage():
    session_token = (request.headers.get("X-Session-Token") or "").strip()
    if not session_token:
        return jsonify({"error": "Thiếu X-Session-Token."}), 401

    user_obj, tenant_obj = get_user_and_tenant_from_session(session_token)
    if not user_obj or not tenant_obj:
        return jsonify({"error": "Session không hợp lệ hoặc tenant không tồn tại."}), 401

    tenant_id = user_obj.get("tenant_id")

    try:
        days = int(request.args.get("days") or "30")
    except ValueError:
        days = 30

    timeseries = get_tenant_usage_timeseries(tenant_id, days=days)

    return jsonify(
        {
            "tenant_id": tenant_id,
            "days": days,
            "items": timeseries,
        }
    )

# =====================================================================
#   HEALTHCHECK
# =====================================================================
@app.route("/", methods=["GET"])
def home():
    return "🔥 Greenway / Welllab Chatbot Gateway đang chạy ngon lành!", 200


if __name__ == "__main__":

    app.run(host="0.0.0.0", port=8080)


