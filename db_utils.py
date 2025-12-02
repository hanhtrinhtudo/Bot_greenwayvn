# db_utils.py
"""
Tiện ích làm việc với PostgreSQL:
- Kết nối DB (get_db_conn)
- Bảng chat_logs: lưu & đọc lịch sử hội thoại
- Bảng tvv_users: quản lý hồ sơ tư vấn viên
"""

from typing import List, Dict, Any, Optional

import psycopg2
from psycopg2.extras import DictCursor

from config import DATABASE_URL


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
