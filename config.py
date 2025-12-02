# config.py
import os
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")

if not OPENAI_API_KEY:
    raise Exception("Thiếu biến môi trường OPENAI_API_KEY")

HOTLINE = os.getenv("HOTLINE", "09xx.xxx.xxx")
FANPAGE_URL = os.getenv("FANPAGE_URL", "https://facebook.com/ten-fanpage")
ZALO_OA_URL = os.getenv("ZALO_OA_URL", "https://zalo.me/ten-oa")
WEBSITE_URL = os.getenv("WEBSITE_URL", "https://greenwayglobal.vn")

LOG_WEBHOOK_URL = os.getenv("LOG_WEBHOOK_URL", "")
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")

# OpenAI client dùng chung
client = OpenAI(api_key=OPENAI_API_KEY)
