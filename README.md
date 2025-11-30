# Welllab / Greenway CX Webhook

Webhook Flask cho Dialogflow CX + OpenAI, dùng để tư vấn combo / sản phẩm theo dữ liệu `products.json` và `combos.json`.

## Cấu trúc

- app.py
- requirements.txt
- .env.example
- products.json
- combos.json
- health_tags_config.json
- combos_meta.json
- multi_issue_rules.json

## Chạy thử local

```bash
python -m venv venv
source venv/bin/activate   # hoặc venv\Scripts\activate trên Windows
pip install -r requirements.txt

cp .env.example .env       # sửa giá trị thật trong .env
python app.py
