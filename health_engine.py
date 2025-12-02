# health_engine.py
"""
Bộ máy xử lý dữ liệu sức khỏe:
- Load JSON: products, combos, health_tags_config, combos_meta, multi_issue_rules
- Hàm extract_tags_from_text: map câu hỏi -> health_tags
- Hàm select_combos_for_tags: chọn combo phù hợp nhất
- Hàm search_products_by_tags: tìm sản phẩm phù hợp
"""

import json
import unicodedata
from typing import List, Tuple, Dict, Any


# =====================================================================
#   TEXT UTIL
# =====================================================================

def strip_accents(text: str) -> str:
    """Bỏ dấu tiếng Việt, chuyển về lowercase để so khớp."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text


# =====================================================================
#   LOAD JSON DATA
# =====================================================================

def load_json_file(path: str, default=None):
    """Đọc file JSON an toàn, nếu lỗi trả về default."""
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Không đọc được file {path}: {e}")
        return default


# Dữ liệu sản phẩm & combo
PRODUCTS_DATA: Dict[str, Any] = load_json_file("products.json", {"products": []})
COMBOS_DATA: Dict[str, Any] = load_json_file("combos.json", {"combos": []})

# Cấu hình health_tags & rule nhiều vấn đề
HEALTH_TAGS_CONFIG: Dict[str, Any] = load_json_file("health_tags_config.json", {})
COMBOS_META: Dict[str, Any] = load_json_file("combos_meta.json", {})
MULTI_ISSUE_RULES: Dict[str, Any] = load_json_file("multi_issue_rules.json", {"rules": []})

PRODUCTS: List[Dict[str, Any]] = PRODUCTS_DATA.get("products", [])
COMBOS: List[Dict[str, Any]] = COMBOS_DATA.get("combos", [])


# =====================================================================
#   TAG & RULE ENGINE
# =====================================================================

def extract_tags_from_text(text: str) -> List[str]:
    """
    Dựa trên HEALTH_TAGS_CONFIG, map câu hỏi sang health_tags.
    HEALTH_TAGS_CONFIG dạng:
    {
      "gan": {
         "synonyms": ["gan nhiem mo", "men gan cao", ...]
      },
      ...
    }
    """
    text_norm = strip_accents(text)
    found = set()

    for tag, cfg in HEALTH_TAGS_CONFIG.items():
        for syn in cfg.get("synonyms", []):
            syn_norm = strip_accents(syn)
            if syn_norm and syn_norm in text_norm:
                found.add(tag)
                break

    return list(found)


def apply_multi_issue_rules(text: str) -> Dict[str, Any] | None:
    """
    Thử match các rule nhiều vấn đề trong multi_issue_rules.json.
    File dạng:
    {
      "rules": [
        {
          "name": "...",
          "match_phrases": ["tieu duong", "mo mau cao", ...],
          "recommended_combos": ["combo_id_1", "combo_id_2"]
        },
        ...
      ]
    }
    """
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


def score_combo_for_tags(combo: Dict[str, Any], requested_tags: List[str]) -> Tuple[float, List[str]]:
    """
    Chấm điểm combo theo danh sách requested_tags.
    - Mỗi tag trùng: +3 điểm
    - role = core: +2, support: +1
    - cộng thêm tỉ lệ phủ: len(intersection) / len(requested_tags)
    """
    requested_tags_set = set(requested_tags)
    combo_tags = set(combo.get("health_tags", []) or [])
    intersection = requested_tags_set & combo_tags
    score = 0.0

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
    if combo_tags and requested_tags_set:
        overlap_ratio = len(intersection) / len(requested_tags_set)
        score += overlap_ratio

    return score, list(intersection)


def select_combos_for_tags(requested_tags: List[str], user_text: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Chọn 1–3 combo phù hợp nhất với tập requested_tags.
    - Nếu không có requested_tags, thử extract từ user_text
    - Nếu match rule nhiều vấn đề -> ưu tiên chỉ các combo trong rule đó
    - Trả về: (danh sách combo, danh sách tags đã được cover)
    """
    # Nếu chưa có tag thì tự rút từ câu hỏi
    if (not requested_tags) and user_text:
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

    scored: List[Tuple[float, Dict[str, Any], List[str]]] = []
    for combo in candidates:
        s, matched = score_combo_for_tags(combo, requested_tags)
        if s > 0:
            scored.append((s, combo, matched))

    # Sắp xếp theo score giảm dần
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:3]

    selected_combos: List[Dict[str, Any]] = [item[1] for item in top]
    covered_tags = set()
    for _, _, matched in top:
        covered_tags.update(matched)

    return selected_combos, list(covered_tags)


def search_products_by_tags(requested_tags: List[str], limit: int = 5) -> List[Dict[str, Any]]:
    """
    Tìm sản phẩm theo requested_tags.
    Logic:
    - Mỗi product có field: health_tags (list) + group (gan, tieu_hoa, than, ...)
    - Tạo tập tags của product = health_tags ∪ {group}
    - Nếu giao với requested_tags ≠ rỗng -> match
    - Trả về tối đa `limit` sản phẩm
    """
    requested_tags_set = set(requested_tags)
    if not requested_tags_set:
        return []

    results: List[Dict[str, Any]] = []
    for p in PRODUCTS:
        tags = set(p.get("health_tags") or [])
        group = p.get("group")  # group: gan, tieu_hoa, than, tim_mach...
        if group:
            tags.add(group)
        if tags & requested_tags_set:
            results.append(p)

    return results[:limit]
