from __future__ import annotations

import json
import hashlib
import html
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TypedDict
from urllib.parse import quote_plus

import requests
from langgraph.graph import END, START, StateGraph


try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None
    HumanMessage = None
    SystemMessage = None

from .config import CACHE_DIR, CACHE_TTL_HOURS, env_value, get_assembly_api_key, get_google_api_key

WORKFLOW_VERSION = "member_activity_v17_bill_scope_option"
BASE_URL = "https://open.assembly.go.kr/portal/openapi"
ENDPOINT_MEMBER_BILLS = "nzmimeepazxkubdpn"  # 국회의원 발의법률안
ENDPOINT_MEMBER_VOTES = "nojepdqqaweusdfbi"  # 국회의원 본회의 표결정보
ENDPOINT_VOTE_BILLS = "ncocpgfiaoituanbr"  # 의안별 표결현황
ENDPOINT_MEMBERS = "ALLNAMEMBER"  # 국회의원 정보 통합 API
HTTP_TIMEOUT = 20
HTTP_RETRY_ATTEMPTS = 3
HTTP_RETRY_BACKOFF_SECONDS = 0.35
NEWS_HTTP_TIMEOUT = 8
SUPPORTED_MEMBER_TERMS = list(range(1, 23))
SUPPORTED_BILL_TERMS = list(range(10, 23))
SUPPORTED_VOTE_TERMS = [20, 21, 22]
DEFAULT_TERMS = list(range(22, 0, -1))
DEFAULT_RECENT_LIMIT = 10
DEFAULT_MAX_BILL_PAGES = 2
DEFAULT_MAX_VOTE_PAGES = 200
DEFAULT_MAX_VOTE_BILLS_TO_SCAN = 0  # 0이면 해당 대수의 표결 의안 전체 조회
DEFAULT_VOTE_FETCH_WORKERS = 8
DEFAULT_PARTY_ALIGNMENT_LIMIT = 100
DEFAULT_LLM_INSIGHTS_ENABLED = True
DEFAULT_LLM_MODEL = "gemini-2.5-flash-lite"
DEFAULT_RECENT_NEWS_LIMIT = 10
RECENT_NEWS_CACHE_TTL_HOURS = 6
DEFAULT_ENABLE_COSPONSOR_SCAN = True
DEFAULT_MAX_COSPONSOR_SCAN_PAGES = 5
DEFAULT_SHOW_PROGRESS = True
DEFAULT_BILL_TERM_SCOPE = "recent"


class MemberActivityState(TypedDict, total=False):
    member_name: str
    assembly_terms: List[int]
    member_terms: List[int]
    bill_terms: List[int]
    vote_terms: List[int]
    all_vote_terms: List[int]
    additional_vote_terms: List[int]
    unsupported_member_terms: List[int]
    unsupported_bill_terms: List[int]
    unsupported_vote_terms: List[int]
    served_terms: List[int]
    analysis_terms: List[int]
    non_served_terms: List[int]
    recent_limit: int
    bills_enabled: bool
    bill_term_scope: str
    vote_detail_enabled: bool
    max_bill_pages: int
    max_vote_pages: int
    max_vote_bills_to_scan: int
    vote_fetch_workers: int
    party_alignment_enabled: bool
    party_alignment_vote_limit: int
    llm_insights_enabled: bool
    recent_news_enabled: bool
    max_cosponsor_scan_pages: int
    member_party_lookup: Dict[str, str]
    enable_cosponsor_scan: bool
    show_progress: bool
    progress_callback: Any
    llm_quota_exhausted: bool
    member_info: List[Dict[str, Any]]
    sponsored_bills: List[Dict[str, Any]]
    representative_bills: List[Dict[str, Any]]
    cosponsored_bills: List[Dict[str, Any]]
    bill_result_stats: List[Dict[str, Any]]
    bill_committee_stats: List[Dict[str, Any]]
    legislative_interest_context: Dict[str, Any]
    legislative_interest_analysis: str
    legislative_interest_analysis_source: str
    cosponsor_network_context: Dict[str, Any]
    cosponsor_network_analysis: str
    cosponsor_network_analysis_source: str
    cosponsor_network_partners: List[Dict[str, Any]]
    cosponsor_network_committees: List[Dict[str, Any]]
    cosponsor_network_summary: Dict[str, Any]
    recent_news_query: str
    recent_news_items: List[Dict[str, Any]]
    recent_news_analysis: str
    recent_news_analysis_source: str
    vote_records: List[Dict[str, Any]]
    yes_votes: List[Dict[str, Any]]
    no_votes: List[Dict[str, Any]]
    abstain_votes: List[Dict[str, Any]]
    absent_votes: List[Dict[str, Any]]
    vote_summary_by_term: List[Dict[str, Any]]
    vote_fetch_stats: List[Dict[str, Any]]
    party_alignment_summary: List[Dict[str, Any]]
    party_alignment_records: List[Dict[str, Any]]
    party_alignment_dissent_votes: List[Dict[str, Any]]
    party_alignment_excluded_votes: List[Dict[str, Any]]
    party_alignment_fetch_stats: List[Dict[str, Any]]
    vote_interpretation_context: Dict[str, Any]
    vote_interpretation_analysis: str
    vote_interpretation_analysis_source: str
    activity_briefing_context: Dict[str, Any]
    activity_briefing: str
    activity_briefing_source: str
    profile_summary: str
    summary: str
    errors: List[str]


def sanitize_error(error: Exception | str) -> str:
    text = str(error)
    for key_name in ["OPENCONGRESS_KEY", "OPENCONGRESS_key", "ASSEMBLY_API_KEY"]:
        key = env_value(key_name)
        if key:
            text = text.replace(key, "***")
    text = re.sub(r"([?&]KEY=)[^&\s]+", r"\1***", text)
    return text


def is_llm_quota_error(error: Exception | str) -> bool:
    text = normalize_space(error)
    upper = text.upper()
    return "RESOURCE_EXHAUSTED" in upper or "QUOTA EXCEEDED" in upper or "429" in upper


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_markdown(value: Any) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    collapsed: List[str] = []
    previous_blank = False
    for line in lines:
        is_blank = not line
        if is_blank and previous_blank:
            continue
        collapsed.append(line)
        previous_blank = is_blank
    return "\n".join(collapsed).strip()


def strip_html(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    return normalize_space(text)


def first_present(record: Dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = normalize_space(record.get(key))
        if value:
            return value
    return ""


def member_name_from_row(record: Dict[str, Any]) -> str:
    return first_present(record, ["HG_NM", "NAAS_NM", "MEMBER_NM", "MNA_NM", "NAME", "VOTER_NM"])


def member_code_from_row(record: Dict[str, Any]) -> str:
    return first_present(record, ["MONA_CD", "NAAS_CD", "NASS_CD", "MEMBER_NO"])


def party_name_from_row(record: Dict[str, Any]) -> str:
    return first_present(record, ["POLY_NM", "PLPT_NM", "PARTY_NM", "POLY_NM_REAL"])


def normalize_party_name(value: Any) -> str:
    text = normalize_space(value)
    if not text:
        return ""
    parts = [normalize_space(part) for part in re.split(r"\s*/\s*", text) if normalize_space(part)]
    text = parts[-1] if parts else text
    text = re.sub(r"\s+", "", text)
    if text in {"정당미확인", "미확인", "정보없음", "없음", "-"}:
        return ""
    return normalize_space(text)


def district_name_from_row(record: Dict[str, Any]) -> str:
    return first_present(record, ["ORIG_NM", "ELECD_NM", "DISTRICT_NM"])


def contains_name(value: Any, member_name: str) -> bool:
    return normalize_space(member_name) in normalize_space(value)


def member_name_variants(member_name: str) -> List[str]:
    name = normalize_space(member_name).replace("의원", "")
    variants = [name, f"{name}의원"]
    return [variant for index, variant in enumerate(variants) if variant and variant not in variants[:index]]


def normalize_terms(value: Any) -> List[int]:
    if value is None:
        return DEFAULT_TERMS[:]
    if isinstance(value, int):
        raw_terms = [value]
    elif isinstance(value, str):
        raw_terms = [int(part) for part in re.findall(r"\d+", value)]
    else:
        raw_terms = [int(term) for term in value]
    terms: List[int] = []
    for term in raw_terms:
        if 1 <= term <= 22 and term not in terms:
            terms.append(term)
    return terms or DEFAULT_TERMS[:]


def intersect_terms(requested_terms: List[int], supported_terms: List[int]) -> List[int]:
    supported = set(supported_terms)
    return [term for term in requested_terms if term in supported]


def unsupported_terms(requested_terms: List[int], supported_terms: List[int]) -> List[int]:
    supported = set(supported_terms)
    return [term for term in requested_terms if term not in supported]


def format_terms(terms: Iterable[int]) -> str:
    values = list(terms)
    return ", ".join(f"{term}대" for term in values) if values else "없음"


def parse_term_selection(value: str, allowed_terms: List[int]) -> List[int]:
    allowed = set(int(term) for term in allowed_terms)
    selected: List[int] = []
    for number in re.findall(r"\d+", normalize_space(value)):
        term = int(number)
        if term in allowed and term not in selected:
            selected.append(term)
    return selected


def extract_terms_from_text(value: Any) -> List[int]:
    text = normalize_space(value)
    if not text:
        return []
    terms: List[int] = []
    for number in re.findall(r"\d+", text):
        term = int(number)
        if 1 <= term <= 22 and term not in terms:
            terms.append(term)
    return terms


def extract_served_terms(member_records: List[Dict[str, Any]]) -> List[int]:
    terms: List[int] = []
    # 재임 대수는 GTELT_ERACO 같은 구조화 필드만 사용합니다.
    # BRF_HST 약력은 연도/선거명/경력 숫자가 섞여 있어 재임 대수 추출에 쓰면 오분류가 납니다.
    for record in member_records:
        for key in ["GTELT_ERACO", "ERACO", "_REQUEST_AGE"]:
            for term in extract_terms_from_text(record.get(key)):
                if term not in terms:
                    terms.append(term)
    return sorted(terms, reverse=True)


def progress_log(state: MemberActivityState, message: str) -> None:
    callback = state.get("progress_callback")
    if callable(callback):
        try:
            callback(message)
        except Exception:
            pass
    if state.get("show_progress", DEFAULT_SHOW_PROGRESS):
        print(f"[진행] {message}", flush=True)


def cache_path(namespace: str, key_parts: Iterable[Any]) -> Path:
    key = json.dumps([str(part) for part in key_parts], ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    folder = CACHE_DIR / namespace
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{digest}.json"


def is_cache_fresh(namespace: str, key_parts: Iterable[Any], ttl_hours: int = CACHE_TTL_HOURS) -> bool:
    path = cache_path(namespace, key_parts)
    if not path.exists():
        return False
    try:
        age_seconds = datetime.now().timestamp() - path.stat().st_mtime
        return ttl_hours <= 0 or age_seconds <= ttl_hours * 3600
    except Exception:
        return False


def load_cache(namespace: str, key_parts: Iterable[Any], ttl_hours: int = CACHE_TTL_HOURS) -> Optional[Any]:
    path = cache_path(namespace, key_parts)
    if not path.exists():
        return None
    try:
        age_seconds = datetime.now().timestamp() - path.stat().st_mtime
        if ttl_hours > 0 and age_seconds > ttl_hours * 3600:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_cache(namespace: str, key_parts: Iterable[Any], value: Any) -> None:
    path = cache_path(namespace, key_parts)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def parse_openapi_rows(data: Any, api_res: str) -> List[Dict[str, Any]]:
    if not isinstance(data, dict):
        return []
    candidates: List[Any] = []
    if api_res in data:
        candidates.append(data[api_res])
    candidates.extend(data.values())
    for candidate in candidates:
        if isinstance(candidate, list):
            for item in candidate:
                if isinstance(item, dict) and isinstance(item.get("row"), list):
                    return item["row"]
        if isinstance(candidate, dict) and isinstance(candidate.get("row"), list):
            return candidate["row"]
    return []


@dataclass
class AssemblyAPIClient:
    api_key: str

    def get_rows(self, api_res: str, params: Dict[str, Any], max_pages: int = 1) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for page in range(1, max_pages + 1):
            p_size = int(params.get("pSize", 100))
            query_params = {key: value for key, value in params.items() if key != "pSize" and value not in (None, "")}
            payload = {"KEY": self.api_key, "Type": "json", "pIndex": page, "pSize": p_size, **query_params}
            last_error: Optional[Exception] = None
            for attempt in range(1, HTTP_RETRY_ATTEMPTS + 1):
                try:
                    response = requests.get(f"{BASE_URL}/{api_res}", params=payload, timeout=HTTP_TIMEOUT)
                    response.raise_for_status()
                    data = response.json()
                    break
                except Exception as error:
                    last_error = error
                    if attempt >= HTTP_RETRY_ATTEMPTS:
                        raise
                    time.sleep(HTTP_RETRY_BACKOFF_SECONDS * attempt)
            else:
                if last_error:
                    raise last_error
                data = {}
            result = data.get("RESULT") if isinstance(data, dict) else None
            if isinstance(result, dict) and normalize_space(result.get("CODE")).startswith("ERROR"):
                raise RuntimeError(normalize_space(result.get("MESSAGE")) or normalize_space(result.get("CODE")))
            page_rows = parse_openapi_rows(data, api_res)
            if not page_rows:
                break
            rows.extend(page_rows)
            if len(page_rows) < p_size:
                break
        return rows


def dedupe_records(records: Iterable[Dict[str, Any]], key_candidates: List[str]) -> List[Dict[str, Any]]:
    seen = set()
    result: List[Dict[str, Any]] = []
    for record in records:
        key_parts = [normalize_space(record.get(key)) for key in key_candidates if normalize_space(record.get(key))]
        key = "|".join(key_parts) if key_parts else json.dumps(record, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        result.append(record)
    return result


def parse_date(value: Any) -> Optional[datetime]:
    text = normalize_space(value)
    if not text:
        return None
    text = text.replace("/", "-").replace(".", "-")
    candidates = [text[:10], text[:8], text[:9], text[:11], text]
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%d-%b-%y", "%d-%b-%Y"):
        for candidate in candidates:
            try:
                return datetime.strptime(candidate, fmt)
            except ValueError:
                continue
    return None


def date_sort_value(record: Dict[str, Any], *keys: str) -> datetime:
    for key in keys:
        parsed = parse_date(record.get(key))
        if parsed:
            return parsed
    return datetime.min


def classify_bill_role(record: Dict[str, Any], member_name: str) -> str:
    if contains_name(record.get("RST_PROPOSER"), member_name):
        return "대표발의"
    if contains_name(record.get("PUBL_PROPOSER"), member_name):
        return "공동발의"
    if contains_name(record.get("PROPOSER"), member_name):
        return "발의자"
    return "확인필요"


def normalize_vote_result(record: Dict[str, Any]) -> str:
    raw = first_present(record, ["RESULT_VOTE_MOD", "RESULT_VOTE", "VOTE_RESULT", "VOTE_RESULT_NM", "RESULT", "RESULT_NM"])
    if "찬성" in raw:
        return "찬성"
    if "반대" in raw:
        return "반대"
    if "기권" in raw:
        return "기권"
    if "불참" in raw or "결석" in raw:
        return "불참"
    return raw or "미확인"


def select_columns(records: List[Dict[str, Any]], columns: List[str]) -> List[Dict[str, Any]]:
    return [{column: record.get(column, "") for column in columns} for record in records]


def count_by_field(records: List[Dict[str, Any]], field: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        key = normalize_space(record.get(field)) or "미확인"
        counts[key] = counts.get(key, 0) + 1
    return counts


def format_counts(counts: Dict[str, int]) -> str:
    if not counts:
        return "없음"
    return " / ".join(f"{key} {value}건" for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def counts_to_records(counts: Dict[str, int], label_name: str, count_name: str = "건수") -> List[Dict[str, Any]]:
    return [
        {label_name: key, count_name: value}
        for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def bill_result_summary_text(bills: List[Dict[str, Any]]) -> str:
    counts = count_by_field(bills, "PROC_RESULT")
    if not counts:
        return f"발의 법안: 총 {len(bills)}건"
    return f"발의 법안: 총 {len(bills)}건 / {format_counts(counts)}"


COMMITTEE_THEME_HINTS = {
    "과학기술정보방송통신위원회": "과학기술·디지털·방송통신",
    "행정안전위원회": "행정제도·지방자치·안전",
    "정무위원회": "금융·공정거래·국정운영",
    "교육위원회": "교육정책",
    "기후에너지환경노동위원회": "기후·에너지·환경·노동",
    "국방위원회": "국방·안보",
    "국토교통위원회": "국토·교통·주거",
    "농림축산식품해양수산위원회": "농림축산·식품·해양수산",
    "문화체육관광위원회": "문화·체육·관광",
    "법제사법위원회": "사법·법무·입법체계",
    "보건복지위원회": "보건·복지",
    "재정경제기획위원회": "재정·경제정책",
    "기획재정위원회": "재정·세제·경제정책",
    "산업통상자원중소벤처기업위원회": "산업·통상·중소벤처",
    "외교통일위원회": "외교·통일",
}


def compact_bill_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "법안명": normalize_space(record.get("BILL_NAME")),
        "발의일": normalize_space(record.get("PROPOSE_DT") or record.get("PPSL_DT")),
        "소관위원회": normalize_space(record.get("COMMITTEE")),
        "처리결과": normalize_space(record.get("PROC_RESULT")) or "미확인",
        "역할": normalize_space(record.get("_ROLE")),
    }


def format_percent(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "-"
    return f"{(numerator / denominator) * 100:.1f}%"


def ordered_vote_count_text(counts: Dict[str, int]) -> str:
    keys = ["찬성", "반대", "기권", "불참", "미확인"]
    parts = [f"{key} {int(counts.get(key, 0))}" for key in keys if int(counts.get(key, 0)) > 0]
    return " / ".join(parts) if parts else "없음"


def parse_nonnegative_int(value: str, default: int) -> int:
    text = normalize_space(value)
    if not text:
        return default
    match = re.search(r"\d+", text)
    return int(match.group()) if match else default


def split_multi_value(value: Any) -> List[str]:
    text = normalize_space(value)
    if not text:
        return []
    parts = re.split(r"\s*/\s*|\s*,\s*", text)
    seen: List[str] = []
    for part in parts:
        item = normalize_space(part)
        if item and item not in seen:
            seen.append(item)
    return seen


def clean_lawmaker_name(value: Any) -> str:
    text = normalize_space(value)
    if not text:
        return ""
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\b\d+\s*인\b", "", text)
    text = re.sub(r"(국회의원|의원|대표발의|공동발의|발의자|찬성자|소개의원)$", "", text)
    text = re.sub(r"\s*(등|외)\s*\d*\s*인?$", "", text)
    text = re.sub(r"[^가-힣A-Za-z·ㆍ\s]", " ", text)
    text = normalize_space(text.replace("ㆍ", " ").replace("·", " "))
    if len(text) < 2 or text in {"등", "외", "인"}:
        return ""
    return text


def parse_lawmaker_names(value: Any) -> List[str]:
    text = normalize_space(value)
    if not text:
        return []
    text = re.sub(r"\([^)]*\)", "", text)
    text = text.replace("ㆍ", ",").replace("·", ",").replace("/", ",").replace("，", ",")
    raw_parts = re.split(r"\s*,\s*|\s+및\s+|\s+와\s+|\s+과\s+", text)
    names: List[str] = []
    for part in raw_parts:
        name = clean_lawmaker_name(part)
        if name and name not in names:
            names.append(name)
    return names


def names_contain_member(names: List[str], member_name: str) -> bool:
    target = clean_lawmaker_name(member_name)
    if not target:
        return False
    return any(name == target or target in name or name in target for name in names)


def compact_list_text(items: List[str], limit: int = 5) -> str:
    if not items:
        return "미확인"
    shown = items[:limit]
    extra = len(items) - len(shown)
    text = ", ".join(shown)
    if extra > 0:
        text += f" 외 {extra}개"
    return text


def markdown_escape(value: Any) -> str:
    return normalize_space(value).replace("|", "&#124;")


def member_profile_markdown_table(member_records: List[Dict[str, Any]], limit: int = 5) -> str:
    if not member_records:
        return ""
    lines = ["| 대수 | 정당 | 선거구 | 당선구분 | 주요 위원회 |", "|---|---|---|---|---|"]
    for row in member_records[:limit]:
        term = normalize_space(row.get("AGE") or row.get("GTELT_ERACO") or row.get("_SERVED_TERMS"))
        if term and term.isdigit():
            term = f"{term}대"
        party = markdown_escape(row.get("PLPT_NM")) or "미확인"
        district = markdown_escape(row.get("ELECD_NM")) or "미확인"
        elected = markdown_escape(row.get("RLCT_DIV_NM") or row.get("ELECD_DIV_NM")) or "미확인"
        committees = split_multi_value(row.get("BLNG_CMIT_NM") or row.get("CMIT_NM"))
        committee_text = markdown_escape(compact_list_text(committees, limit=3))
        lines.append(f"| {markdown_escape(term)} | {party} | {district} | {elected} | {committee_text} |")
    if len(member_records) > limit:
        lines.append(f"| 외 {len(member_records) - limit}건 |  |  |  |  |")
    return "\n".join(lines)


def build_profile_summary(member_name: str, member_records: List[Dict[str, Any]], served_terms: List[int]) -> str:
    if not member_records:
        return f"{member_name} (의원 기본정보 없음)"
    parties: List[str] = []
    districts: List[str] = []
    committees: List[str] = []
    for record in member_records:
        for item in split_multi_value(record.get("PLPT_NM")):
            if item not in parties:
                parties.append(item)
        for item in split_multi_value(record.get("ELECD_NM")):
            if item not in districts:
                districts.append(item)
        for item in split_multi_value(record.get("BLNG_CMIT_NM") or record.get("CMIT_NM")):
            if item not in committees:
                committees.append(item)
    lines = [
        f"**{member_name}**",
        f"- 정당: {compact_list_text(parties, limit=3)}",
        f"- 선거구: {compact_list_text(districts, limit=3)}",
        f"- 재임 대수: {format_terms(served_terms)}",
        f"- 주요 위원회: {compact_list_text(committees, limit=6)}",
    ]
    table = member_profile_markdown_table(member_records)
    if table:
        lines.extend(["", "#### 의원 기본정보", table])
    return "\n".join(lines)


def normalize_input_node(state: MemberActivityState) -> MemberActivityState:
    member_name = normalize_space(state.get("member_name"))
    if not member_name:
        raise ValueError("분석할 의원 이름이 필요합니다.")
    requested_terms = normalize_terms(state.get("assembly_terms"))
    normalized = {
        "member_name": member_name,
        "assembly_terms": requested_terms,
        "member_terms": intersect_terms(requested_terms, SUPPORTED_MEMBER_TERMS),
        "bill_terms": intersect_terms(requested_terms, SUPPORTED_BILL_TERMS),
        "vote_terms": [],
        "all_vote_terms": [],
        "additional_vote_terms": [],
        "unsupported_member_terms": unsupported_terms(requested_terms, SUPPORTED_MEMBER_TERMS),
        "unsupported_bill_terms": unsupported_terms(requested_terms, SUPPORTED_BILL_TERMS),
        "unsupported_vote_terms": unsupported_terms(requested_terms, SUPPORTED_VOTE_TERMS),
        "served_terms": [],
        "analysis_terms": requested_terms,
        "non_served_terms": [],
        "recent_limit": int(state.get("recent_limit") or DEFAULT_RECENT_LIMIT),
        "bills_enabled": bool(state.get("bills_enabled", True)),
        "bill_term_scope": normalize_space(state.get("bill_term_scope")) or DEFAULT_BILL_TERM_SCOPE,
        "vote_detail_enabled": bool(state.get("vote_detail_enabled", True)),
        "max_bill_pages": int(state.get("max_bill_pages") or DEFAULT_MAX_BILL_PAGES),
        "max_vote_pages": int(state.get("max_vote_pages") or DEFAULT_MAX_VOTE_PAGES),
        "max_vote_bills_to_scan": int(state.get("max_vote_bills_to_scan", DEFAULT_MAX_VOTE_BILLS_TO_SCAN)),
        "vote_fetch_workers": int(state.get("vote_fetch_workers") or DEFAULT_VOTE_FETCH_WORKERS),
        "party_alignment_enabled": bool(state.get("party_alignment_enabled", True)),
        "party_alignment_vote_limit": int(state.get("party_alignment_vote_limit", DEFAULT_PARTY_ALIGNMENT_LIMIT)),
        "llm_insights_enabled": bool(state.get("llm_insights_enabled", DEFAULT_LLM_INSIGHTS_ENABLED)),
        "recent_news_enabled": bool(state.get("recent_news_enabled", True)),
        "max_cosponsor_scan_pages": int(state.get("max_cosponsor_scan_pages") or DEFAULT_MAX_COSPONSOR_SCAN_PAGES),
        "enable_cosponsor_scan": bool(state.get("enable_cosponsor_scan", DEFAULT_ENABLE_COSPONSOR_SCAN)),
        "show_progress": bool(state.get("show_progress", DEFAULT_SHOW_PROGRESS)),
        "progress_callback": state.get("progress_callback"),
        "llm_quota_exhausted": bool(state.get("llm_quota_exhausted", False)),
        "errors": [],
    }
    progress_log(normalized, f"입력 정리 완료: {member_name}, 요청 대수 {format_terms(requested_terms)}")
    return normalized


def make_fetch_member_info_node(client: AssemblyAPIClient):
    def fetch_member_info_node(state: MemberActivityState) -> MemberActivityState:
        errors = list(state.get("errors", []))
        member_name = state["member_name"]
        requested_terms = state.get("assembly_terms", DEFAULT_TERMS)
        progress_log(state, "의원 기본정보 조회 시작")
        try:
            rows = client.get_rows(ENDPOINT_MEMBERS, {"NAAS_NM": member_name, "pSize": 100}, max_pages=1)
            exact_rows = [row for row in rows if normalize_space(row.get("NAAS_NM")) == member_name]
            rows = dedupe_records(exact_rows or rows, ["NAAS_CD", "NAAS_NM", "GTELT_ERACO"])
            served_terms = extract_served_terms(rows)
            if served_terms:
                analysis_terms = [term for term in requested_terms if term in served_terms]
                non_served_terms = [term for term in requested_terms if term not in served_terms]
            else:
                analysis_terms = requested_terms
                non_served_terms = []
                errors.append("의원 기본정보에서 재임 대수(GTELT_ERACO)를 추출하지 못해 요청 대수 전체를 분석 대상으로 사용합니다.")
            for row in rows:
                row["_SERVED_TERMS"] = format_terms(served_terms)
            profile_summary = build_profile_summary(member_name, rows, served_terms or analysis_terms)
            all_bill_terms = intersect_terms(analysis_terms, SUPPORTED_BILL_TERMS)
            bill_term_scope = normalize_space(state.get("bill_term_scope")) or DEFAULT_BILL_TERM_SCOPE
            bill_terms = all_bill_terms[:1] if bill_term_scope == "recent" else all_bill_terms
            all_vote_terms = intersect_terms(analysis_terms, SUPPORTED_VOTE_TERMS)
            initial_vote_terms = all_vote_terms[:1]
            additional_vote_terms = all_vote_terms[1:]
            update = {
                "member_info": rows,
                "profile_summary": profile_summary,
                "served_terms": served_terms,
                "analysis_terms": analysis_terms,
                "non_served_terms": non_served_terms,
                "member_terms": intersect_terms(analysis_terms, SUPPORTED_MEMBER_TERMS),
                "bill_terms": bill_terms,
                "all_vote_terms": all_vote_terms,
                "vote_terms": initial_vote_terms,
                "additional_vote_terms": additional_vote_terms,
                "unsupported_member_terms": unsupported_terms(analysis_terms, SUPPORTED_MEMBER_TERMS),
                "unsupported_bill_terms": unsupported_terms(analysis_terms, SUPPORTED_BILL_TERMS),
                "unsupported_vote_terms": unsupported_terms(analysis_terms, SUPPORTED_VOTE_TERMS),
                "errors": errors,
            }
            progress_log(state, f"의원 기본정보 조회 완료: {len(rows)}건, 재임 대수 {format_terms(served_terms or analysis_terms)}")
            return update
        except Exception as error:
            errors.append(f"의원정보 조회 실패: {sanitize_error(error)}")
            progress_log(state, "의원 기본정보 조회 실패")
            return {"member_info": [], "errors": errors}
    return fetch_member_info_node


def member_bills_term_cache_key(
    age: int,
    member_name: str,
    max_pages: int,
    enable_cosponsor_scan: bool,
    max_cosponsor_pages: int,
) -> List[Any]:
    return [
        age,
        normalize_space(member_name),
        max_pages,
        int(bool(enable_cosponsor_scan)),
        max_cosponsor_pages,
    ]


def load_member_bills_term_cache(
    age: int,
    member_name: str,
    max_pages: int,
    enable_cosponsor_scan: bool,
    max_cosponsor_pages: int,
) -> Optional[List[Dict[str, Any]]]:
    return load_cache(
        "member_bills_by_term_v1",
        member_bills_term_cache_key(age, member_name, max_pages, enable_cosponsor_scan, max_cosponsor_pages),
    )


def save_member_bills_term_cache(
    age: int,
    member_name: str,
    max_pages: int,
    enable_cosponsor_scan: bool,
    max_cosponsor_pages: int,
    rows: List[Dict[str, Any]],
) -> None:
    save_cache(
        "member_bills_by_term_v1",
        member_bills_term_cache_key(age, member_name, max_pages, enable_cosponsor_scan, max_cosponsor_pages),
        rows,
    )


def make_fetch_bills_node(client: AssemblyAPIClient):
    def fetch_bills_node(state: MemberActivityState) -> MemberActivityState:
        errors = list(state.get("errors", []))
        if not bool(state.get("bills_enabled", True)):
            progress_log(state, "발의법안 조회 생략: 옵션 꺼짐")
            return {
                "sponsored_bills": [],
                "representative_bills": [],
                "cosponsored_bills": [],
                "bill_result_stats": [],
                "bill_committee_stats": [],
                "errors": errors,
            }
        member_name = state["member_name"]
        max_pages = int(state.get("max_bill_pages") or DEFAULT_MAX_BILL_PAGES)
        max_cosponsor_pages = int(state.get("max_cosponsor_scan_pages") or DEFAULT_MAX_COSPONSOR_SCAN_PAGES)
        enable_cosponsor_scan = bool(state.get("enable_cosponsor_scan", DEFAULT_ENABLE_COSPONSOR_SCAN))
        rows: List[Dict[str, Any]] = []
        bill_terms = state.get("bill_terms", SUPPORTED_BILL_TERMS)
        progress_log(state, f"발의법안 조회 시작: {format_terms(bill_terms)}")

        for age in bill_terms:
            try:
                cached_term_rows = load_member_bills_term_cache(
                    age,
                    member_name,
                    max_pages,
                    enable_cosponsor_scan,
                    max_cosponsor_pages,
                )
                if cached_term_rows is not None:
                    rows.extend(cached_term_rows)
                    progress_log(state, f"{age}대 발의법안 대수 단위 캐시 사용: {len(cached_term_rows)}건")
                    continue

                before = len(rows)
                term_rows: List[Dict[str, Any]] = []
                for proposer in member_name_variants(member_name):
                    page_rows = client.get_rows(ENDPOINT_MEMBER_BILLS, {"AGE": age, "PROPOSER": proposer, "pSize": 100}, max_pages=max_pages)
                    for row in page_rows:
                        row["_REQUEST_AGE"] = age
                        row["_ROLE"] = classify_bill_role(row, member_name)
                    term_rows.extend(page_rows)
                if enable_cosponsor_scan:
                    progress_log(state, f"{age}대 발의법안 공동발의 확인: 최대 {max_cosponsor_pages}페이지")
                    scan_rows = client.get_rows(ENDPOINT_MEMBER_BILLS, {"AGE": age, "pSize": 100}, max_pages=max_cosponsor_pages)
                    for row in scan_rows:
                        if contains_name(row.get("PUBL_PROPOSER"), member_name):
                            row["_REQUEST_AGE"] = age
                            row["_ROLE"] = "공동발의"
                            term_rows.append(row)
                term_rows = dedupe_records(term_rows, ["BILL_ID", "BILL_NO", "BILL_NAME"])
                rows.extend(term_rows)
                save_member_bills_term_cache(
                    age,
                    member_name,
                    max_pages,
                    enable_cosponsor_scan,
                    max_cosponsor_pages,
                    term_rows,
                )
                progress_log(state, f"{age}대 발의법안 조회 완료: {len(rows) - before}건 추가")
            except Exception as error:
                errors.append(f"{age}대 발의법안 조회 실패: {sanitize_error(error)}")
                progress_log(state, f"{age}대 발의법안 조회 실패")

        rows = dedupe_records(rows, ["BILL_ID", "BILL_NO", "BILL_NAME"])
        rows = sorted(rows, key=lambda row: date_sort_value(row, "PROPOSE_DT", "PPSL_DT"), reverse=True)
        representative = [row for row in rows if row.get("_ROLE") == "대표발의"]
        cosponsored = [row for row in rows if row.get("_ROLE") == "공동발의"]
        bill_result_stats = counts_to_records(count_by_field(rows, "PROC_RESULT"), "처리결과")
        bill_committee_stats = counts_to_records(count_by_field(rows, "COMMITTEE"), "소관위원회")
        if not enable_cosponsor_scan:
            errors.append("공동발의 전체 스캔이 꺼져 있습니다. 공동발의 누락이 의심되면 enable_cosponsor_scan=True로 다시 실행하세요.")
        elif not cosponsored:
            errors.append("발의법안 공동발의 확인 범위에서 공동발의 법안을 찾지 못했습니다. 더 넓게 보려면 max_cosponsor_scan_pages를 늘리세요.")
        progress_log(state, f"발의법안 정리 완료: 전체 {len(rows)}건, 대표 {len(representative)}건, 공동 {len(cosponsored)}건")
        return {
            "sponsored_bills": rows,
            "representative_bills": representative,
            "cosponsored_bills": cosponsored,
            "bill_result_stats": bill_result_stats,
            "bill_committee_stats": bill_committee_stats,
            "errors": errors,
        }
    return fetch_bills_node


def build_legislative_interest_context(state: MemberActivityState) -> Dict[str, Any]:
    bills = state.get("sponsored_bills", [])
    recent_limit = int(state.get("recent_limit") or DEFAULT_RECENT_LIMIT)
    representative = state.get("representative_bills", [])[:recent_limit]
    cosponsored = state.get("cosponsored_bills", [])[:recent_limit]
    committee_stats = state.get("bill_committee_stats", [])
    result_stats = state.get("bill_result_stats", [])
    top_committees = committee_stats[:5]
    return {
        "member_name": state.get("member_name", ""),
        "profile_summary": state.get("profile_summary", ""),
        "bill_total": len(bills),
        "representative_total": len(state.get("representative_bills", [])),
        "cosponsored_total": len(state.get("cosponsored_bills", [])),
        "by_result": result_stats,
        "by_committee": committee_stats,
        "top_committees": [
            {
                **item,
                "의제힌트": COMMITTEE_THEME_HINTS.get(normalize_space(item.get("소관위원회")), normalize_space(item.get("소관위원회"))),
            }
            for item in top_committees
        ],
        "recent_representative_bills": [compact_bill_record(row) for row in representative],
        "recent_cosponsored_bills": [compact_bill_record(row) for row in cosponsored],
    }


def fallback_legislative_interest_analysis(context: Dict[str, Any]) -> str:
    member_name = normalize_space(context.get("member_name")) or "해당 의원"
    top_committees = context.get("top_committees", [])
    recent_rep = context.get("recent_representative_bills", [])[:3]
    recent_co = context.get("recent_cosponsored_bills", [])[:3]
    if not context.get("bill_total"):
        return "발의법안 데이터가 없어 입법 관심 분야를 분석할 수 없습니다."

    lines = [
        f"- {member_name} 의원의 발의법안은 총 {context.get('bill_total', 0)}건이며, 대표발의 {context.get('representative_total', 0)}건 / 공동발의 {context.get('cosponsored_total', 0)}건으로 구성됩니다.",
    ]
    if top_committees:
        theme_text = " / ".join(
            f"{item.get('의제힌트')}({item.get('건수')}건)" for item in top_committees[:3]
        )
        lines.append(f"- 소관위원회 기준으로는 {theme_text} 분야가 상대적으로 두드러집니다.")
        lines.append("- 이는 해당 의원의 입법 관심이 위 분야의 제도 개선, 산업·행정 규율, 정책 조정 의제에 집중되어 있을 가능성을 보여줍니다.")
    if recent_rep:
        bill_names = ", ".join(row.get("법안명", "") for row in recent_rep if row.get("법안명"))
        if bill_names:
            lines.append(f"- 최근 대표발의 법안 근거: {bill_names}")
    if recent_co:
        bill_names = ", ".join(row.get("법안명", "") for row in recent_co if row.get("법안명"))
        if bill_names:
            lines.append(f"- 최근 공동발의 법안 근거: {bill_names}")
    lines.append("- 이 분석은 법안명과 소관위원회 기준의 1차 해석입니다. 법안 원문, 제안이유, 회의록까지 결합하면 의제 성격을 더 정밀하게 분류할 수 있습니다.")
    return "\n".join(lines)


def generate_legislative_interest_analysis(context: Dict[str, Any]) -> tuple[str, str]:
    cache_key = [
        "google",
        "readable_v2",
        DEFAULT_LLM_MODEL,
        context.get("member_name"),
        context.get("bill_total"),
        json.dumps(context.get("by_committee", []), ensure_ascii=False, sort_keys=True),
        json.dumps(context.get("recent_representative_bills", [])[:5], ensure_ascii=False, sort_keys=True),
        json.dumps(context.get("recent_cosponsored_bills", [])[:5], ensure_ascii=False, sort_keys=True),
    ]
    cached = load_cache("legislative_interest_analysis", cache_key)
    if cached:
        return str(cached), "cache"

    google_api_key = get_google_api_key()
    if not google_api_key or ChatGoogleGenerativeAI is None:
        analysis = fallback_legislative_interest_analysis(context)
        return analysis, "rule_fallback"

    system_prompt = (
        "당신은 국회의원 활동 데이터를 분석하는 정책 리서치 어시스턴트입니다. "
        "정치적 호불호나 성과 평가를 하지 말고, 제공된 발의법안 통계와 법안 목록만 근거로 입법 관심 분야를 해석하세요. "
        "추정은 반드시 '가능성', '신호', '해석상 주의'처럼 조심스럽게 표현하세요."
    )
    user_prompt = f"""
다음 JSON context를 바탕으로 의원의 입법 관심 분야를 분석하세요.

요구 출력:
- 가독성이 좋도록 줄바꿈을 충분히 사용하세요.
- 이모지는 섹션 구분용으로만 3~5개 정도 제한적으로 사용하세요.
- 주요 관심 의제 3~5개를 `- 🔎 **의제명**` 형식으로 정리하세요.
- 각 의제 아래에 `근거:`와 `해석:`을 짧게 나눠 쓰세요.
- 대표발의/공동발의 비중에 대한 간단한 해석을 별도 bullet로 쓰세요.
- 처리결과가 미확인인 경우 성과 없음으로 단정하지 말라는 주의를 마지막에 쓰세요.
- 제목(`#`, `##`)은 쓰지 말고 Markdown bullet만 사용하세요.

context:
{json.dumps(context, ensure_ascii=False, indent=2)}
""".strip()
    llm = ChatGoogleGenerativeAI(model=DEFAULT_LLM_MODEL, temperature=0.2)
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    analysis = normalize_markdown(getattr(response, "content", ""))
    if not analysis:
        analysis = fallback_legislative_interest_analysis(context)
        return analysis, "rule_fallback"
    save_cache("legislative_interest_analysis", cache_key, analysis)
    return analysis, "llm"


def analyze_legislative_interests_node(state: MemberActivityState) -> MemberActivityState:
    errors = list(state.get("errors", []))
    context = build_legislative_interest_context(state)
    if not bool(state.get("bills_enabled", True)):
        return {"legislative_interest_context": context, "legislative_interest_analysis": "", "legislative_interest_analysis_source": "disabled_bills", "errors": errors}
    if not bool(state.get("llm_insights_enabled", DEFAULT_LLM_INSIGHTS_ENABLED)):
        return {"legislative_interest_context": context, "legislative_interest_analysis": "", "legislative_interest_analysis_source": "disabled", "errors": errors}
    if bool(state.get("llm_quota_exhausted", False)):
        analysis = fallback_legislative_interest_analysis(context)
        return {
            "legislative_interest_context": context,
            "legislative_interest_analysis": analysis,
            "legislative_interest_analysis_source": "rule_fallback_quota",
            "errors": errors,
            "llm_quota_exhausted": True,
        }
    progress_log(state, "입법 관심 분야 분석 시작")
    try:
        analysis, source = generate_legislative_interest_analysis(context)
        progress_log(state, f"입법 관심 분야 분석 완료: {source}")
        return {
            "legislative_interest_context": context,
            "legislative_interest_analysis": analysis,
            "legislative_interest_analysis_source": source,
            "errors": errors,
            "llm_quota_exhausted": bool(state.get("llm_quota_exhausted", False)),
        }
    except Exception as error:
        quota_exhausted = is_llm_quota_error(error)
        if quota_exhausted:
            errors.append("입법 관심 분야 LLM 분석은 Gemini 한도 초과로 규칙 기반 요약으로 대체했습니다.")
        else:
            errors.append(f"입법 관심 분야 LLM 분석 실패: {sanitize_error(error)}")
        analysis = fallback_legislative_interest_analysis(context)
        progress_log(state, "입법 관심 분야 분석 실패: 규칙 기반 요약으로 대체")
        return {
            "legislative_interest_context": context,
            "legislative_interest_analysis": analysis,
            "legislative_interest_analysis_source": "rule_fallback_after_error",
            "errors": errors,
            "llm_quota_exhausted": bool(state.get("llm_quota_exhausted", False)) or quota_exhausted,
        }


def build_cosponsor_network_context(state: MemberActivityState) -> Dict[str, Any]:
    member_name = clean_lawmaker_name(state.get("member_name"))
    bills = state.get("sponsored_bills", [])
    party_lookup = {
        clean_lawmaker_name(name): normalize_party_name(party)
        for name, party in dict(state.get("member_party_lookup", {}) or {}).items()
        if clean_lawmaker_name(name) and normalize_party_name(party)
    }
    member_party = (
        normalize_party_name(state.get("member_party"))
        or party_lookup.get(member_name, "")
        or normalize_party_name(member_party_for_news_query(state))
    )
    partner_stats: Dict[str, Dict[str, Any]] = {}
    committee_counts: Dict[str, int] = {}
    total_links = 0
    representative_link_count = 0
    cosponsored_link_count = 0
    same_party_links = 0
    cross_party_links = 0
    unknown_party_links = 0

    for bill in bills:
        representative_names = parse_lawmaker_names(bill.get("RST_PROPOSER"))
        cosponsor_names = parse_lawmaker_names(bill.get("PUBL_PROPOSER"))
        role = normalize_space(bill.get("_ROLE")) or classify_bill_role(bill, member_name)
        target_is_representative = names_contain_member(representative_names, member_name) or role == "대표발의"
        target_is_cosponsor = names_contain_member(cosponsor_names, member_name) or role == "공동발의"
        if not target_is_representative and not target_is_cosponsor:
            continue

        partners: List[str] = []
        if target_is_representative:
            partners.extend(cosponsor_names)
        if target_is_cosponsor:
            partners.extend(representative_names)
            partners.extend(cosponsor_names)

        cleaned_partners: List[str] = []
        for partner in partners:
            partner_name = clean_lawmaker_name(partner)
            if not partner_name or names_contain_member([partner_name], member_name):
                continue
            if partner_name not in cleaned_partners:
                cleaned_partners.append(partner_name)

        if not cleaned_partners:
            continue

        committee = normalize_space(bill.get("COMMITTEE")) or "미확인"
        bill_date = normalize_space(bill.get("PROPOSE_DT") or bill.get("PPSL_DT"))
        bill_name = normalize_space(bill.get("BILL_NAME")) or "법안명 미확인"
        bill_link = normalize_space(bill.get("DETAIL_LINK"))
        committee_counts[committee] = committee_counts.get(committee, 0) + len(cleaned_partners)

        for partner_name in cleaned_partners:
            total_links += 1
            if target_is_representative:
                representative_link_count += 1
            elif target_is_cosponsor:
                cosponsored_link_count += 1
            stat = partner_stats.setdefault(
                partner_name,
                {
                    "공동발의자": partner_name,
                    "정당": party_lookup.get(partner_name, "정당 미확인"),
                    "공동발의건수": 0,
                    "당내협업": 0,
                    "당외협업": 0,
                    "정당미확인협업": 0,
                    "소관위원회별": {},
                    "최근공동발의일": "",
                    "대표법안": "",
                    "함께한법안": [],
                },
            )
            partner_party = normalize_party_name(stat.get("정당")) or "정당 미확인"
            stat["공동발의건수"] += 1
            if not member_party or partner_party in {"", "정당 미확인"}:
                stat["정당미확인협업"] += 1
                unknown_party_links += 1
                party_relation = "미확인"
            elif partner_party == member_party:
                stat["당내협업"] += 1
                same_party_links += 1
                party_relation = "당내"
            else:
                stat["당외협업"] += 1
                cross_party_links += 1
                party_relation = "당외"
            stat["정당관계"] = party_relation if stat["공동발의건수"] == 1 else stat.get("정당관계", party_relation)
            stat["소관위원회별"][committee] = stat["소관위원회별"].get(committee, 0) + 1
            if not stat["최근공동발의일"] or (bill_date and bill_date > stat["최근공동발의일"]):
                stat["최근공동발의일"] = bill_date
                stat["대표법안"] = bill_name
            if len(stat["함께한법안"]) < 8:
                stat["함께한법안"].append(
                    {
                        "법안명": bill_name,
                        "발의일": bill_date,
                        "소관위원회": committee,
                        "대상역할": "대표발의" if target_is_representative else "공동발의",
                        "링크": bill_link,
                    }
                )

    partner_rows: List[Dict[str, Any]] = []
    for stat in partner_stats.values():
        committee_items = sorted(stat["소관위원회별"].items(), key=lambda item: (-item[1], item[0]))
        partner_rows.append(
            {
                "공동발의자": stat["공동발의자"],
                "정당": stat.get("정당") or "정당 미확인",
                "정당관계": stat.get("정당관계") or "미확인",
                "공동발의건수": stat["공동발의건수"],
                "당내협업": stat.get("당내협업", 0),
                "당외협업": stat.get("당외협업", 0),
                "정당미확인협업": stat.get("정당미확인협업", 0),
                "주요소관위원회": ", ".join(f"{name} {count}건" for name, count in committee_items[:3]) or "미확인",
                "최근공동발의일": stat["최근공동발의일"],
                "대표법안": stat["대표법안"],
                "함께한법안": stat["함께한법안"],
            }
        )
    partner_rows = sorted(partner_rows, key=lambda row: (-int(row.get("공동발의건수", 0)), row.get("공동발의자", "")))
    committee_rows = counts_to_records(committee_counts, "소관위원회", "협업건수")
    top3_links = sum(int(row.get("공동발의건수", 0) or 0) for row in partner_rows[:3])
    concentration = round(top3_links / total_links * 100, 1) if total_links else 0.0
    top_partner = partner_rows[0]["공동발의자"] if partner_rows else ""
    cross_party_rate = round(cross_party_links / total_links * 100, 1) if total_links else 0.0
    same_party_rate = round(same_party_links / total_links * 100, 1) if total_links else 0.0
    unknown_party_rate = round(unknown_party_links / total_links * 100, 1) if total_links else 0.0
    bipartisanship_type = classify_bipartisanship_type(cross_party_rate, total_links, unknown_party_rate)
    network_type = classify_cosponsor_network_type(len(partner_rows), concentration, committee_rows)

    return {
        "member_name": member_name or state.get("member_name", ""),
        "member_party": member_party or "정당 미확인",
        "partner_count": len(partner_rows),
        "total_links": total_links,
        "top_partner": top_partner,
        "top3_concentration": concentration,
        "network_type": network_type,
        "same_party_links": same_party_links,
        "cross_party_links": cross_party_links,
        "unknown_party_links": unknown_party_links,
        "same_party_rate": same_party_rate,
        "cross_party_rate": cross_party_rate,
        "unknown_party_rate": unknown_party_rate,
        "bipartisanship_type": bipartisanship_type,
        "representative_bill_links": representative_link_count,
        "cosponsored_bill_links": cosponsored_link_count,
        "partners": partner_rows[:50],
        "top_partners": partner_rows[:10],
        "committee_stats": committee_rows[:10],
        "data_basis": "현재 발의법안 API 응답의 대표발의자·공동발의자 문자열을 파싱해 대상 의원 중심 ego-network로 계산했습니다.",
    }


def classify_cosponsor_network_type(partner_count: int, top3_concentration: float, committee_rows: List[Dict[str, Any]]) -> str:
    if partner_count == 0:
        return "공동발의 네트워크 데이터 부족"
    if partner_count <= 5 and top3_concentration >= 70:
        return "소수 반복 협업형"
    if partner_count >= 20 and top3_concentration <= 40:
        return "분산 협업형"
    if committee_rows:
        total = sum(int(row.get("협업건수", 0) or 0) for row in committee_rows)
        top = int(committee_rows[0].get("협업건수", 0) or 0)
        if total and top / total >= 0.5:
            return "분야 집중 협업형"
    return "균형 협업형"


def classify_bipartisanship_type(cross_party_rate: float, total_links: int, unknown_party_rate: float) -> str:
    if total_links <= 0:
        return "초당성 데이터 부족"
    if unknown_party_rate >= 50:
        return "정당 매칭 제한"
    if cross_party_rate < 20:
        return "정당 내부 협업 중심"
    if cross_party_rate < 40:
        return "제한적 초당 협업"
    if cross_party_rate < 60:
        return "혼합형 협업"
    return "초당 협업 비중 높음"


def fallback_cosponsor_network_analysis(context: Dict[str, Any]) -> str:
    member_name = normalize_space(context.get("member_name")) or "해당 의원"
    partner_count = int(context.get("partner_count", 0) or 0)
    total_links = int(context.get("total_links", 0) or 0)
    network_type = normalize_space(context.get("network_type")) or "유형 미확인"
    bipartisanship_type = normalize_space(context.get("bipartisanship_type")) or "초당성 미확인"
    concentration = float(context.get("top3_concentration", 0) or 0)
    cross_party_rate = float(context.get("cross_party_rate", 0) or 0)
    same_party_rate = float(context.get("same_party_rate", 0) or 0)
    top_partners = context.get("top_partners", [])
    committees = context.get("committee_stats", [])
    if not partner_count:
        return "- 공동발의자 문자열에서 반복 협업 네트워크를 구성할 충분한 데이터를 찾지 못했습니다."

    lines = [
        f"- {member_name} 의원의 공동발의 네트워크는 파트너 {partner_count:,}명, 연결 {total_links:,}건으로 구성되며 현재 조회 범위에서는 **{network_type}**으로 볼 수 있습니다.",
        f"- 정당 매칭 기준으로는 당내 협업 {same_party_rate:.1f}%, 당외 협업 {cross_party_rate:.1f}%로 **{bipartisanship_type}**에 가깝습니다.",
        f"- 상위 3명 파트너가 전체 협업 연결의 {concentration:.1f}%를 차지하므로, 이 비중을 통해 협업이 소수에게 집중되는지 확인할 수 있습니다.",
    ]
    if top_partners:
        partner_text = ", ".join(f"{row.get('공동발의자')}({row.get('공동발의건수')}건)" for row in top_partners[:5])
        lines.append(f"- 반복 협업 상위 파트너는 {partner_text}입니다.")
    if committees:
        committee_text = ", ".join(f"{row.get('소관위원회')} {row.get('협업건수')}건" for row in committees[:3])
        lines.append(f"- 협업은 소관위원회 기준 {committee_text}에서 상대적으로 많이 나타납니다.")
    lines.append("- 공동발의는 정책 공조, 입장 표명, 의례적 참여가 섞일 수 있으므로 실제 정치적 연대나 정책 동의로 단정하지 않아야 합니다.")
    return "\n".join(lines)


def generate_cosponsor_network_analysis(context: Dict[str, Any]) -> tuple[str, str]:
    cache_key = [
        "google",
        "cosponsor_network_v1",
        DEFAULT_LLM_MODEL,
        context.get("member_name"),
        json.dumps(
            {
                "partner_count": context.get("partner_count"),
                "total_links": context.get("total_links"),
                "top3_concentration": context.get("top3_concentration"),
                "network_type": context.get("network_type"),
                "member_party": context.get("member_party"),
                "same_party_rate": context.get("same_party_rate"),
                "cross_party_rate": context.get("cross_party_rate"),
                "unknown_party_rate": context.get("unknown_party_rate"),
                "bipartisanship_type": context.get("bipartisanship_type"),
                "top_partners": context.get("top_partners", [])[:10],
                "committee_stats": context.get("committee_stats", [])[:10],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    ]
    cached = load_cache("cosponsor_network_analysis", cache_key)
    if cached:
        return str(cached), "cache"
    google_api_key = get_google_api_key()
    if not google_api_key or ChatGoogleGenerativeAI is None:
        return fallback_cosponsor_network_analysis(context), "rule_fallback"

    system_prompt = (
        "당신은 국회의원 공동발의 네트워크를 해석하는 입법 데이터 분석가입니다. "
        "제공된 통계만 근거로 협업 패턴을 설명하고, 정치적 친소관계나 실제 정책 동의를 단정하지 마세요."
    )
    user_prompt = f"""
아래 JSON context를 바탕으로 공동발의 네트워크를 해석하세요.

규칙:
- 3~5개 Markdown bullet로 작성하세요.
- 이 의원이 소수 반복 협업형, 분산 협업형, 분야 집중 협업형 중 어디에 가까운지 설명하세요.
- 정당 내/정당 외 협업 비중을 바탕으로 초당적 협업 성격을 설명하세요.
- 반복 협업 의원과 주요 소관위원회 분야를 근거로 언급하세요.
- 공동발의는 강한 정책 공조로 단정할 수 없다는 주의문을 포함하세요.
- 정당 정보는 현재/최근 의원 정보 매칭 기준이므로 발의 당시 정당과 다를 수 있다는 한계를 언급하세요.
- 제목은 쓰지 마세요.

context:
{json.dumps(context, ensure_ascii=False, indent=2)}
""".strip()
    llm = ChatGoogleGenerativeAI(model=DEFAULT_LLM_MODEL, temperature=0.15)
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    analysis = normalize_markdown(getattr(response, "content", ""))
    if not analysis:
        return fallback_cosponsor_network_analysis(context), "rule_fallback"
    save_cache("cosponsor_network_analysis", cache_key, analysis)
    return analysis, "llm"


def analyze_cosponsor_network_node(state: MemberActivityState) -> MemberActivityState:
    errors = list(state.get("errors", []))
    context = build_cosponsor_network_context(state)
    partners = context.get("partners", [])
    committees = context.get("committee_stats", [])
    summary = {
        "협업파트너": context.get("partner_count", 0),
        "협업연결": context.get("total_links", 0),
        "상위3명집중도": context.get("top3_concentration", 0),
        "협업유형": context.get("network_type", ""),
        "최다협업자": context.get("top_partner", ""),
        "당내협업": context.get("same_party_links", 0),
        "당외협업": context.get("cross_party_links", 0),
        "정당미확인협업": context.get("unknown_party_links", 0),
        "당내협업비율": context.get("same_party_rate", 0),
        "당외협업비율": context.get("cross_party_rate", 0),
        "정당미확인비율": context.get("unknown_party_rate", 0),
        "초당협업유형": context.get("bipartisanship_type", ""),
    }
    if not bool(state.get("bills_enabled", True)):
        return {
            "cosponsor_network_context": context,
            "cosponsor_network_analysis": "",
            "cosponsor_network_analysis_source": "disabled_bills",
            "cosponsor_network_partners": partners,
            "cosponsor_network_committees": committees,
            "cosponsor_network_summary": summary,
            "errors": errors,
        }
    if not partners:
        return {
            "cosponsor_network_context": context,
            "cosponsor_network_analysis": fallback_cosponsor_network_analysis(context),
            "cosponsor_network_analysis_source": "empty",
            "cosponsor_network_partners": partners,
            "cosponsor_network_committees": committees,
            "cosponsor_network_summary": summary,
            "errors": errors,
        }
    if not bool(state.get("llm_insights_enabled", DEFAULT_LLM_INSIGHTS_ENABLED)):
        return {
            "cosponsor_network_context": context,
            "cosponsor_network_analysis": fallback_cosponsor_network_analysis(context),
            "cosponsor_network_analysis_source": "rule_fallback_disabled",
            "cosponsor_network_partners": partners,
            "cosponsor_network_committees": committees,
            "cosponsor_network_summary": summary,
            "errors": errors,
        }
    if bool(state.get("llm_quota_exhausted", False)):
        return {
            "cosponsor_network_context": context,
            "cosponsor_network_analysis": fallback_cosponsor_network_analysis(context),
            "cosponsor_network_analysis_source": "rule_fallback_quota",
            "cosponsor_network_partners": partners,
            "cosponsor_network_committees": committees,
            "cosponsor_network_summary": summary,
            "errors": errors,
            "llm_quota_exhausted": True,
        }
    try:
        progress_log(state, "공동발의 네트워크 분석 시작")
        analysis, source = generate_cosponsor_network_analysis(context)
        progress_log(state, f"공동발의 네트워크 분석 완료: {source}")
        return {
            "cosponsor_network_context": context,
            "cosponsor_network_analysis": analysis,
            "cosponsor_network_analysis_source": source,
            "cosponsor_network_partners": partners,
            "cosponsor_network_committees": committees,
            "cosponsor_network_summary": summary,
            "errors": errors,
            "llm_quota_exhausted": bool(state.get("llm_quota_exhausted", False)),
        }
    except Exception as error:
        quota_exhausted = is_llm_quota_error(error)
        if quota_exhausted:
            errors.append("공동발의 네트워크 LLM 분석은 Gemini 한도 초과로 규칙 기반 요약으로 대체했습니다.")
        else:
            errors.append(f"공동발의 네트워크 LLM 분석 실패: {sanitize_error(error)}")
        return {
            "cosponsor_network_context": context,
            "cosponsor_network_analysis": fallback_cosponsor_network_analysis(context),
            "cosponsor_network_analysis_source": "rule_fallback_after_error",
            "cosponsor_network_partners": partners,
            "cosponsor_network_committees": committees,
            "cosponsor_network_summary": summary,
            "errors": errors,
            "llm_quota_exhausted": bool(state.get("llm_quota_exhausted", False)) or quota_exhausted,
        }


def member_code_candidates(state: MemberActivityState) -> List[str]:
    codes: List[str] = []
    for member in state.get("member_info", []):
        for key in ["MONA_CD", "NAAS_CD", "NASS_CD"]:
            code = normalize_space(member.get(key))
            if code and code not in codes:
                codes.append(code)
    return codes


def fetch_vote_bill_candidates(client: AssemblyAPIClient, age: int, max_pages: int, limit: int = 0) -> List[Dict[str, Any]]:
    cached = load_cache("vote_bills", [age, max_pages, limit])
    if cached is not None:
        return cached
    rows = client.get_rows(ENDPOINT_VOTE_BILLS, {"AGE": age, "pSize": 100}, max_pages=max_pages)
    rows = dedupe_records(rows, ["BILL_ID", "BILL_NO", "BILL_NAME"])
    rows = sorted(rows, key=lambda row: date_sort_value(row, "PROC_DT", "VOTE_DATE"), reverse=True)
    if limit and limit > 0:
        rows = rows[:limit]
    save_cache("vote_bills", [age, max_pages, limit], rows)
    return rows


def member_votes_term_cache_key(age: int, member_name: str, max_pages: int, max_vote_bills: int) -> List[Any]:
    return [age, normalize_space(member_name), max_pages, max_vote_bills]


def load_member_votes_term_cache(age: int, member_name: str, max_pages: int, max_vote_bills: int) -> Optional[Dict[str, Any]]:
    return load_cache("member_votes_by_term_v1", member_votes_term_cache_key(age, member_name, max_pages, max_vote_bills))


def save_member_votes_term_cache(age: int, member_name: str, max_pages: int, max_vote_bills: int, value: Dict[str, Any]) -> None:
    save_cache("member_votes_by_term_v1", member_votes_term_cache_key(age, member_name, max_pages, max_vote_bills), value)


def normalize_cached_vote_rows(rows: List[Dict[str, Any]], age: int) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        fixed = dict(row)
        fixed["_REQUEST_AGE"] = int(fixed.get("_REQUEST_AGE") or fixed.get("AGE") or age)
        fixed["_VOTE_RESULT"] = normalize_vote_result(fixed)
        normalized.append(fixed)
    return normalized


def normalize_member_vote_rows(rows: List[Dict[str, Any]], age: int) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        fixed = dict(row)
        fixed["_REQUEST_AGE"] = age
        fixed["_VOTE_RESULT"] = normalize_vote_result(fixed)
        normalized.append(fixed)
    return normalized


def filter_member_vote_rows(rows: List[Dict[str, Any]], member_name: str, mona_codes: List[str]) -> List[Dict[str, Any]]:
    normalized_name = normalize_space(member_name).replace("의원", "")
    code_set = {normalize_space(code) for code in mona_codes if normalize_space(code)}
    matched: List[Dict[str, Any]] = []
    for row in rows:
        row_name = member_name_from_row(row).replace("의원", "")
        row_code = member_code_from_row(row)
        if row_name and row_name == normalized_name:
            matched.append(row)
        elif row_code and row_code in code_set:
            matched.append(row)
    return matched


def fetch_member_votes_direct_for_term(
    client: AssemblyAPIClient,
    age: int,
    member_name: str,
    mona_codes: List[str],
    max_pages: int,
    limit: int = 0,
) -> List[Dict[str, Any]]:
    query_attempts: List[Dict[str, Any]] = [{"AGE": age, "HG_NM": member_name, "pSize": 100}]
    query_attempts.extend({"AGE": age, "MONA_CD": code, "pSize": 100} for code in mona_codes)

    for query in query_attempts:
        page_rows = client.get_rows(ENDPOINT_MEMBER_VOTES, query, max_pages=max_pages)
        matched_rows = filter_member_vote_rows(page_rows, member_name, mona_codes)
        if not matched_rows and normalize_space(query.get("HG_NM")) == member_name:
            # Some API responses already apply HG_NM but omit a name/code column in edge cases.
            matched_rows = page_rows
        if not matched_rows:
            continue

        rows = normalize_member_vote_rows(matched_rows, age)
        rows = dedupe_records(rows, ["BILL_ID", "BILL_NO", "HG_NM", "VOTE_DATE", "RESULT_VOTE_MOD"])
        rows = sorted(rows, key=lambda row: date_sort_value(row, "VOTE_DATE", "PROC_DT"), reverse=True)
        if limit and limit > 0:
            rows = rows[:limit]
        return rows
    return []


def fetch_member_vote_for_bill(
    client: AssemblyAPIClient,
    age: int,
    bill: Dict[str, Any],
    member_name: str,
    mona_codes: List[str],
) -> List[Dict[str, Any]]:
    bill_id = normalize_space(bill.get("BILL_ID"))
    bill_no = normalize_space(bill.get("BILL_NO"))
    if not bill_id:
        return []
    cache_key = [age, bill_id, normalize_space(member_name)]
    cached = load_cache("member_vote", cache_key)
    if cached is None:
        legacy_cache_key = [age, bill_id, member_name, ",".join(mona_codes)]
        cached = load_cache("member_vote", legacy_cache_key)
        if cached is not None:
            save_cache("member_vote", cache_key, cached)
    if cached is not None:
        return cached

    query_attempts = [{"AGE": age, "BILL_ID": bill_id, "HG_NM": member_name, "pSize": 300}]
    query_attempts.extend({"AGE": age, "BILL_ID": bill_id, "MONA_CD": code, "pSize": 300} for code in mona_codes)

    rows: List[Dict[str, Any]] = []
    for query in query_attempts:
        page_rows = client.get_rows(ENDPOINT_MEMBER_VOTES, query, max_pages=1)
        matched_rows = filter_member_vote_rows(page_rows, member_name, mona_codes)
        if matched_rows:
            rows = matched_rows
            break
        if page_rows and normalize_space(query.get("HG_NM")) == member_name:
            rows = page_rows
            break

    for row in rows:
        row["_REQUEST_AGE"] = age
        row["_VOTE_RESULT"] = normalize_vote_result(row)
        if not normalize_space(row.get("BILL_ID")):
            row["BILL_ID"] = bill_id
        if not normalize_space(row.get("BILL_NO")):
            row["BILL_NO"] = bill_no
        if not normalize_space(row.get("BILL_NAME")):
            row["BILL_NAME"] = bill.get("BILL_NAME", "")
        if not normalize_space(row.get("VOTE_DATE")):
            row["VOTE_DATE"] = bill.get("PROC_DT", "")
        if not normalize_space(row.get("CURR_COMMITTEE")):
            row["CURR_COMMITTEE"] = bill.get("CURR_COMMITTEE", "")
        if not normalize_space(row.get("BILL_URL")):
            row["BILL_URL"] = bill.get("LINK_URL", "")

    save_cache("member_vote", cache_key, rows)
    return rows


def is_full_bill_vote_cached(age: int, bill: Dict[str, Any]) -> bool:
    bill_id = normalize_space(bill.get("BILL_ID"))
    return bool(bill_id) and is_cache_fresh("bill_votes_full_v2", [age, bill_id])


def fetch_all_votes_for_bill(client: AssemblyAPIClient, age: int, bill: Dict[str, Any]) -> List[Dict[str, Any]]:
    bill_id = normalize_space(bill.get("BILL_ID"))
    bill_no = normalize_space(bill.get("BILL_NO"))
    if not bill_id:
        return []
    # v2: pSize=100으로 여러 페이지를 가져옵니다. pSize=300은 API가 100건으로 잘라 반환할 때 앞 페이지만 캐시할 수 있습니다.
    cached = load_cache("bill_votes_full_v2", [age, bill_id])
    if cached is not None:
        return cached

    rows = client.get_rows(ENDPOINT_MEMBER_VOTES, {"AGE": age, "BILL_ID": bill_id, "pSize": 100}, max_pages=5)
    rows = dedupe_records(rows, ["BILL_ID", "BILL_NO", "HG_NM", "NAAS_NM", "MONA_CD", "VOTE_DATE", "RESULT_VOTE_MOD"])
    for row in rows:
        row["_REQUEST_AGE"] = age
        row["_VOTE_RESULT"] = normalize_vote_result(row)
        row.setdefault("BILL_ID", bill_id)
        row.setdefault("BILL_NO", bill_no)
        row.setdefault("BILL_NAME", bill.get("BILL_NAME", ""))
        row.setdefault("VOTE_DATE", bill.get("PROC_DT", ""))
        row.setdefault("CURR_COMMITTEE", bill.get("CURR_COMMITTEE", ""))
        row.setdefault("BILL_URL", bill.get("LINK_URL", ""))
    save_cache("bill_votes_full_v2", [age, bill_id], rows)
    return rows


def vote_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"찬성": 0, "반대": 0, "기권": 0, "불참": 0, "미확인": 0}
    for row in rows:
        result = normalize_vote_result(row)
        counts[result if result in counts else "미확인"] += 1
    return counts


def determine_party_majority(counts: Dict[str, int]) -> tuple[Optional[str], str]:
    valid_counts = {key: int(counts.get(key, 0)) for key in ["찬성", "반대", "기권"]}
    max_count = max(valid_counts.values()) if valid_counts else 0
    if max_count <= 0:
        return None, "정당 내 유효 표결 없음"
    winners = [key for key, value in valid_counts.items() if value == max_count]
    if len(winners) != 1:
        return None, "정당 다수 입장 동률"
    return winners[0], ""


def infer_member_party_for_age(state: MemberActivityState, age: int) -> str:
    party_counts: Dict[str, int] = {}
    for row in state.get("vote_records", []):
        row_age = int(row.get("AGE") or row.get("_REQUEST_AGE") or 0)
        if row_age != int(age):
            continue
        party = party_name_from_row(row)
        if party:
            party_counts[party] = party_counts.get(party, 0) + 1
    if party_counts:
        return sorted(party_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    for row in state.get("member_info", []):
        party = party_name_from_row(row)
        if party:
            return party
    return ""


def analyze_party_alignment_for_bill(
    client: AssemblyAPIClient,
    age: int,
    bill: Dict[str, Any],
    member_name: str,
    mona_codes: List[str],
    known_target_row: Optional[Dict[str, Any]] = None,
    target_party_hint: str = "",
) -> Dict[str, Any]:
    all_rows = fetch_all_votes_for_bill(client, age, bill)
    target_rows = filter_member_vote_rows(all_rows, member_name, mona_codes)
    if not target_rows and known_target_row:
        target_rows = [known_target_row]
    bill_id = normalize_space(bill.get("BILL_ID"))
    base = {
        "AGE": age,
        "BILL_ID": bill_id,
        "BILL_NO": normalize_space(bill.get("BILL_NO")),
        "BILL_NAME": normalize_space(bill.get("BILL_NAME")),
        "VOTE_DATE": normalize_space(bill.get("PROC_DT") or bill.get("VOTE_DATE")),
        "정당": "",
        "의원표결": "",
        "정당다수입장": "",
        "정당내분포": "",
        "분류": "판정 제외",
        "제외사유": "",
    }
    if not all_rows:
        base["제외사유"] = "의안별 전체 표결정보 없음"
        return base

    target_missing_from_bill_rows = False
    if target_rows:
        target = target_rows[0]
        target_vote = normalize_vote_result(target)
        target_party = party_name_from_row(target) or normalize_space(target_party_hint) or "정당 미확인"
    else:
        target_missing_from_bill_rows = True
        target_vote = "불참(추정)"
        target_party = normalize_space(target_party_hint) or "정당 미확인"

    party_rows = [row for row in all_rows if party_name_from_row(row) == target_party]
    counts = vote_counts(party_rows)
    if target_missing_from_bill_rows and target_party not in {"", "정당 미확인", "무소속"}:
        counts["불참"] = int(counts.get("불참", 0)) + 1
    majority, reason = determine_party_majority(counts)
    party_member_count = len(party_rows) + (1 if target_missing_from_bill_rows and target_party not in {"", "정당 미확인", "무소속"} else 0)

    base.update({
        "정당": target_party,
        "의원표결": target_vote,
        "정당다수입장": majority or "",
        "정당내분포": ordered_vote_count_text(counts),
    })
    if target_party in {"무소속", "정당 미확인"}:
        base["제외사유"] = "정당 다수 입장 분석 대상 아님"
    elif party_member_count <= 1:
        base["제외사유"] = "정당 내 표결자 1명"
    elif target_vote.startswith("불참"):
        base["분류"] = "불참"
    elif not majority:
        base["제외사유"] = reason
    elif target_vote == majority:
        base["분류"] = "일치"
    else:
        base["분류"] = "이탈"
    return base


def summarize_party_alignment_by_term(records: List[Dict[str, Any]], vote_terms: List[int]) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for age in vote_terms:
        term_rows = [row for row in records if int(row.get("AGE") or 0) == int(age)]
        agree = sum(1 for row in term_rows if row.get("분류") == "일치")
        dissent = sum(1 for row in term_rows if row.get("분류") == "이탈")
        absent = sum(1 for row in term_rows if row.get("분류") == "불참")
        excluded = sum(1 for row in term_rows if row.get("분류") == "판정 제외")
        judged = agree + dissent
        summary.append({
            "AGE": age,
            "분석표결": len(term_rows),
            "일치": agree,
            "이탈": dissent,
            "불참": absent,
            "판정제외": excluded,
            "일치율": format_percent(agree, judged),
            "불참포함일치율": format_percent(agree, len(term_rows)),
        })
    return summary


def compact_vote_date(value: Any) -> str:
    text = normalize_space(value)
    digits = re.sub(r"\D", "", text)
    if len(digits) >= 8:
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    if len(text) >= 10:
        return text[:10]
    return text


def build_vote_interpretation_context(state: MemberActivityState) -> Dict[str, Any]:
    limit = int(state.get("party_alignment_vote_limit", DEFAULT_PARTY_ALIGNMENT_LIMIT))
    scope_label = "전체" if limit == 0 else f"최근 {limit}건"
    vote_terms = state.get("vote_terms", [])
    records = state.get("party_alignment_records", [])
    date_ranges: List[Dict[str, Any]] = []
    absent_source: List[Dict[str, Any]] = []
    for age in vote_terms:
        term_records = [row for row in records if int(row.get("AGE") or 0) == int(age)]
        dates = sorted({date for date in (compact_vote_date(row.get("VOTE_DATE")) for row in term_records) if date})
        date_ranges.append({
            "AGE": age,
            "start": dates[0] if dates else "",
            "end": dates[-1] if dates else "",
            "records": len(term_records),
        })
        absent_source.append({
            "AGE": age,
            "api_absent_rows": sum(1 for row in term_records if normalize_space(row.get("의원표결")) == "불참"),
            "estimated_absent_rows": sum(1 for row in term_records if normalize_space(row.get("의원표결")).startswith("불참(추정)")),
        })
    return {
        "member_name": state.get("member_name", ""),
        "vote_terms": vote_terms,
        "analysis_scope": scope_label,
        "party_alignment_vote_limit": limit,
        "vote_summary_by_term": state.get("vote_summary_by_term", []),
        "party_alignment_summary": state.get("party_alignment_summary", []),
        "party_alignment_fetch_stats": state.get("party_alignment_fetch_stats", []),
        "date_ranges": date_ranges,
        "absent_source": absent_source,
        "data_basis": "표결값은 열린국회정보 API의 RESULT_VOTE_MOD 및 BILL_ID별 전체 의원 표결 행을 기준으로 계산했습니다.",
    }


def fallback_vote_interpretation_analysis(context: Dict[str, Any]) -> str:
    member_name = normalize_space(context.get("member_name")) or "해당 의원"
    scope = normalize_space(context.get("analysis_scope")) or "선택 범위"
    summaries = context.get("party_alignment_summary", [])
    date_ranges = context.get("date_ranges", [])
    lines = [f"- 이 해석은 {member_name} 의원의 표결 전체 경력이 아니라 현재 선택된 분석 범위인 대수별 {scope} 기준입니다."]
    for item in date_ranges:
        if item.get("start") and item.get("end"):
            lines.append(f"- {item.get('AGE')}대 분석 표결은 {item.get('start')}~{item.get('end')} 기간에 해당하므로, 특정 시기에 표결이 몰려 있으면 전체 재임 기간의 성향으로 일반화하면 안 됩니다.")
    for item in summaries:
        total = int(item.get("분석표결", 0) or 0)
        absent = int(item.get("불참", 0) or 0)
        judged = int(item.get("일치", 0) or 0) + int(item.get("이탈", 0) or 0)
        if total and absent == total:
            lines.append(f"- {item.get('AGE')}대는 분석 범위 {total}건이 모두 불참으로 분류되어 정당 다수 입장과의 일치·이탈 성향을 판단할 실제 찬성/반대/기권 표결이 없습니다.")
        elif total and judged == 0:
            lines.append(f"- {item.get('AGE')}대는 일치율 계산에 사용할 유효 표결이 없어 일치율을 성향 지표로 해석하기 어렵습니다.")
    lines.append("- 정당 다수 입장은 공식 당론이 아니라 해당 표결에서 같은 정당 의원 다수가 선택한 표결값 기준입니다.")
    return "\n".join(dict.fromkeys(lines))


def generate_vote_interpretation_analysis(context: Dict[str, Any]) -> tuple[str, str]:
    cache_key = [
        "google",
        DEFAULT_LLM_MODEL,
        context.get("member_name"),
        context.get("party_alignment_vote_limit"),
        json.dumps(context.get("vote_terms", []), ensure_ascii=False, sort_keys=True),
        json.dumps(context.get("vote_summary_by_term", []), ensure_ascii=False, sort_keys=True),
        json.dumps(context.get("party_alignment_summary", []), ensure_ascii=False, sort_keys=True),
        json.dumps(context.get("date_ranges", []), ensure_ascii=False, sort_keys=True),
    ]
    cached = load_cache("vote_interpretation_analysis", cache_key)
    if cached:
        return str(cached), "cache"

    google_api_key = get_google_api_key()
    if not google_api_key or ChatGoogleGenerativeAI is None:
        return fallback_vote_interpretation_analysis(context), "rule_fallback"

    system_prompt = (
        "당신은 국회의원 표결 통계를 해석할 때 사용자의 과잉 일반화를 막는 데이터 리서치 어시스턴트입니다. "
        "제공된 JSON 통계만 근거로 해석 주의사항을 작성하고, 통계에 없는 정치적 의도나 사실을 추정하지 마세요."
    )
    user_prompt = f"""
아래 JSON context를 바탕으로 표결 통계 해석 시 주의할 점을 작성하세요.

규칙:
- 분석 범위가 최근 N건인지 전체인지 반드시 언급하세요.
- 날짜 범위가 특정 기간에 몰려 있으면 전체 재임 기간으로 일반화하지 말라고 설명하세요.
- 불참이 많거나 전부 불참이면 일치율/이탈률 해석 한계를 분명히 설명하세요.
- '정당 다수 입장'은 공식 당론이 아니라 같은 정당 의원 다수 표결값 기준임을 언급하세요.
- API 원자료 기준 불참과 추정 불참이 구분되어 있으면 그 차이를 반영하세요.
- 3~5개 Markdown bullet로 간결하게 작성하세요.

context:
{json.dumps(context, ensure_ascii=False, indent=2)}
""".strip()
    llm = ChatGoogleGenerativeAI(model=DEFAULT_LLM_MODEL, temperature=0.1)
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    analysis = normalize_markdown(getattr(response, "content", ""))
    if not analysis:
        return fallback_vote_interpretation_analysis(context), "rule_fallback"
    save_cache("vote_interpretation_analysis", cache_key, analysis)
    return analysis, "google"


def analyze_vote_interpretation_node(state: MemberActivityState) -> MemberActivityState:
    errors = list(state.get("errors", []))
    context = build_vote_interpretation_context(state)
    if not state.get("party_alignment_summary"):
        return {"vote_interpretation_context": context, "vote_interpretation_analysis": "", "vote_interpretation_analysis_source": "empty", "errors": errors}
    if not bool(state.get("llm_insights_enabled", DEFAULT_LLM_INSIGHTS_ENABLED)):
        return {"vote_interpretation_context": context, "vote_interpretation_analysis": fallback_vote_interpretation_analysis(context), "vote_interpretation_analysis_source": "rule_fallback_disabled", "errors": errors}
    if bool(state.get("llm_quota_exhausted", False)):
        return {
            "vote_interpretation_context": context,
            "vote_interpretation_analysis": fallback_vote_interpretation_analysis(context),
            "vote_interpretation_analysis_source": "rule_fallback_quota",
            "errors": errors,
            "llm_quota_exhausted": True,
        }
    try:
        progress_log(state, "표결 통계 해석 메모 생성 시작")
        analysis, source = generate_vote_interpretation_analysis(context)
        progress_log(state, "표결 통계 해석 메모 생성 완료")
        return {
            "vote_interpretation_context": context,
            "vote_interpretation_analysis": analysis,
            "vote_interpretation_analysis_source": source,
            "errors": errors,
            "llm_quota_exhausted": bool(state.get("llm_quota_exhausted", False)),
        }
    except Exception as error:
        quota_exhausted = is_llm_quota_error(error)
        if quota_exhausted:
            errors.append("표결 통계 해석 메모는 Gemini 한도 초과로 규칙 기반 해석으로 대체했습니다.")
        else:
            errors.append(f"표결 통계 해석 메모 생성 실패: {sanitize_error(error)}")
        return {
            "vote_interpretation_context": context,
            "vote_interpretation_analysis": fallback_vote_interpretation_analysis(context),
            "vote_interpretation_analysis_source": "rule_fallback_after_error",
            "errors": errors,
            "llm_quota_exhausted": bool(state.get("llm_quota_exhausted", False)) or quota_exhausted,
        }


def build_activity_briefing_context(state: MemberActivityState) -> Dict[str, Any]:
    party_alignment_summary = state.get("party_alignment_summary", [])
    aligned = sum(int(row.get("일치", 0) or 0) for row in party_alignment_summary)
    dissented = sum(int(row.get("이탈", 0) or 0) for row in party_alignment_summary)
    judged = aligned + dissented
    alignment_rate = round(aligned / judged * 100, 1) if judged else None
    return {
        "member_name": state.get("member_name", ""),
        "analysis_terms": state.get("analysis_terms", []),
        "bill_total": len(state.get("sponsored_bills", [])),
        "representative_total": len(state.get("representative_bills", [])),
        "cosponsored_total": len(state.get("cosponsored_bills", [])),
        "top_bill_committees": state.get("bill_committee_stats", [])[:5],
        "vote_total": len(state.get("vote_records", [])),
        "vote_summary_by_term": state.get("vote_summary_by_term", []),
        "party_alignment_rate": alignment_rate,
        "party_alignment_summary": party_alignment_summary,
        "recent_news_total": len(state.get("recent_news_items", [])),
        "recent_news_titles": [
            {
                "title": normalize_space(item.get("title")),
                "source": normalize_space(item.get("source")),
            }
            for item in state.get("recent_news_items", [])[:5]
        ],
    }


def fallback_activity_briefing(context: Dict[str, Any]) -> str:
    member_name = normalize_space(context.get("member_name")) or "해당 의원"
    bill_total = int(context.get("bill_total", 0) or 0)
    rep_total = int(context.get("representative_total", 0) or 0)
    co_total = int(context.get("cosponsored_total", 0) or 0)
    vote_total = int(context.get("vote_total", 0) or 0)
    recent_news_total = int(context.get("recent_news_total", 0) or 0)
    alignment_rate = context.get("party_alignment_rate")
    top_committees = context.get("top_bill_committees", [])

    lines = [
        f"- {member_name} 의원은 선택된 조회 범위에서 발의법안 {bill_total:,}건, 표결 {vote_total:,}건이 확인됩니다.",
    ]
    if bill_total:
        lines.append(f"- 발의법안은 대표발의 {rep_total:,}건과 공동발의 {co_total:,}건으로 구성되어 입법 활동의 역할 구성을 함께 봐야 합니다.")
    if top_committees:
        committee_text = ", ".join(f"{row.get('소관위원회')} {row.get('건수')}건" for row in top_committees[:3])
        lines.append(f"- 소관위원회 기준으로는 {committee_text} 분야가 상대적으로 많이 나타납니다.")
    if alignment_rate is not None:
        lines.append(f"- 정당 다수 입장과의 일치율은 유효 판정 표결 기준 약 {alignment_rate:.1f}%입니다.")
    if recent_news_total:
        lines.append(f"- 최근 뉴스 검색 결과는 {recent_news_total:,}건이며, 웹 검색 기반 이슈 맥락으로만 참고해야 합니다.")
    lines.append("- 이 브리핑은 원자료를 빠르게 읽기 위한 요약이며, 정치적 성과나 의도를 단정하지 않습니다.")
    return "\n".join(lines[:5])


def generate_activity_briefing(context: Dict[str, Any]) -> tuple[str, str]:
    cache_key = [
        "google",
        "activity_briefing_v1",
        DEFAULT_LLM_MODEL,
        context.get("member_name"),
        json.dumps(context, ensure_ascii=False, sort_keys=True),
    ]
    cached = load_cache("activity_briefing", cache_key)
    if cached:
        return str(cached), "cache"

    google_api_key = get_google_api_key()
    if not google_api_key or ChatGoogleGenerativeAI is None:
        return fallback_activity_briefing(context), "rule_fallback"

    system_prompt = (
        "당신은 국회의원 활동 데이터의 첫 화면 브리핑을 작성하는 데이터 리서치 어시스턴트입니다. "
        "제공된 JSON 지표만 근거로 사용하고, 정치적 성과·의도·호불호를 단정하지 마세요."
    )
    user_prompt = f"""
아래 JSON context를 바탕으로 사용자가 차트와 표를 보기 전에 읽을 핵심 브리핑을 작성하세요.

규칙:
- 3~5개 Markdown bullet로 작성하세요.
- 각 bullet은 한 문장 중심으로 짧게 쓰세요.
- 발의법안 수, 대표/공동발의 비중, 소관위원회 분포, 표결 수, 정당 일치율, 최근 뉴스 수 중 데이터가 있는 항목을 골고루 반영하세요.
- '두드러진다', '확인된다', '참고할 수 있다'처럼 조심스럽게 표현하세요.
- 최근 뉴스는 웹 검색 기반 맥락이며 공식 의정활동 데이터와 구분된다고 언급하세요.
- 데이터에 없는 정치적 의도, 성과 평가, 사실관계 단정은 하지 마세요.
- 제목은 쓰지 마세요.

context:
{json.dumps(context, ensure_ascii=False, indent=2)}
""".strip()
    llm = ChatGoogleGenerativeAI(model=DEFAULT_LLM_MODEL, temperature=0.15)
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    briefing = normalize_markdown(getattr(response, "content", ""))
    if not briefing:
        return fallback_activity_briefing(context), "rule_fallback"
    save_cache("activity_briefing", cache_key, briefing)
    return briefing, "llm"


def generate_activity_briefing_node(state: MemberActivityState) -> MemberActivityState:
    errors = list(state.get("errors", []))
    context = build_activity_briefing_context(state)
    if not bool(state.get("llm_insights_enabled", DEFAULT_LLM_INSIGHTS_ENABLED)):
        return {
            "activity_briefing_context": context,
            "activity_briefing": fallback_activity_briefing(context),
            "activity_briefing_source": "rule_fallback_disabled",
            "errors": errors,
        }
    if bool(state.get("llm_quota_exhausted", False)):
        return {
            "activity_briefing_context": context,
            "activity_briefing": fallback_activity_briefing(context),
            "activity_briefing_source": "rule_fallback_quota",
            "errors": errors,
            "llm_quota_exhausted": True,
        }
    try:
        progress_log(state, "의원 활동 핵심 브리핑 생성 시작")
        briefing, source = generate_activity_briefing(context)
        progress_log(state, f"의원 활동 핵심 브리핑 생성 완료: {source}")
        return {
            "activity_briefing_context": context,
            "activity_briefing": briefing,
            "activity_briefing_source": source,
            "errors": errors,
            "llm_quota_exhausted": bool(state.get("llm_quota_exhausted", False)),
        }
    except Exception as error:
        quota_exhausted = is_llm_quota_error(error)
        if quota_exhausted:
            errors.append("의원 활동 핵심 브리핑은 Gemini 한도 초과로 규칙 기반 요약으로 대체했습니다.")
        else:
            errors.append(f"의원 활동 핵심 브리핑 생성 실패: {sanitize_error(error)}")
        return {
            "activity_briefing_context": context,
            "activity_briefing": fallback_activity_briefing(context),
            "activity_briefing_source": "rule_fallback_after_error",
            "errors": errors,
            "llm_quota_exhausted": bool(state.get("llm_quota_exhausted", False)) or quota_exhausted,
        }


def make_analyze_party_alignment_node(client: AssemblyAPIClient):
    def analyze_party_alignment_node(state: MemberActivityState) -> MemberActivityState:
        errors = list(state.get("errors", []))
        if not bool(state.get("party_alignment_enabled", True)):
            return {"party_alignment_summary": [], "party_alignment_records": [], "party_alignment_dissent_votes": [], "party_alignment_excluded_votes": [], "party_alignment_fetch_stats": [], "errors": errors}

        member_name = state["member_name"]
        max_pages = int(state.get("max_vote_pages") or DEFAULT_MAX_VOTE_PAGES)
        workers = max(1, int(state.get("vote_fetch_workers") or DEFAULT_VOTE_FETCH_WORKERS))
        limit = int(state.get("party_alignment_vote_limit", DEFAULT_PARTY_ALIGNMENT_LIMIT))
        vote_terms = state.get("vote_terms", [])
        mona_codes = member_code_candidates(state)
        known_member_vote_by_bill_id = {normalize_space(row.get("BILL_ID")): row for row in state.get("vote_records", []) if normalize_space(row.get("BILL_ID"))}
        member_party_by_age = {int(age): infer_member_party_for_age(state, int(age)) for age in vote_terms}
        records: List[Dict[str, Any]] = []
        stats: List[Dict[str, Any]] = []

        if not vote_terms:
            progress_log(state, "정당 다수 입장 일치 분석 생략: 표결 조회 대수 없음")
            return {"party_alignment_summary": [], "party_alignment_records": [], "party_alignment_dissent_votes": [], "party_alignment_excluded_votes": [], "party_alignment_fetch_stats": [], "errors": errors}

        scope_label = "전체" if limit == 0 else f"최근 {limit}건"
        progress_log(state, f"정당 다수 입장 일치 분석 시작: {format_terms(vote_terms)}, 대수별 {scope_label}, 병렬 {workers}개")
        for age in vote_terms:
            term_stat = {"AGE": age, "분석대상표결": 0, "BILL_ID캐시적중": 0, "API신규조회대상": 0, "상세조회성공": 0, "상세조회실패": 0, "분석완료": 0}
            try:
                vote_bills = fetch_vote_bill_candidates(client, age, max_pages=max_pages, limit=limit)
                term_stat["분석대상표결"] = len(vote_bills)
                term_stat["BILL_ID캐시적중"] = sum(1 for bill in vote_bills if is_full_bill_vote_cached(age, bill))
                term_stat["API신규조회대상"] = len(vote_bills) - term_stat["BILL_ID캐시적중"]
                progress_log(state, f"{age}대 정당 일치도 BILL_ID 캐시: 적중 {term_stat['BILL_ID캐시적중']}건 / 신규조회대상 {term_stat['API신규조회대상']}건")
                completed = 0
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [
                        executor.submit(
                            analyze_party_alignment_for_bill,
                            client,
                            age,
                            bill,
                            member_name,
                            mona_codes,
                            known_member_vote_by_bill_id.get(normalize_space(bill.get("BILL_ID"))),
                            member_party_by_age.get(int(age), ""),
                        )
                        for bill in vote_bills
                    ]
                    for future in as_completed(futures):
                        completed += 1
                        try:
                            record = future.result()
                            records.append(record)
                            term_stat["상세조회성공"] += 1
                            term_stat["분석완료"] += 1
                        except Exception as error:
                            term_stat["상세조회실패"] += 1
                            errors.append(f"{age}대 정당 일치도 분석 실패: {sanitize_error(error)}")
                        if completed % 10 == 0 or completed == len(futures):
                            progress_log(state, f"{age}대 정당 일치도 분석 진행: {completed}/{len(futures)} (성공 {term_stat['상세조회성공']}, 실패 {term_stat['상세조회실패']})")
            except Exception as error:
                term_stat["상세조회실패"] += 1
                errors.append(f"{age}대 정당 일치도 표결의안 조회 실패: {sanitize_error(error)}")
                progress_log(state, f"{age}대 정당 일치도 분석 실패")
            stats.append(term_stat)

        records = sorted(records, key=lambda row: date_sort_value(row, "VOTE_DATE"), reverse=True)
        summary = summarize_party_alignment_by_term(records, vote_terms)
        progress_log(state, f"정당 다수 입장 일치 분석 완료: 전체 {len(records)}건")
        return {
            "party_alignment_summary": summary,
            "party_alignment_records": records,
            "party_alignment_dissent_votes": [row for row in records if row.get("분류") == "이탈"],
            "party_alignment_excluded_votes": [row for row in records if row.get("분류") == "판정 제외"],
            "party_alignment_fetch_stats": stats,
            "errors": errors,
        }
    return analyze_party_alignment_node


def summarize_votes_by_term(rows: List[Dict[str, Any]], vote_terms: List[int]) -> List[Dict[str, Any]]:
    summary = []
    for age in vote_terms:
        term_rows = [row for row in rows if int(row.get("AGE") or row.get("_REQUEST_AGE") or 0) == int(age)]
        summary.append(
            {
                "AGE": age,
                "전체": len(term_rows),
                "찬성": sum(1 for row in term_rows if row.get("_VOTE_RESULT") == "찬성"),
                "반대": sum(1 for row in term_rows if row.get("_VOTE_RESULT") == "반대"),
                "기권": sum(1 for row in term_rows if row.get("_VOTE_RESULT") == "기권"),
                "불참": sum(1 for row in term_rows if row.get("_VOTE_RESULT") == "불참"),
            }
        )
    return summary


def make_fetch_votes_node(client: AssemblyAPIClient):
    def fetch_votes_node(state: MemberActivityState) -> MemberActivityState:
        errors = list(state.get("errors", []))
        if not bool(state.get("vote_detail_enabled", True)):
            progress_log(state, "표결정보 조회 생략: 옵션 꺼짐")
            return {
                "vote_records": [],
                "yes_votes": [],
                "no_votes": [],
                "abstain_votes": [],
                "absent_votes": [],
                "vote_summary_by_term": [],
                "vote_fetch_stats": [],
                "errors": errors,
            }
        member_name = state["member_name"]
        max_pages = int(state.get("max_vote_pages") or DEFAULT_MAX_VOTE_PAGES)
        max_vote_bills = int(state.get("max_vote_bills_to_scan", DEFAULT_MAX_VOTE_BILLS_TO_SCAN))
        workers = max(1, int(state.get("vote_fetch_workers") or DEFAULT_VOTE_FETCH_WORKERS))
        vote_terms = state.get("vote_terms", SUPPORTED_VOTE_TERMS)
        mona_codes = member_code_candidates(state)
        rows: List[Dict[str, Any]] = []
        vote_fetch_stats: List[Dict[str, Any]] = []

        if not vote_terms:
            progress_log(state, "표결정보 조회 생략: 재임 대수 중 표결 API 지원 대수가 없음")
            return {"vote_records": [], "yes_votes": [], "no_votes": [], "abstain_votes": [], "absent_votes": [], "vote_summary_by_term": [], "vote_fetch_stats": [], "errors": errors}

        scope_label = "전체" if max_vote_bills == 0 else f"최근 {max_vote_bills}건"
        progress_log(state, f"표결정보 조회 시작: {format_terms(vote_terms)}, 대수별 {scope_label}, 병렬 {workers}개, 캐시 TTL {CACHE_TTL_HOURS}시간")
        for age in vote_terms:
            term_stat = {"AGE": age, "표결의안": 0, "상세조회성공": 0, "상세조회실패": 0, "매칭표결": 0, "캐시사용": "N"}
            try:
                cached_term = load_member_votes_term_cache(age, member_name, max_pages, max_vote_bills)
                if cached_term is not None:
                    cached_rows = normalize_cached_vote_rows(cached_term.get("rows", []), age)
                    cached_stat = dict(cached_term.get("stat", {}))
                    term_stat.update({key: cached_stat.get(key, term_stat.get(key)) for key in term_stat})
                    term_stat["매칭표결"] = len(cached_rows)
                    term_stat["캐시사용"] = "Y"
                    rows.extend(cached_rows)
                    vote_fetch_stats.append(term_stat)
                    progress_log(state, f"{age}대 표결정보 대수 단위 캐시 사용: {len(cached_rows)}건")
                    continue

                progress_log(state, f"{age}대 의원명 기준 표결 일괄 조회 시도")
                try:
                    direct_rows = fetch_member_votes_direct_for_term(client, age, member_name, mona_codes, max_pages=max_pages, limit=max_vote_bills)
                except Exception as error:
                    direct_rows = []
                    progress_log(state, f"{age}대 의원명 기준 일괄 조회 실패: BILL_ID별 조회로 전환")
                if direct_rows:
                    term_stat["표결의안"] = len(direct_rows)
                    term_stat["상세조회성공"] = len(direct_rows)
                    term_stat["매칭표결"] = len(direct_rows)
                    term_stat["조회방식"] = "의원명일괄조회"
                    rows.extend(direct_rows)
                    save_member_votes_term_cache(age, member_name, max_pages, max_vote_bills, {"rows": direct_rows, "stat": term_stat})
                    progress_log(state, f"{age}대 의원명 기준 표결 일괄 조회 완료: {len(direct_rows)}건")
                    vote_fetch_stats.append(term_stat)
                    continue

                progress_log(state, f"{age}대 의원명 기준 일괄 조회 결과 없음: BILL_ID별 병렬 조회로 전환")
                vote_bills = fetch_vote_bill_candidates(client, age, max_pages=max_pages, limit=max_vote_bills)
                term_stat["표결의안"] = len(vote_bills)
                term_stat["조회방식"] = "BILL_ID별조회"
                progress_log(state, f"{age}대 표결 의안 후보 {len(vote_bills)}건 확보")
                completed = 0
                term_rows: List[Dict[str, Any]] = []
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [executor.submit(fetch_member_vote_for_bill, client, age, bill, member_name, mona_codes) for bill in vote_bills]
                    for future in as_completed(futures):
                        completed += 1
                        try:
                            member_rows = future.result()
                            term_rows.extend(member_rows)
                            term_stat["상세조회성공"] += 1
                            term_stat["매칭표결"] += len(member_rows)
                        except Exception as error:
                            term_stat["상세조회실패"] += 1
                            errors.append(f"{age}대 표결 상세 조회 실패: {sanitize_error(error)}")
                        if completed % 25 == 0 or completed == len(futures):
                            progress_log(state, f"{age}대 표결 상세 조회 진행: {completed}/{len(futures)} (성공 {term_stat['상세조회성공']}, 실패 {term_stat['상세조회실패']})")
                rows.extend(term_rows)
                if term_stat["상세조회실패"] == 0 and term_stat["상세조회성공"] == term_stat["표결의안"]:
                    save_member_votes_term_cache(age, member_name, max_pages, max_vote_bills, {"rows": term_rows, "stat": term_stat})
                    progress_log(state, f"{age}대 표결정보 대수 단위 캐시 저장: {len(term_rows)}건")
            except Exception as error:
                term_stat["상세조회실패"] += 1
                errors.append(f"{age}대 표결의안 목록 조회 실패: {sanitize_error(error)}")
                progress_log(state, f"{age}대 표결정보 조회 실패")
            vote_fetch_stats.append(term_stat)

        if not rows:
            errors.append(f"표결 의안 조회 범위({format_terms(vote_terms)}, 대수별 {scope_label})에서 해당 의원 표결을 찾지 못했습니다.")
        rows = dedupe_records(rows, ["BILL_ID", "BILL_NO", "HG_NM", "VOTE_DATE", "RESULT_VOTE_MOD"])
        rows = sorted(rows, key=lambda row: date_sort_value(row, "VOTE_DATE", "PROC_DT"), reverse=True)
        vote_summary_by_term = summarize_votes_by_term(rows, vote_terms)
        progress_log(state, f"표결정보 정리 완료: 전체 {len(rows)}건")
        return {
            "vote_records": rows,
            "yes_votes": [row for row in rows if row.get("_VOTE_RESULT") == "찬성"],
            "no_votes": [row for row in rows if row.get("_VOTE_RESULT") == "반대"],
            "abstain_votes": [row for row in rows if row.get("_VOTE_RESULT") == "기권"],
            "absent_votes": [row for row in rows if row.get("_VOTE_RESULT") == "불참"],
            "vote_summary_by_term": vote_summary_by_term,
            "vote_fetch_stats": vote_fetch_stats,
            "errors": errors,
        }
    return fetch_votes_node


def member_party_for_news_query(state: MemberActivityState) -> str:
    for row in state.get("member_info", []):
        party = party_name_from_row(row)
        if party:
            return party
    for row in state.get("vote_records", []):
        party = party_name_from_row(row)
        if party:
            return party
    return ""


def parse_google_news_rss(xml_text: str, limit: int) -> List[Dict[str, Any]]:
    root = ET.fromstring(xml_text)
    channel = root.find("channel")
    if channel is None:
        return []
    items: List[Dict[str, Any]] = []
    for item in channel.findall("item"):
        source_node = item.find("source")
        title = normalize_space(item.findtext("title"))
        source_name = normalize_space(source_node.text if source_node is not None else "")
        if source_name and title.endswith(f" - {source_name}"):
            title = title[: -(len(source_name) + 3)].strip()
        news_item = {
            "title": title,
            "source": source_name,
            "published": normalize_space(item.findtext("pubDate")),
            "url": normalize_space(item.findtext("link")),
            "snippet": strip_html(item.findtext("description")),
        }
        if news_item["title"] and news_item["url"]:
            items.append(news_item)
        if len(items) >= limit:
            break
    return items


def fetch_recent_member_news(member_name: str, party_name: str, limit: int = DEFAULT_RECENT_NEWS_LIMIT) -> tuple[str, List[Dict[str, Any]], str]:
    query = normalize_space(f"{member_name} 국회의원 {party_name}")
    cached = load_cache("recent_member_news", [query, limit], ttl_hours=RECENT_NEWS_CACHE_TTL_HOURS)
    if cached:
        return query, cached, "cache"
    rss_url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=ko&gl=KR&ceid=KR:ko"
    response = requests.get(rss_url, timeout=NEWS_HTTP_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    items = parse_google_news_rss(response.text, limit)
    save_cache("recent_member_news", [query, limit], items)
    return query, items, "google_news_rss"


def fallback_recent_news_analysis(member_name: str, party_name: str, items: List[Dict[str, Any]]) -> str:
    if not items:
        return "최근 뉴스 검색 결과가 없어 의원 관련 이슈를 요약할 수 없습니다."
    lines = [
        f"- 최근 뉴스 {len(items)}건의 제목과 요약을 기준으로 보면, {member_name} 의원은 아래 기사들에서 공개적으로 언급되었습니다.",
    ]
    for item in items[:5]:
        title = item.get("title", "제목 없음")
        source = item.get("source") or "출처 미확인"
        url = item.get("url") or ""
        citation = f"[{source}]({url})" if url else source
        lines.append(f"- {title} - 근거: {citation}")
    lines.append("- 이 섹션은 열린국회정보 API가 아니라 최근 웹 검색 결과 기반입니다. 기사 제목과 요약만으로 의원의 의도, 실제 정책 성과, 논란의 사실관계를 단정할 수 없습니다.")
    return "\n".join(lines)


def generate_recent_news_analysis(member_name: str, party_name: str, query: str, items: List[Dict[str, Any]]) -> tuple[str, str]:
    cache_key = [
        "google",
        "separated_evidence_v3",
        DEFAULT_LLM_MODEL,
        query,
        json.dumps([{k: item.get(k, "") for k in ["title", "source", "published", "url", "snippet"]} for item in items], ensure_ascii=False, sort_keys=True),
    ]
    cached = load_cache("recent_news_analysis", cache_key, ttl_hours=RECENT_NEWS_CACHE_TTL_HOURS)
    if cached:
        return str(cached), "cache"
    google_api_key = get_google_api_key()
    if not google_api_key or ChatGoogleGenerativeAI is None:
        return fallback_recent_news_analysis(member_name, party_name, items), "rule_fallback"
    system_prompt = (
        "당신은 국회의원 관련 최근 공개 이슈를 요약하는 리서치 어시스턴트입니다. "
        "제공된 뉴스 검색 결과만 근거로 요약하고, 사실관계나 의도를 단정하지 마세요. "
        "공식 의정활동 데이터가 아니라 웹 검색 기반 맥락이라는 점을 명확히 구분하세요."
    )
    user_prompt = f"""
다음은 '{query}' 검색으로 수집한 최근 뉴스 최대 10건입니다.

요구 출력:
- 최근 의원 관련 이슈 3~5개를 Markdown bullet로 요약
- 기사 URL이나 언론사 링크 목록은 출력하지 말 것
- 각 bullet은 이슈명과 요약 문장만 포함할 것
- 기사 제목/요약 기반의 해석임을 밝히고, 사실관계 단정 금지
- 열린국회정보 API 기반 공식 의정활동 데이터와 구분된 웹 검색 기반 맥락임을 마지막에 주의문으로 포함

뉴스 JSON:
{json.dumps(items, ensure_ascii=False, indent=2)}
""".strip()
    llm = ChatGoogleGenerativeAI(model=DEFAULT_LLM_MODEL, temperature=0.2)
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    analysis = normalize_markdown(getattr(response, "content", ""))
    if not analysis:
        return fallback_recent_news_analysis(member_name, party_name, items), "rule_fallback"
    save_cache("recent_news_analysis", cache_key, analysis)
    return analysis, "gemini"


def search_recent_member_news_node(state: MemberActivityState) -> MemberActivityState:
    errors = list(state.get("errors", []))
    if not bool(state.get("recent_news_enabled", True)):
        progress_log(state, "최근 의원 관련 뉴스 검색 생략: 옵션 꺼짐")
        return {
            "recent_news_query": "",
            "recent_news_items": [],
            "recent_news_analysis": "",
            "recent_news_analysis_source": "disabled",
            "errors": errors,
            "llm_quota_exhausted": bool(state.get("llm_quota_exhausted", False)),
        }
    member_name = state.get("member_name", "")
    party_name = member_party_for_news_query(state)
    progress_log(state, "최근 의원 관련 뉴스 검색 시작")
    try:
        query, items, fetch_source = fetch_recent_member_news(member_name, party_name, DEFAULT_RECENT_NEWS_LIMIT)
        if bool(state.get("llm_quota_exhausted", False)):
            analysis = fallback_recent_news_analysis(member_name, party_name, items)
            analysis_source = "rule_fallback_quota"
        else:
            analysis, analysis_source = generate_recent_news_analysis(member_name, party_name, query, items)
        progress_log(state, f"최근 의원 관련 뉴스 검색 완료: {len(items)}건, {fetch_source}, {analysis_source}")
        return {
            "recent_news_query": query,
            "recent_news_items": items,
            "recent_news_analysis": analysis,
            "recent_news_analysis_source": analysis_source,
            "errors": errors,
            "llm_quota_exhausted": bool(state.get("llm_quota_exhausted", False)),
        }
    except Exception as error:
        quota_exhausted = is_llm_quota_error(error)
        query = normalize_space(f"{member_name} 국회의원 {party_name}")
        if quota_exhausted:
            errors.append("최근 의원 관련 이슈 LLM 요약은 Gemini 한도 초과로 기사 목록 기반 요약으로 대체했습니다.")
            _, items, _ = fetch_recent_member_news(member_name, party_name, DEFAULT_RECENT_NEWS_LIMIT)
            analysis = fallback_recent_news_analysis(member_name, party_name, items)
            progress_log(state, "최근 의원 관련 뉴스 검색 완료: Gemini 한도 초과, 기사 목록 기반 요약으로 대체")
            return {
                "recent_news_query": query,
                "recent_news_items": items,
                "recent_news_analysis": analysis,
                "recent_news_analysis_source": "rule_fallback_after_quota",
                "errors": errors,
                "llm_quota_exhausted": True,
            }
        errors.append(f"최근 의원 관련 뉴스 검색 실패: {sanitize_error(error)}")
        progress_log(state, "최근 의원 관련 뉴스 검색 실패")
        return {
            "recent_news_query": query,
            "recent_news_items": [],
            "recent_news_analysis": "최근 뉴스 검색에 실패해 의원 관련 이슈를 요약하지 못했습니다.",
            "recent_news_analysis_source": "error",
            "errors": errors,
            "llm_quota_exhausted": bool(state.get("llm_quota_exhausted", False)),
        }


def summarize_node(state: MemberActivityState) -> MemberActivityState:
    member_name = state["member_name"]
    requested_terms = state.get("assembly_terms", DEFAULT_TERMS)
    served_terms = state.get("served_terms", [])
    analysis_terms = state.get("analysis_terms", requested_terms)
    non_served_terms = state.get("non_served_terms", [])
    member_terms = state.get("member_terms", [])
    bill_terms = state.get("bill_terms", [])
    vote_terms = state.get("vote_terms", [])
    all_vote_terms = state.get("all_vote_terms", vote_terms)
    additional_vote_terms = state.get("additional_vote_terms", [])
    unsupported_bill = state.get("unsupported_bill_terms", [])
    unsupported_vote = state.get("unsupported_vote_terms", [])
    bills = state.get("sponsored_bills", [])
    vote_summary = state.get("vote_summary_by_term", [])
    vote_fetch_stats = state.get("vote_fetch_stats", [])
    party_alignment_summary = state.get("party_alignment_summary", [])
    party_alignment_stats = state.get("party_alignment_fetch_stats", [])
    party_alignment_limit = int(state.get("party_alignment_vote_limit", DEFAULT_PARTY_ALIGNMENT_LIMIT))

    lines = [
        f"## {member_name} 의원 활동 조회 결과",
        "",
        "### 의원 프로필",
        state.get("profile_summary", f"{member_name} (프로필 정보 없음)"),
        "",
        "### 조회 범위",
        f"- 의원 재임 대수: {format_terms(served_terms or analysis_terms)}",
        f"- 의원 기본정보 조회 대수: {format_terms(member_terms)}",
        f"- 발의법안 조회 대수: {format_terms(bill_terms)}",
        f"- 본회의 표결정보 초기 조회 대수: {format_terms(vote_terms)}",
        f"- 표결정보 추가 선택 가능 대수: {format_terms(additional_vote_terms)}",
        "",
        "### 발의 법안 요약",
        f"- {bill_result_summary_text(bills)}",
        f"- 대표발의: {len(state.get('representative_bills', []))}건 / 공동발의: {len(state.get('cosponsored_bills', []))}건",
        f"- 처리결과별: {format_counts(count_by_field(bills, 'PROC_RESULT'))}",
        f"- 소관위원회별: {format_counts(count_by_field(bills, 'COMMITTEE'))}",
        "",
        "### 입법 관심 분야 분석",
        state.get("legislative_interest_analysis") or "입법 관심 분야 분석 결과가 없습니다.",
        "",
        "### 표결 요약",
        f"- 전체: {len(state.get('vote_records', []))}건",
        f"- 찬성: {len(state.get('yes_votes', []))}건 / 반대: {len(state.get('no_votes', []))}건 / 기권: {len(state.get('abstain_votes', []))}건 / 불참: {len(state.get('absent_votes', []))}건",
    ]

    for item in vote_summary:
        lines.append(f"- {item['AGE']}대: 찬성 {item['찬성']} / 반대 {item['반대']} / 기권 {item['기권']} / 불참 {item['불참']} (전체 {item['전체']})")

    if party_alignment_summary:
        alignment_scope = "전체" if party_alignment_limit == 0 else f"최근 {party_alignment_limit}건"
        lines.extend(["", "### 정당 다수 입장 일치 분석", f"- 분석 기준: 대수별 {alignment_scope}", "- 공식 당론이 아니라 해당 표결에서 같은 정당 의원 다수가 선택한 표결값 기준입니다."])
        for item in party_alignment_summary:
            lines.append(f"- {item['AGE']}대: 일치 {item['일치']} / 이탈 {item['이탈']} / 불참 {item['불참']} / 판정 제외 {item['판정제외']} (일치율 {item['일치율']}, 불참 포함 {item['불참포함일치율']})")
        if party_alignment_stats:
            total_target = sum(int(item.get("분석대상표결", 0)) for item in party_alignment_stats)
            total_success = sum(int(item.get("상세조회성공", 0)) for item in party_alignment_stats)
            total_failed = sum(int(item.get("상세조회실패", 0)) for item in party_alignment_stats)
            lines.append(f"- 분석 상태: 표결 {total_target}건 중 성공 {total_success}건, 실패 {total_failed}건")
        if state.get("vote_interpretation_analysis"):
            lines.extend(["", "### 표결 통계 해석 시 주의할 점", state.get("vote_interpretation_analysis", "")])

    if vote_fetch_stats:
        total_bills = sum(int(item.get("표결의안", 0)) for item in vote_fetch_stats)
        total_success = sum(int(item.get("상세조회성공", 0)) for item in vote_fetch_stats)
        total_failed = sum(int(item.get("상세조회실패", 0)) for item in vote_fetch_stats)
        lines.extend([
            "",
            "### 표결 조회 상태",
            f"- 표결 의안 {total_bills}건 중 상세조회 성공 {total_success}건, 실패 {total_failed}건",
        ])
        for item in vote_fetch_stats:
            lines.append(f"- {item['AGE']}대: 표결의안 {item['표결의안']} / 성공 {item['상세조회성공']} / 실패 {item['상세조회실패']} / 매칭표결 {item['매칭표결']}")

    if unsupported_bill or unsupported_vote:
        lines.extend(["", "### API 제공 범위 안내"])
        if unsupported_bill:
            lines.append(f"- 발의법안 API(`nzmimeepazxkubdpn`)는 실제 호출 기준 10~22대에서 데이터가 확인됩니다. 재임 대수 중 {format_terms(unsupported_bill)} 발의법안은 API 제공 범위 밖이라 불러오지 않습니다.")
        if unsupported_vote:
            lines.append(f"- 본회의 표결정보 API(`nojepdqqaweusdfbi`)와 의안별 표결현황 API(`ncocpgfiaoituanbr`)는 실제 호출 기준 20~22대에서 데이터가 확인됩니다. 재임 대수 중 {format_terms(unsupported_vote)} 표결정보는 '조회 결과 없음'이 아니라 API 제공 범위 밖이라 불러올 수 없습니다.")
    if additional_vote_terms:
        lines.extend(["", "### 추가 표결 상세 조회 안내"])
        lines.append(f"- 초기 결과는 가장 최근 표결 지원 대수인 {format_terms(vote_terms)}만 조회했습니다. 추가로 {format_terms(additional_vote_terms)} 표결 상세 조회를 선택할 수 있습니다.")
    elif all_vote_terms:
        lines.extend(["", "### 추가 표결 상세 조회 안내"])
        lines.append("- 표결 API가 지원하는 재임 대수는 초기 조회 대수뿐입니다. 추가로 조회할 표결 대수가 없습니다.")
    if state.get("errors"):
        lines.extend(["", "### 조회 중 확인할 점"])
        lines.extend(f"- {error}" for error in state["errors"])
    progress_log(state, "요약 생성 완료")
    return {"summary": "\n".join(lines)}


def build_member_activity_graph(client: AssemblyAPIClient):
    workflow = StateGraph(MemberActivityState)
    workflow.add_node("normalize_user_request", normalize_input_node)
    workflow.add_node("get_member_info", make_fetch_member_info_node(client))
    workflow.add_node("search_member_bills", make_fetch_bills_node(client))
    workflow.add_node("analyze_cosponsor_network", analyze_cosponsor_network_node)
    workflow.add_node("analyze_legislative_interests", analyze_legislative_interests_node)
    workflow.add_node("get_all_member_votes", make_fetch_votes_node(client))
    workflow.add_node("analyze_party_alignment", make_analyze_party_alignment_node(client))
    workflow.add_node("interpret_vote_statistics", analyze_vote_interpretation_node)
    workflow.add_node("search_recent_member_news", search_recent_member_news_node)
    workflow.add_node("generate_activity_briefing", generate_activity_briefing_node)
    workflow.add_node("summarize_member_activity", summarize_node)
    workflow.add_edge(START, "normalize_user_request")
    workflow.add_edge("normalize_user_request", "get_member_info")
    workflow.add_edge("get_member_info", "search_member_bills")
    workflow.add_edge("search_member_bills", "analyze_legislative_interests")
    workflow.add_edge("analyze_legislative_interests", "analyze_cosponsor_network")
    workflow.add_edge("analyze_cosponsor_network", "get_all_member_votes")
    workflow.add_edge("get_all_member_votes", "analyze_party_alignment")
    workflow.add_edge("analyze_party_alignment", "interpret_vote_statistics")
    workflow.add_edge("interpret_vote_statistics", "search_recent_member_news")
    workflow.add_edge("search_recent_member_news", "generate_activity_briefing")
    workflow.add_edge("generate_activity_briefing", "summarize_member_activity")
    workflow.add_edge("summarize_member_activity", END)
    return workflow.compile()
