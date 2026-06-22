from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import html
import io
import json
import math
import os
import re
from typing import Any, Dict, Iterable, List

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    HumanMessage = None
    SystemMessage = None
    ChatGoogleGenerativeAI = None

from src.member_activity_workflow import (
    DEFAULT_ASYNC_OPEN_ASSEMBLY_CONCURRENCY,
    DEFAULT_ASYNC_OPEN_ASSEMBLY_ENABLED,
    DEFAULT_ENABLE_COSPONSOR_SCAN,
    DEFAULT_MAX_BILL_PAGES,
    DEFAULT_MAX_COSPONSOR_SCAN_PAGES,
    DEFAULT_MAX_VOTE_PAGES,
    DEFAULT_PARTY_ALIGNMENT_LIMIT,
    DEFAULT_TERMS,
    DEFAULT_VOTE_FETCH_WORKERS,
    DEFAULT_LLM_MODEL,
    WORKFLOW_VERSION,
    analyze_vote_interpretation_node,
    format_terms,
    make_analyze_party_alignment_node,
    make_fetch_votes_node,
    parse_term_selection,
)
from src.member_directory import (
    DEFAULT_MEMBER_DIRECTORY_TERM,
    MEMBER_DIRECTORY_TERMS,
    fetch_member_directory,
    filter_member_directory,
)
from src.streamlit_runtime import get_client, get_member_activity_app


st.set_page_config(
    page_title="국회의원 의정활동 추적",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_BILL_PAGE_SCAN = DEFAULT_MAX_BILL_PAGES
APP_COSPONSOR_SCAN_PAGES = DEFAULT_MAX_COSPONSOR_SCAN_PAGES
APP_DEFAULT_VOTE_DETAIL_LIMIT = 100
APP_VOTE_DETAIL_SLIDER_MAX = 1000
APP_PARTY_ALIGNMENT_SLIDER_MAX = 500
MEMBER_DIRECTORY_CACHE_TTL_SECONDS = 24 * 60 * 60
MEMBER_DIRECTORY_PAGE_SIZE = 9
APP_DEFAULT_BILL_TERM_SCOPE = "recent"
ALIGNMENT_RESULT_KEYS = [
    "party_alignment_vote_limit",
    "party_alignment_summary",
    "party_alignment_records",
    "party_alignment_dissent_votes",
    "party_alignment_excluded_votes",
    "party_alignment_fetch_stats",
    "vote_interpretation_context",
    "vote_interpretation_analysis",
    "vote_interpretation_analysis_source",
    "llm_quota_exhausted",
]
BILL_QA_EXAMPLE_QUESTIONS = [
    "이 의원이 가장 많이 발의한 분야는 뭐야?",
    "대표발의와 공동발의 비중을 설명해줘.",
    "통과된 법안이 있어?",
    "법사위 관련 법안만 요약해줘.",
    "청년 관련 법안이 있어?",
    "최근 발의한 법안 5개 알려줘.",
]
BILL_QA_STOPWORDS = {
    "이",
    "의원",
    "의원이",
    "가장",
    "많이",
    "발의",
    "발의한",
    "법안",
    "관련",
    "요약",
    "설명",
    "설명해줘",
    "알려줘",
    "뭐야",
    "있어",
    "있는지",
    "최근",
    "통과된",
    "분야",
}
BILL_QA_COMMITTEE_ALIASES = {
    "법사위": "법제사법위원회",
    "과방위": "과학기술정보방송통신위원회",
    "정무위": "정무위원회",
    "행안위": "행정안전위원회",
    "국토위": "국토교통위원회",
    "교육위": "교육위원회",
    "복지위": "보건복지위원회",
    "기재위": "기획재정위원회",
    "문체위": "문화체육관광위원회",
    "국방위": "국방위원회",
    "환노위": "환경노동위원회",
}
BILL_QA_OUT_OF_SCOPE_TERMS = ["표결", "투표", "찬성", "반대", "기권", "불참", "정당", "뉴스", "최근 이슈", "여론", "지지율"]
BILL_QA_STAT_QUESTION_TERMS = [
    "분야",
    "소관",
    "위원회",
    "비중",
    "통계",
    "처리결과",
    "대표발의",
    "공동발의",
    "대표",
    "공동",
    "통과",
    "가결",
    "계류",
    "폐기",
    "전체",
    "총",
    "몇",
]


@st.cache_resource(show_spinner=False)
def get_alignment_reanalysis_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="party-alignment-reanalysis")


@st.cache_data(ttl=MEMBER_DIRECTORY_CACHE_TTL_SECONDS, show_spinner=False)
def load_member_directory_cached(term: int | None = None) -> List[Dict[str, Any]]:
    return fetch_member_directory(get_client(), term=term)


def build_member_party_lookup_from_directory(members: Iterable[Dict[str, Any]]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for member in members:
        name = str(member.get("name") or "").strip()
        party = str(member.get("party") or "").strip()
        terms = [int(term) for term in member.get("terms", []) if str(term).isdigit()]
        if not name or not party:
            continue
        existing = lookup.get(name)
        if not existing or DEFAULT_MEMBER_DIRECTORY_TERM in terms:
            lookup[name] = party
    return lookup


def get_member_party_lookup_cached() -> Dict[str, str]:
    try:
        return build_member_party_lookup_from_directory(load_member_directory_cached(None))
    except Exception:
        return {}


def as_dataframe(rows: Iterable[Dict[str, Any]], columns: List[str] | None = None) -> pd.DataFrame:
    df = pd.DataFrame(list(rows or []))
    if df.empty:
        return df
    if columns:
        existing = [column for column in columns if column in df.columns]
        return df[existing]
    return df


def render_dataframe(
    title: str,
    rows: Iterable[Dict[str, Any]],
    columns: List[str] | None = None,
    *,
    height: int = 360,
) -> None:
    st.subheader(title)
    df = as_dataframe(rows, columns)
    if df.empty:
        st.info("조회된 데이터가 없습니다.")
        return
    st.dataframe(df, use_container_width=True, height=height)
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        f"{title} CSV 다운로드",
        data=csv,
        file_name=f"{title.replace(' ', '_')}.csv",
        mime="text/csv",
        key=f"download_{title}",
    )


def render_vote_summary_text(title: str, rows: Iterable[Dict[str, Any]]) -> None:
    st.subheader(title)
    summary_rows = list(rows or [])
    if not summary_rows:
        st.info("표결 요약 데이터가 없습니다.")
        return

    for row in summary_rows:
        age = row.get("AGE", "-")
        total = as_int(row.get("전체"))
        yes = as_int(row.get("찬성"))
        no = as_int(row.get("반대"))
        abstain = as_int(row.get("기권"))
        absent = as_int(row.get("불참"))
        st.markdown(
            f"- **{age}대**: 전체 **{total:,}건** · "
            f"찬성 {yes:,}건 / 반대 {no:,}건 / 기권 {abstain:,}건 / 불참 {absent:,}건"
        )


def render_vote_fetch_status_text(title: str, rows: Iterable[Dict[str, Any]]) -> None:
    st.subheader(title)
    status_rows = list(rows or [])
    if not status_rows:
        st.info("표결 조회 상태 데이터가 없습니다.")
        return

    total_bills = sum(as_int(row.get("표결의안")) for row in status_rows)
    total_success = sum(as_int(row.get("상세조회성공")) for row in status_rows)
    total_failed = sum(as_int(row.get("상세조회실패")) for row in status_rows)
    total_matched = sum(as_int(row.get("매칭표결")) for row in status_rows)
    st.markdown(
        f"표결 의안 **{total_bills:,}건** 중 상세조회 **{total_success:,}건 성공**, "
        f"**{total_failed:,}건 실패**, 의원 매칭 표결 **{total_matched:,}건**입니다."
    )

    for row in status_rows:
        age = row.get("AGE", "-")
        cache_status = row.get("캐시사용")
        cache_text = f" · 캐시 {cache_status}" if cache_status not in (None, "") else ""
        st.markdown(
            f"- **{age}대**: 표결의안 {as_int(row.get('표결의안')):,}건 / "
            f"상세조회 성공 {as_int(row.get('상세조회성공')):,}건 / "
            f"실패 {as_int(row.get('상세조회실패')):,}건 / "
            f"매칭표결 {as_int(row.get('매칭표결')):,}건{cache_text}"
        )


def render_alignment_summary_text(title: str, rows: Iterable[Dict[str, Any]]) -> None:
    st.subheader(title)
    summary_rows = list(rows or [])
    if not summary_rows:
        st.info("정당 다수 입장 일치 분석 요약 데이터가 없습니다.")
        return

    for row in summary_rows:
        age = row.get("AGE", "-")
        analyzed = as_int(row.get("분석표결"))
        matched = as_int(row.get("일치"))
        dissented = as_int(row.get("이탈"))
        absent = as_int(row.get("불참"))
        excluded = as_int(row.get("판정제외"))
        match_rate = row.get("일치율", "-")
        match_rate_with_absent = row.get("불참포함일치율", "-")
        st.markdown(
            f"- **{age}대**: 분석표결 **{analyzed:,}건** · "
            f"일치 {matched:,}건 / 이탈 {dissented:,}건 / 불참 {absent:,}건 / "
            f"판정 제외 {excluded:,}건 "
            f"(일치율 {match_rate}, 불참 포함 {match_rate_with_absent})"
        )


def render_alignment_fetch_status_text(title: str, rows: Iterable[Dict[str, Any]]) -> None:
    st.subheader(title)
    status_rows = list(rows or [])
    if not status_rows:
        st.info("정당 일치도 분석 상태 데이터가 없습니다.")
        return

    total_targets = sum(as_int(row.get("분석대상표결")) for row in status_rows)
    total_cache_hits = sum(as_int(row.get("BILL_ID캐시적중")) for row in status_rows)
    total_api_targets = sum(as_int(row.get("API신규조회대상")) for row in status_rows)
    total_success = sum(as_int(row.get("상세조회성공")) for row in status_rows)
    total_failed = sum(as_int(row.get("상세조회실패")) for row in status_rows)
    total_completed = sum(as_int(row.get("분석완료")) for row in status_rows)
    st.markdown(
        f"정당 일치도 분석 대상 **{total_targets:,}건** 중 "
        f"캐시 적중 **{total_cache_hits:,}건**, 신규 API 조회 대상 **{total_api_targets:,}건**입니다. "
        f"상세조회는 **{total_success:,}건 성공**, **{total_failed:,}건 실패**, "
        f"분석 완료 **{total_completed:,}건**입니다."
    )

    for row in status_rows:
        age = row.get("AGE", "-")
        st.markdown(
            f"- **{age}대**: 분석대상 {as_int(row.get('분석대상표결')):,}건 / "
            f"캐시적중 {as_int(row.get('BILL_ID캐시적중')):,}건 / "
            f"신규조회대상 {as_int(row.get('API신규조회대상')):,}건 / "
            f"상세조회 성공 {as_int(row.get('상세조회성공')):,}건 / "
            f"실패 {as_int(row.get('상세조회실패')):,}건 / "
            f"분석완료 {as_int(row.get('분석완료')):,}건"
        )


def render_alignment_method_note() -> None:
    with st.expander("정당 일치도 계산 방식 보기", expanded=False):
        st.markdown(
            """
**정당 다수 입장**은 공식적인 당론이 아니라, 같은 정당 소속 의원들이 해당 표결에서 실제로 가장 많이 선택한 표결값을 뜻합니다.

계산 방식은 다음과 같습니다.

1. 분석 대상 의안의 `BILL_ID`를 기준으로 의원별 본회의 표결정보를 조회합니다.
2. 해당 의원의 표결값을 `찬성`, `반대`, `기권`, `불참`으로 정규화합니다.
3. 같은 정당 의원들의 표결값을 모아 `찬성/반대/기권/불참` 분포를 계산합니다.
4. 정당 다수 입장은 `찬성/반대/기권` 중 가장 많은 값으로 정합니다. 불참은 다수 입장 계산에서 제외합니다.
5. 대상 의원의 표결이 정당 다수 입장과 같으면 `일치`, 다르면 `이탈`로 분류합니다.
6. 대상 의원이 불참한 표결은 `불참`으로 분류합니다.
7. 무소속, 정당 내 유효 표결 없음, 찬성/반대/기권 동률처럼 다수 입장을 정하기 어려운 경우는 `판정 제외`로 분리합니다.

따라서 이 지표는 “공식 당론 준수율”이 아니라 **같은 정당 의원들의 실제 표결 다수 흐름과 얼마나 비슷하게 투표했는지**를 보는 참고 지표입니다.
            """
        )


def render_theoretical_background_sidebar() -> None:
    st.divider()
    st.subheader("📚 이론 배경")

    with st.expander("🏛️ 의정활동이란?", expanded=False):
        st.markdown(
            """
의정활동은 국회의원이 헌법적 권한을 바탕으로 개별적인 자율성과 개성을 발휘하여 원내외 활동을 수행하고, 국가와 국민 사이를 매개하여 사회적 합의를 도출해가는 과정입니다.

📌 참고: 이재윤·김의진·박태선(2024), 「국회의원 의정활동 기록의 개념과 특성에 관한 고찰」
            """
        )

    with st.expander("🗂️ 의정활동 기록의 의미", expanded=False):
        st.markdown(
            """
국회의정활동 기록은 국가와 지역에 관련된 다양하고 복합적인 사회적 현안·의제를 포함하는 동시에 의원의 관점과 개성이 연결된 결과물입니다.

따라서 발의법안이나 표결 기록은 단순한 행정 데이터를 넘어서 의원이 어떤 의제에 관심을 두고 어떤 판단을 반복해왔는지 보여주는 기록으로 볼 수 있습니다.

📌 참고: 이재윤·김의진·박태선(2024), 「국회의원 의정활동 기록의 개념과 특성에 관한 고찰」
            """
        )

    with st.expander("🗣️ 공론장과 정치 커뮤니케이션", expanded=False):
        st.markdown(
            """
- 하버마스(2006)는 공론장을 국가와 사회 사이를 매개하는 의사소통 체계로 설명합니다.

- 숙의 민주주의가 작동하려면 정치적 의견 형성 과정에서 공개성과 투명성이 보장되어야 하며, 시민들이 공적 쟁점에 접근하고 이해할 수 있어야 합니다.

- 이 웹은 단순한 정보 조회 서비스를 넘어, 제도정치와 시민사회 사이의 정보 격차를 줄이는 정치 커뮤니케이션 도구입니다.

📌 참고: Habermas(2006), 「Political Communication in Media Society:
Does Democracy Still Enjoy an Epistemic
Dimension? The Impact of Normative
Theory on Empirical Research」

            """
        )

    with st.expander("📰 감시저널리즘과 민주적 책임성", expanded=False):
        st.markdown(
            """
- Strömbäck(2005)은 민주주의와 저널리즘의 관계를 사회계약으로 설명합니다. 민주주의는 정보의 흐름, 공적 토론, 독립적인 권력 감시 기능을 필요로 하며, 저널리즘은 시민의 자율적 판단을 위한 정보를 제공하고 권력 남용을 감시합니다.

- 이 웹은 감시저널리즘의 실천을 보조하고, 민주적 책임성과 근거 있는 정치적 토론을 강화하는 정치 커뮤니케이션 도구입니다.
  - 시민은 의원의 이미지나 정쟁적 발언이 아니라 객관적 자료를 바탕으로 의정활동을 평가할 수 있습니다.
  - 언론인은 표결 이탈, 불참 비중, 반복적 공동발의 관계, 특정 정책 분야에 집중된 입법 활동을 확인해 후속 취재의 단서를 얻을 수 있습니다.
  - 참여 민주주의와 숙의 민주주의 관점에서, 이 웹은 산재된 의정활동 데이터를 시민이 이해할 수 있는 쟁점 중심 정보로 재구성하여 시민과 언론인의 근거 있는 정치적 토론을 돕습니다.

📌 참고: Strömbäck(2005), 「In Search of a Standard: Four Models of Democracy and Their Normative Implications for Journalism」
            """
        )

    with st.expander("⚖️ 국회의원의 복합적 지위", expanded=False):
        st.markdown(
            """
국회의원은 단순히 개인 정치인이 아니라 복합적 지위를 가진 행위자입니다.

국회의원은 헌법상 다음과 같은 세 가지 지위를 가집니다.

- 국회라는 합의체 결정기관의 구성원
- 지역구민을 넘어 국민 전체를 대표하는 대표자
- 정당 당원으로서 정당정치에 참여하는 행위자

📌 참고: 이재윤·김의진·박태선(2024), 「국회의원 의정활동 기록의 개념과 특성에 관한 고찰」

            """
        )

    with st.expander("🤝 정당 다수 입장 일치도의 중요성", expanded=False):
        st.markdown(
            """
- 국회의원은 자신의 활동을 통해 당론을 직접 만들어가는 주체인 동시에 그 당론을 지침 삼아 정당과 발맞추어 활동하는 존재입니다. 따라서 의원 개인의 소신과 정당의 전략은 서로 떼려야 뗄 수 없이 긴밀하게 연결되어 있습니다.

- 국회의원의 의정활동을 제대로 이해하기 위해서는 개별 의원의 표결이 단순한 개인 선택인지, 정당 다수 입장과의 결속 속에서 이루어진 선택인지 함께 살펴볼 필요가 있습니다.

- `정당 다수 입장 일치도 분석`은 의원의 표결이 같은 정당의 다수 입장과 얼마나 일치하거나 이탈하는지를 보여줍니다. 이를 통해 국회의원의 의정활동이 개인적 판단과 정당정치의 관계 속에서 어떻게 형성되는지 파악하도록 도울 수 있습니다.

            """
        )


def toggle_directory_member(member_name: str, member_key: str) -> None:
    if st.session_state.get("selected_directory_member_key") == member_key:
        st.session_state["member_name_input"] = ""
        st.session_state.pop("selected_directory_member", None)
        st.session_state.pop("selected_directory_member_key", None)
        return
    st.session_state["member_name_input"] = member_name
    st.session_state["selected_directory_member"] = member_name
    st.session_state["selected_directory_member_key"] = member_key


def render_member_directory_styles() -> None:
    st.markdown(
        """
        <style>
        .member-card {
            height: 324px;
            padding: 16px;
            border: 1px solid #dbe4ee;
            border-radius: 18px;
            background:
                linear-gradient(145deg, rgba(255,255,255,0.96), rgba(244,248,252,0.92));
            box-shadow: 0 8px 24px rgba(34, 56, 84, 0.08);
            box-sizing: border-box;
            overflow: hidden;
        }
        .member-card--selected {
            border-color: #5091f1;
            background:
                linear-gradient(145deg, rgba(239,246,255,0.98), rgba(248,251,255,0.94));
            box-shadow: 0 8px 24px rgba(34, 56, 84, 0.08);
        }
        .member-card__badge {
            display: inline-block;
            margin-bottom: 8px;
            padding: 4px 9px;
            border-radius: 999px;
            background: rgba(80, 145, 241, 0.14);
            color: #2563eb;
            font-size: 0.78rem;
            font-weight: 800;
        }
        .member-card__badge-spacer {
            height: 30px;
            margin-bottom: 8px;
        }
        .member-card__top {
            display: flex;
            gap: 14px;
            align-items: center;
            margin-bottom: 14px;
        }
        .member-card__image {
            width: 76px;
            height: 88px;
            object-fit: cover;
            border-radius: 16px;
            border: 1px solid #d7e2ee;
            background: #eef3f8;
        }
        .member-card__placeholder {
            width: 76px;
            height: 88px;
            border-radius: 16px;
            border: 1px solid #d7e2ee;
            background: linear-gradient(145deg, #eaf2fb, #f7fafc);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #6b7a90;
            font-weight: 700;
        }
        .member-card__name {
            margin: 0;
            font-size: 1.1rem;
            font-weight: 800;
            color: #1e344d;
        }
        .member-card__party {
            display: inline-block;
            margin-top: 6px;
            padding: 4px 9px;
            border-radius: 999px;
            background: rgba(80, 145, 241, 0.12);
            color: #315f9f;
            font-size: 0.82rem;
            font-weight: 700;
        }
        .member-card__row {
            margin: 7px 0;
            color: #4d5b6d;
            font-size: 0.92rem;
            line-height: 1.42;
            display: grid;
            grid-template-columns: 56px minmax(0, 1fr);
            gap: 8px;
            align-items: start;
        }
        .member-card__label {
            color: #8190a3;
            font-weight: 700;
        }
        .member-card__value {
            min-width: 0;
            overflow: hidden;
            display: -webkit-box;
            -webkit-box-orient: vertical;
            -webkit-line-clamp: 2;
        }
        .member-card__value--committee {
            -webkit-line-clamp: 2;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def member_directory_sort_key(member: Dict[str, Any], sort_label: str) -> tuple:
    if sort_label == "정당순":
        return (member.get("party", ""), member.get("name", ""))
    if sort_label == "지역순":
        return (member.get("region_bucket", ""), member.get("region", ""), member.get("name", ""))
    if sort_label == "위원회순":
        return (member.get("committee", ""), member.get("name", ""))
    return (member.get("name", ""), member.get("party", ""))


def change_member_directory_page(key_prefix: str, delta: int) -> None:
    page_key = f"{key_prefix}_page"
    st.session_state[page_key] = max(1, int(st.session_state.get(page_key, 1)) + delta)


def show_member_directory(key_prefix: str = "member_directory") -> None:
    st.session_state["show_member_directory"] = True
    st.session_state[f"{key_prefix}_term"] = f"{DEFAULT_MEMBER_DIRECTORY_TERM}대"
    st.session_state[f"{key_prefix}_page"] = 1


def render_member_card(member: Dict[str, Any], *, key_prefix: str) -> None:
    selected_name = str(st.session_state.get("selected_directory_member") or "")
    selected_key = str(st.session_state.get("selected_directory_member_key") or "")
    member_name = str(member.get("name") or "")
    member_key = str(member.get("_key") or member_name)
    is_selected = bool(member_key and selected_key == member_key) or bool(not selected_key and member_name and selected_name == member_name)
    name = html.escape(str(member.get("name") or "이름 미확인"))
    party = html.escape(str(member.get("party") or "정당 미확인"))
    region = html.escape(str(member.get("region") or "지역 미확인"))
    committee = html.escape(str(member.get("committee") or "위원회 미확인"))
    term_label = html.escape(str(member.get("term_label") or "대수 미확인"))
    reelection = html.escape(str(member.get("reelection") or "선수 미확인"))
    image_src = html.escape(str(member.get("image_url") or "").strip())
    image_html = (
        f"<img class='member-card__image' src='{image_src}' alt='{name} 의원 사진' loading='lazy' decoding='async' data-member-key='{html.escape(member_key)}'>"
        if image_src
        else f"<div class='member-card__placeholder'>{name[:1]}</div>"
    )
    card_class = "member-card member-card--selected" if is_selected else "member-card"
    selected_badge = "<div class='member-card__badge'>현재 선택</div>" if is_selected else "<div class='member-card__badge-spacer'></div>"

    st.markdown(
        f"""
        <div class="{card_class}">
            {selected_badge}
            <div class="member-card__top">
                {image_html}
                <div>
                    <p class="member-card__name">{name}</p>
                    <span class="member-card__party">{party}</span>
                </div>
            </div>
            <div class="member-card__row"><span class="member-card__label">지역</span><span class="member-card__value" title="{region}">{region}</span></div>
            <div class="member-card__row"><span class="member-card__label">위원회</span><span class="member-card__value member-card__value--committee" title="{committee}">{committee}</span></div>
            <div class="member-card__row"><span class="member-card__label">대수</span><span class="member-card__value" title="{term_label}">{term_label}</span></div>
            <div class="member-card__row"><span class="member-card__label">선수</span><span class="member-card__value" title="{reelection}">{reelection}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.button(
        "선택 해제" if is_selected else f"{member.get('name', '의원')} 선택",
        key=f"{key_prefix}_{member_key}",
        use_container_width=True,
        on_click=toggle_directory_member,
        args=(member_name, member_key),
    )


def render_member_directory_panel(*, key_prefix: str = "member_directory") -> None:
    render_member_directory_styles()

    try:
        all_members = load_member_directory_cached(None)
    except Exception as error:
        st.error(f"의원 목록을 불러오지 못했습니다: {error}")
        st.caption("목록 조회가 실패해도 왼쪽 사이드바에서 의원 이름을 직접 입력해 분석할 수 있습니다.")
        return

    if not all_members:
        st.info("조회된 의원 목록이 없습니다.")
        return

    available_terms = [term for term in MEMBER_DIRECTORY_TERMS if 2 <= int(term) <= 22]
    term_options = ["전체"] + [f"{term}대" for term in available_terms]
    default_term_index = term_options.index(f"{DEFAULT_MEMBER_DIRECTORY_TERM}대")
    term_key = f"{key_prefix}_term"
    if st.session_state.get(term_key) not in term_options:
        st.session_state[term_key] = f"{DEFAULT_MEMBER_DIRECTORY_TERM}대"
    selected_term_label = st.selectbox("대수", term_options, index=default_term_index, key=term_key)
    st.caption("2대 ~ 22대 의원 목록 조회가 가능합니다.")
    selected_term = None if selected_term_label == "전체" else int(str(selected_term_label).replace("대", ""))
    members = [
        member
        for member in all_members
        if selected_term is None or selected_term in [int(term) for term in member.get("terms", [])]
    ]

    if not members:
        st.info("선택한 대수에 해당하는 의원이 없습니다.")
        return

    selected_name = str(st.session_state.get("selected_directory_member") or "").strip()
    if selected_name:
        st.success(f"{selected_name} 의원이 선택되었습니다. 왼쪽 사이드바에서 분석 시작을 누르세요.")

    parties = sorted({str(member.get("party")) for member in members if member.get("party")})
    regions = sorted({str(member.get("region_bucket")) for member in members if member.get("region_bucket")})
    committees = sorted({str(member.get("committee")) for member in members if member.get("committee")})

    search_col, party_col, region_col = st.columns([1.3, 1, 1])
    with search_col:
        query = st.text_input(
            "이름·지역·위원회 검색",
            key=f"{key_prefix}_query",
        )
    with party_col:
        selected_parties = st.multiselect("정당", parties, key=f"{key_prefix}_parties")
    with region_col:
        selected_regions = st.multiselect("지역", regions, key=f"{key_prefix}_regions")

    option_col, sort_col = st.columns([1.4, 1])
    with option_col:
        selected_committees = st.multiselect("위원회", committees, key=f"{key_prefix}_committees")
    with sort_col:
        sort_label = st.selectbox("정렬", ["이름순", "정당순", "지역순", "위원회순"], key=f"{key_prefix}_sort")

    filtered = filter_member_directory(
        members,
        query=query,
        parties=selected_parties,
        regions=selected_regions,
        committees=selected_committees,
    )
    filtered = sorted(filtered, key=lambda member: member_directory_sort_key(member, sort_label))

    term_caption = "전체 대수" if selected_term is None else f"{selected_term}대"
    st.caption(
        f"{term_caption} 의원 {len(members)}명 중 "
        f"{len(filtered)}명이 현재 조건에 맞습니다. 목록은 24시간 캐시됩니다."
    )

    if not filtered:
        st.info("조건에 맞는 의원이 없습니다.")
        return

    page_key = f"{key_prefix}_page"
    current_page = int(st.session_state.get(page_key, 1))
    total_pages = max(1, (len(filtered) + MEMBER_DIRECTORY_PAGE_SIZE - 1) // MEMBER_DIRECTORY_PAGE_SIZE)
    current_page = min(max(1, current_page), total_pages)
    st.session_state[page_key] = current_page

    start_index = (current_page - 1) * MEMBER_DIRECTORY_PAGE_SIZE
    visible = filtered[start_index : start_index + MEMBER_DIRECTORY_PAGE_SIZE]
    columns = st.columns(3)
    for index, member in enumerate(visible):
        with columns[index % 3]:
            render_member_card(member, key_prefix=f"{key_prefix}_select")

    nav_left, nav_center, nav_right = st.columns([1, 1, 1])
    with nav_left:
        st.button(
            "이전",
            disabled=current_page <= 1,
            use_container_width=True,
            key=f"{key_prefix}_prev",
            on_click=change_member_directory_page,
            args=(key_prefix, -1),
        )
    with nav_center:
        st.markdown(
            f"<div style='text-align:center;padding-top:0.45rem;font-weight:700;color:#334155;'>{current_page} / {total_pages}</div>",
            unsafe_allow_html=True,
        )
    with nav_right:
        st.button(
            "다음",
            disabled=current_page >= total_pages,
            use_container_width=True,
            key=f"{key_prefix}_next",
            on_click=change_member_directory_page,
            args=(key_prefix, 1),
        )


def render_member_directory_section() -> None:
    if "show_member_directory" not in st.session_state:
        st.session_state["show_member_directory"] = False

    directory_expanded = bool(st.session_state["show_member_directory"]) or not st.session_state.get("member_activity_result")
    with st.expander("의원 목록 조회", expanded=directory_expanded):
        st.caption("카드에서 의원을 선택하면 왼쪽 사이드바의 국회의원 이름 입력칸에 반영됩니다.")
        st.caption("의원 목록을 처음 조회할 때는 API 호출과 캐시 생성으로 아주 잠깐 시간이 걸릴 수 있습니다.")
        st.button("의원 목록 불러오기", use_container_width=True, on_click=show_member_directory)
        if st.session_state["show_member_directory"]:
            render_member_directory_panel()
        else:
            st.info("의원 목록은 처음 불러올 때 열린국회정보 API를 호출하고, 이후 24시간 동안 캐시를 사용합니다.")


def metric_value(rows: List[Dict[str, Any]], key: str) -> int:
    return sum(int(row.get(key, 0) or 0) for row in rows)


def as_int(value: Any) -> int:
    text = str(value or "").replace(",", "").strip()
    try:
        return int(float(text))
    except ValueError:
        return 0


def chart_has_data(items: List[Dict[str, Any]]) -> bool:
    return any(as_int(item.get("value")) > 0 for item in items)


def clean_vote_interpretation_text(value: Any) -> str:
    text = str(value or "").strip()
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        heading_text = stripped.lstrip("#").strip()
        if heading_text in {"표결 통계 해석 시 주의할 점"}:
            continue
        if "정당 다수 입장" in stripped and ("공식 당론" in stripped or "공식적인 당론" in stripped):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def clean_recent_news_analysis_text(value: Any) -> str:
    lines: List[str] = []
    for line in str(value or "").splitlines():
        stripped = line.strip()
        if re.match(r"^[-*]\s*\[[^\]]+\]\([^)]+\)\s*$", stripped):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def compact_bill_for_qa(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "역할": normalize_text(row.get("_ROLE")),
        "대수": normalize_text(row.get("AGE") or row.get("_REQUEST_AGE")),
        "의안번호": normalize_text(row.get("BILL_NO")),
        "법안명": normalize_text(row.get("BILL_NAME")),
        "발의일": normalize_text(row.get("PROPOSE_DT") or row.get("PPSL_DT")),
        "소관위원회": normalize_text(row.get("COMMITTEE")),
        "처리결과": normalize_text(row.get("PROC_RESULT")) or "미확인",
        "링크": normalize_text(row.get("DETAIL_LINK") or row.get("LINK_URL")),
    }


def compact_bill_for_recent_qa(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "역할": normalize_text(row.get("_ROLE")),
        "대수": normalize_text(row.get("AGE") or row.get("_REQUEST_AGE")),
        "의안번호": normalize_text(row.get("BILL_NO")),
        "법안명": normalize_text(row.get("BILL_NAME")),
        "발의일": normalize_text(row.get("PROPOSE_DT") or row.get("PPSL_DT")),
        "소관위원회": normalize_text(row.get("COMMITTEE")),
        "링크": normalize_text(row.get("DETAIL_LINK") or row.get("LINK_URL")),
    }


def bill_qa_search_terms(question: str) -> List[str]:
    normalized = normalize_text(question)
    terms: List[str] = []
    for source, target in BILL_QA_COMMITTEE_ALIASES.items():
        if source in normalized:
            terms.extend([source, target])
    for token in re.findall(r"[가-힣A-Za-z0-9]{2,}", normalized):
        if token in BILL_QA_STOPWORDS:
            continue
        terms.append(token)
    result: List[str] = []
    for term in terms:
        if term and term not in result:
            result.append(term)
    return result


def is_bill_qa_stat_question(question: str) -> bool:
    question_text = normalize_text(question)
    return any(term in question_text for term in BILL_QA_STAT_QUESTION_TERMS)


def is_recent_bill_list_question(question: str) -> bool:
    question_text = normalize_text(question)
    return "최근" in question_text and any(term in question_text for term in ["법안", "발의", "의안"])


def requested_bill_count(question: str, default: int = 5, maximum: int = 20) -> int:
    numbers = [int(number) for number in re.findall(r"\d+", normalize_text(question))]
    if not numbers:
        return default
    return max(1, min(max(numbers), maximum))


def top_stat_rows(rows: List[Dict[str, Any]], label_key: str, count_key: str = "건수", limit: int = 5) -> List[Dict[str, Any]]:
    normalized_rows = [
        {label_key: normalize_text(row.get(label_key)), count_key: as_int(row.get(count_key))}
        for row in rows
        if normalize_text(row.get(label_key))
    ]
    return sorted(normalized_rows, key=lambda row: row.get(count_key, 0), reverse=True)[:limit]


def row_matches_bill_question(row: Dict[str, Any], terms: List[str]) -> bool:
    if not terms:
        return False
    text = " ".join(
        normalize_text(row.get(key))
        for key in ["BILL_NAME", "COMMITTEE", "PROC_RESULT", "_ROLE", "BILL_NO", "PROPOSE_DT", "PPSL_DT"]
    )
    return any(term in text for term in terms)


def select_bill_qa_evidence(result: Dict[str, Any], question: str, limit: int = 30) -> List[Dict[str, Any]]:
    bills = list(result.get("sponsored_bills", []) or [])
    if not bills:
        return []

    terms = bill_qa_search_terms(question)
    question_text = normalize_text(question)
    matched = [row for row in bills if row_matches_bill_question(row, terms)]

    if any(word in question_text for word in ["분야", "소관", "위원회", "많이 발의"]):
        top_committees = top_stat_rows(result.get("bill_committee_stats", []) or [], "소관위원회", limit=3)
        committee_names = {row.get("소관위원회") for row in top_committees}
        if committee_names:
            matched = [row for row in bills if normalize_text(row.get("COMMITTEE")) in committee_names]

    if "대표" in question_text and "공동" in question_text:
        representative_rows = [row for row in bills if normalize_text(row.get("_ROLE")) == "대표발의"]
        cosponsored_rows = [row for row in bills if normalize_text(row.get("_ROLE")) == "공동발의"]
        half = max(1, limit // 2)
        matched = representative_rows[:half] + cosponsored_rows[:half]
    elif "대표" in question_text and "공동" not in question_text:
        role_rows = [row for row in bills if normalize_text(row.get("_ROLE")) == "대표발의"]
        matched = role_rows[:limit] if not matched else [row for row in matched if normalize_text(row.get("_ROLE")) == "대표발의"] or matched
    elif "공동" in question_text and "대표" not in question_text:
        role_rows = [row for row in bills if normalize_text(row.get("_ROLE")) == "공동발의"]
        matched = role_rows[:limit] if not matched else [row for row in matched if normalize_text(row.get("_ROLE")) == "공동발의"] or matched

    if any(word in question_text for word in ["통과", "가결", "의결"]):
        passed = [row for row in bills if any(word in normalize_text(row.get("PROC_RESULT")) for word in ["가결", "의결", "공포"])]
        matched = passed if passed else matched

    if "최근" in question_text and not matched:
        matched = bills[:limit]

    if not matched and terms:
        return []
    if not matched:
        matched = bills[:limit]
    return [compact_bill_for_qa(row) for row in matched[:limit]]


def build_bill_qa_context(result: Dict[str, Any], question: str) -> Dict[str, Any]:
    bills = list(result.get("sponsored_bills", []) or [])
    evidence_bills = select_bill_qa_evidence(result, question)
    recent_list_question = is_recent_bill_list_question(question)
    recent_limit = requested_bill_count(question, default=5)
    return {
        "member_name": result.get("member_name", ""),
        "question": question,
        "scope": "현재 조회된 발의법안 데이터",
        "answer_mode": "recent_bill_list" if recent_list_question else "general",
        "recent_list_limit": recent_limit,
        "summary": {
            "total_bills": len(bills),
            "representative_bills": len(result.get("representative_bills", []) or []),
            "cosponsored_bills": len(result.get("cosponsored_bills", []) or []),
            "by_result": result.get("bill_result_stats", []),
            "by_committee": result.get("bill_committee_stats", []),
        },
        "matched_bills": [compact_bill_for_recent_qa(row) for row in bills[:recent_limit]] if recent_list_question else evidence_bills,
        "recent_bills": [compact_bill_for_recent_qa(row) for row in bills[:10]],
    }


def fallback_bill_qa_answer(context: Dict[str, Any]) -> str:
    summary = context.get("summary", {})
    matched = context.get("matched_bills", [])
    question = normalize_text(context.get("question"))
    if not summary.get("total_bills"):
        return "현재 조회된 발의법안 데이터가 없어 답변할 수 없습니다."

    if context.get("answer_mode") == "recent_bill_list":
        limit = int(context.get("recent_list_limit") or 5)
        recent_rows = matched[:limit]
        if not recent_rows:
            return "현재 조회된 발의법안 데이터에서 최근 발의 법안을 확인하지 못했습니다."
        lines = [f"최근 발의한 법안 {len(recent_rows):,}건은 다음과 같습니다."]
        for index, row in enumerate(recent_rows, start=1):
            committee = row.get("소관위원회") or "소관위원회 미확인"
            role = row.get("역할") or "역할 미확인"
            date = row.get("발의일") or "발의일 미확인"
            lines.append(f"{index}. 「{row.get('법안명')}」 - {date}, {role}, {committee}")
        lines.append("확인 자료: 현재 조회된 발의법안 목록을 발의일 기준 최근순으로 정렬한 결과입니다.")
        return "\n\n".join(lines)

    if any(word in question for word in ["분야", "소관", "위원회", "많이 발의"]):
        top_committees = top_stat_rows(summary.get("by_committee", []) or [], "소관위원회", limit=5)
        if top_committees:
            top = top_committees[0]
            ranking = " / ".join(
                f"{row.get('소관위원회')} {as_int(row.get('건수')):,}건"
                for row in top_committees
            )
            evidence_text = "; ".join(
                f"「{row.get('법안명')}」({row.get('역할')}, {row.get('소관위원회')}, {row.get('발의일')}, {row.get('처리결과')})"
                for row in matched[:5]
                if row.get("법안명")
            )
            lines = [
                f"현재 조회된 발의법안 기준으로 가장 많이 나타난 분야는 `{top.get('소관위원회')}`입니다.",
                f"상위 소관위원회는 {ranking} 순입니다.",
                "소관위원회는 법안의 정책 분야를 대략적으로 보여주는 기준으로 볼 수 있습니다.",
                "판단 기준: 법안의 소관위원회는 정책 분야를 대략적으로 분류하는 기준이므로, 소관위원회별 발의 건수를 분야 판단의 1차 기준으로 봅니다.",
            ]
            if evidence_text:
                lines.append(f"확인 자료: {evidence_text}")
            return "\n\n".join(lines)

    if not matched and bill_qa_search_terms(question) and not is_bill_qa_stat_question(question):
        return "현재 조회된 발의법안 데이터에서는 질문과 직접 관련된 법안을 확인하지 못했습니다.\n\n판단 기준: 질문 키워드가 현재 조회된 법안명·소관위원회·처리결과와 연결되는지 확인했습니다."

    lines = [
        f"현재 조회된 발의법안은 총 {as_int(summary.get('total_bills')):,}건이며, 대표발의 {as_int(summary.get('representative_bills')):,}건 / 공동발의 {as_int(summary.get('cosponsored_bills')):,}건입니다.",
        "판단 기준: 발의법안의 역할 구분은 대표발의와 공동발의 건수를 나누어 봅니다.",
    ]
    if matched:
        evidence_text = "; ".join(
            f"「{row.get('법안명')}」({row.get('역할')}, {row.get('소관위원회')}, {row.get('발의일')}, {row.get('처리결과')})"
            for row in matched[:5]
            if row.get("법안명")
        )
        if evidence_text:
            lines.append(f"질문과 관련해 우선 확인할 수 있는 법안은 {len(matched):,}건입니다.")
            lines.append(f"확인 자료: {evidence_text}")
    return "\n\n".join(lines)


def gemini_key_validation_error(key: str) -> str:
    key = str(key or "").strip()
    if not key:
        return "missing"
    try:
        key.encode("ascii")
    except UnicodeEncodeError:
        return "non_ascii"
    if any(ch.isspace() or ord(ch) < 33 or ord(ch) > 126 for ch in key):
        return "invalid_character"
    return ""


def current_gemini_key_validation_error() -> str:
    return gemini_key_validation_error(str(st.session_state.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""))


def generate_bill_qa_answer(result: Dict[str, Any], question: str) -> Dict[str, Any]:
    context = build_bill_qa_context(result, question)
    evidence = context.get("matched_bills", [])
    question_text = normalize_text(question)
    if any(term in question_text for term in BILL_QA_OUT_OF_SCOPE_TERMS) and not any(term in question_text for term in ["발의", "법안", "의안"]):
        return {
            "answer": "이 챗봇은 현재 조회된 발의법안 데이터만 기준으로 답변합니다. 표결, 정당 일치도, 뉴스 관련 질문은 각각의 탭과 원자료를 확인해 주세요.\n\n판단 기준: 발의법안 Q&A 탭의 답변 범위는 발의법안 목록, 처리결과 통계, 소관위원회 통계로 제한됩니다.",
            "evidence": [],
            "source": "out_of_scope",
        }
    if not context.get("summary", {}).get("total_bills"):
        return {"answer": fallback_bill_qa_answer(context), "evidence": evidence, "source": "empty"}
    if context.get("answer_mode") == "recent_bill_list":
        return {"answer": fallback_bill_qa_answer(context), "evidence": evidence, "source": "rule_recent_list"}
    if not evidence and bill_qa_search_terms(question) and not is_bill_qa_stat_question(question):
        return {"answer": fallback_bill_qa_answer(context), "evidence": evidence, "source": "no_match"}
    key_error = gemini_key_validation_error(os.environ.get("GOOGLE_API_KEY", ""))
    if key_error:
        return {
            "answer": "발의법안 Q&A 챗봇을 사용하려면 올바른 Gemini API 키가 필요합니다. 키에 한글, 공백, 줄바꿈이 섞이지 않았는지 확인해 주세요.",
            "evidence": evidence,
            "source": "invalid_api_key",
        }
    if ChatGoogleGenerativeAI is None or SystemMessage is None or HumanMessage is None or not os.environ.get("GOOGLE_API_KEY"):
        return {"answer": fallback_bill_qa_answer(context), "evidence": evidence, "source": "rule_fallback"}

    system_prompt = (
        "당신은 현재 조회된 국회의원 발의법안 데이터만 설명하는 Q&A 어시스턴트입니다. "
        "제공된 발의법안 목록, 처리결과 통계, 소관위원회 통계만 사용해 답하세요. "
        "표결, 뉴스, 정당 일치도, 의원의 정치적 의도나 성과 평가는 답하지 마세요. "
        "데이터에 없으면 '현재 조회된 발의법안 데이터에서는 확인되지 않습니다'라고 답하세요. "
        "해석에는 `판단 기준:`을, 법안 예시에는 `확인 자료:`를 사용하세요. 숫자를 그대로 반복하는 근거 문장은 피하세요. "
        "최근 발의 법안 목록을 묻는 질문에서는 처리결과를 말하지 말고, 법안명·발의일·대표/공동발의 역할·소관위원회만 제시하세요."
    )
    user_prompt = f"""
아래 JSON context와 사용자의 질문을 바탕으로 답변하세요.

규칙:
- 3~6문장 이내로 간결하게 답하세요.
- 질문이 발의법안 범위를 벗어나면 이 챗봇은 현재 조회된 발의법안 데이터만 답한다고 안내하세요.
- 필요한 경우 `판단 기준:` 또는 `확인 자료:`를 포함하세요.
- `판단 기준:`에는 어떤 통계나 분류 기준으로 답했는지 쓰고, `확인 자료:`에는 법안명, 역할, 소관위원회, 발의일을 사람이 읽을 수 있게 쓰세요.
- context.answer_mode가 `recent_bill_list`이면 처리결과를 언급하지 마세요. 특히 "처리 결과는 미확인입니다"를 반복하지 마세요.
- 본문 숫자를 그대로 반복하는 확인 문장은 쓰지 마세요.
- matched_bills가 비어 있으면 관련 법안을 확인하지 못했다고 답하세요.

context:
{json.dumps(context, ensure_ascii=False, indent=2)}
""".strip()
    llm = ChatGoogleGenerativeAI(model=DEFAULT_LLM_MODEL, temperature=0.1)
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    answer = str(getattr(response, "content", "") or "").strip()
    return {"answer": answer or fallback_bill_qa_answer(context), "evidence": evidence, "source": "llm" if answer else "rule_fallback"}


def render_echarts(title: str, option: Dict[str, Any], *, height: int = 360) -> None:
    st.subheader(title)
    option.pop("_axisStyle", None)
    option_json = json.dumps(option, ensure_ascii=False)
    components.html(
        f"""
        <div id="chart" style="width:100%;height:{height}px;"></div>
        <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
        <script>
          const el = document.getElementById('chart');
          if (window.echarts) {{
            const chart = echarts.init(el, null, {{ renderer: 'canvas' }});
            chart.setOption({option_json});
            window.addEventListener('resize', () => chart.resize());
          }} else {{
            el.innerHTML = '<div style="padding:16px;color:#666;">차트 라이브러리를 불러오지 못했습니다.</div>';
          }}
        </script>
        """,
        height=height + 20,
    )


def base_chart_option() -> Dict[str, Any]:
    axis_style = {
        "axisLine": {"lineStyle": {"color": "#dfe5ec"}},
        "axisTick": {"lineStyle": {"color": "#dfe5ec"}},
        "axisLabel": {"color": "#687384"},
        "splitLine": {"lineStyle": {"color": "#edf1f5"}},
    }
    return {
        "color": [
            "rgba(80, 145, 241, 0.82)",
            "rgba(248, 92, 96, 0.78)",
            "rgba(255, 178, 44, 0.82)",
            "rgba(48, 190, 142, 0.80)",
            "rgba(166, 111, 236, 0.78)",
            "rgba(72, 197, 214, 0.78)",
        ],
        "backgroundColor": "rgba(255,255,255,0)",
        "textStyle": {"fontFamily": "sans-serif", "color": "#596270"},
        "tooltip": {"trigger": "item"},
        "legend": {"bottom": 0, "type": "scroll", "textStyle": {"color": "#687384"}},
        "_axisStyle": axis_style,
    }


def chart_axis_style(option: Dict[str, Any]) -> Dict[str, Any]:
    return dict(option.get("_axisStyle", {}))


def build_initial_state(member_name: str, options: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "member_name": member_name.strip(),
        "assembly_terms": DEFAULT_TERMS,
        "recent_limit": 10,
        "bills_enabled": options["bills_enabled"],
        "bill_term_scope": options["bill_term_scope"],
        "vote_detail_enabled": options["vote_detail_enabled"],
        "max_bill_pages": options["max_bill_pages"],
        "max_vote_pages": DEFAULT_MAX_VOTE_PAGES,
        "max_vote_bills_to_scan": options["max_vote_bills_to_scan"],
        "force_vote_refresh": options.get("force_vote_refresh", False),
        "vote_fetch_workers": options["vote_fetch_workers"],
        "party_alignment_enabled": options["party_alignment_enabled"],
        "party_alignment_vote_limit": options["party_alignment_vote_limit"],
        "llm_insights_enabled": options["llm_insights_enabled"],
        "recent_news_enabled": options["recent_news_enabled"],
        "enable_cosponsor_scan": options["enable_cosponsor_scan"],
        "max_cosponsor_scan_pages": options["max_cosponsor_scan_pages"],
        "member_party_lookup": options.get("member_party_lookup", {}),
        "async_open_assembly_enabled": options.get("async_open_assembly_enabled", DEFAULT_ASYNC_OPEN_ASSEMBLY_ENABLED),
        "async_open_assembly_concurrency": options.get("async_open_assembly_concurrency", DEFAULT_ASYNC_OPEN_ASSEMBLY_CONCURRENCY),
        "show_progress": False,
    }


WORKFLOW_STEPS = [
    ("normalize_user_request", "입력값 정리", "의원명과 조회 옵션을 정리합니다."),
    ("get_member_info", "의원 기본정보 조회", "재임 대수, 정당, 선거구, 위원회 정보를 확인합니다."),
    ("search_member_bills", "발의법안 조회", "대표발의와 공동발의 법안을 수집합니다."),
    ("analyze_legislative_interests", "입법 관심 분야 분석", "발의법안 통계를 바탕으로 관심 의제를 요약합니다."),
    ("analyze_cosponsor_network", "공동발의 네트워크 분석", "반복 공동발의 파트너와 협업 분야를 계산합니다."),
    ("get_all_member_votes", "표결 상세 조회", "본회의 표결 기록을 조회하고 찬성·반대·기권·불참으로 분류합니다."),
    ("analyze_party_alignment", "정당 다수 입장 일치도 분석", "같은 정당 의원 다수 입장과의 일치 여부를 계산합니다."),
    ("interpret_vote_statistics", "표결 해석 메모 생성", "분석 범위와 불참 비중을 고려한 해석 주의사항을 만듭니다."),
    ("search_recent_member_news", "최근 이슈 검색", "최근 뉴스 검색 결과를 바탕으로 공개 이슈를 요약합니다."),
    ("generate_activity_briefing", "핵심 브리핑 생성", "주요 활동 지표를 3~5문장으로 정리합니다."),
    ("summarize_member_activity", "최종 결과 정리", "전체 분석 결과를 화면에 표시할 형태로 정리합니다."),
]

WORKFLOW_STEP_LABELS = {node: label for node, label, _ in WORKFLOW_STEPS}
WORKFLOW_STEP_DESCRIPTIONS = {node: description for node, _, description in WORKFLOW_STEPS}
WORKFLOW_PROGRESS = {
    node: int((index + 1) / len(WORKFLOW_STEPS) * 100)
    for index, (node, _, _) in enumerate(WORKFLOW_STEPS)
}


def configure_user_gemini_key(gemini_api_key: str) -> None:
    key = gemini_api_key.strip()
    if key:
        st.session_state["GEMINI_API_KEY"] = key
        os.environ["GOOGLE_API_KEY"] = key
        os.environ["GEMINI_API_KEY"] = key


def run_analysis(member_name: str, options: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    initial_state = build_initial_state(member_name, options)
    app = get_member_activity_app(WORKFLOW_VERSION)

    progress = st.progress(0, text="분석을 시작합니다.")
    step_box = st.empty()
    detail_box = st.empty()
    completed_box = st.container()
    completed_steps: List[str] = []
    max_progress_percent = 0

    def progress_callback(message: str) -> None:
        detail_box.caption(f"세부 진행: {message}")

    initial_state["progress_callback"] = progress_callback

    with st.status("의원 활동을 조회하고 있습니다.", expanded=True) as status:
        st.write("의원 기본정보 확인 후 발의법안, 표결정보, 최근 이슈를 병렬로 분석합니다.")
        for event in app.stream(initial_state, stream_mode="updates"):
            for node_name, node_update in event.items():
                label = WORKFLOW_STEP_LABELS.get(node_name, node_name)
                description = WORKFLOW_STEP_DESCRIPTIONS.get(node_name, "")
                percent = WORKFLOW_PROGRESS.get(node_name, min(95, len(completed_steps) * 10))
                max_progress_percent = max(max_progress_percent, percent)

                progress.progress(max_progress_percent, text=f"{label} 완료 ({max_progress_percent}%)")
                step_box.info(f"현재 단계: {label}\n\n{description}")

                if node_update:
                    result.update(node_update)
                    result.pop("progress_callback", None)

                if label not in completed_steps:
                    completed_steps.append(label)
                    completed_box.write(f"✅ {label} 완료")

                if node_name == "get_all_member_votes" and result.get("vote_fetch_stats"):
                    stats = result.get("vote_fetch_stats", [])
                    total_bills = sum(int(item.get("표결의안", 0) or 0) for item in stats)
                    total_success = sum(int(item.get("상세조회성공", 0) or 0) for item in stats)
                    total_failed = sum(int(item.get("상세조회실패", 0) or 0) for item in stats)
                    st.caption(f"표결 상세 조회 상태: 대상 {total_bills:,}건 / 성공 {total_success:,}건 / 실패 {total_failed:,}건")

                if node_name == "analyze_party_alignment" and result.get("party_alignment_fetch_stats"):
                    stats = result.get("party_alignment_fetch_stats", [])
                    total_target = sum(int(item.get("분석대상표결", 0) or 0) for item in stats)
                    cache_hits = sum(int(item.get("BILL_ID캐시적중", 0) or 0) for item in stats)
                    api_targets = sum(int(item.get("API신규조회대상", 0) or 0) for item in stats)
                    st.caption(f"정당 일치도 분석 상태: 대상 {total_target:,}건 / 캐시 {cache_hits:,}건 / 신규 API 조회 {api_targets:,}건")

        progress.progress(100, text="분석 완료")
        status.update(label="분석 완료", state="complete", expanded=False)

    if not result:
        raise RuntimeError("워크플로우 결과가 비어 있습니다.")
    result.pop("progress_callback", None)
    result["_workflow_version"] = WORKFLOW_VERSION
    return result


def run_party_alignment_reanalysis_job(client: Any, state: Dict[str, Any]) -> Dict[str, Any]:
    update = make_analyze_party_alignment_node(client)(state)
    interpretation = analyze_vote_interpretation_node({**state, **update})
    return {**update, **interpretation}


def start_party_alignment_reanalysis(result: Dict[str, Any], limit: int) -> None:
    state = {
        **result,
        "party_alignment_vote_limit": int(limit),
        "errors": [],
        "show_progress": False,
    }
    state.pop("progress_callback", None)
    future = get_alignment_reanalysis_executor().submit(run_party_alignment_reanalysis_job, get_client(), state)
    st.session_state["alignment_reanalysis_future"] = future
    st.session_state["alignment_reanalysis_limit"] = int(limit)
    st.session_state["alignment_reanalysis_status"] = "running"
    st.session_state["alignment_reanalysis_message"] = (
        f"정당 일치도만 최근 {int(limit):,}건 기준으로 재계산하고 있습니다. "
        "발의법안, 표결 목록, 최근 이슈는 다시 조회하지 않습니다."
    )


def merge_party_alignment_update(result: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(result)
    for key in ALIGNMENT_RESULT_KEYS:
        if key in update:
            merged[key] = update[key]

    existing_errors = list(result.get("errors", []) or [])
    new_errors = [error for error in list(update.get("errors", []) or []) if error not in existing_errors]
    if new_errors:
        merged["errors"] = existing_errors + new_errors
    elif "errors" in result:
        merged["errors"] = existing_errors

    merged["_alignment_reanalysis_last_limit"] = update.get(
        "party_alignment_vote_limit",
        st.session_state.get("alignment_reanalysis_limit"),
    )
    return merged


def apply_completed_party_alignment_reanalysis() -> None:
    future = st.session_state.get("alignment_reanalysis_future")
    if not isinstance(future, Future) or not future.done():
        return

    try:
        update = future.result()
    except Exception as error:
        st.session_state["alignment_reanalysis_status"] = "error"
        st.session_state["alignment_reanalysis_message"] = f"정당 일치도 재분석 중 오류가 발생했습니다: {error}"
    else:
        result = st.session_state.get("member_activity_result")
        if isinstance(result, dict):
            st.session_state["member_activity_result"] = merge_party_alignment_update(result, update)
        st.session_state["alignment_reanalysis_status"] = "complete"
        st.session_state["alignment_reanalysis_message"] = "정당 일치도 재분석이 완료되어 현재 정당 일치도 탭에 반영되었습니다."
    finally:
        st.session_state.pop("alignment_reanalysis_future", None)


def render_summary_kpis(result: Dict[str, Any]) -> None:
    alignment_summary = result.get("party_alignment_summary", [])
    aligned = metric_value(alignment_summary, "일치")
    dissented = metric_value(alignment_summary, "이탈")
    judged = aligned + dissented
    alignment_rate = f"{aligned / judged * 100:.1f}%" if judged else "-"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("발의법안", f"{len(result.get('sponsored_bills', [])):,}건")
    c2.metric("표결", f"{len(result.get('vote_records', [])):,}건")
    c3.metric("정당 일치율", alignment_rate)
    c4.metric("최근 뉴스", f"{len(result.get('recent_news_items', [])):,}건")


def render_summary_scope(result: Dict[str, Any]) -> None:
    st.subheader("조회 범위")
    lines = [
        f"- 의원 재임 대수: {format_terms(result.get('served_terms') or result.get('analysis_terms', []))}",
        f"- 의원 기본정보 조회 대수: {format_terms(result.get('member_terms', []))}",
        f"- 발의법안 조회 대수: {format_terms(result.get('bill_terms', []))}",
        f"- 본회의 표결정보 초기 조회 대수: {format_terms(result.get('vote_terms', []))}",
        f"- 표결정보 추가 선택 가능 대수: {format_terms(result.get('additional_vote_terms', []))}",
    ]
    st.markdown("\n".join(lines))


def render_activity_briefing(result: Dict[str, Any]) -> None:
    briefing = str(result.get("activity_briefing") or "").strip()
    if not briefing:
        return
    with st.container(border=True):
        st.markdown("#### 핵심 브리핑")
        st.caption("발의법안, 표결, 정당 일치도, 최근 뉴스 수를 바탕으로 한 요약입니다.")
        st.markdown(briefing)


def render_bill_summary_charts(result: Dict[str, Any]) -> None:
    rep_count = len(result.get("representative_bills", []))
    co_count = len(result.get("cosponsored_bills", []))
    other_count = max(0, len(result.get("sponsored_bills", [])) - rep_count - co_count)
    role_data = [
        {"name": "대표발의", "value": rep_count},
        {"name": "공동발의", "value": co_count},
    ]
    if other_count:
        role_data.append({"name": "역할 확인필요", "value": other_count})

    committee_data = [
        {"name": str(row.get("소관위원회") or "미확인"), "value": as_int(row.get("건수"))}
        for row in result.get("bill_committee_stats", [])
    ]

    left, right = st.columns(2)
    with left:
        if chart_has_data(role_data):
            option = base_chart_option()
            option.update(
                {
                    "series": [
                        {
                            "type": "pie",
                            "radius": ["48%", "72%"],
                            "center": ["50%", "45%"],
                            "avoidLabelOverlap": True,
                            "label": {"formatter": "{b}\n{c}건", "color": "#56616f"},
                            "itemStyle": {"borderColor": "#ffffff", "borderWidth": 2},
                            "data": role_data,
                        }
                    ]
                }
            )
            render_echarts("발의 역할 구성", option, height=320)
        else:
            st.subheader("발의 역할 구성")
            st.info("발의법안 데이터가 없습니다.")
    with right:
        if chart_has_data(committee_data):
            option = base_chart_option()
            option.update(
                {
                    "tooltip": {"formatter": "{b}: {c}건"},
                    "series": [
                        {
                            "type": "treemap",
                            "roam": False,
                            "breadcrumb": {"show": False},
                            "label": {"show": True, "formatter": "{b}\n{c}건", "color": "#ffffff", "fontWeight": 700},
                            "upperLabel": {"show": False},
                            "itemStyle": {"borderColor": "rgba(255,255,255,0.82)", "borderWidth": 2},
                            "data": committee_data,
                        }
                    ],
                }
            )
            render_echarts("소관위원회별 발의법안", option, height=320)
        else:
            st.subheader("소관위원회별 발의법안")
            st.info("소관위원회 통계가 없습니다.")


def render_cosponsor_network_graph(result: Dict[str, Any], partners_override: List[Dict[str, Any]] | None = None) -> None:
    context = result.get("cosponsor_network_context", {})
    partners = list(partners_override if partners_override is not None else result.get("cosponsor_network_partners", []) or [])[:30]
    member_name = str(context.get("member_name") or result.get("member_name") or "대상 의원")
    if not partners:
        st.info("공동발의 네트워크를 구성할 파트너 데이터가 없습니다.")
        return

    max_count = max(as_int(partner.get("공동발의건수")) for partner in partners) or 1

    def partner_node_size(count: int) -> int:
        # Power scaling separates high-frequency collaborators more clearly than sqrt,
        # while keeping low-frequency nodes readable.
        scaled = (max(count, 1) / max_count) ** 0.75
        return int(round((10 + scaled * 36) * 0.9))

    def edge_width(count: int) -> float:
        scaled = (max(count, 1) / max_count) ** 0.75
        return round(0.8 + scaled * 7.0, 2)

    nodes = [
        {
            "id": "center",
            "label": member_name,
            "title": f"{member_name} 의원",
            "shape": "dot",
            "size": 36,
            "x": 0,
            "y": 0,
            "fixed": {"x": True, "y": True},
            "color": {"background": "#5091f1", "border": "#2563eb"},
            "font": {"size": 18, "color": "#1e344d", "bold": True},
        }
    ]
    edges = []
    for index, partner in enumerate(partners):
        count = as_int(partner.get("공동발의건수"))
        partner_name = str(partner.get("공동발의자") or f"파트너 {index + 1}")
        party_relation = str(partner.get("정당관계") or "정당 미확인")
        partner_id = f"partner_{index}"
        angle = (2 * math.pi * index) / max(len(partners), 1)
        radius = 210 if len(partners) <= 18 else 250
        if len(partners) > 24 and index % 2:
            radius = 175
        if party_relation == "당내":
            color = {"background": "#30be8e", "border": "#19946b"}
        elif party_relation == "당외":
            color = {"background": "#ffb22c", "border": "#d28b13"}
        else:
            color = {"background": "#a66fec", "border": "#7c3fc9"}
        nodes.append(
            {
                "id": partner_id,
                "label": f"{partner_name}\n{count}건",
                "title": f"{partner_name}<br>정당: {partner.get('정당', '정당 미확인')} ({party_relation})<br>{partner.get('주요소관위원회', '')}<br>최근: {partner.get('최근공동발의일', '')}",
                "shape": "dot",
                "size": partner_node_size(count),
                "x": round(math.cos(angle) * radius, 2),
                "y": round(math.sin(angle) * radius, 2),
                "fixed": {"x": True, "y": True},
                "color": color,
                "font": {
                    "size": 14 if count >= max_count * 0.6 else 12,
                    "color": "#334155",
                    "bold": count >= max_count * 0.6,
                },
            }
        )
        edges.append(
            {
                "from": "center",
                "to": partner_id,
                "value": max(1, count),
                "width": edge_width(count),
                "title": f"공동발의 {count}건",
                "color": {"color": "rgba(80, 145, 241, 0.42)"},
            }
        )

    nodes_json = json.dumps(nodes, ensure_ascii=False)
    edges_json = json.dumps(edges, ensure_ascii=False)
    components.html(
        f"""
        <div id="cosponsor-network" style="height:520px;border:1px solid #dbe4ee;border-radius:16px;background:linear-gradient(145deg,#ffffff,#f6f9fc);"></div>
        <script src="https://cdn.jsdelivr.net/npm/vis-network@9.1.9/dist/vis-network.min.js"></script>
        <script>
          const container = document.getElementById('cosponsor-network');
          if (window.vis) {{
            const data = {{
              nodes: new vis.DataSet({nodes_json}),
              edges: new vis.DataSet({edges_json})
            }};
            const options = {{
              autoResize: true,
              interaction: {{ hover: true, tooltipDelay: 120 }},
              nodes: {{ physics: false }},
              physics: false,
              layout: {{ improvedLayout: false }},
              edges: {{ smooth: {{ type: 'continuous' }} }}
            }};
            const network = new vis.Network(container, data, options);
            const centerNetwork = () => {{
              const width = Math.max(container.clientWidth || 0, 320);
              const height = Math.max(container.clientHeight || 0, 360);
              const nodes = data.nodes.get();
              const maxExtent = nodes.reduce((value, node) => {{
                const nodeSize = Number(node.size || 20);
                const extent = Math.max(Math.abs(Number(node.x || 0)), Math.abs(Number(node.y || 0))) + nodeSize * 2.2;
                return Math.max(value, extent);
              }}, 260);
              const scale = Math.max(0.5, Math.min(1.2, Math.min((width - 80) / (maxExtent * 2), (height - 70) / (maxExtent * 2))));
              network.moveTo({{ position: {{ x: 0, y: 0 }}, scale, animation: false }});
              network.redraw();
            }};
            requestAnimationFrame(centerNetwork);
            network.once('afterDrawing', centerNetwork);
            setTimeout(centerNetwork, 80);
            setTimeout(centerNetwork, 250);
            setTimeout(centerNetwork, 700);
            if (window.ResizeObserver) {{
              new ResizeObserver(centerNetwork).observe(container);
            }}
          }} else {{
            container.innerHTML = '<div style="padding:16px;color:#666;">네트워크 라이브러리를 불러오지 못했습니다.</div>';
          }}
        </script>
        """,
        height=545,
    )


def render_cosponsor_network_section(result: Dict[str, Any]) -> None:
    partners = list(result.get("cosponsor_network_partners", []) or [])
    committees = list(result.get("cosponsor_network_committees", []) or [])
    summary = result.get("cosponsor_network_summary", {}) or {}
    analysis = str(result.get("cosponsor_network_analysis") or "").strip()

    st.subheader("공동발의 네트워크")
    st.caption("대상 의원을 중심으로 같은 법안에 함께 이름이 등장한 공동발의 파트너를 연결한 1차 ego-network입니다.")
    st.info(
        "참고: 국회의원이 법률안을 발의하려면 의원 10인 이상의 찬성이 필요합니다. "
        "이 발의 명단은 보통 대표발의자와 공동발의자로 구성되므로, 공동발의 네트워크는 "
        "법안 발의 요건을 충족하는 과정에서 함께 이름을 올린 의원들의 입법 협력 관계를 보여줍니다."
    )
    if not partners:
        st.info("공동발의 네트워크 데이터가 없습니다. 발의법안 조회 범위가 좁거나 공동발의자 문자열을 확인할 수 없는 경우입니다.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("협업 파트너", f"{as_int(summary.get('협업파트너')):,}명")
    c2.metric("협업 연결", f"{as_int(summary.get('협업연결')):,}건")
    c3.metric("상위 3명 집중도", f"{float(summary.get('상위3명집중도') or 0):.1f}%")
    c4.metric("협업 유형", str(summary.get("협업유형") or "-"))
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("당내 협업", f"{float(summary.get('당내협업비율') or 0):.1f}%")
    b2.metric("당외 협업", f"{float(summary.get('당외협업비율') or 0):.1f}%")
    b3.metric("정당 미확인", f"{float(summary.get('정당미확인비율') or 0):.1f}%")
    b4.metric("초당성 유형", str(summary.get("초당협업유형") or "-"))

    if analysis:
        with st.container(border=True):
            st.markdown("#### 협업 유형 해석")
            st.markdown(analysis)

    party_data = [
        {"name": "당내", "value": as_int(summary.get("당내협업"))},
        {"name": "당외", "value": as_int(summary.get("당외협업"))},
        {"name": "정당 미확인", "value": as_int(summary.get("정당미확인협업"))},
    ]
    left_chart, _right_spacer = st.columns([0.55, 0.45])
    with left_chart:
        if chart_has_data(party_data):
            option = base_chart_option()
            option["color"] = ["rgba(48, 190, 142, 0.82)", "rgba(255, 178, 44, 0.82)", "rgba(166, 111, 236, 0.78)"]
            option.update(
                {
                    "series": [
                        {
                            "type": "pie",
                            "radius": ["48%", "72%"],
                            "center": ["50%", "45%"],
                            "label": {"formatter": "{b}\n{c}건", "color": "#56616f"},
                            "itemStyle": {"borderColor": "#ffffff", "borderWidth": 2},
                            "data": party_data,
                        }
                    ]
                }
            )
            render_echarts("정당 내/외 협업 비중", option, height=300)
            st.caption(
                "정당 색상 기준: 초록은 당내, 주황은 당외, 보라는 정당 미확인입니다. "
                "정당 매칭은 의원 목록 API의 현재/최근 정당명을 기준으로 하므로 발의 당시 정당과 다를 수 있습니다."
            )

    committee_options = ["전체"] + [str(row.get("소관위원회")) for row in committees if row.get("소관위원회")]
    selected_committee = st.selectbox(
        "소관위원회별 네트워크 필터",
        committee_options,
        key="cosponsor_network_committee_filter",
        help="선택한 소관위원회에서 함께 발의한 파트너만 네트워크에 표시합니다.",
    )
    filtered_partners = partners
    if selected_committee != "전체":
        filtered_partners = []
        for partner in partners:
            filtered_bills = [bill for bill in partner.get("함께한법안", []) if bill.get("소관위원회") == selected_committee]
            if not filtered_bills:
                continue
            filtered_partner = dict(partner)
            filtered_partner["공동발의건수"] = len(filtered_bills)
            filtered_partner["함께한법안"] = filtered_bills
            filtered_partner["주요소관위원회"] = f"{selected_committee} {len(filtered_bills)}건"
            filtered_partners.append(filtered_partner)
        filtered_partners = sorted(filtered_partners, key=lambda row: (-as_int(row.get("공동발의건수")), str(row.get("공동발의자") or "")))
    st.caption(f"네트워크 시각화는 렌더링 속도를 위해 현재 조건의 상위 {min(30, len(filtered_partners))}명까지만 표시합니다.")
    render_cosponsor_network_graph(result, filtered_partners)

    top_partners = partners[:10]
    if top_partners:
        option = base_chart_option()
        axis = chart_axis_style(option)
        option.update(
            {
                "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
                "grid": {"left": 70, "right": 20, "top": 30, "bottom": 80},
                "xAxis": {"type": "category", "data": [row.get("공동발의자") for row in top_partners], **axis},
                "yAxis": {"type": "value", "name": "공동발의 건수", **axis},
                "series": [
                    {
                        "name": "공동발의",
                        "type": "bar",
                        "data": [as_int(row.get("공동발의건수")) for row in top_partners],
                        "itemStyle": {"borderRadius": [8, 8, 0, 0]},
                    }
                ],
            }
        )
        render_echarts("반복 협업 의원 Top 10", option, height=330)

    table_rows = [
        {
            "공동발의자": row.get("공동발의자"),
            "정당": row.get("정당"),
            "정당관계": row.get("정당관계"),
            "공동발의건수": row.get("공동발의건수"),
            "주요소관위원회": row.get("주요소관위원회"),
            "최근공동발의일": row.get("최근공동발의일"),
            "대표법안": row.get("대표법안"),
        }
        for row in top_partners
    ]
    left, right = st.columns([1.25, 0.75])
    with left:
        render_dataframe("반복 협업 의원", table_rows, ["공동발의자", "정당", "정당관계", "공동발의건수", "주요소관위원회", "최근공동발의일", "대표법안"], height=320)
    with right:
        render_dataframe("협업 소관위원회", committees[:10], ["소관위원회", "협업건수"], height=320)

    partner_names = [str(row.get("공동발의자")) for row in partners if row.get("공동발의자")]
    if partner_names:
        selected_partner_name = st.selectbox("파트너 상세 보기", partner_names[:30], key="cosponsor_partner_detail")
        selected_partner = next((row for row in partners if row.get("공동발의자") == selected_partner_name), None)
        if selected_partner:
            st.markdown(
                f"**{selected_partner_name}** · {selected_partner.get('정당', '정당 미확인')} · "
                f"{selected_partner.get('공동발의건수', 0)}건"
            )
            render_dataframe(
                "함께 발의한 법안",
                selected_partner.get("함께한법안", []),
                ["법안명", "발의일", "소관위원회", "대상역할", "링크"],
                height=280,
            )

    st.caption("주의: 공동발의는 정책 공조, 입장 표명, 의례적 참여가 섞일 수 있으므로 실제 정치적 연대나 정책 동의로 단정하지 않습니다.")


def render_vote_summary_chart(result: Dict[str, Any]) -> None:
    rows = result.get("vote_summary_by_term", [])
    if not rows:
        st.subheader("표결 요약")
        st.info("표결 요약 데이터가 없습니다.")
        return

    x_axis = [f"{row.get('AGE')}대" for row in rows]
    option = base_chart_option()
    axis = chart_axis_style(option)
    option.update(
        {
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
            "xAxis": {"type": "category", "data": x_axis, **axis},
            "yAxis": {"type": "value", "name": "표결 수", **axis},
            "series": [
                {"name": "찬성", "type": "bar", "stack": "vote", "data": [as_int(row.get("찬성")) for row in rows]},
                {"name": "반대", "type": "bar", "stack": "vote", "data": [as_int(row.get("반대")) for row in rows]},
                {"name": "기권", "type": "bar", "stack": "vote", "data": [as_int(row.get("기권")) for row in rows]},
                {"name": "불참", "type": "bar", "stack": "vote", "data": [as_int(row.get("불참")) for row in rows]},
            ],
            "grid": {"left": 48, "right": 24, "top": 32, "bottom": 64},
        }
    )
    render_echarts("표결 요약", option, height=340)


def render_alignment_summary_charts(result: Dict[str, Any]) -> None:
    rows = result.get("party_alignment_summary", [])
    if not rows:
        st.subheader("정당 다수 입장 일치 분석")
        st.info("정당 일치도 데이터가 없습니다.")
        return

    total_data = [
        {"name": "일치", "value": metric_value(rows, "일치")},
        {"name": "이탈", "value": metric_value(rows, "이탈")},
        {"name": "불참", "value": metric_value(rows, "불참")},
        {"name": "판정 제외", "value": metric_value(rows, "판정제외")},
    ]
    x_axis = [f"{row.get('AGE')}대" for row in rows]

    left, right = st.columns(2)
    with left:
        option = base_chart_option()
        option.update(
            {
                "series": [
                    {
                        "type": "pie",
                        "radius": ["48%", "72%"],
                        "center": ["50%", "45%"],
                        "label": {"formatter": "{b}\n{c}건", "color": "#56616f"},
                        "itemStyle": {"borderColor": "#ffffff", "borderWidth": 2},
                        "data": total_data,
                    }
                ]
            }
        )
        render_echarts("정당 일치도 구성", option, height=320)
    with right:
        option = base_chart_option()
        axis = chart_axis_style(option)
        option.update(
            {
                "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
                "xAxis": {"type": "category", "data": x_axis, **axis},
                "yAxis": {"type": "value", "name": "표결 수", **axis},
                "series": [
                    {"name": "일치", "type": "bar", "stack": "alignment", "data": [as_int(row.get("일치")) for row in rows]},
                    {"name": "이탈", "type": "bar", "stack": "alignment", "data": [as_int(row.get("이탈")) for row in rows]},
                    {"name": "불참", "type": "bar", "stack": "alignment", "data": [as_int(row.get("불참")) for row in rows]},
                    {"name": "판정 제외", "type": "bar", "stack": "alignment", "data": [as_int(row.get("판정제외")) for row in rows]},
                ],
                "grid": {"left": 48, "right": 24, "top": 32, "bottom": 64},
            }
        )
        render_echarts("대수별 정당 일치도", option, height=320)


def render_vote_fetch_text(result: Dict[str, Any]) -> None:
    stats = result.get("vote_fetch_stats", [])
    st.subheader("표결 조회 상태")
    if not stats:
        st.info("표결 조회 상태 데이터가 없습니다.")
        return
    total_bills = sum(as_int(row.get("표결의안")) for row in stats)
    total_success = sum(as_int(row.get("상세조회성공")) for row in stats)
    total_failed = sum(as_int(row.get("상세조회실패")) for row in stats)
    lines = [f"- 표결 의안 {total_bills:,}건 중 상세조회 성공 {total_success:,}건, 실패 {total_failed:,}건"]
    for row in stats:
        lines.append(
            f"- {row.get('AGE')}대: 표결의안 {as_int(row.get('표결의안')):,} / "
            f"성공 {as_int(row.get('상세조회성공')):,} / "
            f"실패 {as_int(row.get('상세조회실패')):,} / "
            f"매칭표결 {as_int(row.get('매칭표결')):,}"
        )
    st.markdown("\n".join(lines))


def render_member_profile_photo(result: Dict[str, Any]) -> None:
    for row in result.get("member_info", []):
        image_url = str(row.get("NAAS_PIC") or "").strip()
        if image_url:
            st.image(image_url, width=150)
            return


def render_news_table_timeline(result: Dict[str, Any]) -> None:
    items = result.get("recent_news_items", [])
    st.subheader("최근 이슈")
    analysis = clean_recent_news_analysis_text(result.get("recent_news_analysis"))
    with st.container(border=True):
        st.markdown("#### LLM 분석 요약")
        st.caption("아래 요약은 바로 밑의 뉴스 검색 결과를 근거 자료로 사용해 생성한 해석입니다.")
        st.markdown(analysis or "최근 이슈 분석 결과가 없습니다.")
    if not items:
        st.info("최근 뉴스 데이터가 없습니다.")
        return

    st.markdown("#### 분석 근거 뉴스")
    st.caption("LLM 분석에 사용된 원 뉴스 검색 결과입니다. 기사 제목을 눌러 원문을 확인할 수 있습니다.")

    safe_items = [
        {
            "title": html.escape(str(item.get("title") or "")),
            "source": html.escape(str(item.get("source") or "")),
            "published": html.escape(str(item.get("published") or "")),
            "url": html.escape(str(item.get("url") or "")),
        }
        for item in items
    ]
    rows_html = "\n".join(
        f"""
        <tr>
          <td><a href="{item['url']}" target="_blank" rel="noopener noreferrer">{item['title'] or '제목 없음'}</a></td>
          <td>{item['source'] or '출처 미확인'}</td>
          <td>{item['published'] or '발행일 미확인'}</td>
        </tr>
        """
        for item in safe_items
    )
    timeline_html = "\n".join(
        f"""
        <div class="timeline-item">
          <div class="dot"></div>
          <div class="time">{item['published'] or '발행일 미확인'}</div>
          <a href="{item['url']}" target="_blank" rel="noopener noreferrer">{item['title'] or '제목 없음'}</a>
          <span>{item['source'] or '출처 미확인'}</span>
        </div>
        """
        for item in safe_items[:8]
    )
    component_height = max(520, min(900, 170 + len(safe_items) * 72))
    components.html(
        f"""
        <style>
          body {{ margin: 0; font-family: sans-serif; color: #1f2933; }}
          .wrap {{ display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 18px; }}
          .panel {{ border: 1px solid #d9e0e6; border-radius: 8px; padding: 14px; background: #fff; }}
          input {{ width: 100%; box-sizing: border-box; padding: 10px 12px; border: 1px solid #c8d0d8; border-radius: 6px; margin-bottom: 10px; }}
          table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
          th, td {{ border-bottom: 1px solid #edf1f4; padding: 9px 8px; text-align: left; vertical-align: top; }}
          th {{ background: #f6f8fa; font-weight: 700; }}
          a {{ color: #2F6F73; text-decoration: none; }}
          .timeline {{ position: relative; padding-left: 18px; }}
          .timeline:before {{ content: ""; position: absolute; left: 6px; top: 5px; bottom: 5px; width: 2px; background: #d6dde4; }}
          .timeline-item {{ position: relative; padding: 0 0 16px 12px; }}
          .dot {{ position: absolute; left: -17px; top: 5px; width: 10px; height: 10px; border-radius: 999px; background: #C84C31; }}
          .time {{ font-size: 12px; color: #607080; margin-bottom: 3px; }}
          .timeline-item span {{ display: block; font-size: 12px; color: #607080; margin-top: 3px; }}
          @media (max-width: 760px) {{ .wrap {{ grid-template-columns: 1fr; }} }}
        </style>
        <div class="wrap">
          <div class="panel">
            <input id="filter" placeholder="기사 제목, 언론사, 날짜 검색" />
            <table id="news-table">
              <thead><tr><th>제목</th><th>출처</th><th>발행</th></tr></thead>
              <tbody>{rows_html}</tbody>
            </table>
          </div>
          <div class="panel">
            <div class="timeline">{timeline_html}</div>
          </div>
        </div>
        <script>
          const input = document.getElementById('filter');
          const rows = Array.from(document.querySelectorAll('#news-table tbody tr'));
          input.addEventListener('input', () => {{
            const q = input.value.trim().toLowerCase();
            rows.forEach(row => {{
              row.style.display = row.innerText.toLowerCase().includes(q) ? '' : 'none';
            }});
          }});
        </script>
        """,
        height=component_height,
    )


def render_summary_tab(result: Dict[str, Any]) -> None:
    render_summary_kpis(result)
    render_activity_briefing(result)
    st.divider()

    st.subheader("의원 프로필")
    render_member_profile_photo(result)
    st.markdown(result.get("profile_summary", "프로필 정보가 없습니다."))
    render_summary_scope(result)
    st.divider()

    st.subheader("발의 법안 요약")
    st.markdown(
        f"- 발의법안: {len(result.get('sponsored_bills', [])):,}건\n"
        f"- 대표발의: {len(result.get('representative_bills', [])):,}건 / 공동발의: {len(result.get('cosponsored_bills', [])):,}건"
    )
    render_bill_summary_charts(result)
    st.divider()

    st.subheader("입법 관심 분야 분석")
    st.markdown(result.get("legislative_interest_analysis") or "입법 관심 분야 분석 결과가 없습니다.")
    st.divider()

    render_vote_summary_chart(result)
    render_vote_fetch_text(result)


def render_bills_tab(result: Dict[str, Any]) -> None:
    bills = result.get("sponsored_bills", [])
    rep = result.get("representative_bills", [])
    co = result.get("cosponsored_bills", [])

    c1, c2, c3 = st.columns(3)
    c1.metric("발의 법안", f"{len(bills):,}건")
    c2.metric("대표발의", f"{len(rep):,}건")
    c3.metric("공동발의", f"{len(co):,}건")
    render_bill_summary_charts(result)
    st.divider()
    render_cosponsor_network_section(result)
    st.divider()

    stat_cols = ["처리결과", "건수"]
    committee_cols = ["소관위원회", "건수"]
    bill_cols = ["_ROLE", "AGE", "BILL_NO", "BILL_NAME", "PROPOSE_DT", "COMMITTEE", "PROC_RESULT", "DETAIL_LINK"]

    with st.expander("처리결과별 통계"):
        left, right = st.columns(2)
        with left:
            render_dataframe("처리결과별 통계", result.get("bill_result_stats", []), stat_cols, height=260)
        with right:
            render_dataframe("소관위원회별 통계", result.get("bill_committee_stats", []), committee_cols, height=260)

    with st.expander("최근 대표발의 법안"):
        render_dataframe("대표발의", rep[:20], bill_cols)
    with st.expander("최근 공동발의 법안"):
        render_dataframe("공동발의", co[:20], bill_cols)
    with st.expander("발의 법안 전체 원자료"):
        render_dataframe("발의 법안 전체", bills, bill_cols, height=520)


def set_bill_qa_example_question(question: str) -> None:
    st.session_state["bill_qa_question_input"] = question


def bill_qa_result_signature(result: Dict[str, Any]) -> str:
    bills = list(result.get("sponsored_bills", []) or [])
    parts = [
        str(result.get("member_name", "")),
        str(len(bills)),
        "|".join(str(row.get("BILL_NO") or row.get("BILL_ID") or "") for row in bills[:5]),
    ]
    return "::".join(parts)


def reset_bill_qa_if_needed(result: Dict[str, Any]) -> None:
    signature = bill_qa_result_signature(result)
    if st.session_state.get("bill_qa_signature") == signature:
        return
    st.session_state["bill_qa_signature"] = signature
    st.session_state["bill_qa_messages"] = []
    st.session_state["bill_qa_question_input"] = ""
    st.session_state.pop("bill_qa_pending_question", None)
    st.session_state.pop("bill_qa_clear_input", None)


def render_bill_qa_message(message: Dict[str, Any], index: int) -> None:
    role = message.get("role", "assistant")
    with st.chat_message(role):
        st.markdown(str(message.get("content", "")))
        evidence = list(message.get("evidence", []) or [])
        if role == "assistant" and evidence:
            with st.expander("사용한 근거 법안", expanded=False):
                df = as_dataframe(evidence, ["역할", "대수", "의안번호", "법안명", "발의일", "소관위원회", "처리결과", "링크"])
                st.dataframe(df, use_container_width=True, height=min(360, 80 + len(df) * 36))
                csv = df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "근거 법안 CSV 다운로드",
                    data=csv,
                    file_name=f"bill_qa_evidence_{index}.csv",
                    mime="text/csv",
                    key=f"bill_qa_evidence_download_{index}",
                )


def render_bill_qa_tab(result: Dict[str, Any]) -> None:
    reset_bill_qa_if_needed(result)
    bills = list(result.get("sponsored_bills", []) or [])

    st.subheader("발의법안 Q&A 챗봇")
    st.caption("현재 조회된 발의법안 데이터만 기준으로 답변합니다. 표결, 뉴스, 정당 일치도 질문은 해당 탭의 원자료를 확인해 주세요.")
    if not bills:
        st.info("현재 조회된 발의법안 데이터가 없어 Q&A를 실행할 수 없습니다.")
        return

    with st.expander("사용 방식과 답변 범위", expanded=False):
        st.markdown(
            """
- 이 챗봇은 현재 분석 결과에 포함된 발의법안 목록, 처리결과 통계, 소관위원회 통계만 사용합니다.
- 답변에는 필요에 따라 `판단 기준:` 또는 `확인 자료:`를 붙여 해석 기준과 법안 예시를 구분해 제시합니다.
- 현재 조회 데이터에서 확인되지 않는 내용은 확인되지 않는다고 답변합니다.
"""
        )

    key_error = current_gemini_key_validation_error()
    if key_error == "missing":
        st.warning(
            "발의법안 Q&A 챗봇을 사용하려면 페이지 상단의 `Gemini API 키` 입력란에 본인의 키를 입력해야 합니다. "
            "키 입력 후 다시 이 탭에서 예상 질문을 클릭하거나 직접 질문할 수 있습니다."
        )
        st.caption("키 발급: https://aistudio.google.com/apikey")
        return
    if key_error:
        st.error("Gemini API 키 형식이 올바르지 않습니다. 키에 한글, 공백, 줄바꿈이 섞이지 않았는지 확인해 주세요.")
        st.caption("키 발급: https://aistudio.google.com/apikey")
        return

    st.markdown("#### 예상 질문")
    question_columns = st.columns(2)
    for index, question in enumerate(BILL_QA_EXAMPLE_QUESTIONS):
        with question_columns[index % 2]:
            st.button(
                question,
                key=f"bill_qa_example_{index}",
                use_container_width=True,
                on_click=set_bill_qa_example_question,
                args=(question,),
            )

    st.divider()
    if st.button("대화 기록 지우기", key="clear_bill_qa_chat"):
        st.session_state["bill_qa_messages"] = []
        st.session_state["bill_qa_question_input"] = ""
        st.session_state.pop("bill_qa_clear_input", None)
        st.rerun()

    messages = list(st.session_state.get("bill_qa_messages", []) or [])
    if not messages:
        st.info("예상 질문을 클릭하거나 직접 질문을 입력해 보세요.")
    for index, message in enumerate(messages):
        render_bill_qa_message(message, index)

    st.divider()
    if st.session_state.pop("bill_qa_clear_input", False):
        st.session_state["bill_qa_question_input"] = ""

    question_value = st.text_input(
        "발의법안 데이터에 대해 질문하세요",
        key="bill_qa_question_input",
        placeholder="예: 청년 관련 법안이 있어?",
    )
    ask_clicked = st.button("질문하기", type="primary", key="submit_bill_qa_question")

    question_to_submit = normalize_text(question_value) if ask_clicked else ""

    if question_to_submit:
        st.session_state.setdefault("bill_qa_messages", []).append({"role": "user", "content": question_to_submit})
        with st.spinner("현재 조회된 발의법안 데이터를 기준으로 답변을 생성하고 있습니다."):
            try:
                answer_payload = generate_bill_qa_answer(result, question_to_submit)
            except Exception as error:
                answer_payload = {
                    "answer": f"답변 생성 중 오류가 발생했습니다: {error}",
                    "evidence": [],
                    "source": "error",
                }
        st.session_state["bill_qa_messages"].append(
            {
                "role": "assistant",
                "content": answer_payload.get("answer", ""),
                "evidence": answer_payload.get("evidence", []),
                "source": answer_payload.get("source", ""),
            }
        )
        st.session_state["bill_qa_clear_input"] = True
        st.rerun()


def render_votes_tab(result: Dict[str, Any]) -> None:
    summary = result.get("vote_summary_by_term", [])
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("전체 표결", f"{len(result.get('vote_records', [])):,}건")
    c2.metric("찬성", f"{len(result.get('yes_votes', [])):,}건")
    c3.metric("반대", f"{len(result.get('no_votes', [])):,}건")
    c4.metric("기권", f"{len(result.get('abstain_votes', [])):,}건")
    c5.metric("불참", f"{len(result.get('absent_votes', [])):,}건")
    render_vote_summary_chart(result)
    vote_interpretation_text = clean_vote_interpretation_text(result.get("vote_interpretation_analysis"))
    if vote_interpretation_text:
        st.subheader("표결 통계 해석 시 주의할 점")
        st.markdown(vote_interpretation_text)
    st.divider()

    vote_cols = ["_VOTE_RESULT", "AGE", "VOTE_DATE", "BILL_NO", "BILL_NAME", "LAW_TITLE", "CURR_COMMITTEE", "POLY_NM", "ORIG_NM", "BILL_URL"]

    render_vote_summary_text("대수별 표결 요약", summary)
    render_vote_fetch_status_text("표결 조회 상태", result.get("vote_fetch_stats", []))

    vote_tabs = st.tabs(["찬성", "반대", "기권", "불참", "전체"])
    with vote_tabs[0]:
        render_dataframe("찬성 표결", result.get("yes_votes", []), vote_cols)
    with vote_tabs[1]:
        render_dataframe("반대 표결", result.get("no_votes", []), vote_cols)
    with vote_tabs[2]:
        render_dataframe("기권 표결", result.get("abstain_votes", []), vote_cols)
    with vote_tabs[3]:
        render_dataframe("불참 표결", result.get("absent_votes", []), vote_cols)
    with vote_tabs[4]:
        render_dataframe("전체 표결", result.get("vote_records", []), vote_cols, height=520)


def render_alignment_tab(result: Dict[str, Any]) -> None:
    apply_completed_party_alignment_reanalysis()
    result = st.session_state.get("member_activity_result", result)
    summary = result.get("party_alignment_summary", [])
    if summary:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("일치", f"{metric_value(summary, '일치'):,}건")
        c2.metric("이탈", f"{metric_value(summary, '이탈'):,}건")
        c3.metric("불참", f"{metric_value(summary, '불참'):,}건")
        c4.metric("판정 제외", f"{metric_value(summary, '판정제외'):,}건")

    record_cols = ["분류", "AGE", "VOTE_DATE", "BILL_NO", "BILL_NAME", "정당", "의원표결", "정당다수입장", "정당내분포", "제외사유"]

    render_alignment_summary_charts(result)
    render_alignment_method_note()
    st.divider()
    render_alignment_summary_text("정당 다수 입장 일치 분석 요약", summary)
    render_alignment_fetch_status_text("정당 일치도 분석 상태", result.get("party_alignment_fetch_stats", []))

    with st.expander("정당 다수 입장 이탈 표결", expanded=True):
        render_dataframe("이탈 표결", result.get("party_alignment_dissent_votes", []), record_cols)
    with st.expander("판정 제외 표결"):
        render_dataframe("판정 제외 표결", result.get("party_alignment_excluded_votes", []), record_cols)

    st.divider()
    st.subheader("정당 일치도 범위 재분석")
    st.caption("초기 조회된 표결 대수 안에서 최근 N건 기준으로 정당 일치도만 다시 계산합니다. 발의법안, 표결 목록, 최근 이슈는 다시 조회하지 않습니다.")
    reanalysis_status = st.session_state.get("alignment_reanalysis_status")
    reanalysis_message = st.session_state.get("alignment_reanalysis_message")
    reanalysis_future = st.session_state.get("alignment_reanalysis_future")
    if isinstance(reanalysis_future, Future) and not reanalysis_future.done():
        st.info(reanalysis_message or "정당 일치도 재분석이 진행 중입니다. 다른 탭을 확인하다가 돌아와 결과를 반영할 수 있습니다.")
        if st.button("재분석 완료 확인", type="secondary"):
            apply_completed_party_alignment_reanalysis()
            st.rerun()
    elif reanalysis_status == "complete" and reanalysis_message:
        st.success(reanalysis_message)
    elif reanalysis_status == "error" and reanalysis_message:
        st.error(reanalysis_message)

    last_limit = result.get("_alignment_reanalysis_last_limit")
    if last_limit:
        st.caption(f"현재 정당 일치도 탭은 최근 {int(last_limit):,}건 기준 재분석 결과를 반영하고 있습니다.")

    current_limit = int(result.get("party_alignment_vote_limit", DEFAULT_PARTY_ALIGNMENT_LIMIT) or DEFAULT_PARTY_ALIGNMENT_LIMIT)
    current_limit = min(max(current_limit, 50), APP_PARTY_ALIGNMENT_SLIDER_MAX)
    limit = st.slider(
        "정당 일치도 최근 N건",
        min_value=50,
        max_value=APP_PARTY_ALIGNMENT_SLIDER_MAX,
        value=current_limit,
        step=50,
        help=(
            "정당 다수 입장과 의원 표결을 비교할 최근 표결 수입니다. "
            "값을 높이면 더 넓은 범위를 보지만 의안별 전체 표결을 추가 조회해 시간이 길어질 수 있습니다."
        ),
    )
    running = isinstance(st.session_state.get("alignment_reanalysis_future"), Future)
    if st.button("정당 일치도 재분석", type="secondary", disabled=running):
        start_party_alignment_reanalysis(result, int(limit))
        st.rerun()


def render_news_tab(result: Dict[str, Any]) -> None:
    st.caption(f"검색 쿼리: {result.get('recent_news_query', '')}")
    render_news_table_timeline(result)
    st.warning("이 섹션은 웹 검색 결과 기반이며 열린국회정보 API의 공식 의정활동 데이터가 아닙니다.", icon="⚠️")
    with st.expander("최근 기사 목록"):
        render_dataframe(
            "최근 기사 목록",
            result.get("recent_news_items", []),
            ["title", "source", "published", "url"],
            height=420,
        )


def render_raw_tab(result: Dict[str, Any]) -> None:
    with st.expander("워크플로우 요약 Markdown", expanded=True):
        st.code(result.get("summary", ""), language="markdown")
    with st.expander("오류 및 확인 메시지"):
        errors = result.get("errors", [])
        if errors:
            for error in errors:
                st.error(error)
        else:
            st.success("오류 메시지가 없습니다.")
    with st.expander("전체 결과 JSON"):
        st.json(result, expanded=False)


def render_additional_vote_lookup(result: Dict[str, Any]) -> None:
    additional_terms = result.get("additional_vote_terms", [])
    if not additional_terms:
        return
    st.divider()
    st.subheader("다른 재임 대수 표결 상세 조회")
    st.caption(f"추가 조회 가능 대수: {format_terms(additional_terms)}")
    selected = st.multiselect(
        "추가로 조회할 대수",
        options=additional_terms,
        format_func=lambda term: f"{term}대",
    )
    if st.button("선택 대수 표결 조회", disabled=not selected):
        state = {
            **result,
            "vote_terms": selected,
            "errors": [],
            "show_progress": False,
        }
        with st.status("선택한 대수의 표결을 조회하고 있습니다.", expanded=True) as status:
            update = make_fetch_votes_node(get_client())(state)
            status.update(label="추가 표결 조회 완료", state="complete", expanded=False)
        st.session_state["additional_vote_update"] = update
        st.rerun()

    if st.session_state.get("additional_vote_update"):
        update = st.session_state["additional_vote_update"]
        st.info("아래는 추가 대수 표결 조회 결과입니다.")
        render_vote_summary_text("추가 대수별 표결 요약", update.get("vote_summary_by_term", []))
        render_vote_fetch_status_text("추가 표결 조회 상태", update.get("vote_fetch_stats", []))


def render_result(result: Dict[str, Any]) -> None:
    tabs = st.tabs(["요약", "발의 법안", "발의법안 Q&A 챗봇", "표결", "정당 일치도", "최근 이슈", "원자료"])
    with tabs[0]:
        render_summary_tab(result)
    with tabs[1]:
        render_bills_tab(result)
    with tabs[2]:
        render_bill_qa_tab(result)
    with tabs[3]:
        render_votes_tab(result)
        render_additional_vote_lookup(result)
    with tabs[4]:
        render_alignment_tab(result)
    with tabs[5]:
        render_news_tab(result)
    with tabs[6]:
        render_raw_tab(result)


st.title("🕵️‍♂️국회의원 의정활동 추적")
st.caption("열린국회정보 API와 LangGraph를 이용해 의원의 발의법안, 표결, 정당 다수 입장 일치도, 최근 이슈를 조회합니다.")

if "member_name_input" not in st.session_state:
    st.session_state["member_name_input"] = ""

st.info("본인의 Gemini API 키를 입력하세요. 키는 브라우저에만 저장됩니다.")
gemini_api_key = st.text_input(
    "Gemini API 키",
    type="password",
    value=st.session_state.get("GEMINI_API_KEY", ""),
    placeholder="발급 받은 key를 입력하세요",
)
if gemini_api_key.strip():
    configure_user_gemini_key(gemini_api_key)
st.caption("키 발급: https://aistudio.google.com/apikey")
st.caption(f"사용 모델: `{DEFAULT_LLM_MODEL}`")
if gemini_api_key:
    configure_user_gemini_key(gemini_api_key)

with st.sidebar:
    st.header("조회 설정")
    st.caption(f"워크플로우 버전: {WORKFLOW_VERSION}")
    member_name = st.text_input("국회의원 이름", key="member_name_input")
    st.caption("의원 목록 조회 카드에서 의원을 선택하거나, 직접 입력하여 분석을 시작하세요.")
    selected_directory_member = str(st.session_state.get("selected_directory_member") or "").strip()
    if selected_directory_member:
        st.caption(f"현재 선택: {selected_directory_member} 의원")
    st.divider()
    st.subheader("분석 옵션")
    bills_enabled = st.checkbox(
        "발의법안",
        value=DEFAULT_ENABLE_COSPONSOR_SCAN,
        help=(
            "의원의 대표발의와 공동발의 법안을 모두 조회합니다. "
            "공동발의 누락을 줄이기 위해 발의법안 목록을 추가 스캔하므로 끄면 더 빠르지만 발의법안 탭과 입법 관심 분야 분석이 비어 있을 수 있습니다."
        ),
    )
    vote_detail_enabled = st.checkbox(
        "초기 표결 상세 조회",
        value=True,
        help=(
            "초기 분석에서 의원별 찬성·반대·기권·불참 표결 원자료를 조회합니다. "
            "끄면 표결 탭은 비어 있지만 정당 다수 입장 일치 분석은 별도 옵션으로 실행할 수 있습니다."
        ),
    )
    party_alignment_enabled = st.checkbox(
        "정당 다수 입장 일치 분석",
        value=True,
        help=(
            "각 표결에서 같은 정당 의원들의 다수 표결값을 계산하고, 해당 의원의 표결이 그 다수 입장과 일치했는지 비교합니다. "
            "의안별 전체 의원 표결을 조회하므로 범위를 키우면 시간이 길어질 수 있습니다."
        ),
    )
    recent_news_enabled = st.checkbox(
        "최근 뉴스 검색",
        value=True,
        help=(
            "Google News RSS에서 의원명과 정당명을 검색해 최근 공개 이슈를 수집합니다. "
            "공식 의정활동 데이터가 아니라 웹 검색 기반 맥락입니다."
        ),
    )
    llm_insights_enabled = st.checkbox(
        "LLM 분석 사용",
        value=bool(gemini_api_key.strip()),
        disabled=not bool(gemini_api_key.strip()),
        help=(
            "Gemini로 발의법안 관심 분야, 표결 해석 메모, 최근 뉴스 요약을 생성합니다. "
            "끄면 가능한 항목은 규칙 기반 요약 또는 원자료 중심으로 표시됩니다."
        ),
    )
    if not gemini_api_key.strip():
        st.caption("LLM 분석을 사용하려면 페이지 상단에 Gemini API 키를 입력하세요.")

    st.divider()
    st.subheader("세부 설정")
    st.caption("조회 범위와 실행 비용을 조정합니다. 기본값으로도 바로 분석할 수 있습니다.")
    st.caption(
        f"기본값: 발의법안 최근 재임 대수 · 표결 최근 {APP_DEFAULT_VOTE_DETAIL_LIMIT}건 · 정당 일치도 최근 {DEFAULT_PARTY_ALIGNMENT_LIMIT}건"
    )
    advanced_options_enabled = st.toggle(
        "세부 설정 열기",
        value=False,
        help="발의법안 조회 범위, 초기 표결 상세 범위, 정당 일치도 조회 범위를 직접 조정합니다.",
    )
    bill_scope_options = {
        "recent": "최근 재임 대수",
        "all": "전체 재임 대수",
    }
    bill_term_scope = APP_DEFAULT_BILL_TERM_SCOPE
    vote_detail_limit = APP_DEFAULT_VOTE_DETAIL_LIMIT
    party_alignment_limit = DEFAULT_PARTY_ALIGNMENT_LIMIT
    if advanced_options_enabled:
        bill_term_scope_label = st.radio(
            "발의법안 조회 범위",
            list(bill_scope_options.values()),
            index=list(bill_scope_options.keys()).index(APP_DEFAULT_BILL_TERM_SCOPE),
            horizontal=True,
            disabled=not bills_enabled,
            help=(
                "최근 재임 대수는 현재 활동 맥락을 보기 좋고, 전체 재임 대수는 누적 입법 활동을 보기 좋습니다. "
                "공동발의 네트워크와 입법 관심 분야 분석도 이 범위를 기준으로 계산됩니다."
            ),
        )
        bill_term_scope = next(key for key, label in bill_scope_options.items() if label == bill_term_scope_label)
        vote_detail_limit = st.slider(
            "초기 표결 상세 범위",
            min_value=50,
            max_value=APP_VOTE_DETAIL_SLIDER_MAX,
            value=APP_DEFAULT_VOTE_DETAIL_LIMIT,
            step=50,
            disabled=not vote_detail_enabled,
            help=(
                "초기 표결 상세 조회 범위입니다. 최솟값은 최근 50건입니다. "
                f"슬라이더를 오른쪽 끝({APP_VOTE_DETAIL_SLIDER_MAX})까지 당기면 해당 대수 전체 표결을 조회합니다."
            ),
        )
        party_alignment_limit = st.slider(
            "정당 일치도 최근 N건",
            min_value=50,
            max_value=APP_PARTY_ALIGNMENT_SLIDER_MAX,
            value=DEFAULT_PARTY_ALIGNMENT_LIMIT,
            step=50,
            disabled=not party_alignment_enabled,
            help=(
                "정당 다수 입장 일치도를 계산할 최근 표결 수입니다. "
                f"최댓값은 {APP_PARTY_ALIGNMENT_SLIDER_MAX}건이며, 의안별 전체 의원 표결을 조회하는 비용을 제한하기 위한 상한입니다."
            ),
        )
        max_vote_bills = 0 if vote_detail_enabled and vote_detail_limit == APP_VOTE_DETAIL_SLIDER_MAX else int(vote_detail_limit)
        if vote_detail_enabled:
            st.caption("초기 표결 상세 선택 범위: 전체 조회" if max_vote_bills == 0 else f"초기 표결 상세 선택 범위: 최근 {max_vote_bills}건")
        if vote_detail_enabled and max_vote_bills == 0:
            st.warning("전체 표결 조회는 첫 실행 또는 캐시 만료 시 수 분 이상 걸릴 수 있습니다.", icon="⏳")
    else:
        max_vote_bills = int(vote_detail_limit)

    run_button = st.button("분석 시작", type="primary", use_container_width=True)
    render_theoretical_background_sidebar()

options = {
    "party_alignment_vote_limit": int(party_alignment_limit),
    "max_vote_bills_to_scan": int(max_vote_bills),
    "force_vote_refresh": False,
    "max_bill_pages": APP_BILL_PAGE_SCAN,
    "bill_term_scope": bill_term_scope,
    "vote_fetch_workers": DEFAULT_VOTE_FETCH_WORKERS,
    "bills_enabled": bool(bills_enabled),
    "vote_detail_enabled": bool(vote_detail_enabled),
    "enable_cosponsor_scan": bool(bills_enabled),
    "max_cosponsor_scan_pages": APP_COSPONSOR_SCAN_PAGES,
    "party_alignment_enabled": bool(party_alignment_enabled),
    "llm_insights_enabled": bool(llm_insights_enabled),
    "recent_news_enabled": bool(recent_news_enabled),
    "member_party_lookup": get_member_party_lookup_cached(),
    "async_open_assembly_enabled": DEFAULT_ASYNC_OPEN_ASSEMBLY_ENABLED,
    "async_open_assembly_concurrency": DEFAULT_ASYNC_OPEN_ASSEMBLY_CONCURRENCY,
}

if run_button:
    st.session_state.pop("alignment_update", None)
    st.session_state.pop("alignment_reanalysis_future", None)
    st.session_state.pop("alignment_reanalysis_limit", None)
    st.session_state.pop("alignment_reanalysis_status", None)
    st.session_state.pop("alignment_reanalysis_message", None)
    st.session_state.pop("additional_vote_update", None)
    if not member_name.strip():
        st.error("분석할 국회의원 이름을 입력하세요.")
    elif llm_insights_enabled and not gemini_api_key.strip():
        st.error("LLM 분석을 사용하려면 Gemini API 키를 입력하세요.")
    else:
        try:
            if gemini_api_key.strip():
                configure_user_gemini_key(gemini_api_key)
            st.session_state["member_activity_result"] = run_analysis(member_name, options)
        except Exception as error:
            st.error(f"분석 중 오류가 발생했습니다: {error}")

if st.session_state.get("member_activity_result") and st.session_state["member_activity_result"].get("_workflow_version") != WORKFLOW_VERSION:
    st.session_state.pop("member_activity_result", None)

render_member_directory_section()

if st.session_state.get("member_activity_result"):
    apply_completed_party_alignment_reanalysis()
    render_result(st.session_state["member_activity_result"])
