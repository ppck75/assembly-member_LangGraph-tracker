from __future__ import annotations

import io
import os
from typing import Any, Dict, Iterable, List

import pandas as pd
import streamlit as st

from src.member_activity_workflow import (
    DEFAULT_ENABLE_COSPONSOR_SCAN,
    DEFAULT_MAX_BILL_PAGES,
    DEFAULT_MAX_COSPONSOR_SCAN_PAGES,
    DEFAULT_MAX_VOTE_BILLS_TO_SCAN,
    DEFAULT_MAX_VOTE_PAGES,
    DEFAULT_PARTY_ALIGNMENT_LIMIT,
    DEFAULT_TERMS,
    DEFAULT_VOTE_FETCH_WORKERS,
    DEFAULT_LLM_MODEL,
    analyze_vote_interpretation_node,
    format_terms,
    make_analyze_party_alignment_node,
    make_fetch_votes_node,
    parse_term_selection,
)
from src.streamlit_runtime import get_client, get_member_activity_app


st.set_page_config(
    page_title="국회의원 활동 추적",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_BILL_PAGE_SCAN = DEFAULT_MAX_BILL_PAGES
APP_COSPONSOR_SCAN_PAGES = DEFAULT_MAX_COSPONSOR_SCAN_PAGES


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


def metric_value(rows: List[Dict[str, Any]], key: str) -> int:
    return sum(int(row.get(key, 0) or 0) for row in rows)


def build_initial_state(member_name: str, options: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "member_name": member_name.strip(),
        "assembly_terms": DEFAULT_TERMS,
        "recent_limit": 10,
        "max_bill_pages": options["max_bill_pages"],
        "max_vote_pages": DEFAULT_MAX_VOTE_PAGES,
        "max_vote_bills_to_scan": options["max_vote_bills_to_scan"],
        "vote_fetch_workers": options["vote_fetch_workers"],
        "party_alignment_enabled": options["party_alignment_enabled"],
        "party_alignment_vote_limit": options["party_alignment_vote_limit"],
        "llm_insights_enabled": options["llm_insights_enabled"],
        "enable_cosponsor_scan": options["enable_cosponsor_scan"],
        "max_cosponsor_scan_pages": options["max_cosponsor_scan_pages"],
        "show_progress": False,
    }


WORKFLOW_STEPS = [
    ("normalize_user_request", "입력값 정리", "의원명과 조회 옵션을 정리합니다."),
    ("get_member_info", "의원 기본정보 조회", "재임 대수, 정당, 선거구, 위원회 정보를 확인합니다."),
    ("search_member_bills", "발의법안 조회", "대표발의와 공동발의 법안을 수집합니다."),
    ("analyze_legislative_interests", "입법 관심 분야 분석", "발의법안 통계를 바탕으로 관심 의제를 요약합니다."),
    ("get_all_member_votes", "표결 상세 조회", "본회의 표결 기록을 조회하고 찬성·반대·기권·불참으로 분류합니다."),
    ("analyze_party_alignment", "정당 다수 입장 일치도 분석", "같은 정당 의원 다수 입장과의 일치 여부를 계산합니다."),
    ("interpret_vote_statistics", "표결 해석 메모 생성", "분석 범위와 불참 비중을 고려한 해석 주의사항을 만듭니다."),
    ("search_recent_member_news", "최근 이슈 검색", "최근 뉴스 검색 결과를 바탕으로 공개 이슈를 요약합니다."),
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
    app = get_member_activity_app()

    progress = st.progress(0, text="분석을 시작합니다.")
    step_box = st.empty()
    detail_box = st.empty()
    completed_box = st.container()
    completed_steps: List[str] = []

    def progress_callback(message: str) -> None:
        detail_box.caption(f"세부 진행: {message}")

    initial_state["progress_callback"] = progress_callback

    with st.status("의원 활동을 조회하고 있습니다.", expanded=True) as status:
        st.write("의원 기본정보, 발의법안, 표결정보, 정당 일치도, 최근 이슈를 순서대로 분석합니다.")
        for event in app.stream(initial_state, stream_mode="updates"):
            for node_name, node_update in event.items():
                label = WORKFLOW_STEP_LABELS.get(node_name, node_name)
                description = WORKFLOW_STEP_DESCRIPTIONS.get(node_name, "")
                percent = WORKFLOW_PROGRESS.get(node_name, min(95, len(completed_steps) * 10))

                progress.progress(percent, text=f"{label} 완료 ({percent}%)")
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
    return result


def render_summary_tab(result: Dict[str, Any]) -> None:
    st.markdown(result.get("summary", "요약 결과가 없습니다."))


def render_bills_tab(result: Dict[str, Any]) -> None:
    bills = result.get("sponsored_bills", [])
    rep = result.get("representative_bills", [])
    co = result.get("cosponsored_bills", [])

    c1, c2, c3 = st.columns(3)
    c1.metric("발의 법안", f"{len(bills):,}건")
    c2.metric("대표발의", f"{len(rep):,}건")
    c3.metric("공동발의", f"{len(co):,}건")

    stat_cols = ["처리결과", "건수"]
    committee_cols = ["소관위원회", "건수"]
    bill_cols = ["_ROLE", "AGE", "BILL_NO", "BILL_NAME", "PROPOSE_DT", "COMMITTEE", "PROC_RESULT", "DETAIL_LINK"]

    left, right = st.columns(2)
    with left:
        render_dataframe("처리결과별 통계", result.get("bill_result_stats", []), stat_cols, height=260)
    with right:
        render_dataframe("소관위원회별 통계", result.get("bill_committee_stats", []), committee_cols, height=260)

    with st.expander("최근 대표발의 법안", expanded=True):
        render_dataframe("대표발의", rep[:20], bill_cols)
    with st.expander("최근 공동발의 법안", expanded=True):
        render_dataframe("공동발의", co[:20], bill_cols)
    with st.expander("발의 법안 전체 원자료"):
        render_dataframe("발의 법안 전체", bills, bill_cols, height=520)


def render_votes_tab(result: Dict[str, Any]) -> None:
    summary = result.get("vote_summary_by_term", [])
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("전체 표결", f"{len(result.get('vote_records', [])):,}건")
    c2.metric("찬성", f"{len(result.get('yes_votes', [])):,}건")
    c3.metric("반대", f"{len(result.get('no_votes', [])):,}건")
    c4.metric("기권", f"{len(result.get('abstain_votes', [])):,}건")
    c5.metric("불참", f"{len(result.get('absent_votes', [])):,}건")

    vote_summary_cols = ["AGE", "전체", "찬성", "반대", "기권", "불참"]
    vote_fetch_cols = ["AGE", "표결의안", "상세조회성공", "상세조회실패", "매칭표결", "캐시사용"]
    vote_cols = ["_VOTE_RESULT", "AGE", "VOTE_DATE", "BILL_NO", "BILL_NAME", "LAW_TITLE", "CURR_COMMITTEE", "POLY_NM", "ORIG_NM", "BILL_URL"]

    render_dataframe("대수별 표결 요약", summary, vote_summary_cols, height=180)
    render_dataframe("표결 조회 상태", result.get("vote_fetch_stats", []), vote_fetch_cols, height=180)

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
    summary = result.get("party_alignment_summary", [])
    if summary:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("일치", f"{metric_value(summary, '일치'):,}건")
        c2.metric("이탈", f"{metric_value(summary, '이탈'):,}건")
        c3.metric("불참", f"{metric_value(summary, '불참'):,}건")
        c4.metric("판정 제외", f"{metric_value(summary, '판정제외'):,}건")

    summary_cols = ["AGE", "분석표결", "일치", "이탈", "불참", "판정제외", "일치율", "불참포함일치율"]
    stat_cols = ["AGE", "분석대상표결", "BILL_ID캐시적중", "API신규조회대상", "상세조회성공", "상세조회실패", "분석완료"]
    record_cols = ["분류", "AGE", "VOTE_DATE", "BILL_NO", "BILL_NAME", "정당", "의원표결", "정당다수입장", "정당내분포", "제외사유"]

    render_dataframe("정당 다수 입장 일치 분석 요약", summary, summary_cols, height=180)
    render_dataframe("정당 일치도 분석 상태", result.get("party_alignment_fetch_stats", []), stat_cols, height=180)

    if result.get("vote_interpretation_analysis"):
        st.subheader("표결 통계 해석 시 주의할 점")
        st.markdown(result["vote_interpretation_analysis"])

    with st.expander("정당 다수 입장 이탈 표결", expanded=True):
        render_dataframe("이탈 표결", result.get("party_alignment_dissent_votes", []), record_cols)
    with st.expander("판정 제외 표결"):
        render_dataframe("판정 제외 표결", result.get("party_alignment_excluded_votes", []), record_cols)

    st.divider()
    st.subheader("정당 일치도 범위 재분석")
    st.caption("초기 조회된 표결 대수 안에서 최근 N건 또는 전체 기준으로 정당 일치도만 다시 계산합니다.")
    limit = st.number_input("최근 N건 기준, 전체는 0", min_value=0, max_value=5000, value=int(result.get("party_alignment_vote_limit", DEFAULT_PARTY_ALIGNMENT_LIMIT)), step=10)
    if st.button("정당 일치도 재분석", type="secondary"):
        state = {
            **result,
            "party_alignment_vote_limit": int(limit),
            "errors": [],
            "show_progress": False,
        }
        with st.status("정당 일치도를 재분석하고 있습니다.", expanded=True) as status:
            update = make_analyze_party_alignment_node(get_client())(state)
            interpretation = analyze_vote_interpretation_node({**state, **update})
            update = {**update, **interpretation}
            status.update(label="재분석 완료", state="complete", expanded=False)
        st.session_state["alignment_update"] = update
        st.rerun()

    if st.session_state.get("alignment_update"):
        st.info("아래는 방금 실행한 정당 일치도 재분석 결과입니다.")
        update = st.session_state["alignment_update"]
        render_dataframe("재분석 요약", update.get("party_alignment_summary", []), summary_cols, height=180)
        render_dataframe("재분석 상태", update.get("party_alignment_fetch_stats", []), stat_cols, height=180)
        if update.get("vote_interpretation_analysis"):
            st.markdown("#### 재분석 표결 통계 해석 시 주의할 점")
            st.markdown(update["vote_interpretation_analysis"])


def render_news_tab(result: Dict[str, Any]) -> None:
    st.subheader("최근 의원 관련 이슈")
    st.caption(f"검색 쿼리: {result.get('recent_news_query', '')}")
    st.markdown(result.get("recent_news_analysis") or "최근 이슈 분석 결과가 없습니다.")
    st.warning("이 섹션은 웹 검색 결과 기반이며 열린국회정보 API의 공식 의정활동 데이터가 아닙니다.", icon="⚠️")
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
        render_dataframe("추가 대수별 표결 요약", update.get("vote_summary_by_term", []), ["AGE", "전체", "찬성", "반대", "기권", "불참"], height=180)
        render_dataframe("추가 표결 조회 상태", update.get("vote_fetch_stats", []), ["AGE", "표결의안", "상세조회성공", "상세조회실패", "매칭표결", "캐시사용"], height=180)


def render_result(result: Dict[str, Any]) -> None:
    tabs = st.tabs(["요약", "발의 법안", "표결", "정당 일치도", "최근 이슈", "원자료"])
    with tabs[0]:
        render_summary_tab(result)
    with tabs[1]:
        render_bills_tab(result)
    with tabs[2]:
        render_votes_tab(result)
        render_additional_vote_lookup(result)
    with tabs[3]:
        render_alignment_tab(result)
    with tabs[4]:
        render_news_tab(result)
    with tabs[5]:
        render_raw_tab(result)


st.title("국회의원 활동 추적")
st.caption("열린국회정보 API와 LangGraph를 이용해 의원의 발의법안, 표결, 정당 다수 입장 일치도, 최근 이슈를 조회합니다.")

st.info("본인의 Gemini API 키를 입력하세요. 키는 브라우저에만 저장됩니다.")
gemini_api_key = st.text_input(
    "Gemini API 키",
    type="password",
    value=st.session_state.get("GEMINI_API_KEY", ""),
    placeholder="AIza...",
)
st.caption("키 발급: https://aistudio.google.com/apikey")
st.caption(f"사용 모델: `{DEFAULT_LLM_MODEL}`")
if gemini_api_key:
    configure_user_gemini_key(gemini_api_key)

with st.sidebar:
    st.header("조회 설정")
    member_name = st.text_input("국회의원 이름", placeholder="예: 이준석, 추미애, 주광덕")
    st.divider()
    st.subheader("분석 옵션")
    party_alignment_limit = st.number_input(
        "정당 일치도 최근 N건, 전체는 0",
        min_value=0,
        max_value=5000,
        value=DEFAULT_PARTY_ALIGNMENT_LIMIT,
        step=10,
        help=(
            "정당 다수 입장과 의원 표결을 비교할 표결 수입니다. "
            "50은 빠르게 최근 경향을 봅니다. 값을 높이면 더 많은 표결을 반영하지만 오래 걸릴 수 있습니다. "
            "0은 해당 대수 전체 표결을 분석하므로 가장 무겁습니다."
        ),
    )
    vote_scan_mode = st.radio(
        "초기 표결 상세 조회 범위",
        options=["전체", "최근 50건", "최근 200건"],
        index=2,
        help=(
            "처음 화면에 보여줄 의원별 찬성·반대·기권·불참 표결의 범위입니다. "
            "최근 50건은 가장 빠르고, 최근 200건은 속도와 정보량의 균형형입니다. "
            "전체는 해당 대수 표결을 모두 조회해 가장 포괄적이지만 첫 실행 시 수 분 이상 걸릴 수 있습니다."
        ),
    )
    if vote_scan_mode == "전체":
        st.warning("전체 표결 조회는 첫 실행 또는 캐시 만료 시 수 분 이상 걸릴 수 있습니다.", icon="⏳")
    max_vote_bills = {
        "전체": DEFAULT_MAX_VOTE_BILLS_TO_SCAN,
        "최근 50건": 50,
        "최근 200건": 200,
    }[vote_scan_mode]
    enable_cosponsor_scan = st.checkbox(
        "공동발의 보강 스캔",
        value=DEFAULT_ENABLE_COSPONSOR_SCAN,
        help="공동발의 법안 누락을 줄이기 위해 발의법안 목록을 추가 확인합니다. 끄면 더 빠르지만 공동발의가 일부 빠질 수 있습니다.",
    )
    party_alignment_enabled = st.checkbox("정당 다수 입장 일치 분석", value=True)
    llm_insights_enabled = st.checkbox("Gemini 해석 사용", value=True, disabled=not bool(gemini_api_key.strip()))
    if not gemini_api_key.strip():
        st.caption("Gemini 해석을 사용하려면 페이지 상단에 API 키를 입력하세요.")
    run_button = st.button("분석 시작", type="primary", use_container_width=True)

options = {
    "party_alignment_vote_limit": int(party_alignment_limit),
    "max_vote_bills_to_scan": int(max_vote_bills),
    "max_bill_pages": APP_BILL_PAGE_SCAN,
    "vote_fetch_workers": DEFAULT_VOTE_FETCH_WORKERS,
    "enable_cosponsor_scan": bool(enable_cosponsor_scan),
    "max_cosponsor_scan_pages": APP_COSPONSOR_SCAN_PAGES,
    "party_alignment_enabled": bool(party_alignment_enabled),
    "llm_insights_enabled": bool(llm_insights_enabled),
}

if run_button:
    st.session_state.pop("alignment_update", None)
    st.session_state.pop("additional_vote_update", None)
    if not member_name.strip():
        st.error("분석할 국회의원 이름을 입력하세요.")
    elif not gemini_api_key.strip():
        st.error("Gemini API 키를 입력하세요.")
    else:
        try:
            configure_user_gemini_key(gemini_api_key)
            st.session_state["member_activity_result"] = run_analysis(member_name, options)
        except Exception as error:
            st.error(f"분석 중 오류가 발생했습니다: {error}")

if st.session_state.get("member_activity_result"):
    render_result(st.session_state["member_activity_result"])
else:
    st.info("왼쪽 사이드바에서 의원 이름을 입력하고 분석을 시작하세요.")
