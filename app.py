from __future__ import annotations

import html
import io
import json
import os
from typing import Any, Dict, Iterable, List

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.member_activity_workflow import (
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
    page_title="국회의원 활동 추적",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_BILL_PAGE_SCAN = DEFAULT_MAX_BILL_PAGES
APP_COSPONSOR_SCAN_PAGES = DEFAULT_MAX_COSPONSOR_SCAN_PAGES
APP_VOTE_DETAIL_SLIDER_MAX = 1000
APP_PARTY_ALIGNMENT_SLIDER_MAX = 500
MEMBER_DIRECTORY_CACHE_TTL_SECONDS = 24 * 60 * 60
MEMBER_DIRECTORY_PAGE_SIZE = 9


@st.cache_data(ttl=MEMBER_DIRECTORY_CACHE_TTL_SECONDS, show_spinner=False)
def load_member_directory_cached(term: int | None = None) -> List[Dict[str, Any]]:
    return fetch_member_directory(get_client(), term=term)


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


def select_directory_member(member_name: str) -> None:
    st.session_state["member_name_input"] = member_name


def render_member_directory_styles() -> None:
    st.markdown(
        """
        <style>
        .member-card {
            min-height: 282px;
            padding: 16px;
            border: 1px solid #dbe4ee;
            border-radius: 18px;
            background:
                linear-gradient(145deg, rgba(255,255,255,0.96), rgba(244,248,252,0.92));
            box-shadow: 0 8px 24px rgba(34, 56, 84, 0.08);
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
        }
        .member-card__label {
            display: inline-block;
            min-width: 54px;
            color: #8190a3;
            font-weight: 700;
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


def render_member_card(member: Dict[str, Any], *, key_prefix: str) -> None:
    name = html.escape(str(member.get("name") or "이름 미확인"))
    party = html.escape(str(member.get("party") or "정당 미확인"))
    region = html.escape(str(member.get("region") or "지역 미확인"))
    committee = html.escape(str(member.get("committee") or "위원회 미확인"))
    term_label = html.escape(str(member.get("term_label") or "대수 미확인"))
    reelection = html.escape(str(member.get("reelection") or "선수 미확인"))
    image_url = html.escape(str(member.get("image_url") or ""))
    image_html = (
        f"<img class='member-card__image' src='{image_url}' alt='{name} 의원 사진'>"
        if image_url
        else f"<div class='member-card__placeholder'>{name[:1]}</div>"
    )

    st.markdown(
        f"""
        <div class="member-card">
            <div class="member-card__top">
                {image_html}
                <div>
                    <p class="member-card__name">{name}</p>
                    <span class="member-card__party">{party}</span>
                </div>
            </div>
            <div class="member-card__row"><span class="member-card__label">지역</span>{region}</div>
            <div class="member-card__row"><span class="member-card__label">위원회</span>{committee}</div>
            <div class="member-card__row"><span class="member-card__label">대수</span>{term_label}</div>
            <div class="member-card__row"><span class="member-card__label">선수</span>{reelection}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.button(
        f"{member.get('name', '의원')} 선택",
        key=f"{key_prefix}_{member.get('_key', member.get('name', 'unknown'))}",
        use_container_width=True,
        on_click=select_directory_member,
        args=(str(member.get("name") or ""),),
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

    term_options = ["전체"] + [f"{term}대" for term in MEMBER_DIRECTORY_TERMS]
    default_term_index = term_options.index(f"{DEFAULT_MEMBER_DIRECTORY_TERM}대")
    selected_term_label = st.selectbox("대수", term_options, index=default_term_index, key=f"{key_prefix}_term")
    st.caption("참고: 열린국회정보 API 기준 20대, 21대, 22대 의원 목록 조회가 가능합니다.")
    selected_term = None if selected_term_label == "전체" else int(str(selected_term_label).replace("대", ""))
    members = [
        member
        for member in all_members
        if selected_term is None or selected_term in [int(term) for term in member.get("terms", [])]
    ]

    if not members:
        st.info("선택한 대수에 해당하는 의원이 없습니다.")
        return

    parties = sorted({str(member.get("party")) for member in members if member.get("party")})
    regions = sorted({str(member.get("region_bucket")) for member in members if member.get("region_bucket")})
    committees = sorted({str(member.get("committee")) for member in members if member.get("committee")})

    search_col, party_col, region_col = st.columns([1.3, 1, 1])
    with search_col:
        query = st.text_input(
            "이름·지역·위원회 검색",
            placeholder="예: 이준석, 경기, 법제사법위원회",
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

    with st.expander("의원 목록 조회", expanded=not st.session_state.get("member_activity_result")):
        st.caption("카드에서 의원을 선택하면 왼쪽 사이드바의 국회의원 이름 입력칸에 반영됩니다.")
        if st.button("의원 목록 불러오기", use_container_width=True):
            st.session_state["show_member_directory"] = True
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
        if heading_text in {"표결 통계 해석 시 주의할 점", "표결 통계 해석 시 주의사항"}:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


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
        "vote_detail_enabled": options["vote_detail_enabled"],
        "max_bill_pages": options["max_bill_pages"],
        "max_vote_pages": DEFAULT_MAX_VOTE_PAGES,
        "max_vote_bills_to_scan": options["max_vote_bills_to_scan"],
        "vote_fetch_workers": options["vote_fetch_workers"],
        "party_alignment_enabled": options["party_alignment_enabled"],
        "party_alignment_vote_limit": options["party_alignment_vote_limit"],
        "llm_insights_enabled": options["llm_insights_enabled"],
        "recent_news_enabled": options["recent_news_enabled"],
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
    app = get_member_activity_app(WORKFLOW_VERSION)

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
    result["_workflow_version"] = WORKFLOW_VERSION
    return result


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


def render_news_table_timeline(result: Dict[str, Any]) -> None:
    items = result.get("recent_news_items", [])
    st.subheader("최근 이슈")
    st.markdown(result.get("recent_news_analysis") or "최근 이슈 분석 결과가 없습니다.")
    if not items:
        st.info("최근 뉴스 데이터가 없습니다.")
        return

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
    st.divider()

    st.subheader("의원 프로필")
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

    render_alignment_summary_charts(result)
    st.divider()
    render_dataframe("정당 다수 입장 일치 분석 요약", summary, summary_cols, height=180)
    render_dataframe("정당 일치도 분석 상태", result.get("party_alignment_fetch_stats", []), stat_cols, height=180)

    with st.expander("정당 다수 입장 이탈 표결", expanded=True):
        render_dataframe("이탈 표결", result.get("party_alignment_dissent_votes", []), record_cols)
    with st.expander("판정 제외 표결"):
        render_dataframe("판정 제외 표결", result.get("party_alignment_excluded_votes", []), record_cols)

    st.divider()
    st.subheader("정당 일치도 범위 재분석")
    st.caption("초기 조회된 표결 대수 안에서 최근 N건 기준으로 정당 일치도만 다시 계산합니다.")
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
        updated_vote_interpretation_text = clean_vote_interpretation_text(update.get("vote_interpretation_analysis"))
        if updated_vote_interpretation_text:
            st.markdown("#### 재분석 표결 통계 해석 시 주의할 점")
            st.markdown(updated_vote_interpretation_text)


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


st.title("🕵️‍♂️국회의원 활동 추적")
st.caption("열린국회정보 API와 LangGraph를 이용해 의원의 발의법안, 표결, 정당 다수 입장 일치도, 최근 이슈를 조회합니다.")

if "member_name_input" not in st.session_state:
    st.session_state["member_name_input"] = ""

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
    member_name = st.text_input("국회의원 이름", placeholder="예: 이준석, 추미애, 주광덕", key="member_name_input")
    st.caption("의원 목록 조회 카드에서 의원을 선택하거나, 직접 입력하여 분석을 시작하세요.")
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
    vote_detail_limit = st.slider(
        "초기 표결 상세 범위",
        min_value=50,
        max_value=APP_VOTE_DETAIL_SLIDER_MAX,
        value=200,
        step=50,
        disabled=not vote_detail_enabled,
        help=(
            "초기 표결 상세 조회 범위입니다. 최솟값은 최근 50건입니다. "
            f"슬라이더를 오른쪽 끝({APP_VOTE_DETAIL_SLIDER_MAX})까지 당기면 해당 대수 전체 표결을 조회합니다."
        ),
    )
    max_vote_bills = 0 if vote_detail_enabled and vote_detail_limit == APP_VOTE_DETAIL_SLIDER_MAX else int(vote_detail_limit)
    if vote_detail_enabled:
        st.caption("선택 범위: 전체 조회" if max_vote_bills == 0 else f"선택 범위: 최근 {max_vote_bills}건")
    if vote_detail_enabled and max_vote_bills == 0:
        st.warning("전체 표결 조회는 첫 실행 또는 캐시 만료 시 수 분 이상 걸릴 수 있습니다.", icon="⏳")
    party_alignment_enabled = st.checkbox(
        "정당 다수 입장 일치 분석",
        value=True,
        help=(
            "각 표결에서 같은 정당 의원들의 다수 표결값을 계산하고, 해당 의원의 표결이 그 다수 입장과 일치했는지 비교합니다. "
            "의안별 전체 의원 표결을 조회하므로 범위를 키우면 시간이 길어질 수 있습니다."
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
    run_button = st.button("분석 시작", type="primary", use_container_width=True)

options = {
    "party_alignment_vote_limit": int(party_alignment_limit),
    "max_vote_bills_to_scan": int(max_vote_bills),
    "max_bill_pages": APP_BILL_PAGE_SCAN,
    "vote_fetch_workers": DEFAULT_VOTE_FETCH_WORKERS,
    "bills_enabled": bool(bills_enabled),
    "vote_detail_enabled": bool(vote_detail_enabled),
    "enable_cosponsor_scan": bool(bills_enabled),
    "max_cosponsor_scan_pages": APP_COSPONSOR_SCAN_PAGES,
    "party_alignment_enabled": bool(party_alignment_enabled),
    "llm_insights_enabled": bool(llm_insights_enabled),
    "recent_news_enabled": bool(recent_news_enabled),
}

if run_button:
    st.session_state.pop("alignment_update", None)
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
    render_result(st.session_state["member_activity_result"])
