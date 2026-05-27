from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

from .member_activity_workflow import AssemblyAPIClient, ENDPOINT_MEMBERS, normalize_space


DEFAULT_MEMBER_DIRECTORY_TERM = 22
MEMBER_DIRECTORY_TERMS = list(range(22, 0, -1))


def _split_latest(value: Any) -> str:
    text = normalize_space(value)
    if not text:
        return ""
    parts = [part.strip() for part in text.split("/") if part.strip()]
    return parts[-1] if parts else text


def _extract_terms(value: Any) -> List[int]:
    terms: List[int] = []
    for number in re.findall(r"\d+", normalize_space(value)):
        term = int(number)
        if 1 <= term <= 30 and term not in terms:
            terms.append(term)
    return terms


def _region_bucket(region: str) -> str:
    text = normalize_space(region)
    if not text:
        return "지역 미확인"
    if text == "비례대표":
        return text
    return text.split()[0]


def normalize_member_directory_row(row: Dict[str, Any], term: int = DEFAULT_MEMBER_DIRECTORY_TERM) -> Dict[str, Any]:
    name = normalize_space(row.get("NAAS_NM"))
    party = _split_latest(row.get("PLPT_NM")) or "정당 미확인"
    region = _split_latest(row.get("ELECD_NM")) or _split_latest(row.get("ELECD_DIV_NM")) or "지역 미확인"
    committee = _split_latest(row.get("BLNG_CMIT_NM") or row.get("CMIT_NM")) or "위원회 미확인"
    terms = _extract_terms(row.get("GTELT_ERACO"))

    return {
        "name": name,
        "party": party,
        "region": region,
        "region_bucket": _region_bucket(region),
        "committee": committee,
        "terms": terms,
        "term_label": ", ".join(f"{value}대" for value in terms) or f"{term}대",
        "reelection": _split_latest(row.get("RLCT_DIV_NM")),
        "duty": _split_latest(row.get("DTY_NM")),
        "election_type": _split_latest(row.get("ELECD_DIV_NM")),
        "image_url": normalize_space(row.get("NAAS_PIC")),
        "search_text": normalize_space(
            " ".join(
                [
                    name,
                    party,
                    region,
                    committee,
                    _split_latest(row.get("RLCT_DIV_NM")),
                    _split_latest(row.get("ELECD_DIV_NM")),
                ]
            )
        ),
        "_key": normalize_space(row.get("NAAS_CD")) or f"{name}-{party}-{region}",
    }


def fetch_member_directory(
    client: AssemblyAPIClient,
    *,
    term: int | None = DEFAULT_MEMBER_DIRECTORY_TERM,
    page_size: int = 1000,
    max_pages: int = 20,
) -> List[Dict[str, Any]]:
    rows = client.get_rows(ENDPOINT_MEMBERS, {"pSize": page_size}, max_pages=max_pages)
    members: List[Dict[str, Any]] = []
    seen = set()

    for row in rows:
        terms = _extract_terms(row.get("GTELT_ERACO"))
        if term is not None and term not in terms:
            continue

        member = normalize_member_directory_row(row, term=term or DEFAULT_MEMBER_DIRECTORY_TERM)
        if not member["name"]:
            continue

        dedupe_key = member["_key"]
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        members.append(member)

    return sorted(members, key=lambda item: (item["name"], item["party"], item["region"]))


def filter_member_directory(
    members: Iterable[Dict[str, Any]],
    *,
    query: str = "",
    parties: List[str] | None = None,
    regions: List[str] | None = None,
    committees: List[str] | None = None,
) -> List[Dict[str, Any]]:
    query_text = normalize_space(query).lower()
    party_set = set(parties or [])
    region_set = set(regions or [])
    committee_set = set(committees or [])

    filtered: List[Dict[str, Any]] = []
    for member in members:
        if query_text and query_text not in normalize_space(member.get("search_text")).lower():
            continue
        if party_set and member.get("party") not in party_set:
            continue
        if region_set and member.get("region_bucket") not in region_set:
            continue
        if committee_set and member.get("committee") not in committee_set:
            continue
        filtered.append(member)

    return filtered
