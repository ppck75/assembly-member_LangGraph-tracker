from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

for env_path in [PROJECT_ROOT / ".env", PROJECT_ROOT.parent / ".env"]:
    if env_path.exists():
        load_dotenv(env_path)


def _streamlit_secret(name: str) -> Optional[str]:
    try:
        import streamlit as st

        value = st.secrets.get(name)
        if value:
            return str(value).strip().strip("'\"")
    except Exception:
        return None
    return None


def env_value(name: str) -> Optional[str]:
    value = _streamlit_secret(name) or os.getenv(name)
    if value:
        return str(value).strip().strip("'\"")
    for key, val in os.environ.items():
        if key.strip() == name and val:
            return val.strip().strip("'\"")
    return None


def get_assembly_api_key() -> str:
    api_key = (
        env_value("OPENCONGRESS_key")
        or env_value("OPENCONGRESS_KEY")
        or env_value("ASSEMBLY_API_KEY")
    )
    if not api_key:
        raise ValueError(
            "열린국회정보 API KEY가 필요합니다. Streamlit secrets 또는 .env에 "
            "OPENCONGRESS_key를 설정하세요."
        )
    return api_key


def get_google_api_key() -> Optional[str]:
    api_key = env_value("GOOGLE_API_KEY") or env_value("GEMINI_API_KEY")
    if api_key and not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = api_key
    return api_key


CACHE_TTL_HOURS = int(env_value("CACHE_TTL_HOURS") or "24")
CACHE_DIR = Path(env_value("MEMBER_ACTIVITY_CACHE_DIR") or PROJECT_ROOT / "cache" / "member_activity")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
