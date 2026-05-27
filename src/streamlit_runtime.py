from __future__ import annotations

from typing import Any, Dict

from .config import get_assembly_api_key
from .member_activity_workflow import AssemblyAPIClient, WORKFLOW_VERSION, build_member_activity_graph


def get_streamlit_cached_resource():
    try:
        import streamlit as st

        return st.cache_resource
    except Exception:
        return lambda func=None, **_: func if func is not None else (lambda wrapped: wrapped)


cache_resource = get_streamlit_cached_resource()


@cache_resource(show_spinner=False)
def get_client() -> AssemblyAPIClient:
    return AssemblyAPIClient(api_key=get_assembly_api_key())


@cache_resource(show_spinner=False)
def get_member_activity_app(workflow_version: str = WORKFLOW_VERSION):
    return build_member_activity_graph(get_client())


def run_member_activity_workflow(initial_state: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    app = get_member_activity_app(WORKFLOW_VERSION)
    for event in app.stream(initial_state, stream_mode="updates"):
        for _, node_update in event.items():
            if node_update:
                result.update(node_update)
    return result
