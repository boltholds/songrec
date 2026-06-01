from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from songrec.audio import load_audio
from songrec.db.repositories import FingerprintMatchRow, FingerprintRepository
from songrec.db.session import create_session_factory
from songrec.fingerprint import Fingerprint, fingerprint_audio
from songrec.matcher import MatchResult, match


class RecognitionState(TypedDict, total=False):
    audio_path: Path
    db_path: Path

    audio: Any
    query_fingerprints: list[Fingerprint]
    db_rows: list[FingerprintMatchRow]
    result: MatchResult | None
    error: str | None


def validate_input(state: RecognitionState) -> RecognitionState:
    audio_path = state["audio_path"]
    db_path = state["db_path"]

    if not audio_path.exists():
        return {"error": f"Audio file does not exist: {audio_path}"}

    if not db_path.exists():
        return {"error": f"Database does not exist: {db_path}"}

    return {"error": None}


def should_continue_after_validation(state: RecognitionState) -> str:
    if state.get("error"):
        return "error"
    return "continue"


def load_audio_node(state: RecognitionState) -> RecognitionState:
    audio = load_audio(state["audio_path"])
    return {"audio": audio}


def extract_fingerprints_node(state: RecognitionState) -> RecognitionState:
    fingerprints = fingerprint_audio(state["audio"])
    return {"query_fingerprints": fingerprints}


def find_db_matches_node(state: RecognitionState) -> RecognitionState:
    session_factory = create_session_factory(state["db_path"])

    with session_factory() as session:
        repository = FingerprintRepository(session)
        rows = repository.find_matches(state["query_fingerprints"])

    return {"db_rows": rows}


def vote_match_node(state: RecognitionState) -> RecognitionState:
    result = match(
        query_fingerprints=state["query_fingerprints"],
        db_rows=state["db_rows"],
    )
    return {"result": result}


def build_recognition_graph():
    graph = StateGraph(RecognitionState)

    graph.add_node("validate_input", validate_input)
    graph.add_node("load_audio", load_audio_node)
    graph.add_node("extract_fingerprints", extract_fingerprints_node)
    graph.add_node("find_db_matches", find_db_matches_node)
    graph.add_node("vote_match", vote_match_node)

    graph.add_edge(START, "validate_input")

    graph.add_conditional_edges(
        "validate_input",
        should_continue_after_validation,
        {
            "continue": "load_audio",
            "error": END,
        },
    )

    graph.add_edge("load_audio", "extract_fingerprints")
    graph.add_edge("extract_fingerprints", "find_db_matches")
    graph.add_edge("find_db_matches", "vote_match")
    graph.add_edge("vote_match", END)

    return graph.compile()