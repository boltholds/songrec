from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from songrec.audio import load_audio
from songrec.db.repositories import FingerprintMatchRow, FingerprintRepository
from songrec.db.session import create_session_factory
from songrec.fingerprint import Fingerprint
from songrec.matcher import MatchDecision, MatchResult
from songrec.recognition.speed import DEFAULT_SPEED_FACTORS, recognize_audio


class RecognitionState(TypedDict, total=False):
    audio_path: Path
    db_path: Path
    mode: str
    speed_factors: list[float]
    decision: MatchDecision

    audio: Any
    query_fingerprints_count: int
    query_fingerprints: list[Fingerprint]
    db_rows: list[FingerprintMatchRow]
    result: MatchResult | None
    error: str | None


def validate_input(state: RecognitionState) -> RecognitionState:
    audio_path = state["audio_path"]
    db_path = state["db_path"]
    mode = state.get("mode", "fast")

    if not audio_path.exists():
        return {"error": f"Audio file does not exist: {audio_path}"}

    if not db_path.exists():
        return {"error": f"Database does not exist: {db_path}"}

    if mode not in {"fast", "multi_speed", "scale_aware"}:
        return {"error": f"Unknown recognition mode: {mode}"}

    return {"error": None, "mode": mode}


def should_continue_after_validation(state: RecognitionState) -> str:
    if state.get("error"):
        return "error"
    return "continue"


def load_audio_node(state: RecognitionState) -> RecognitionState:
    audio = load_audio(state["audio_path"])
    return {"audio": audio}


def recognize_node(state: RecognitionState) -> RecognitionState:
    session_factory = create_session_factory(state["db_path"])
    decision = state.get("decision") or MatchDecision()
    speed_factors = state.get("speed_factors") or list(DEFAULT_SPEED_FACTORS)

    with session_factory() as session:
        repository = FingerprintRepository(session)
        result, fingerprints_count = recognize_audio(
            audio=state["audio"],
            fingerprint_repo=repository,
            decision=decision,
            mode=state.get("mode", "fast"),
            speed_factors=speed_factors,
        )

    return {
        "result": result,
        "query_fingerprints_count": fingerprints_count,
    }


def build_recognition_graph():
    graph = StateGraph(RecognitionState)

    graph.add_node("validate_input", validate_input)
    graph.add_node("load_audio", load_audio_node)
    graph.add_node("recognize", recognize_node)

    graph.add_edge(START, "validate_input")

    graph.add_conditional_edges(
        "validate_input",
        should_continue_after_validation,
        {
            "continue": "load_audio",
            "error": END,
        },
    )

    graph.add_edge("load_audio", "recognize")
    graph.add_edge("recognize", END)

    return graph.compile()
