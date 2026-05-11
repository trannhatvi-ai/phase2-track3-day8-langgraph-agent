from __future__ import annotations

import sqlite3

import pytest

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.metrics import MetricsReport, summarize_metrics
from langgraph_agent_lab.nodes import classify_node
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.report import render_report_stub
from langgraph_agent_lab.state import Route, Scenario, initial_state


def test_classify_uses_priority_and_word_boundaries() -> None:
    assert classify_node({"query": "Please check status and refund order 123"})["route"] == Route.RISKY.value
    assert classify_node({"query": "Can you fix item 123?"})["route"] == Route.SIMPLE.value
    assert classify_node({"query": "Can you fix it?"})["route"] == Route.MISSING_INFO.value
    assert classify_node({"query": "The support portal is unavailable"})["route"] == Route.ERROR.value


@pytest.mark.parametrize(
    ("query", "expected_route"),
    [
        ("Cancel the subscription and send a receipt", Route.RISKY.value),
        ("Track shipment for order 987", Route.TOOL.value),
        ("System crash during checkout", Route.ERROR.value),
        ("Need help with billing preferences", Route.SIMPLE.value),
    ],
)
def test_graph_handles_hidden_style_scenarios(query: str, expected_route: str) -> None:
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    scenario = Scenario(id="hidden-style", query=query, expected_route=Route(expected_route))
    state = initial_state(scenario)

    result = graph.invoke(state, config={"configurable": {"thread_id": state["thread_id"]}})

    assert result["route"] == expected_route
    assert result.get("final_answer") or result.get("pending_question")


def test_error_route_dead_letters_when_max_attempts_is_one() -> None:
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    scenario = Scenario(
        id="dead-letter",
        query="System unavailable and cannot recover",
        expected_route=Route.ERROR,
        max_attempts=1,
    )
    state = initial_state(scenario)

    result = graph.invoke(state, config={"configurable": {"thread_id": state["thread_id"]}})
    event_nodes = [event["node"] for event in result["events"]]

    assert result["route"] == Route.ERROR.value
    assert result["attempt"] == 1
    assert "dead_letter" in event_nodes
    assert result["final_answer"]


def test_sqlite_checkpointer_is_directly_usable(tmp_path) -> None:
    db_path = tmp_path / "checkpoints.db"
    checkpointer = build_checkpointer("sqlite", str(db_path))

    try:
        assert db_path.exists()
        with sqlite3.connect(db_path) as conn:
            journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]

        assert journal_mode == "wal"
        assert hasattr(checkpointer, "get_tuple")
    finally:
        checkpointer.conn.close()


def test_report_contains_completed_lab_sections() -> None:
    report = summarize_metrics([])
    text = render_report_stub(report)

    assert "TODO(student)" not in text
    assert "| Scenario | Expected route | Actual route | Success | Retries | Interrupts |" in text
    assert "Persistence / recovery evidence" in text
    assert "Improvement plan" in text


def test_summarize_metrics_allows_empty_for_report_drafts() -> None:
    report = summarize_metrics([])

    assert isinstance(report, MetricsReport)
    assert report.total_scenarios == 0
    assert report.success_rate == 0
