"""Node skeletons for the LangGraph workflow.

Each function should be small, testable, and return a partial state update. Avoid mutating the
input state in place.
"""

from __future__ import annotations

import re

from .state import AgentState, ApprovalDecision, Route, make_event

RISKY_KEYWORDS = {"refund", "delete", "send", "cancel", "remove", "revoke", "close", "terminate"}
TOOL_KEYWORDS = {"status", "order", "lookup", "check", "track", "find", "search", "shipment"}
ERROR_KEYWORDS = {
    "timeout",
    "failure",
    "fail",
    "failed",
    "error",
    "crash",
    "unavailable",
    "cannot",
    "recover",
}
VAGUE_PRONOUNS = {"it", "this", "that", "them", "they", "thing"}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def intake_node(state: AgentState) -> dict:
    """Normalize raw query into state fields.

    Keeps state serializable and records the normalized query for auditability.
    """
    query = " ".join(state.get("query", "").strip().split())
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized", query_length=len(query))],
    }


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route.

    Uses keyword sets rather than exact scenario matching so hidden scenarios with the same
    intent route correctly. Risky actions take priority over tool calls, then missing-info
    checks, error handling, and finally a safe simple answer.
    """
    query = state.get("query", "")
    clean_words = _tokens(query)
    word_set = set(clean_words)
    route = Route.SIMPLE
    risk_level = "low"
    if word_set & RISKY_KEYWORDS:
        route = Route.RISKY
        risk_level = "high"
    elif word_set & TOOL_KEYWORDS:
        route = Route.TOOL
    elif len(clean_words) <= 5 and word_set & VAGUE_PRONOUNS:
        route = Route.MISSING_INFO
    elif word_set & ERROR_KEYWORDS:
        route = Route.ERROR
    return {
        "route": route.value,
        "risk_level": risk_level,
        "events": [
            make_event(
                "classify",
                "completed",
                f"route={route.value}",
                matched_keywords=sorted(
                    word_set & (RISKY_KEYWORDS | TOOL_KEYWORDS | ERROR_KEYWORDS)
                ),
            )
        ],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    The prompt names the missing artifact instead of attempting a risky or fabricated action.
    """
    question = "Can you provide the affected account, order id, or the missing support context?"
    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event("clarify", "completed", "missing information requested")],
    }


def tool_node(state: AgentState) -> dict:
    """Call a mock tool.

    Simulates idempotent tool execution. Error-route scenarios fail for the first two tool
    attempts, which lets the graph demonstrate bounded retry and dead-letter behavior.
    """
    attempt = int(state.get("attempt", 0))
    if state.get("route") == Route.ERROR.value and attempt < 2:
        result = (
            "ERROR: transient failure "
            f"attempt={attempt} scenario={state.get('scenario_id', 'unknown')}"
        )
    else:
        result = (
            "OK: mock support lookup completed "
            f"scenario={state.get('scenario_id', 'unknown')} attempt={attempt}"
        )
    return {
        "tool_results": [result],
        "events": [make_event("tool", "completed", f"tool executed attempt={attempt}")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for approval.

    Produces a serializable approval package with the query and risk justification.
    """
    query = state.get("query", "")
    return {
        "proposed_action": f"Review and approve high-risk support action for query: {query}",
        "events": [
            make_event(
                "risky_action",
                "pending_approval",
                "approval required before external action",
                risk_level=state.get("risk_level", "unknown"),
            )
        ],
    }


def approval_node(state: AgentState) -> dict:
    """Human approval step with optional LangGraph interrupt().

    Set LANGGRAPH_INTERRUPT=true to use real interrupt() for HITL demos.
    Default uses mock decision so tests and CI run offline.

    In local and CI runs, approval defaults to an explicit mock approval. Setting
    LANGGRAPH_INTERRUPT=true switches to real LangGraph interrupt/resume behavior.
    """
    import os

    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        from langgraph.types import interrupt

        value = interrupt({
            "proposed_action": state.get("proposed_action"),
            "risk_level": state.get("risk_level"),
        })
        if isinstance(value, dict):
            decision = ApprovalDecision(**value)
        else:
            decision = ApprovalDecision(approved=bool(value))
    else:
        decision = ApprovalDecision(approved=True, comment="mock approval for lab")
    return {
        "approval": decision.model_dump(),
        "events": [make_event("approval", "completed", f"approved={decision.approved}")],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt or fallback decision.

    The conditional edge after this node enforces the attempt bound.
    """
    attempt = int(state.get("attempt", 0)) + 1
    errors = [f"transient failure attempt={attempt}"]
    return {
        "attempt": attempt,
        "errors": errors,
        "events": [make_event("retry", "completed", "retry attempt recorded", attempt=attempt)],
    }


def answer_node(state: AgentState) -> dict:
    """Produce a final response.

    Grounds answers in tool output and approval state when those paths were used.
    """
    if state.get("tool_results"):
        approval = state.get("approval")
        prefix = "Approved action completed. " if approval and approval.get("approved") else ""
        answer = f"{prefix}I found: {state['tool_results'][-1]}"
    else:
        answer = "I can help with that support request. No external tool or approval was required."
    return {
        "final_answer": answer,
        "events": [make_event("answer", "completed", "answer generated")],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the 'done?' check that enables retry loops.

    A structured tool contract would be used in production; this lab uses the mock result
    prefix to keep tests offline and deterministic.
    """
    tool_results = state.get("tool_results", [])
    latest = tool_results[-1] if tool_results else ""
    if latest.upper().startswith("ERROR"):
        return {
            "evaluation_result": "needs_retry",
            "events": [
                make_event(
                    "evaluate",
                    "completed",
                    "tool result indicates failure, retry needed",
                )
            ],
        }
    return {
        "evaluation_result": "success",
        "events": [make_event("evaluate", "completed", "tool result satisfactory")],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Log unresolvable failures for manual review.

    Third layer of error strategy: retry, fallback, then dead letter.
    """
    return {
        "final_answer": (
            "Request could not be completed after maximum retry attempts. "
            "Logged for manual review."
        ),
        "events": [
            make_event(
                "dead_letter",
                "completed",
                f"max retries exceeded, attempt={state.get('attempt', 0)}",
                manual_review=True,
            )
        ],
    }


def finalize_node(state: AgentState) -> dict:
    """Finalize the run and emit a final audit event."""
    return {"events": [make_event("finalize", "completed", "workflow finished")]}
