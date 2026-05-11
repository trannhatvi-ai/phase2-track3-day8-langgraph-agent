# Day 08 Lab Report

## 1. Team / student

- Name: Tran Nhat Vi
- Student ID: 2A202600497
- Repo/commit: local workspace
- Date: 2026-05-11

## 2. Architecture

The workflow is a typed LangGraph `StateGraph` for support-ticket orchestration. It starts with
`intake`, normalizes the query, classifies the request, and then routes to one of five paths:
simple answer, tool lookup, clarification, risky approval, or retry/error handling. Tool paths run
through `evaluate`, which is the done-check for retry loops. All terminal paths pass through
`finalize` so every run emits a final audit event.

The graph keeps node boundaries small: classification only chooses route and risk, routing
functions only choose next node names, tools only append tool results, and answer/finalize nodes
produce user-facing output and audit evidence.

## 3. State schema

| Field | Reducer | Why |
|---|---|---|
| `thread_id` | overwrite | stable persistence key for each scenario run |
| `scenario_id` | overwrite | joins state, metrics, and report rows |
| `query` | overwrite | normalized latest user query |
| `route` | overwrite | current route selected by classifier |
| `risk_level` | overwrite | current risk decision for HITL policy |
| `attempt` | overwrite | retry counter bounded by `max_attempts` |
| `evaluation_result` | overwrite | gate from `evaluate` to `retry` or `answer` |
| `messages` | append | compact audit conversation notes |
| `tool_results` | append | preserves every tool attempt/result |
| `errors` | append | preserves retry and dead-letter evidence |
| `events` | append | node-level audit trail used by metrics |

## 4. Scenario results

| Scenario | Expected route | Actual route | Success | Retries | Interrupts |
|---|---|---|---:|---:|---:|
| S01_simple | simple | simple | True | 0 | 0 |
| S02_tool | tool | tool | True | 0 | 0 |
| S03_missing | missing_info | missing_info | True | 0 | 0 |
| S04_risky | risky | risky | True | 0 | 1 |
| S05_error | error | error | True | 2 | 0 |
| S06_delete | risky | risky | True | 0 | 1 |
| S07_dead_letter | error | error | True | 1 | 0 |

## 5. Metrics summary

- Total scenarios: 7
- Success rate: 100.00%
- Average nodes visited: 6.43
- Total retries: 3
- Total interrupts: 2
- Resume/state-history evidence: True

## 6. Failure analysis

1. Retry or tool failure: error-route requests enter `retry`, increment `attempt`, then route back
   to `tool` only while `attempt < max_attempts`. If the bound is reached, the graph sends the run
   to `dead_letter` and returns a manual-review answer.
2. Risky action without approval: risky keywords such as refund, delete, send, cancel, remove, and
   revoke route to `risky_action` before any tool call. `approval` must produce an approved decision
   before the graph continues to `tool`; rejected decisions route to clarification.

## 7. Persistence / recovery evidence

Each scenario is invoked with `configurable.thread_id` set from state, so the checkpointer stores a
separate timeline per run. The default config uses `MemorySaver` for deterministic tests. The
`sqlite` option creates a real SQLite checkpointer with WAL mode, suitable for local crash-resume
or state-history demos.

## 8. Extension work

Completed extensions:

- SQLite checkpointer support using `SqliteSaver(conn=sqlite3.connect(...))`.
- Real LangGraph interrupt hook when `LANGGRAPH_INTERRUPT=true`.
- Generated report from metrics instead of a static template.

## 9. Improvement plan

With one more day, the first production improvements would be replacing keyword classification
with a policy-tested LLM classifier, moving dead-letter records to durable storage, and adding a
human approval UI that can edit or reject proposed actions with reviewer identity.
