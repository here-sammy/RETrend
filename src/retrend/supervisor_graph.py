"""The supervisor as a LangGraph graph (optional: needs ``pip install -e ".[graph]"``).

Same node functions as ``Supervisor.run``; the graph just wires them with the retry loop:

    run_workers --(retryable failures left)--> run_workers
         |
         v (otherwise)
       critic -> synthesize -> narrate -> END

NOT TESTED in the build environment (langgraph could not be installed there). The plain
``Supervisor.run`` is the tested path; ``tests/test_phase3.py`` has a test that exercises
this graph and is skipped automatically when langgraph is absent.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, TypedDict

from retrend.location import Location
from retrend.outlook import Outlook
from retrend.supervisor import Supervisor, new_state


class GraphState(TypedDict, total=False):
    location: Any
    as_of: Any
    ref: Any
    reports: dict
    errors: dict
    retryable: dict
    attempts: dict
    findings: list
    outlook: Any


def build_graph(sup: Supervisor):
    from langgraph.graph import END, StateGraph  # imported lazily: optional dependency

    graph = StateGraph(GraphState)
    graph.add_node("run_workers", sup.run_workers)
    graph.add_node("critic", sup.critic)
    graph.add_node("synthesize", sup.synthesize)
    graph.add_node("narrate", sup.narrate)
    graph.set_entry_point("run_workers")
    graph.add_conditional_edges(
        "run_workers",
        lambda state: "retry" if sup.needs_retry(state) else "critic",
        {"retry": "run_workers", "critic": "critic"},
    )
    graph.add_edge("critic", "synthesize")
    graph.add_edge("synthesize", "narrate")
    graph.add_edge("narrate", END)
    return graph.compile()


def run_with_graph(sup: Supervisor, location: Location, as_of: datetime | None = None) -> Outlook:
    ref = as_of.astimezone(timezone.utc).date() if as_of else sup.store.clock().date()
    final = build_graph(sup).invoke(new_state(location, as_of, ref))
    return final["outlook"]
