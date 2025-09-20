from typing import Callable

from langgraph.graph import StateGraph, START, END

from llm import analyze_intent_and_summary
from models import RouterGraphState
from router import llm_call_by_intent


def _analyze(state: RouterGraphState) -> RouterGraphState:
    text = state["input_text"]
    analysis = analyze_intent_and_summary(text)
    state["analysis"] = analysis
    return state


def _execute(state: RouterGraphState) -> RouterGraphState:
    intent = state["analysis"].intent
    text = state["input_text"]
    response = llm_call_by_intent(intent, text)
    state["response"] = response
    return state


def build_router_graph() -> Callable[[RouterGraphState], RouterGraphState]:
    graph = StateGraph(RouterGraphState)

    # Register nodes
    graph.add_node("analyze", _analyze)
    graph.add_node("execute", _execute)

    # Register edges
    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "execute")
    graph.add_edge("execute", END)

    # Compile the graph
    return graph.compile()


