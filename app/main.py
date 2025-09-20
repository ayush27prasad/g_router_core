import os
import sys

from graph import build_router_graph
from models.models import RouterGraphState, ToolResponse
from dotenv import load_dotenv


def run(query: str) -> None:
    print("Building router graph...")
    graph = build_router_graph()
    state = RouterGraphState(input_text=query)
    result = graph.invoke(state)
    analysis = result.get("analysis")
    response = result.get("response")

    print("Intent:", analysis.intent.value)
    print("Confidence:", getattr(analysis, "confidence", None))
    print("Response type:", response.type.value)
    print("Response:")
    print(response.content)


if __name__ == "__main__":
    load_dotenv()
    query = input("Hi Try asking me something! : \n > ")
    run(query)


