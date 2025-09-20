import os
import sys

from graph import build_router_graph
from models import RouterGraphState
from dotenv import load_dotenv


def run(query: str) -> None:
    print("Building router graph...")
    app = build_router_graph()
    state: RouterGraphState = {"input_text": query}
    result = app.invoke(state)
    analysis = result.get("analysis")
    response = result.get("response")

    print("Intent:", analysis.intent.value)
    print("Confidence:", getattr(analysis, "confidence", None))
    if analysis.needs_summary and analysis.summary:
        print("Summary:")
        print(analysis.summary)
    print("Response type:", response.type.value)
    print("Response:")
    print(response.content)


if __name__ == "__main__":
    load_dotenv()
    query = input("Hi Try asking me something! : \n > ")
    
    run(query)


