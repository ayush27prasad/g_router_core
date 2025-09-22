from graph import build_router_graph
from app.schemas.models import RouterGraphState
from dotenv import load_dotenv


def run(query: str) -> None:
    
    graph = build_router_graph()
    state = RouterGraphState(input_text=query)
    result = graph.invoke(state)
    
    analysis = result.get("analysis")
    response = result.get("response")

    print("Intent:", analysis.intent.value)
    print("Confidence:", getattr(analysis, "confidence", None))
    print("Response type:", response.type.value)
    print("Response generated via:", response.response_generated_via)
    print("Response:")
    print(response.content)

# Run via terminal
if __name__ == "__main__":
    load_dotenv()
    print("Hi Try asking me something! : ")
    while True:
        query = input("\n > ")
        run(query)

