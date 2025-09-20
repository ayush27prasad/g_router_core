from typing import Callable, Literal, Dict

from langgraph.graph import StateGraph, START, END

from tools import analyze_intent
from models.models import RouterGraphState, Intent, ToolResponse
from models.enums import Intent, ModelProvider, get_model_provider, ToolResponseType



allowed_nodes = [
    "d2llm",
    "reasoning",
    "real_time_info",
    "coding",
    "image_generation",
    "localized_india",
    "other",
]

def _analyze_user_query(state: RouterGraphState) ->  RouterGraphState:
    
    # if model_name is provided, directly call the LLM using d2llm node
    req_model_name = state.get("request_model_name")
    
    if (req_model_name == None):
        text = state["input_text"]    
        # if model_name is not provided, analyze the intent
        analysis = analyze_intent(text)
        print(f"User Query Analysis : {analysis.__pretty__}")
    
        # Update the analysis in the state
        state["analysis"] = analysis

    else:
        print("Model Name is provided, no need to analyze the intent...")
    
    # return the updated state with the analysis from the classifier model
    return state

def _route_user_query(state: RouterGraphState) ->  Literal[*allowed_nodes]:

    # if model_name is provided, directly call the LLM using d2llm node
    req_model_name = state.get("request_model_name")
    print(f"Request Model Name : {req_model_name}")
    if (req_model_name != None):
        return "d2llm"

    node_by_intent : Dict[Intent, Literal[*allowed_nodes]]  = {
        Intent.REASONING: "reasoning",
        Intent.CURRENT_AFFAIRS: "real_time_info",
        Intent.CODE_GENERATION: "coding",
        Intent.DEBUG_CODE: "coding",
        Intent.IMAGE_GENERATION: "image_generation",
        Intent.LOCALIZED_INDIA: "localized_india",
        Intent.OTHER: "other",
    }
    intent = state["analysis"].intent
    routed_node = node_by_intent.get(intent, "other")

    print(f"For intent {intent}, routing to node {routed_node}")

    # route the user query to the appropriate node
    return routed_node

def _resolve_reasoning_query(state: RouterGraphState) -> RouterGraphState:
    # call the reasoning model tool
    print("Resolving reasoning query...")
    state["response"] : ToolResponse = ToolResponse(type=ToolResponseType.TEXT, content="This is a response from the reasoning model")
    return state

def _resolve_coding_query(state: RouterGraphState) -> RouterGraphState:
    # call the coding model tool
    return state

def _generate_image(state: RouterGraphState) -> RouterGraphState:
    # call the coding model tool
    return state

def _fetch_real_time_info(state: RouterGraphState) -> RouterGraphState:
    # call the RAG model tool
    return state

def _resolve_localized_india_query(state: RouterGraphState) -> RouterGraphState:
    # call the RAG model tool
    return state

def _default_llm_call(state: RouterGraphState) -> RouterGraphState:
    # call the default LLM tool
    return state

def _call_model_by_name(state: RouterGraphState) -> RouterGraphState:
    
    req_model_name = state["request_model_name"]
    
    model_provider = get_model_provider(req_model_name)

    user_query = state["input_text"]
    
    response : ToolResponse = None

    if model_provider == ModelProvider.OPEN_AI:
        response = _call_openai(user_query)
    elif model_provider == ModelProvider.ANTHROPIC:
        response = _call_anthropic(user_query)
    elif model_provider == ModelProvider.PERPLEXITY:
        response = _call_perplexity(user_query)
    elif model_provider == ModelProvider.X_AI:
        response = _call_grok(user_query)
    elif model_provider == ModelProvider.GEMINI:
        response = _call_gemini(user_query)
    elif model_provider == ModelProvider.SARVAM:
        response = _call_sarvam(user_query)
    
    state["response"] = response

    return state

def build_router_graph() -> Callable[[RouterGraphState], RouterGraphState]:
    graph = StateGraph(RouterGraphState)

    # Register nodes
    graph.add_node("analyze", _analyze_user_query)

    graph.add_node("d2llm", _call_model_by_name)

    graph.add_node("reasoning", _resolve_reasoning_query)
    graph.add_node("coding", _resolve_coding_query)
    graph.add_node("image_generation", _generate_image)
    graph.add_node("real_time_info", _fetch_real_time_info)
    graph.add_node("localized_india", _resolve_localized_india_query)
    graph.add_node("other", _default_llm_call)
  

    # Register edges
    graph.add_edge(START, "analyze")
    
    graph.add_conditional_edges("analyze", _route_user_query)

    for node in allowed_nodes:
        graph.add_edge(node, END)

    # Compile the graph
    return graph.compile()


