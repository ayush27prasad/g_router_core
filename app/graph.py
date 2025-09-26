from typing import Callable, Literal, Dict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

import app.tools as tools
from app.schemas.models import RouterGraphState
from app.schemas.enums import Intent


allowed_nodes = [
    "d2llm",
    "reasoning",
    "real_time_info",
    "coding",
    "image_generation",
    "localized_india",
    "social_media",
    "other",
]

# Analyze the user query and update the analysis in the state
def _analyze_user_query(state: RouterGraphState) ->  RouterGraphState:
    
    # if model_name is provided, directly call the LLM using d2llm node
    req_model_name = state.get("request_model_name")
    
    if (req_model_name == None):
        text = state["input_text"]

        # if model_name is not provided, analyze the intent
        analysis = tools.analyze_intent(text)
        print(f"User Query Analysis : {analysis}")
    
        # Update the analysis in the state
        state["analysis"] = analysis

    else:
        print("Model Name is provided, no need to analyze the intent...")
    
    # Append the latest user message to history using the messages aggregator
    if state.get("input_text"):
        state["messages"] = [HumanMessage(state["input_text"])]
    
    # return the updated state with the analysis from the classifier model
    return state

# Route the user query to the appropriate node
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
        Intent.SOCIA_MEDIA: "social_media",
        Intent.OTHER: "other",
    }
    intent = state["analysis"].intent
    routed_node = node_by_intent.get(intent, "other")

    print(f"For intent {intent}, routing to node {routed_node}")

    # route the user query to the appropriate node
    return routed_node

def _resolve_reasoning_query(state: RouterGraphState) -> RouterGraphState:
    # call the reasoning model tool
    history = state.get("messages", [])
    tool_response = tools.call_reasoning_model(state["input_text"], messages=history)
    state["response"] = tool_response
    state["response_model_name"] = tool_response.response_generated_via
    # Append assistant response to messages
    state["messages"] = [AIMessage(tool_response.content)]
    return state

def _resolve_coding_query(state: RouterGraphState) -> RouterGraphState:
    # call the coding model tool
    history = state.get("messages", [])
    tool_response = tools.call_coding_model(state["input_text"], messages=history)
    state["response"] = tool_response
    state["response_model_name"] = tool_response.response_generated_via
    state["messages"] = [AIMessage(tool_response.content)]
    return state

def _generate_image(state: RouterGraphState) -> RouterGraphState:
    # call the coding model tool
    history = state.get("messages", [])
    tool_response = tools.call_image_generation_model(state["input_text"], messages=history)
    state["response"] = tool_response
    state["response_model_name"] = tool_response.response_generated_via
    state["messages"] = [AIMessage(tool_response.content)]
    return state

def _fetch_real_time_info(state: RouterGraphState) -> RouterGraphState:
    # call the RAG model tool
    history = state.get("messages", [])
    tool_response = tools.call_realtime_info_model(state["input_text"], messages=history)
    state["response"] = tool_response
    state["response_model_name"] = tool_response.response_generated_via
    state["messages"] = [AIMessage(tool_response.content)]
    return state

def _resolve_localized_india_query(state: RouterGraphState) -> RouterGraphState:
    # call the India facts model
    history = state.get("messages", [])
    tool_response = tools.call_india_facts_model(state["input_text"], messages=history)
    state["response"] = tool_response
    state["response_model_name"] = tool_response.response_generated_via
    state["messages"] = [AIMessage(tool_response.content)]
    return state

def _resolve_social_media_query(state: RouterGraphState) -> RouterGraphState:
    # call the social media model
    history = state.get("messages", [])
    tool_response = tools.call_realtime_info_model(state["input_text"], messages=history) # TODO: Use the social media model
    state["response"] = tool_response
    state["response_model_name"] = tool_response.response_generated_via
    state["messages"] = [AIMessage(tool_response.content)]
    return state

def _default_llm_call(state: RouterGraphState) -> RouterGraphState:
    # call the default LLM tool
    history = state.get("messages", [])
    tool_response = tools.call_default_model(state["input_text"], messages=history)
    state["response"] = tool_response
    state["response_model_name"] = tool_response.response_generated_via
    state["messages"] = [AIMessage(tool_response.content)]
    return state

def _call_model_by_name(state: RouterGraphState) -> RouterGraphState:
    
    req_model_name = state["request_model_name"]
    
    if (req_model_name == None):
        raise ValueError("Invalid request model name!!!")

    # call the model by name
    history = state.get("messages", [])
    tool_response = tools.call_model_by_name(req_model_name, state["input_text"], messages=history)
    state["response"] = tool_response
    state["response_model_name"] = tool_response.response_generated_via
    state["messages"] = [AIMessage(tool_response.content)]

    return state

# Build the router graph
def build_router_graph(checkpointer: MemorySaver | None = None) -> Callable[[RouterGraphState], RouterGraphState]:
    graph = StateGraph(RouterGraphState)

    # Register nodes
    graph.add_node("analyze", _analyze_user_query)

    graph.add_node("d2llm", _call_model_by_name)

    graph.add_node("reasoning", _resolve_reasoning_query)
    graph.add_node("coding", _resolve_coding_query)
    graph.add_node("image_generation", _generate_image)
    graph.add_node("real_time_info", _fetch_real_time_info)
    graph.add_node("localized_india", _resolve_localized_india_query)
    graph.add_node("social_media", _resolve_social_media_query)
    graph.add_node("other", _default_llm_call)
  

    # Register edges
    graph.add_edge(START, "analyze")
    
    graph.add_conditional_edges("analyze", _route_user_query)

    for node in allowed_nodes:
        graph.add_edge(node, END) # Add edges to the end node for all nodes

    # Compile the graph with optional checkpointer for stateful memory across invocations
    if checkpointer is not None:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()


