from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from app.graph import build_router_graph
from langgraph.checkpoint.memory import MemorySaver
from app.llms.openai_compatible import call_onboarded_model
from app.schemas.models import RouterGraphState
from typing import Optional, List
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

load_dotenv()

app = FastAPI()

class ChatRequest(BaseModel):
    query: str
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    messages: Optional[list] = None  # [{"role": "user|assistant|system", "content": "..."}]
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    response_generated_via: str = None

def _convert_messages(raw_messages: Optional[list]) -> Optional[List[BaseMessage]]:
    if not raw_messages:
        return None
    converted: List[BaseMessage] = []
    for m in raw_messages:
        role = m.get("role")
        content = m.get("content")
        if not role or content is None:
            continue
        if role == "system":
            converted.append(SystemMessage(content))
        elif role == "assistant":
            converted.append(AIMessage(content))
        else:
            converted.append(HumanMessage(content))
    return converted


memory = MemorySaver()
compiled_graph = build_router_graph(checkpointer=memory)


def run(user_query: str, messages: Optional[List[BaseMessage]] = None, request_model_name: Optional[str] = None, session_id: Optional[str] = None) -> RouterGraphState:
    graph = compiled_graph
    state: RouterGraphState = RouterGraphState(input_text=user_query)
    if messages:
        state["messages"] = messages
    if request_model_name:
        state["request_model_name"] = request_model_name
    cfg = {"configurable": {"thread_id": session_id or "default"}}
    result = graph.invoke(state, config=cfg)
    return result

@app.post("/ask")
async def ask(request: ChatRequest) -> ChatResponse:
    print(f"Received request: {request}")
    history = _convert_messages(request.messages)
    response_graph = run(request.query, messages=history, request_model_name=request.model_name, session_id=request.session_id)
    response_model_name = response_graph.get("response_model_name")
    return ChatResponse(response=response_graph.get("response").content, response_generated_via=response_model_name)

@app.post("/ask/d2llm")
async def ask(request: ChatRequest) -> ChatResponse:
    print(f"Received request d2llm: {request}")
    history = _convert_messages(request.messages)
    # Direct model call remains stateless unless messages are provided. Memory is not used here by design.
    response = call_onboarded_model(model_name=request.model_name, api_key=request.api_key, base_url=request.base_url, user_query=request.query, messages=history)
    response_model_name = request.model_name
    return ChatResponse(response=response, response_generated_via=response_model_name)
