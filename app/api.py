from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from app.graph import build_router_graph
from app.llms.openai_compatible import call_onboarded_model
from app.schemas.models import RouterGraphState
from typing import Optional

load_dotenv()

app = FastAPI()

class ChatRequest(BaseModel):
    query: str
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    response_generated_via: str = None

def run(user_query: str) -> RouterGraphState:
    graph = build_router_graph()
    state = RouterGraphState(input_text=user_query)
    result = graph.invoke(state)
    return result

@app.post("/ask")
async def ask(request: ChatRequest) -> ChatResponse:
    print(f"Received request: {request}")
    response_graph = run(request.query)
    response_model_name = response_graph.get("response_model_name")
    return ChatResponse(response=response_graph.get("response").content, response_generated_via=response_model_name)

@app.post("/ask/d2llm")
async def ask(request: ChatRequest) -> ChatResponse:
    print(f"Received request d2llm: {request}")
    response = call_onboarded_model(model_name=request.model_name, api_key=request.api_key, base_url=request.base_url, user_query=request.query)
    response_model_name = request.model_name
    return ChatResponse(response=response, response_generated_via=response_model_name)
