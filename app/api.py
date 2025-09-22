from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from app.graph import build_router_graph
from app.schemas.models import RouterGraphState

load_dotenv()

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

def run(query: str):
    graph = build_router_graph()
    state = RouterGraphState(input_text=query)
    result = graph.invoke(state)
    return result  # directly return the dict

@app.post("/ask")
async def ask(request: QueryRequest):
    print(f"Received request: {request}")
    return run(request.query)
