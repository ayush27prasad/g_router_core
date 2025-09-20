from langchain_perplexity import ChatPerplexity
from langchain_core.messages import SystemMessage, HumanMessage
from schemas.models import ToolResponse
from schemas.enums import ToolResponseType
from openai import OpenAI
import os


def call_perplexity(model_name: str, system_msg: str, human_msg: str) -> ToolResponse:
    """Call the Perplexity API."""
    model = ChatPerplexity(model=model_name, temperature=0.3)
    response = model.invoke([SystemMessage(system_msg), HumanMessage(human_msg)])
    return ToolResponse(type=ToolResponseType.TEXT, content=str(response))

def call_perplexity_api(model_name: str, system_msg: str, human_msg: str) -> ToolResponse:
    """Call the Perplexity API directly."""
    API_KEY = os.getenv("PPLX_API_KEY")

    client = OpenAI(
        api_key=API_KEY,
        base_url="https://api.perplexity.ai"
    )

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": human_msg}
        ]
    )
    return ToolResponse(type=ToolResponseType.TEXT, content=resp.choices[0].message.content)







