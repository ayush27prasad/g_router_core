from typing import Optional

from openai import OpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from llm import get_mini_model, get_capable_model
from models import ToolResponse, ToolResponseType


def summary(text: str) -> ToolResponse:
    llm = get_mini_model(temperature=0.1)
    msgs = [
        SystemMessage(content="Summarize the user's request into 3-5 bullet points. Be faithful and concise."),
        HumanMessage(content=text),
    ]
    content = llm.invoke(msgs).content
    return ToolResponse(type=ToolResponseType.MARKDOWN, content=content, meta={"tool": "summary"})


def basic_qa(question: str) -> ToolResponse:
    llm = get_mini_model(temperature=0.2)
    msgs = [
        SystemMessage(content="Answer concisely and helpfully. Use markdown for structure when useful."),
        HumanMessage(content=question),
    ]
    content = llm.invoke(msgs).content
    return ToolResponse(type=ToolResponseType.MARKDOWN, content=content, meta={"tool": "basic_qa"})


def code_generation(instruction: str) -> ToolResponse:
    llm = get_capable_model(temperature=0.2)
    msgs = [
        SystemMessage(content=(
            "Generate correct, production-quality code with minimal explanations. "
            "Prefer idiomatic patterns and include a single fenced code block."
        )),
        HumanMessage(content=instruction),
    ]
    content = llm.invoke(msgs).content
    return ToolResponse(type=ToolResponseType.MARKDOWN, content=content, meta={"tool": "code_generation"})


def debug_code(code: str, issue: Optional[str] = None) -> ToolResponse:
    llm = get_capable_model(temperature=0.2)
    user_content = f"Code:\n```\n{code}\n```\n\nIssue:\n{issue or ''}"
    msgs = [
        SystemMessage(content=(
            "You are a senior engineer. Find the root cause and propose a minimal fix. "
            "If relevant, provide a small diff or corrected snippet."
        )),
        HumanMessage(content=user_content),
    ]
    content = llm.invoke(msgs).content
    return ToolResponse(type=ToolResponseType.MARKDOWN, content=content, meta={"tool": "debug_code"})


def solve_math(problem: str) -> ToolResponse:
    llm = get_mini_model(temperature=0.0)
    msgs = [
        SystemMessage(content="Solve the problem step-by-step. Provide the final numeric or symbolic answer."),
        HumanMessage(content=problem),
    ]
    content = llm.invoke(msgs).content
    return ToolResponse(type=ToolResponseType.MARKDOWN, content=content, meta={"tool": "math_solve"})


def generate_image(prompt_text: str) -> ToolResponse:
    client = OpenAI()
    result = client.images.generate(
        model="gpt-image-1",
        prompt=prompt_text,
        size="1024x1024",
    )
    # Prefer URL output if available
    image_url = None
    if result and getattr(result, "data", None):
        first = result.data[0]
        image_url = getattr(first, "url", None)
    if not image_url:
        # Fallback to base64 if URL is unavailable
        b64 = getattr(getattr(result, "data", [{}])[0], "b64_json", None)
        content = b64 if b64 else ""
        return ToolResponse(
            type=ToolResponseType.IMAGE,
            content=content,
            meta={"tool": "image_generation", "format": "b64"},
        )
    return ToolResponse(
        type=ToolResponseType.IMAGE,
        content=image_url,
        meta={"tool": "image_generation", "format": "url"},
    )


