from typing import Optional

from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate

from llm import get_mini_model, get_capable_model
from models import ToolResponse, ToolResponseType


def summary(text: str) -> ToolResponse:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Summarize the user's request into 3-5 bullet points. Be faithful and concise.",
            ),
            ("user", "{text}"),
        ]
    )
    llm = get_mini_model(temperature=0.1)
    chain = prompt | llm
    content = chain.invoke({"text": text}).content
    return ToolResponse(type=ToolResponseType.MARKDOWN, content=content, meta={"tool": "summary"})


def basic_qa(question: str) -> ToolResponse:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer concisely and helpfully. Use markdown for structure when useful."),
            ("user", "{question}"),
        ]
    )
    llm = get_mini_model(temperature=0.2)
    chain = prompt | llm
    content = chain.invoke({"question": question}).content
    return ToolResponse(type=ToolResponseType.MARKDOWN, content=content, meta={"tool": "basic_qa"})


def code_generation(instruction: str) -> ToolResponse:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Generate correct, production-quality code with minimal explanations. "
                "Prefer idiomatic patterns and include a single fenced code block.",
            ),
            ("user", "{instruction}"),
        ]
    )
    llm = get_capable_model(temperature=0.2)
    chain = prompt | llm
    content = chain.invoke({"instruction": instruction}).content
    return ToolResponse(type=ToolResponseType.MARKDOWN, content=content, meta={"tool": "code_generation"})


def debug_code(code: str, issue: Optional[str] = None) -> ToolResponse:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a senior engineer. Find the root cause and propose a minimal fix. "
                "If relevant, provide a small diff or corrected snippet.",
            ),
            ("user", "Code:\n```\n{code}\n```\n\nIssue:\n{issue}"),
        ]
    )
    llm = get_capable_model(temperature=0.2)
    chain = prompt | llm
    content = chain.invoke({"code": code, "issue": issue or ""}).content
    return ToolResponse(type=ToolResponseType.MARKDOWN, content=content, meta={"tool": "debug_code"})


def solve_math(problem: str) -> ToolResponse:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Solve the problem step-by-step. Provide the final numeric or symbolic answer.",
            ),
            ("user", "{problem}"),
        ]
    )
    llm = get_mini_model(temperature=0.0)
    chain = prompt | llm
    content = chain.invoke({"problem": problem}).content
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


