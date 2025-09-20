from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from models import Intent, IntentAnalysis


def get_mini_model(temperature: float = 0.2) -> ChatOpenAI:
    """Return a GPT-4o mini chat model for lightweight tasks."""
    return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)


def get_capable_model(temperature: float = 0.2) -> ChatOpenAI:
    """Return a more capable model (defaults to GPT-4o)."""
    return ChatOpenAI(model="gpt-4o", temperature=temperature)


def analyze_intent_and_summary(text: str, summary_threshold_chars: int = 600) -> IntentAnalysis:
    """Analyze user intent and optionally produce a short summary using structured output.

    The mini model returns a structured `IntentAnalysis` Pydantic object.
    """
    intents: List[str] = [i.value for i in Intent]

    system = (
        "You are an intent classifier. Determine the user's primary intent from the list. "
        "Also compute input length (characters). If input length exceeds the threshold, set "
        "needs_summary true and include a concise 1-3 sentence summary of the request."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "user",
                "Available intents: {intents}\nThreshold (chars): {threshold}\n\nUser input:\n{text}",
            ),
        ]
    )

    llm = get_mini_model()
    structured_llm = llm.with_structured_output(schema=IntentAnalysis)
    chain = prompt | structured_llm
    return chain.invoke(
        {
            "intents": ", ".join(intents),
            "threshold": summary_threshold_chars,
            "text": text,
        }
    )


