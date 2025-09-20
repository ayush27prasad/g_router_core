from langchain_anthropic import ChatAnthropic


def call_anthropic(system_msg: str, human_msg: str) -> ToolResponse:
    """Call the Anthropic API."""
    model_name = "claude-opus-4-1-20250805"
    model = ChatAnthropic(model=model_name, temperature=0.2).with_structured_output(schema=ToolResponse)
    return model.invoke([SystemMessage(system_msg), HumanMessage(human_msg)])
