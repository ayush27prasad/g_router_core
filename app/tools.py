from langchain_core.messages import SystemMessage, HumanMessage
from typing import Dict
from app.llms.openai import get_mini_model, call_openai
from app.llms.anthropic import call_anthropic
from app.llms.sarvam import call_sarvam
from app.llms.google import call_gemini_image_generation_model, call_gemini
from app.llms.perplexity import call_perplexity, call_perplexity_api
from app.llms.xai import call_grok
from app.schemas.models import IntentAnalysis, ToolResponse
from app.schemas.enums import ModelName, ModelProvider

from app.system_prompts import *

def analyze_intent(user_query: str) -> IntentAnalysis:
    """Analyze user intent only using structured output."""
    classifier_model = get_mini_model().with_structured_output(schema=IntentAnalysis)
    return classifier_model.invoke([SystemMessage(INTENT_CLASSIFIER_PROMPT), HumanMessage(user_query)])

def call_reasoning_model(user_query: str) -> ToolResponse:
    """Resolve reasoning query."""
    reasoning_model_name = "gpt-3.5-turbo"
    reasoning_model_response = call_openai(model_name=reasoning_model_name, system_msg=REASONING_PROMPT, human_msg=user_query)
    reasoning_model_response.response_generated_via = reasoning_model_name
    return reasoning_model_response

def call_coding_model(user_query: str) -> ToolResponse:
    """Resolve coding query."""
    coding_model_name = "claude-opus-4-1-20250805"
    coding_model_response = call_anthropic(model_name=coding_model_name, system_msg=CODING_PROMPT, human_msg=user_query)
    coding_model_response.response_generated_via = coding_model_name
    return coding_model_response

def call_image_generation_model(user_query: str) -> ToolResponse:
    """Generate image as per the user's request."""
    image_generation_model_name = "gemini-2.5-flash-image-preview" # aka gemini-nano-banana
    image_generation_model_response = call_gemini_image_generation_model(model_name=image_generation_model_name, system_msg=IMAGE_GENERATION_PROMPT, human_msg=user_query)
    image_generation_model_response.response_generated_via = image_generation_model_name
    return image_generation_model_response

def call_realtime_info_model(user_query: str) -> ToolResponse:
    """Fetch real time info."""
    realtime_model_name = "sonar-pro"
    realtime_model_response = call_perplexity_api(model_name=realtime_model_name, system_msg=REAL_TIME_INFO_PROMPT, human_msg=user_query)
    realtime_model_formatted_response = _format_model_response(user_query=user_query, unformatted_model_response=realtime_model_response)
    realtime_model_formatted_response.response_generated_via = realtime_model_name
    return realtime_model_formatted_response

def call_sarvam_model(user_query: str) -> ToolResponse:
    """Fetch real time info."""
    sarvam_model_name = "sarvam-v2"
    sarvam_model_response = call_sarvam(model_name=sarvam_model_name, system_msg=SARVAM_PROMPT, human_msg=user_query)
    sarvam_model_response.response_generated_via = sarvam_model_name
    return sarvam_model_response

def call_default_model(user_query: str) -> ToolResponse:
    """Call the default model."""
    default_model_name = "gpt-5-nano"
    default_model_response = call_openai(model_name=default_model_name, system_msg=DEFAULT_PROMPT, human_msg=user_query)
    default_model_response.response_generated_via = default_model_name
    return default_model_response


def _format_model_response(user_query: str, unformatted_model_response: str) -> ToolResponse:
    """Format the model response."""
    formatter_model_name = "gpt-4o-mini"
    message = f"User query: {user_query}\nUnformatted model response: {unformatted_model_response}"
    formatter_model_response = call_openai(model_name=formatter_model_name, system_msg=FORMATTER_PROMPT, human_msg=message)
    return formatter_model_response

def call_model_by_name(model_name: str, user_query: str) -> ToolResponse:
    """Call the model by model name."""
    model_provider_company_mapping : Dict[ModelName, ModelProvider] = {
        ModelName.OPEN_AI_GPT_4O_MINI: ModelProvider.OPEN_AI,
        ModelName.OPEN_AI_GPT_5: ModelProvider.OPEN_AI,
        ModelName.ANTHROPIC_OPUS: ModelProvider.ANTHROPIC,
        ModelName.PERPLEXITY_O3: ModelProvider.PERPLEXITY,
        ModelName.GROK_4: ModelProvider.X_AI,
        ModelName.GEMINI_NANO_BANANA: ModelProvider.GEMINI,
        ModelName.SARVAM_V2: ModelProvider.SARVAM,
    }
    model_provider = model_provider_company_mapping[model_name]
    if model_provider == ModelProvider.OPEN_AI:
        return call_openai(model_name=model_name, system_msg=DEFAULT_PROMPT, human_msg=user_query)
    elif model_provider == ModelProvider.ANTHROPIC:
        return call_anthropic(model_name=model_name, system_msg=DEFAULT_PROMPT, human_msg=user_query)
    elif model_provider == ModelProvider.PERPLEXITY:
        return call_perplexity(model_name=model_name, system_msg=DEFAULT_PROMPT, human_msg=user_query)
    elif model_provider == ModelProvider.X_AI:
        return call_grok(model_name=model_name, system_msg=DEFAULT_PROMPT, human_msg=user_query)
    elif model_provider == ModelProvider.GEMINI:
        return call_gemini(model_name=model_name, system_msg=DEFAULT_PROMPT, human_msg=user_query)
    elif model_provider == ModelProvider.SARVAM:
        return call_sarvam(model_name=model_name, system_msg=DEFAULT_PROMPT, human_msg=user_query)