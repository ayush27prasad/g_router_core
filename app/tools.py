from langchain_core.messages import SystemMessage, HumanMessage
from typing import Dict
from llms.openai import get_mini_model, call_openai
from llms.anthropic import call_anthropic
from llms.sarvam import call_sarvam
from llms.google import call_gemini_image_model, call_gemini
from llms.perplexity import call_perplexity, call_perplexity_api
from llms.xai import call_grok
from schemas.models import IntentAnalysis, ToolResponse
from schemas.enums import ModelName, ModelProvider

from system_prompts import *

def analyze_intent(user_query: str) -> IntentAnalysis:
    """Analyze user intent only using structured output."""
    classifier_model = get_mini_model().with_structured_output(schema=IntentAnalysis)
    return classifier_model.invoke([SystemMessage(INTENT_CLASSIFIER_PROMPT), HumanMessage(user_query)])

def call_reasoning_model(user_query: str) -> ToolResponse:
    """Resolve reasoning query."""
    reasoning_model_name = "gpt-5-mini"
    reasoning_model_response = call_openai(model_name=reasoning_model_name, system_msg=REASONING_PROMPT, human_msg=user_query)
    return reasoning_model_response

def call_coding_model(user_query: str) -> ToolResponse:
    """Resolve coding query."""
    coding_model_name = "claude-opus-4-1-20250805"
    coding_model_response = call_anthropic(model_name=coding_model_name, system_msg=CODING_PROMPT, human_msg=user_query)
    return coding_model_response

def call_image_generation_model(user_query: str) -> ToolResponse:
    """Generate image."""
    image_generation_model_name = "gemini-nano-banana"
    image_generation_model_response = call_gemini_image_model(model_name=image_generation_model_name, system_msg=IMAGE_GENERATION_PROMPT, human_msg=user_query)
    return image_generation_model_response

def call_real_time_info_model(user_query: str) -> ToolResponse:
    """Fetch real time info."""
    real_time_info_model_name = "sonar-pro"
    real_time_info_model_response = call_perplexity_api(model_name=real_time_info_model_name, system_msg=REAL_TIME_INFO_PROMPT, human_msg=user_query)
    return real_time_info_model_response

def call_sarvam_model(user_query: str) -> ToolResponse:
    """Fetch real time info."""
    sarvam_model_name = "sarvam-v2"
    sarvam_model_response = call_sarvam(model_name=sarvam_model_name, system_msg=SARVAM_PROMPT, human_msg=user_query)
    return sarvam_model_response

def call_default_model(user_query: str) -> ToolResponse:
    """Call the default model."""
    default_model_name = "gpt-4o-mini"
    default_model_response = call_openai(model_name=default_model_name, system_msg=DEFAULT_PROMPT, human_msg=user_query)
    return default_model_response

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