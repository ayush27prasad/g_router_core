from enum import Enum
from typing import Dict


class Intent(str, Enum):
    REASONING = "reasoning"
    CURRENT_AFFAIRS = "current_affairs"
    CODE_GENERATION = "code_generation"
    DEBUG_CODE = "debug_code"
    IMAGE_GENERATION = "image_generation"
    LOCALIZED_INDIA = "localized_india"
    OTHER = "other"

class ToolResponseType(str, Enum):
    TEXT = "text"
    MARKDOWN = "markdown"
    IMAGE = "image"
    BYTE_ARRAY = "byte_array"

class ModelName(str, Enum):
    OPEN_AI_GPT_4O_MINI = "gpt-4o-mini"
    OPEN_AI_GPT_5 = "gpt-5"
    ANTHROPIC_OPUS = "claude-3.5-sonnet"
    PERPLEXITY_O3 = "o3"
    GROK_4 = "grok-4"
    GEMINI_NANO_BANANA = "gemini-nano-banana"
    SARVAM_V2 = "sarvam-v2"

class ModelProvider(str, Enum):
    OPEN_AI = "openai"
    ANTHROPIC = "anthropic"
    PERPLEXITY = "perplexity"
    X_AI = "grok"
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"
    SARVAM = "sarvam"

def get_model_provider(model_name: ModelName) -> ModelProvider:
    model_provider_company_mapping : Dict[ModelName, ModelProvider] = {
        ModelName.OPEN_AI_GPT_4O_MINI: ModelProvider.OPEN_AI,
        ModelName.OPEN_AI_GPT_5: ModelProvider.OPEN_AI,
        ModelName.ANTHROPIC_OPUS: ModelProvider.ANTHROPIC,
        ModelName.PERPLEXITY_O3: ModelProvider.PERPLEXITY,
        ModelName.GROK_4: ModelProvider.X_AI,
        ModelName.GEMINI_NANO_BANANA: ModelProvider.GEMINI,
        ModelName.SARVAM_V2: ModelProvider.SARVAM,
    }
    return model_provider_company_mapping[model_name]
