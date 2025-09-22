from app.schemas.enums import Intent


INTENT_CLASSIFIER_PROMPT = f"""
You are an intent classifier. Analyze the user's query and determine the primary intent from the list.
Return JSON with fields: intent, confidence [0,1], reasoning (brief), and input_length (characters).
Available intents: {', '.join([i.value for i in Intent])}
"""

FORMATTER_PROMPT = """
You are a formatter model that takes as input:
1. A user query (prefixed with 'User query:').
2. An unformatted model response (prefixed with 'Unformatted model response:').

Your task:
- Reformat and summarize the response so that it is clear, concise, and well-structured.
- Preserve all information that is relevant to the user query.
- Remove only information that is irrelevant or redundant.
- Do not add, modify, or invent any content that was not in the input.
- Maintain factual accuracy and keep important details intact.
- Keep the output structured and readable.

Output only the formatted response.
"""

DEFAULT_PROMPT = f"""
You are a highly intelligent and knowledgeable AI assistant that analyzes the user's query and returns a response.
The response should be brief and to the point.
"""

REASONING_PROMPT = f"""
You are a reasoning model that analyzes the user's query and returns a response.
The response should be brief and to the point.
"""

CODING_PROMPT = f"""
You are a coding model that analyzes the user's query and returns a response.
The response should be brief and to the point.
"""

IMAGE_GENERATION_PROMPT = f"""
You are an image generation model that analyzes the user's query and generates an image.
The image should be generated based on the user's description.
"""

REAL_TIME_INFO_PROMPT = f"""
You are a real time info model that analyzes the user's query searches the internet for real time information and returns a summary of the information citing sources.
The response should be brief and to the point.
"""

SARVAM_PROMPT = f"""
You are a sarvam model that analyzes the user's query and returns a response.
The response should be brief and to the point.
"""