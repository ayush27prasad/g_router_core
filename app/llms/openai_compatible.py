from openai import OpenAI
from app.system_prompts import DEFAULT_PROMPT

def call_onboarded_model(base_url: str, api_key: str, model_name: str, user_query: str) -> str:
    """Call the Onboarded Model API using OpenAI client."""

    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": DEFAULT_PROMPT},
            {"role": "user", "content": user_query}
        ]
    )
    response_content = resp.choices[0].message.content
    print(f"{model_name} response : {response_content}")
    return response_content