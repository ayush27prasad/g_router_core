from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from schemas.models import ToolResponse
from schemas.enums import ToolResponseType 
from utils import save_base64_to_downloads


def call_gemini_image_generation_model(model_name: str, system_msg: str, human_msg: str) -> ToolResponse:
    """Call the Gemini Image Generation Model."""
    llm = ChatGoogleGenerativeAI(model=model_name)
    response = llm.invoke(
        [HumanMessage(human_msg)],
        generation_config=dict(response_modalities=["TEXT", "IMAGE"]),
    )
    image_block = next(
        block
        for block in response.content
        if isinstance(block, dict) and block.get("image_url")
    )

    image_base64 = image_block["image_url"].get("url").split(",")[-1]
    # TODO : Upload image to storage and return url
    # saved_path = save_base64_to_downloads(image_base64, "cat_hat.png")
    saved_path = save_base64_to_downloads(image_base64, f"{human_msg[:15].replace(' ', '_')}.png")
    print(f"Image saved at: {saved_path}")
    
    return ToolResponse(type=ToolResponseType.IMAGE_URL, content=f"I think this is the image u asked for is saved in the downloads folder at {saved_path}", response_generated_via=model_name)

def call_gemini(model_name: str, system_msg: str, human_msg: str) -> ToolResponse:
    """Call the Gemini API."""
    model = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
    return model.invoke([SystemMessage(system_msg), HumanMessage(human_msg)])
