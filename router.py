from models import Intent, ToolResponse
import tools


def llm_call_by_intent(intent: Intent, user_text: str) -> ToolResponse:
    if intent == Intent.SUMMARY:
        return tools.summary(user_text)
    if intent == Intent.BASIC_QA:
        return tools.basic_qa(user_text)
    if intent == Intent.CODE_GENERATION:
        return tools.code_generation(user_text)
    if intent == Intent.DEBUG_CODE:
        return tools.debug_code(user_text)
    if intent == Intent.IMAGE_GENERATION:
        return tools.generate_image(user_text)
    if intent == Intent.MATH_SOLVE:
        return tools.solve_math(user_text)

    # Fallback to basic QA
    return tools.basic_qa(user_text)


