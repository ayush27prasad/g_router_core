from models.enums import Intent


INTENT_CLASSIFIER_PROMPT = f"""
        You are an intent classifier that analyzes . Determine the user's primary intent from the list.
        Return confidence in [0,1], a brief reasoning, and input_length (characters).
        Available intents: {', '.join([i.value for i in Intent])}
        """