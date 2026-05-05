from rag import retrieve_context
from intents import detect_intent
from utils import call_bedrock

def handle_chat(user_id, message):
    intent = detect_intent(message)
    context = retrieve_context(message)

    prompt = f"""
You are an AI Ecommerce Assistant.

Intent: {intent}

Context:
{context}

User Question:
{message}

Give helpful response:
"""

    return call_bedrock(prompt)
