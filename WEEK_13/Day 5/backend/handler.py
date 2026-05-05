import json
from chatbot import handle_chat

def lambda_handler(event, context):
    body = json.loads(event.get("body", "{}"))
    user_id = body.get("user_id")
    message = body.get("message")

    response = handle_chat(user_id, message)

    return {
        "statusCode": 200,
        "body": json.dumps({"response": response})
    }
