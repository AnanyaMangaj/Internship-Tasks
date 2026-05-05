def detect_intent(message):
    msg = message.lower()

    if "order" in msg:
        return "order_status"
    elif "refund" in msg:
        return "refund_request"
    elif "product" in msg:
        return "product_query"
    else:
        return "general_query"
