import streamlit as st
import boto3
import json

# Page config
st.set_page_config(
    page_title="LLaMA 3 Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 LLaMA 3 Chatbot (Low Cost)")
st.caption("AWS Bedrock - Optimized")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
with st.sidebar:
    st.header("AWS Config")

    aws_access_key = st.text_input("Access Key", type="password")
    aws_secret_key = st.text_input("Secret Key", type="password")

    region = st.selectbox(
        "Region",
        ["us-east-1", "us-west-2"]
    )

    connect = st.button("Connect")

# Connect
if connect and aws_access_key and aws_secret_key:
    try:
        session = boto3.Session(
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region
        )

        st.session_state.bedrock = session.client("bedrock-runtime")
        st.success("Connected ✅")

    except Exception as e:
        st.error(f"Error: {e}")

# Chat UI
if "bedrock" in st.session_state:

    user_input = st.text_input("Message")

    if st.button("Send") and user_input:

        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )

        prompt = ""
        for chat in st.session_state.chat_history:
            role = "User" if chat["role"] == "user" else "Assistant"
            prompt += f"### {role}:\n{chat['content']}\n\n"

        body = {
            "prompt": prompt,
            "temperature": 0.5,
            "top_p": 0.9,
            "max_gen_len": 100   # LOW COST
        }

        try:
            response = st.session_state.bedrock.invoke_model(
                modelId="meta.llama3-8b-instruct-v1:0",
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json"
            )

            result = json.loads(response["body"].read())
            output = result.get("generation", "")

            st.session_state.chat_history.append(
                {"role": "assistant", "content": output}
            )

        except Exception as e:
            st.error(e)

    # Display
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.write(f"🧑: {chat['content']}")
        else:
            st.write(f"🤖: {chat['content']}")

else:
    st.info("Enter AWS credentials to start")