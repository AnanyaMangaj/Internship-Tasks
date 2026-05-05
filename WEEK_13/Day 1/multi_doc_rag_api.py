import streamlit as st
import tempfile
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="Multi-Doc Chatbot", layout="wide")

st.title("📚 Multi-Document Chatbot (RAG)")
st.markdown("Upload multiple documents and chat with them.")

# ---------------- SESSION ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Settings")

    openai_api_key = st.text_input("OpenAI API Key", type="password")

    uploaded_files = st.file_uploader(
        "Upload multiple PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    process_btn = st.button("🚀 Process Documents")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------- PROCESS DOCUMENTS ----------------
if process_btn:
    if not openai_api_key:
        st.error("Enter API key")
    elif not uploaded_files:
        st.error("Upload at least one file")
    else:
        try:
            with st.spinner("Processing all documents..."):

                all_docs = []

                for uploaded_file in uploaded_files:
                    file_extension = os.path.splitext(uploaded_file.name)[1]

                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
                        tmp.write(uploaded_file.read())
                        path = tmp.name

                    if uploaded_file.type == "application/pdf":
                        loader = PyPDFLoader(path)
                    else:
                        loader = TextLoader(path)

                    docs = loader.load()
                    all_docs.extend(docs)

                # -------- SPLIT --------
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                split_docs = splitter.split_documents(all_docs)

                # -------- EMBEDDINGS --------
                embeddings = OpenAIEmbeddings(api_key=openai_api_key)

                # -------- VECTOR STORE --------
                vectorstore = FAISS.from_documents(split_docs, embeddings)

                # -------- LLM --------
                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    api_key=openai_api_key,
                    temperature=0
                )

                # -------- QA CHAIN --------
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=vectorstore.as_retriever()
                )

                st.session_state.qa_chain = qa_chain

                st.success("✅ All documents processed successfully!")

        except Exception as e:
            st.error(str(e))

# ---------------- CHAT DISPLAY ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- CHAT INPUT ----------------
if prompt := st.chat_input("Ask questions from your documents..."):

    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if st.session_state.qa_chain is None:
        response = "Please upload and process documents first."
    else:
        try:
            with st.spinner("Thinking..."):

                chat_history = []
                for i in range(0, len(st.session_state.messages) - 1, 2):
                    if i + 1 < len(st.session_state.messages):
                        chat_history.append((
                            st.session_state.messages[i]["content"],
                            st.session_state.messages[i + 1]["content"]
                        ))

                result = st.session_state.qa_chain.invoke({
                    "question": prompt,
                    "chat_history": chat_history
                })

                response = result["answer"]

        except Exception as e:
            response = f"Error: {str(e)}"

    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
