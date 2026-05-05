import os
import boto3
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings

region = "us-east-1"
session = boto3.Session(region_name=region)

embeddings = BedrockEmbeddings(
    client=session.client("bedrock-runtime"),
    model_id="amazon.titan-embed-text-v1"
)

DB_PATH = "/tmp/faiss_store"

def load_db():
    if os.path.exists(DB_PATH):
        return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    return None

def retrieve_context(query):
    db = load_db()
    if db is None:
        return ""

    results = db.similarity_search(query, k=3)
    return "\n".join([r.page_content for r in results])
