import os
import json
import boto3
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings
from langchain.text_splitter import CharacterTextSplitter

from io import BytesIO
import PyPDF2

# -----------------------------
# AWS Credentials (IAM USER)
# -----------------------------
aws_access_key = input("Enter AWS Access Key: ").strip()
aws_secret_key = input("Enter AWS Secret Key: ").strip()
region = "us-east-1"

session = boto3.Session(
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=region
)

bedrock_runtime = session.client("bedrock-runtime")

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="AI Medical Assistant (Bedrock)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "faiss_medical"

# -----------------------------
# Embeddings (Bedrock Titan)
# -----------------------------
embeddings = BedrockEmbeddings(
    client=session.client("bedrock-runtime"),
    model_id="amazon.titan-embed-text-v1"
)

# -----------------------------
# Helpers
# -----------------------------
def load_db():
    if os.path.exists(DB_PATH):
        return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    return None

def save_db(db):
    db.save_local(DB_PATH)

def extract_pdf(file: UploadFile):
    reader = PyPDF2.PdfReader(BytesIO(file.file.read()))
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def store_pdf(user_id, text):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_text(text)

    db = load_db()

    if db is None:
        db = FAISS.from_texts(
            docs,
            embeddings,
            metadatas=[{"user_id": user_id}] * len(docs)
        )
    else:
        db.add_texts(
            docs,
            metadatas=[{"user_id": user_id}] * len(docs)
        )

    save_db(db)

def retrieve(user_id, query):
    db = load_db()
    if db is None:
        return ""

    results = db.similarity_search(query, k=3)

    return "\n".join([
        r.page_content for r in results if r.metadata["user_id"] == user_id
    ])

def ask_bedrock(prompt):
    body = {
        "prompt": prompt,
        "max_gen_len": 150,
        "temperature": 0.3,
        "top_p": 0.9
    }

    response = bedrock_runtime.invoke_model(
        modelId="meta.llama3-8b-instruct-v1:0",
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json"
    )

    result = json.loads(response["body"].read())
    return result.get("generation", "")

# -----------------------------
# API Models
# -----------------------------
class QuestionRequest(BaseModel):
    user_id: str
    question: str

# -----------------------------
# Routes
# -----------------------------
@app.post("/upload_pdf")
async def upload_pdf(user_id: str = Form(...), file: UploadFile = File(...)):
    try:
        text = extract_pdf(file)
        store_pdf(user_id, text)
        return {"message": "PDF stored successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask(req: QuestionRequest):
    try:
        context = retrieve(req.user_id, req.question)

        if not context:
            return {"answer": "No medical data found. Upload PDF first."}

        prompt = f"""
You are an AI Medical Assistant.

Use the medical document below to answer.

Context:
{context}

Question:
{req.question}

Answer clearly:
"""

        answer = ask_bedrock(prompt)
        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "AI Medical Assistant API running 🚀"}