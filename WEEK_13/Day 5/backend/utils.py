import boto3
import json

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

def call_bedrock(prompt):
    body = {
        "prompt": prompt,
        "max_gen_len": 150,
        "temperature": 0.5,
        "top_p": 0.9
    }

    response = bedrock.invoke_model(
        modelId="meta.llama3-8b-instruct-v1:0",
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json"
    )

    result = json.loads(response["body"].read())
    return result.get("generation", "")
