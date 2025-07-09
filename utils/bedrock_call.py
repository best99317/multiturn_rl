from typing import List
import json
import boto3
from botocore.exceptions import ClientError
import threading

_thread_local = threading.local()

def get_bedrock_client():
    """Get or create a thread-local bedrock client"""
    if not hasattr(_thread_local, 'bedrock_client'):
        session = boto3.session.Session()
        _thread_local.bedrock_client = session.client(
            service_name="bedrock-runtime",
            region_name="us-east-1",
            endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
        )
    return _thread_local.bedrock_client

def bedrock_call(model: str = None, messages: List[dict] = None, max_tokens: int = 4096, temperature: float = 1.0):

    # Get thread-local client
    bedrock_inference_client = get_bedrock_client()

    # Set the model ID
    model_id = model

    # Define the request using Messages API format
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages
    }

    # Convert the request body to JSON
    request = json.dumps(request_body)

    try:
        # Invoke the model with the request
        response = bedrock_inference_client.invoke_model(
            modelId=model_id,
            body=request
        )
        
        # Decode the response body
        response_body = json.loads(response["body"].read())
        
        # Extract and print the response text
        if "content" in response_body:
            # print(response_body["content"][0]["text"])
            return response_body["content"][0]["text"]
        else:
            # print(response_body)
            return response_body
            
    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")