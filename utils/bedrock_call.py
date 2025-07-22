from typing import List
import json
import boto3
from botocore.exceptions import ClientError
import threading
import time

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

def bedrock_call(model: str = None, messages: List[dict] = None, max_tokens: int = 4096, temperature: float = 1.0, num_retries: int = 30):

    # Get thread-local client
    bedrock_inference_client = get_bedrock_client()

    # Set the model ID
    model_id = model

    # Define the request using Messages API format
    if "claude" in model_id:
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages
        }
    elif "llama" in model_id:
        def format_messages_for_llama(messages):
            prompt = "<|begin_of_text|>"
            
            for message in messages:
                role = message["role"]
                content = message["content"]
                
                if role == "system":
                    prompt += f"<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
                elif role == "user":
                    prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
                elif role == "assistant":
                    prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
            
            # Add final assistant header to prompt for response
            prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
            return prompt
            
        request_body = {
            "max_gen_len": max_tokens,
            "temperature": temperature,
            "prompt": format_messages_for_llama(messages)
        }
    else:
        raise "Model not claude nor llama, check request_body format"

    # Convert the request body to JSON
    request = json.dumps(request_body)

    num_tries = 0
    while True:
        try:
            # Invoke the model with the request
            response = bedrock_inference_client.invoke_model(
                modelId=model_id,
                body=request
            )
            
            # Decode the response body
            response_body = json.loads(response["body"].read())
            if "content" in response_body:
                result = response_body["content"][0]["text"]
                # 🔴 ADDED: Check for None response and retry if needed
                if result is None or result == "None":
                    raise Exception("Received None response from bedrock")
                return result
            else:
                result = response_body
                # 🔴 ADDED: Check for None response and retry if needed
                if result is None or result == "None":
                    raise Exception("Received None response from bedrock")
                return result
                
        except (ClientError, Exception) as e:
            if isinstance(e, KeyboardInterrupt): 
                raise e
            print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
            
            num_tries += 1
            if num_tries > num_retries:
                print("Error: Bedrock call too many retries ... Possible error in code ...")
                return None  # 🔴 ADDED: Return None after max retries
            
            time.sleep(30*num_tries)  # 🔴 ADDED: Sleep before retry
            continue