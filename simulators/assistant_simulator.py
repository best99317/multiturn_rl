from typing import List, Dict
import logging
import boto3
import time

from utils.parse_message import parse_messages
from utils.extract_json import extract_json
from utils.bedrock_call import bedrock_call

from concurrent.futures import ThreadPoolExecutor
import asyncio
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class AssistantSimulator(object):
    def __init__(self, assistant_meta_prompt=None, num_retries=100, **llm_kwargs):
        """
        Initialize the LLMAssistant model.
        """
        super().__init__()
        self.assistant_meta_prompt = assistant_meta_prompt
        self.num_retries = num_retries
        
        self.llm_kwargs = {"temperature": 0.8, "max_tokens": 2048, **llm_kwargs}
        self._executor = ThreadPoolExecutor(max_workers=100, thread_name_prefix="bedrock")
        assert 'model' in self.llm_kwargs, "Model name must be provided in llm_kwargs"

    def _format_conversation_history(self, messages: List[Dict[str, str]]) -> str:
        """Format conversation history for context"""
        prompt_parts = []

        if self.assistant_meta_prompt:
            prompt_parts.append(f"System: {self.assistant_meta_prompt}\n")
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
        
            if role == "user":
                prompt_parts.append(f"User: {content}\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n")
        
        prompt_parts.append("Assistant: ")  # Prompt for assistant response
        return "".join(prompt_parts)
    
    def _messages_to_bedrock_format(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Convert conversation to Bedrock format for ASSISTANT simulation"""
        bedrock_messages = []
        
        # Add the conversation history as user message context
        conversation_context = self._format_conversation_history(messages)
        
        bedrock_messages.append({
            "role": "user",
            "content": conversation_context
        })
        
        return bedrock_messages

    def _sync_call(self, messages: List[dict]):
        """
        Forward pass of the LLMAssistant model.
        
        Args:
            messages (List[dict]): A list of message dictionaries with the last message being the user message.
        
        Returns:
            torch.Tensor: The output tensor.
        """
        assert messages[-1]['role'] == 'user'

        prompt = self._messages_to_bedrock_format(messages)
        
        response = bedrock_call(
            model=self.llm_kwargs['model'], 
            max_tokens=self.llm_kwargs['max_tokens'],
            temperature=self.llm_kwargs['temperature'],
            messages=prompt,
            num_retries=self.num_retries
        )


        if response is None:
            print("Error: Assistant simulator bedrock call failed after retries")
            return ""

        try:
            if isinstance(response, str) and self.method != 'none':
                import json
                try:
                    response = extract_json(response)
                except:
                    pass
            
            if isinstance(response, str):
                try:
                    parsed_response = json.loads(response)
                except:
                    parsed_response = response
            else:
                parsed_response = response
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            parsed_response = response
        
        if isinstance(parsed_response, dict):
            keys = parsed_response.keys()
            if {'thought', 'response'}.issubset(keys):
                response = parsed_response.pop('response')
            elif {'generation'}.issubset(keys):
                response = parsed_response.pop('generation')
            else:
                logger.error(f"[AssistantSimulator] Keys {keys} do not match expected keys.")
                response = str(parsed_response)

        return str(response).strip()

    def __call__(self, messages: List[dict]):
        """Call method that works in both sync and async contexts"""
        # Always run in sync context, even if called from async
        return self._sync_call(messages)
    
    async def async_call(self, messages: List[dict]):
        """Async wrapper for use in async contexts"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._sync_call, messages)