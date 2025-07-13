from typing import List, Dict
import logging
import boto3

from prompts import USER_PROMPT, TEST_PROMPT, TERMINATION_SIGNAL
from utils.parse_message import parse_messages
from utils.extract_json import extract_json
from utils.bedrock_call import bedrock_call
from concurrent.futures import ThreadPoolExecutor
import asyncio
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class UserSimulator(object):
    def __init__(self, user_meta_prompt=None, num_retries=100, **llm_kwargs):
        """
        Initialize the UserSimulator model.
        """
        super().__init__()
        self.user_meta_prompt = user_meta_prompt
        self.num_retries = num_retries

        self.llm_kwargs = {"temperature": 1.0, "max_tokens": 1024, **llm_kwargs}
        self._executor = ThreadPoolExecutor(max_workers=100, thread_name_prefix="bedrock")
        
        assert 'model' in self.llm_kwargs, "Model name must be provided in llm_kwargs"

    def _format_conversation_history(self, messages: List[Dict[str, str]]) -> str:
        """Format conversation history for context"""
        formatted_history = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                continue  # Skip system messages in history
            elif role == "user":
                formatted_history.append(f"User: {content}")
            elif role == "assistant":
                formatted_history.append(f"Assistant: {content}")
        
        formatted_history.append("User: ")  # Prompt for user response
        return "\n".join(formatted_history)
    
    def _messages_to_bedrock_format(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Convert conversation to Bedrock format for USER simulation"""
        bedrock_messages = []
        
        # Add the conversation history as user message context
        conversation_context = self.user_meta_prompt.format(
            chat_history = self._format_conversation_history(messages)
        ) 
        
        bedrock_messages.append({
            "role": "user",
            "content": conversation_context
        })
        
        return bedrock_messages

    def _sync_call(self, messages: List[dict]):
        """Synchronous call to bedrock - runs in thread pool"""

        prompt = self._messages_to_bedrock_format(messages)

        # print("++++++++++++++++++++++ User input prompt: ++++++++++++++++++++++\n", prompt)
        # print("++++++++++++++++++++++++++++++++++++++++++++\n")
        
        
        response = bedrock_call(
            model=self.llm_kwargs['model'], 
            max_tokens=self.llm_kwargs['max_tokens'],
            temperature=self.llm_kwargs['temperature'],
            messages=prompt,
            num_retries=self.num_retries
        )
        
        if response is None:
            print("Error: User simulator bedrock call failed after retries")
            return ""

        try:
            if isinstance(response, str):
                import json
                try:
                    parsed_response = json.loads(response)
                except json.JSONDecodeError:
                    parsed_response = response
            else:
                parsed_response = response
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            parsed_response = response
        
        if isinstance(response, dict):
            keys = response.keys()
            if {'thought', 'response'}.issubset(keys):
                response = response.pop('response')
            elif {'generation'}.issubset(keys):
                response = response.pop('generation')
            else:
                logger.error(f"[LLMCollaborator] Keys {keys} do not match expected keys. Retrying...")

        return str(response).strip()

    def __call__(self, messages: List[dict]):
        """Call method that works in both sync and async contexts"""
        # 🔴 IMPORTANT: Always run in sync context, even if called from async
        return self._sync_call(messages)
    
    async def async_call(self, messages: List[dict]):
        """Async wrapper for use in async contexts"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._sync_call, messages)

    def __del__(self):
        """Clean up thread pool on deletion"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)
