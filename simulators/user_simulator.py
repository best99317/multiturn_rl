from typing import List
import logging
import boto3
import time

from prompts import USER_PROMPT, TERMINATION_SIGNAL
from utils.parse_message import parse_messages
from utils.extract_json import extract_json
from utils.bedrock_call import bedrock_call
from concurrent.futures import ThreadPoolExecutor
import asyncio
logger = logging.getLogger(__name__)


class UserSimulator(object):
    def __init__(self, task_desc='', single_turn_prompt='', num_retries=100, **llm_kwargs):
        """
        Initialize the UserSimulator model.
        """
        super().__init__()
        self.task_desc = task_desc
        self.single_turn_prompt = single_turn_prompt
        self.num_retries = num_retries

        self.llm_kwargs = {"temperature": 1.0, "max_tokens": 1024, **llm_kwargs}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bedrock")
        
        assert 'model' in self.llm_kwargs, "Model name must be provided in llm_kwargs"

    def _sync_call(self, messages: List[dict]):
        """Synchronous call to bedrock - runs in thread pool"""
        prompt = USER_PROMPT.format(
            task_desc=self.task_desc,
            single_turn_prompt=self.single_turn_prompt,
            chat_history=parse_messages(messages, strip_sys_prompt=True),
            terminal_signal=TERMINATION_SIGNAL,
        )
        messages = [{"role": "user", "content": prompt}]
        
        num_tries = 0
        while True:
            try:
                response = bedrock_call(
                    model=self.llm_kwargs['model'], 
                    max_tokens=self.llm_kwargs['max_tokens'],
                    temperature=self.llm_kwargs['temperature'],
                    messages=messages
                )
                
            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    raise e
                print(f"Bedrock error: {e}")
                
                num_tries += 1
                if num_tries > self.num_retries:
                    print("Error: User simulator bedrock call too many retries ... Possible error in code ...")
                    break
                
                time.sleep(2)
                continue
            
            try:
                response = eval(response)
            except Exception as e:
                pass
            
            if isinstance(response, dict):
                keys = response.keys()
                if {'thought', 'response'}.issubset(keys):
                    response = response.pop('response')
                    break
                else:
                    logger.error(f"[LLMCollaborator] Keys {keys} do not match expected keys. Retrying...")
                    continue
            else:
                break
        
        return response.strip()

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
