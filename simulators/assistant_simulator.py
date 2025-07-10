from typing import List
import logging
import boto3
import time

from utils.parse_message import parse_messages
from utils.extract_json import extract_json
from utils.bedrock_call import bedrock_call
from prompts import ASSITANT_PROMPT

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class LLMCollaborator(object):
    registered_prompts = {
        'none': None,
        'meta': ASSITANT_PROMPT
    }
    def __init__(self, method='none', num_retries=300, **llm_kwargs):
        """
        Initialize the LLMAssistant model.
        """
        super().__init__()
        self.method = method
        assert method in self.registered_prompts, f"Prompting method {method} not registered. Available methods: {list(self.registered_prompts.keys())}"

        self.num_retries = num_retries
        self.llm_kwargs = {"temperature": 0.8, "max_tokens": 2048, **llm_kwargs}

    def __call__(self, messages: List[dict], **kwargs):
        """
        Forward pass of the LLMAssistant model.
        
        Args:
            messages (List[dict]): A list of message dictionaries with the last message being the user message.
        
        Returns:
            torch.Tensor: The output tensor.
        """
        assert messages[-1]['role'] == 'user'

        if self.method == 'none':
            if len(messages) and messages[0]['role'] == 'system':
                logger.info('System message detected.')
        else:
            kwargs = {}
            prompt = ASSITANT_PROMPT.format(
                chat_history=parse_messages(messages, strip_sys_prompt=True),
                max_new_tokens=self.llm_kwargs.get('max_new_tokens', 1024),
                additional_info=kwargs.get('additional_info', '')
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
    
                if e == KeyboardInterrupt:
                    raise e
                print(e)

                num_tries += 1
                if num_tries > self.num_retries:
                    print("Error: Assistant simulator bedrock call too many retries ... Possible error in code ...")
                    break

                time.sleep(2)
                continue
            
            try:
                if isinstance(response, str) and not (self.method == 'none'):
                    response = extract_json(response)
            except Exception as e:
                logger.error(f"[LLMCollaborator] Error extracting JSON: {e}")
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