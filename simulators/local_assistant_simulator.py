from typing import Dict, List
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class LoRAAssistantSimulator:
    """Assistant simulator with LoRA support using vLLM"""
    
    def __init__(self, assistant_meta_prompt: None, lora_model_path: str, base_model_path: str, num_gpus: int = 1, **generation_kwargs):
        self.lora_model_path = lora_model_path
        self.base_model_path = base_model_path
        self.num_gpus = min(num_gpus, torch.cuda.device_count()) if torch.cuda.is_available() else 1
        self.generation_kwargs = generation_kwargs
        self.llm = None
        self.lora_request = None
        self.sampling_params = None
        self.assistant_meta_prompt = assistant_meta_prompt
        print(f"🚀 Initializing LoRA assistant with {self.num_gpus} GPU(s)...")
        print(f"   Base model: {base_model_path}")
        print(f"   LoRA model: {lora_model_path}")
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize vLLM model with LoRA"""
        try:
            # Initialize vLLM with LoRA support            
            self.llm = LLM(
                model=self.base_model_path,
                enable_lora=True,
                tensor_parallel_size=self.num_gpus if torch.cuda.is_available() else 1,
                gpu_memory_utilization=0.7,  # Increased for LoRA
                max_model_len=2048,  # Adjust based on your needs
                trust_remote_code=True,
                max_lora_rank=64,
                max_num_batched_tokens=4096,
                max_num_seqs=32,
            )

            self.lora_request = LoRARequest(
                "assistant_simulator",  # Human readable name
                1,                      # Unique ID
                self.lora_model_path    # Path to LoRA adapter
            )
            
            self.sampling_params = SamplingParams(
                temperature=self.generation_kwargs.get("temperature", 0.7),
                top_p=self.generation_kwargs.get("top_p", 0.9),
                max_tokens=self.generation_kwargs.get("max_tokens", 512),
                stop=["</s>", "<|endoftext|>", "User:", "Human:", "\n\nUser:", "\n\nHuman:"]
            )
            
            logger.info(f"✅ Successfully initialized vLLM with LoRA: {self.lora_model_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize vLLM with LoRA: {e}")
            logger.error("💡 Make sure your LoRA model path is correct and contains adapter files")
            raise

    def generate_responses_batch(self, message_lists: List[List[Dict[str, str]]], **kwargs) -> List[str]:
        """Generate multiple responses in a single batch - NEW METHOD"""
        try:
            prompts = [self._messages_to_prompt(messages) for messages in message_lists]
            
            # Use LoRA request for batch generation
            outputs = self.llm.generate(
                prompts, 
                self.sampling_params,
                lora_request=self.lora_request  # CHANGED: Use pre-created LoRARequest
            )
            
            responses = [output.outputs[0].text.strip() for output in outputs]
            return [self._clean_response(response) for response in responses]
            
        except Exception as e:
            logger.error(f"Error in LoRA batch generation: {e}")
            return ["I understand. How can I help you further?"] * len(message_lists)
    
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate single response using LoRA"""
        try:
            prompt = self._messages_to_prompt(messages)
            
            # print("===================== Assistant input prompt: =====================\n", prompt)
            # print("==========================================\n")
            
            # Use LoRA request for generation
            
            outputs = self.llm.generate(
                [prompt], 
                self.sampling_params,
                lora_request=self.lora_request
            )
            
            response = outputs[0].outputs[0].text.strip()
            return self._clean_response(response)
            
        except Exception as e:
            logger.error(f"Error in LoRA generation: {e}")
            return "I understand. How can I help you further?"
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to simple prompt format"""
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
    
    def _clean_response(self, response: str) -> str:
        """Clean up generated response"""
        # Remove any potential role prefixes
        prefixes_to_remove = ["Assistant:", "AI:", "Bot:", "Response:"]
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        # Remove any trailing role indicators
        suffixes_to_remove = ["User:", "Human:", "\nUser:", "\nHuman:"]
        for suffix in suffixes_to_remove:
            if response.endswith(suffix):
                response = response[:-len(suffix)].strip()
        
        return response
