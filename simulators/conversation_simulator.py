import asyncio
import copy
from typing import Any, Dict, List
from dataclasses import dataclass
import torch
from concurrent.futures import ThreadPoolExecutor
from simulators.user_simulator import UserSimulator
from simulators.local_assistant_simulator import LoRAAssistantSimulator
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ConversationConfig:
    user_meta_prompt: str
    assistant_meta_prompt: str
    max_total_turns: int = 10
    max_gen_workers: int = 2  # Reduced for testing
    local_model_path: str = "microsoft/DialoGPT-medium"  # Using a smaller model for testing
    base_model_path: str = "microsoft/DialoGPT-medium"
    user_generation_kwargs: Dict[str, Any] = None
    assistant_generation_kwargs: Dict[str, Any] = None
    enable_batching: bool = True  # Enable batch processing for better performance
    bedrock_rate_limit_delay: float = 0.01  # Delay between bedrock calls to avoid rate limits

    def __post_init__(self):
        if self.user_generation_kwargs is None:
            self.user_generation_kwargs = {
                "model": "anthropic.claude-sonnet-4-20250514-v1:0",
                "temperature": 1.0,
                "max_tokens": 512
            }
        if self.assistant_generation_kwargs is None:
            self.assistant_generation_kwargs = {
                "temperature": 0.7,
                "max_tokens": 512
            }

class MultiTurnConversationGenerator:
    """Main class for generating multi-turn conversations"""
    
    def __init__(self, config: ConversationConfig):
        self.config = config
        print("🤖 Initializing conversation generator...")
        
        # Initialize assistant simulator with LoRA
        self.assistant_simulator = LoRAAssistantSimulator(
            assistant_meta_prompt = config.assistant_meta_prompt,
            lora_model_path=config.local_model_path,
            base_model_path=config.base_model_path,
            num_gpus=torch.cuda.device_count() if torch.cuda.is_available() else 1,
            **config.assistant_generation_kwargs
        )

        self.executor = ThreadPoolExecutor(max_workers=config.max_gen_workers)
        
        # Semaphore for controlling concurrent conversations
        self.semaphore = asyncio.Semaphore(config.max_gen_workers)
        print("✅ Conversation generator ready!")
    
    async def generate_conversations_batch(self, prompts: List[str]=None, conv_num: int=1, batch_configs: List[Dict]=None, **kwargs) -> List[List[Dict[str, str]]]:
        """Generate multiple conversations with optimized batching - NEW METHOD"""
        print(f"🚀 Starting batch generation for {len(prompts)} conversations...")

        if batch_configs is not None:
            conv_num = len(batch_configs)
            if prompts is None:
                prompts = [None for config in batch_configs]
            print(f"🚀 Starting batch generation for {conv_num} conversations with custom configs...")
        else:
            print(f"🚀 Starting batch generation for {conv_num} conversations...")
        
        conversation_states = []
        for i in range(conv_num):
            chat_history = []

            if batch_configs and i < len(batch_configs):
                config = batch_configs[i]
                user_meta_prompt = config.get('user_meta_prompt', self.config.user_meta_prompt)
                user_gen_kwargs = config.get('user_generation_kwargs', self.config.user_generation_kwargs)
            else:
                user_meta_prompt = self.config.user_meta_prompt
                user_gen_kwargs = self.config.user_generation_kwargs

            user_sim = UserSimulator(
                        user_meta_prompt=user_meta_prompt,
                        **user_gen_kwargs
                    )

            if prompts[i]: 
                chat_history.append({"role": "user", "content": prompts[i]})
            else:
                chat_history = []

            state = {
                'id': i,
                'prompt': prompts[i],
                'chat_history': chat_history,
                'user_sim': user_sim,
                'completed': False,
                'turn_count': 0
            }
            conversation_states.append(state)

        if not prompts or not prompts[0]: # no initial user prompts, generate user prompts first
            active_states = [s for s in conversation_states if not s['completed']]
            await self._process_user_turn_batch(active_states)


        max_rounds = self.config.max_total_turns // 2
        
        for round_idx in range(max_rounds):
            # Filter active conversations
            active_states = [s for s in conversation_states if not s['completed']]
            
            if not active_states:
                print(f"✅ All conversations completed by round {round_idx}")
                break
            
            print(f"🔄 Round {round_idx + 1}: Processing {len(active_states)} active conversations")

            await self._process_assistant_turn_batch(active_states)
            
            # Check for terminations and generate user responses
            await self._process_user_turn_batch(active_states)

            
            # Update turn counts
            for state in active_states:
                state['turn_count'] += 1
        
        # Extract final conversations
        results = []
        for state in conversation_states:
            results.append(state['chat_history'])
            
            # 🔴 NEW: Generate statistics about conversation lengths
            lengths = [len(state['chat_history']) for state in conversation_states]
            avg_length = sum(lengths) / len(lengths) if lengths else 0
            min_length = min(lengths) if lengths else 0
            max_length = max(lengths) if lengths else 0
        
        successful = sum(1 for state in conversation_states if len(state['chat_history']) > 2)
        print(f"✅ Batch generation complete: {successful}/{len(prompts)} successful conversations")
        print(f"📊 Conversation lengths - Min: {min_length}, Max: {max_length}, Avg: {avg_length:.1f}")
        
        # 🔴 NEW: Show early termination summary
        early_terminations = sum(1 for state in conversation_states if state['completed'] and state['turn_count'] < max_rounds - 1)
        if early_terminations > 0:
            print(f"⏹️  {early_terminations} conversations terminated early")
        
        successful = sum(1 for state in conversation_states if len(state['chat_history']) > 2)
        print(f"✅ Batch generation complete: {successful}/{len(prompts)} successful conversations")
        
        return results
    
    async def _process_assistant_turn_batch(self, states: List[Dict]):
        """Process assistant responses for a batch of conversations"""
        # Prepare batch input
        active_states = [s for s in states if not s['completed']]
        
        if not active_states:
            return
            
        message_lists = [state['chat_history'] for state in states]
        
        # Generate responses in batch using thread pool
        loop = asyncio.get_event_loop()
        responses = await loop.run_in_executor(
            self.executor,
            self.assistant_simulator.generate_responses_batch,
            message_lists
        )
        
        # Update conversations with responses
        for state, response in zip(states, responses):
            state['chat_history'].append({"role": "assistant", "content": response})
            
            # Check termination
            if self._should_terminate_conversation(response):
                state['completed'] = True
                print(f"  🛑 Conversation {state['id']} terminated by assistant at turn {state['turn_count'] + 1}")

    
    async def _process_user_turn_batch(self, states: List[Dict]):
        """Process user responses for active conversations"""
        # Filter out completed conversations
        active_states = [s for s in states if not s['completed']]
        
        if not active_states:
            return
        
        # Generate user responses concurrently
        tasks = []
        for state in active_states:
            task = self._generate_user_response_async(state)
            tasks.append(task)
        
        if hasattr(self.config, 'bedrock_rate_limit_delay'):
            for i, task in enumerate(tasks):
                if i > 0:
                    await asyncio.sleep(self.config.bedrock_rate_limit_delay)
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Update conversations
        for state, response in zip(active_states, responses):
            if isinstance(response, Exception):
                logger.error(f"Error generating user response for conversation {state['id']}: {response}")
                response = "I see. Please continue."
            
            state['chat_history'].append({"role": "user", "content": response})
            
            # Check termination
            if self._should_terminate_conversation(response):
                state['completed'] = True
                print(f"  🛑 Conversation {state['id']} terminated by user")
    
    
    async def _generate_user_response_async(self, state: Dict) -> str:
        """Generate user response asynchronously"""
        user_sim = state['user_sim']
        chat_history = state['chat_history'].copy()  # Make a copy to avoid mutations
        
        # If UserSimulator has async_call method, use it
        if hasattr(user_sim, 'async_call'):
            return await user_sim.async_call(chat_history)
        else:
            # Otherwise, run the sync call in a thread
            loop = asyncio.get_event_loop()
            
            # 🔴 CRITICAL: Don't use lambda or partial with instance methods
            # Create a standalone function
            def make_call():
                return user_sim(chat_history)
            
            return await loop.run_in_executor(self.executor, make_call)

    async def generate_single_conversation(
        self,
        init_user_prompt: str,
        **kwargs
    ) -> List[Dict[str, str]]:
        """Generate a single multi-turn conversation"""
        
        async with self.semaphore:
            try:
                # Initialize user simulator for this conversation
                user_sim = UserSimulator(  
                    user_meta_prompt=self.config.user_meta_prompt,
                    **self.config.user_generation_kwargs
                )

                if init_user_prompt:
                    print(f"🔄 Starting conversation with: '{init_user_prompt[:50]}...'")
                    
                    # Start with initial system message and user prompt
                    chat_history = [
                        {"role": "user", "content": init_user_prompt}
                    ]
                else:
                    chat_history = []
                    user_response = await self._generate_user_response(user_sim, chat_history)
                    chat_history.append({"role": "user", "content": user_response})
                
                
                # Generate conversation turns
                for turn_idx in range(self.config.max_total_turns // 2):
                    print(f"  Turn {turn_idx + 1}: Generating assistant response...")
                    
                    # Generate assistant response
                    assistant_response = await self._generate_assistant_response(chat_history)
                    chat_history.append({"role": "assistant", "content": assistant_response})
                    
                    # Check if conversation should end
                    if self._should_terminate_conversation(assistant_response):
                        print(f"  🛑 Assistant terminated conversation")
                        break
                    
                    print(f"  Turn {turn_idx + 1}: Generating user response...")
                    
                    # Generate user response
                    user_response = await self._generate_user_response(user_sim, chat_history)
                    chat_history.append({"role": "user", "content": user_response})
                    
                    # Check if conversation should end
                    if self._should_terminate_conversation(user_response):
                        print(f"  🛑 User terminated conversation")
                        break
                
                print(f"✅ Conversation completed with {len(chat_history)} messages")
                return chat_history
                
            except Exception as e:
                logger.error(f"❌ Error generating conversation: {e}")
                return None
    
    async def _generate_assistant_response(self, chat_history: List[Dict[str, str]]) -> str:
        """Generate assistant response"""
        try:
            # Run assistant simulator in thread pool
            loop = asyncio.get_event_loop()
            
            # Create a callable function for the executor
            def generate():
                return self.assistant_simulator.generate_response(chat_history)
            
            response = await loop.run_in_executor(
                self.executor,
                generate
            )
            return response
            
        except Exception as e:
            logger.error(f"Error generating assistant response: {e}")
            return "I understand. How can I help you further?"
    
    async def _generate_user_response(self, user_sim: UserSimulator, chat_history: List[Dict[str, str]]) -> str:
        """Generate user response"""
        try:
            # Run user simulator in thread pool
            loop = asyncio.get_event_loop()
            
            # Create a callable function for the executor
            def generate():
                return user_sim(chat_history)
            
            user_response = await loop.run_in_executor(
                self.executor,
                generate
            )
            return user_response
            
        except Exception as e:
            logger.error(f"Error generating user response: {e}")
            return "I see. Please continue."
    
    def _should_terminate_conversation(self, message: str) -> bool:
        """Check if conversation should be terminated"""
        termination_phrases = [
            "thank you for your help", "that's all I need", "i understand now",
            "perfect!", "goodbye", "bye"
        ]
        
        message_lower = message.lower()
    
        # Check for explicit termination signal (adjust based on your TERMINATION_SIGNAL)
        if "[DONE]" in message or "TERMINATE" in message:
            return True
            
        return any(phrase in message_lower for phrase in termination_phrases)