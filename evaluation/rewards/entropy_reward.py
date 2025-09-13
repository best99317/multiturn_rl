"""
Entropy-based reward calculator for conversation diversity measurement.
Modified to generate follow-up user responses and evaluate entropy reduction.
Raises exceptions instead of using fallback implementations.
"""

import math
import re
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import json

from .base_reward import BaseRewardCalculator

# Import required dependencies - raise exception if not available
try:
    from utils.bedrock_call import bedrock_call
except ImportError:
    raise ImportError(
        "bedrock_call is required for ConversationEntropyReward. "
        "Please ensure utils.bedrock_call is available or install the required dependencies."
    )

import logging
logger = logging.getLogger(__name__)


class ConversationEntropyReward(BaseRewardCalculator):
    """
    Entropy-based reward calculator that measures conversation diversity
    by generating follow-up user responses and calculating entropy reduction.
    """
    
    def __init__(
        self, 
        model_id: str = 'us.meta.llama3-1-8b-instruct-v1:0',
        user_model_id: str = 'us.anthropic.claude-3-haiku-20240307-v1:0',
        num_samples: int = 5,
        num_items: int = 5,
        max_workers: int = 10,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.user_model_id = user_model_id
        self.num_samples = num_samples
        self.num_items = num_items
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, 
            thread_name_prefix="entropy_reward"
        )
        
    def calculate_single_completion_reward(
        self, 
        original_conversation: str,
        conversation_history: List[Dict], 
        completion: str,
        ground_truth: str = "",
        **kwargs
    ) -> float:
        """
        Calculate entropy reduction reward for a single completion.
        
        Process:
        1. Calculate entropy before completion (from conversation history)
        2. Generate follow-up user response based on history + completion
        3. Calculate entropy after user response
        4. Return entropy reduction as reward
        """
        # Step 1: Calculate entropy before assistant completion (from history only)
        history_formatted = self.format_conversation(conversation_history)
        entropy_before = self.sample_and_calculate_entropy(history_formatted)
        
        # Step 2: Create full conversation including the completion
        if not isinstance(conversation_history, list):
            conversation_history = json.loads(conversation_history)
        full_conversation = conversation_history + [{"role": "assistant", "content": completion}]
        
        # Step 3: Generate follow-up user response
        user_response = self.generate_user_response(original_conversation, full_conversation, ground_truth)
        
        if not user_response:
            raise RuntimeError("Failed to generate user response for entropy calculation")
        
        # Step 4: Calculate entropy after user response
        extended_conversation = full_conversation + [{"role": "user", "content": user_response}]
        extended_formatted = self.format_conversation(extended_conversation)
        entropy_after = self.sample_and_calculate_entropy(extended_formatted)
        
        # Step 5: Calculate entropy reduction as reward
        entropy_reduction = entropy_before - entropy_after
        
        logger.info(f"Entropy reward: before={entropy_before:.3f}, after={entropy_after:.3f}, reduction={entropy_reduction:.3f}")
        # logger.info(f"Generated user response: {user_response[:100]}...")
        
        return entropy_reduction
    
    def generate_user_response(self, original_conversation, chat_history: List[Dict], ground_truth: str = "") -> str:
        """Generate a follow-up user response based on the conversation."""
        chat_history = self.format_conversation(chat_history)
        original_conversation = self.format_conversation(original_conversation)
        
        prompt = f"""Below is a conversation between a user and a movie recommendation assistant. 

                    ### Conversation:
                    {original_conversation}

                    In this conversation, the user accepted the recommended movie
                    {ground_truth}
                    based on their preference.

                    Now pretend you are the user in this conversation. Ask a new assistant for movie recommendation as if the previous conversation didn't happen. 
                    Directly reply to the assistant's latest response or propose request based on the following chat history:\n
                    {chat_history}

                    ## Guidelines:
                    - Try to mimic the user, including their preference, based on the conversation without exposing the {ground_truth}.
                    - Try to include user preference, instead of asking general questions like "Do you have some movies to recommend?"
                    - With some probability, you can mention a movie you watched and liked or disliked as a reference.
                    - Provide short, vague or incomplete demands in the conversation to minimize your effort. Let the AI ask for clarification rather than providing everything upfront.
                    - Occasionally make spelling mistakes depending on the amount in chat history. No need to be polite.
                    - Terminate the conversation if the {ground_truth} is recommended and happily accepts the recommendation.
                    - Terminate the conversation if you obtained other satisfactory answers, or you think the assistant cannot help further.
                    - Use "[[TERMINATE CHAT]]" as your response when you terminate the conversation. 
                    - Stay in Character: Role-play as a human USER. You are NOT an AI. Maintain a consistent personality throughout the chat.
                    - Pretend the current year is 2023.

                    Now directly generate your response:"""
        
        messages = [{"role": "user", "content": prompt}]
        
        response = bedrock_call(
            model=self.user_model_id,
            max_tokens=128,
            temperature=0.8,
            messages=messages,
            num_retries=10
        )
        
        if not response or not response.strip():
            raise RuntimeError(f"Empty response from model {self.user_model_id}")
        
        return response.strip()
    
    def format_conversation(self, conversation: List[Dict]) -> str:
        """Format conversation history for LLM consumption."""
        formatted = ""
        if not isinstance(conversation, list):
            conversation = json.loads(conversation)
        for turn in conversation:
            role = turn['role'].capitalize()
            content = turn['content'].replace('QUOTATION_MARK', '"')
            formatted += f"{role}: {content}\n"
        
        return formatted.strip()
    
    def call_llm_for_recommendations(self, conversation_history: str) -> str:
        """Call LLM to generate recommendations based on conversation history."""
        prompt = f"""Given the following conversation history:

{conversation_history}

Generate a list of the top {self.num_items} movies the user would like to watch based on this conversation. Format your response as a numbered list with no extra sentences:

1. [First movie]
2. [Second movie]
3. [Third movie]
4. [Fourth movie]  
5. [Fifth movie]

Make sure each recommendation is unique and plausible given the conversation context."""
        
        messages = [{"role": "user", "content": prompt}]
        
        response = bedrock_call(
            model=self.model_id,
            max_tokens=128,
            temperature=0.7,
            messages=messages,
            num_retries=30
        )
        
        if not response:
            raise RuntimeError(f"Empty response from model {self.model_id}")
        
        return response
    
    def extract_responses(self, llm_output: str) -> List[str]:
        """Extract and normalize responses from LLM output."""
        if isinstance(llm_output, dict):
            llm_output = llm_output.get('generation', str(llm_output))

        lines = llm_output.strip().split('\n')
        responses = []
        
        for line in lines:
            match = re.match(r'^\d+\.\s*(.+)', line.strip())
            if match:
                responses.append(match.group(1).strip().lower())
        
        if len(responses) != self.num_items:
            parts = re.split(r'\d+\.', llm_output)
            responses = [part.lower().strip() for part in parts if part.lower().strip()]
        
        if len(responses) < self.num_items:
            raise ValueError(f"Expected {self.num_items} recommendations but only extracted {len(responses)} from: {llm_output[:200]}...")
            
        return responses[:self.num_items]
    
    def calculate_weighted_entropy(self, responses: List[List[str]]) -> float:
        """Calculate weighted entropy from multiple response lists."""
        if not responses or not all(responses):
            raise ValueError("responses list cannot be empty")
        
        top_k = len(responses[0])
        weights = [1.0 / math.log2(r + 2) for r in range(top_k)]
        
        weighted_counts = defaultdict(float)
        for rec_list in responses:
            for rank, item in enumerate(rec_list):
                if rank < len(weights):
                    weighted_counts[item] += weights[rank]
        
        total_weight = sum(weighted_counts.values())
        if total_weight == 0:
            raise ValueError("Total weight is zero, cannot calculate entropy")
            
        probabilities = [count / total_weight for count in weighted_counts.values()]
        entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
        return entropy

    def _single_sample(self, conversation_history: str) -> List[str]:
        """Generate a single sample of recommendations."""
        llm_output = self.call_llm_for_recommendations(conversation_history)
        return self.extract_responses(llm_output)
    
    def sample_and_calculate_entropy(self, conversation_history: str) -> float:
        """Sample multiple recommendation lists and calculate entropy."""
        with ThreadPoolExecutor(max_workers=min(self.num_samples, 5)) as executor:
            futures = [
                executor.submit(self._single_sample, conversation_history)
                for _ in range(self.num_samples)
            ]
            
            all_responses = []
            failed_samples = 0
            
            for future in futures:
                try:
                    response = future.result(timeout=60)
                    if response:
                        all_responses.append(response)
                    else:
                        failed_samples += 1
                except Exception as e:
                    failed_samples += 1
                    logger.error(f"Sample failed: {e}")
                    
        if not all_responses:
            raise RuntimeError(f"All {self.num_samples} sampling attempts failed")
        
        if failed_samples > 0:
            logger.warning(f"{failed_samples}/{self.num_samples} samples failed")
        
        return self.calculate_weighted_entropy(all_responses)
    
    def calculate_batch_completion_rewards(
        self,
        original_conversations: List[str],
        conversation_histories: List[List[Dict]],
        completions: List[str],
        ground_truths: List[str] = None,
        **kwargs
    ) -> List[float]:
        """Optimized batch calculation for entropy rewards."""
        if ground_truths is None:
            ground_truths = [""] * len(completions)
        
        def calculate_single_reward(args):
            original_conversation, history, completion, gt = args
            return self.calculate_single_completion_reward(original_conversation, history, completion, gt, **kwargs)
        
        with ThreadPoolExecutor(max_workers=min(len(completions), 4)) as executor:
            args_list = list(zip(original_conversations, conversation_histories, completions, ground_truths))
            rewards = list(executor.map(calculate_single_reward, args_list))
        
        return rewards
            
    def cleanup(self):
        """Clean up thread pool resources."""
        if hasattr(self, '_executor') and self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None