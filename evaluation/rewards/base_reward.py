"""
Base reward calculator interface for completion-based PPO training.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any
import torch


class BaseRewardCalculator(ABC):
    """Abstract base class for completion reward calculators."""
    
    def __init__(self, **kwargs):
        """Initialize the reward calculator with configuration parameters."""
        pass
    
    @abstractmethod
    def calculate_single_completion_reward(
        self, 
        conversation_history: List[Dict], 
        completion: str,
        ground_truth: str = "",
        **kwargs
    ) -> float:
        """
        Calculate reward for a single completion given conversation history.
        
        Args:
            conversation_history: Previous conversation turns with 'role' and 'content' keys
            completion: The assistant's completion to evaluate
            ground_truth: Optional ground truth for evaluation
            **kwargs: Additional parameters specific to reward type
            
        Returns:
            Reward value for this completion
        """
        pass
    
    def calculate_batch_completion_rewards(
        self,
        conversation_histories: List[List[Dict]],
        completions: List[str],
        ground_truths: List[str] = None,
        **kwargs
    ) -> List[float]:
        """
        Calculate rewards for a batch of completions. Default implementation calls
        calculate_single_completion_reward for each item.
        
        Args:
            conversation_histories: List of conversation histories
            completions: List of completions to evaluate
            ground_truths: Optional list of ground truths
            **kwargs: Additional parameters
            
        Returns:
            List of reward values
        """
        if ground_truths is None:
            ground_truths = [""] * len(completions)
            
        rewards = []
        for i, (history, completion, gt) in enumerate(zip(conversation_histories, completions, ground_truths)):
            try:
                reward = self.calculate_single_completion_reward(
                    history, completion, gt, **kwargs
                )
                rewards.append(reward)
            except Exception as e:
                print(f"Error calculating reward for item {i}: {e}")
                rewards.append(0.0)
        
        return rewards
    
    def cleanup(self):
        """Clean up any resources (connections, thread pools, etc.)."""
        pass


class SimpleRewardCalculator(BaseRewardCalculator):
    """Simple length-based reward calculator for testing."""
    
    def __init__(self, max_length: int = 50, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
    
    def calculate_single_completion_reward(
        self, 
        conversation_history: List[Dict], 
        completion: str,
        ground_truth: str = "",
        **kwargs
    ) -> float:
        """Calculate simple length-based reward."""
        word_count = len(completion.split())
        reward = min(word_count / self.max_length, 1.0)
        
        if ground_truth:
            gt_terms = set(ground_truth.lower().split())
            completion_terms = set(completion.lower().split())
            overlap = len(gt_terms.intersection(completion_terms))
            if overlap > 0:
                reward += 0.1 * (overlap / len(gt_terms))
        
        return reward