"""
Factory for creating different types of reward calculators.
"""

from typing import Dict, Any, Optional
import argparse

from .base_reward import BaseRewardCalculator, SimpleRewardCalculator
from .entropy_reward import ConversationEntropyReward
from .collabllm_reward import CollabLLMRewardCalculator


class RewardCalculatorFactory:
    """Factory class for creating appropriate reward calculators."""
    
    REWARD_TYPES = {
        "simple": SimpleRewardCalculator,
        "entropy": ConversationEntropyReward,
        "collabllm": CollabLLMRewardCalculator,
    }
    
    @classmethod
    def create_reward_calculator(
        cls, 
        reward_type: str, 
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BaseRewardCalculator:
        """
        Create a reward calculator of the specified type.
        
        Args:
            reward_type: Type of reward calculator ("simple", "entropy", "collabllm")
            config: Configuration dictionary for the reward calculator
            **kwargs: Additional keyword arguments
            
        Returns:
            Initialized reward calculator instance
            
        Raises:
            ValueError: If reward_type is not supported
        """
        if reward_type not in cls.REWARD_TYPES:
            available_types = ", ".join(cls.REWARD_TYPES.keys())
            raise ValueError(
                f"Unknown reward type '{reward_type}'. "
                f"Available types: {available_types}"
            )
        
        calculator_class = cls.REWARD_TYPES[reward_type]
        
        # Merge config with kwargs
        final_config = {}
        if config:
            final_config.update(config)
        final_config.update(kwargs)
        
        # Create type-specific configuration
        if reward_type == "entropy":
            final_config = cls._prepare_entropy_config(final_config)
        elif reward_type == "collabllm":
            final_config = cls._prepare_collabllm_config(final_config)
        elif reward_type == "simple":
            final_config = cls._prepare_simple_config(final_config)
        
        return calculator_class(**final_config)
    
    @classmethod
    def create_from_args(cls, args: argparse.Namespace) -> BaseRewardCalculator:
        """
        Create reward calculator from command line arguments.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Initialized reward calculator instance
        """
        reward_type = getattr(args, 'reward_type', 'simple')
        
        # Extract relevant arguments based on reward type
        config = {}
        
        if reward_type == "entropy":
            config = {
                'model_id': getattr(args, 'entropy_model', 'us.meta.llama3-1-8b-instruct-v1:0'),
                'user_model_id': getattr(args, 'entropy_user_model', 'us.anthropic.claude-3-haiku-20240307-v1:0'),
                'num_samples': getattr(args, 'entropy_samples', 5),
                'num_items': getattr(args, 'entropy_items', 5),
                'max_workers': getattr(args, 'max_metric_workers', 10),
            }
        elif reward_type == "collabllm":
            # Map metric_weights to CollabLLM's expected format [hit_rate, interactivity, token_efficiency]
            weights = getattr(args, 'metric_weights', [1.0, 1.0, -0.1])
            if len(weights) != 3:
                weights = [1.0, 1.0, -0.1]  # Default CollabLLM weights
                
            config = {
                'weights': weights,
                'interactivity_model': getattr(args, 'interactivity_model', 'us.anthropic.claude-3-7-sonnet-20250219-v1:0'),
                'max_workers': getattr(args, 'max_metric_workers', 3),
                'max_eval_turns': getattr(args, 'max_eval_turns', 11),
                'encoding_name': getattr(args, 'encoding_name', 'cl100k_base'),
            }
        elif reward_type == "simple":
            config = {
                'max_length': getattr(args, 'max_new_tokens', 256) // 4,  # Conservative estimate
            }
        
        return cls.create_reward_calculator(reward_type, config)
    
    @classmethod
    def _prepare_entropy_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare configuration for entropy reward calculator."""
        entropy_config = {
            'model_id': 'us.meta.llama3-1-8b-instruct-v1:0',
            'user_model_id': 'us.anthropic.claude-3-haiku-20240307-v1:0',
            'num_samples': 10,
            'num_items': 3,
            'max_workers': 10,
        }
        entropy_config.update(config)
        return entropy_config
    
    @classmethod
    def _prepare_collabllm_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare configuration for CollabLLM reward calculator."""
        collabllm_config = {
            'weights': [1.0, 1.0, -0.1],  # [hit_rate, interactivity, token_efficiency]
            'interactivity_model': 'us.anthropic.claude-3-7-sonnet-20250219-v1:0',
            'max_workers': 3,
            'max_eval_turns': 11,
            'max_tokens': 512,
            'encoding_name': 'cl100k_base',
        }
        collabllm_config.update(config)
        return collabllm_config
    
    @classmethod
    def _prepare_simple_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare configuration for simple reward calculator."""
        simple_config = {
            'max_length': 50,
        }
        simple_config.update(config)
        return simple_config
    
    @classmethod
    def get_available_types(cls) -> list:
        """Get list of available reward calculator types."""
        return list(cls.REWARD_TYPES.keys())
    
    @classmethod
    def get_type_info(cls, reward_type: str) -> Dict[str, Any]:
        """
        Get information about a specific reward calculator type.
        
        Args:
            reward_type: Type of reward calculator
            
        Returns:
            Dictionary with type information
        """
        if reward_type not in cls.REWARD_TYPES:
            return {}
        
        calculator_class = cls.REWARD_TYPES[reward_type]
        
        info = {
            'name': reward_type,
            'class': calculator_class.__name__,
            'description': calculator_class.__doc__ or "No description available",
            'available': True
        }
        
        # Add specific information for each type
        if reward_type == "collabllm":
            try:
                from evaluation.metrics.hit_checker import calculate_hit_rate_batch
                from evaluation.metrics.interactivity import evaluate_interactivity_batch_async_with_reasoning
                info['available'] = True
            except ImportError:
                info['available'] = False
                info['error'] = "Evaluation metrics dependencies not available"
        elif reward_type == "entropy":
            try:
                from utils.bedrock_call import bedrock_call
                info['available'] = True
            except ImportError:
                info['available'] = False
                info['error'] = "Bedrock dependencies not available"
        
        return info


# Convenience function for direct use
def create_reward_calculator(reward_type: str, **kwargs) -> BaseRewardCalculator:
    """
    Convenience function to create a reward calculator.
    
    Args:
        reward_type: Type of reward calculator
        **kwargs: Configuration parameters
        
    Returns:
        Initialized reward calculator instance
    """
    return RewardCalculatorFactory.create_reward_calculator(reward_type, **kwargs)