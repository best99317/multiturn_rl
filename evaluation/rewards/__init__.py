"""
Reward calculation module for completion-based PPO training.

This module provides various reward calculators for evaluating completion quality
in multi-turn dialogue systems. All implementations require proper dependencies
and will raise exceptions if dependencies are missing.
"""

from .base_reward import BaseRewardCalculator, SimpleRewardCalculator
from .entropy_reward import ConversationEntropyReward
from .collabllm_reward import CollabLLMRewardCalculator
from .reward_factory import RewardCalculatorFactory, create_reward_calculator

# Export main components
__all__ = [
    # Base classes
    'BaseRewardCalculator',
    'SimpleRewardCalculator',
    
    # Specific implementations
    'ConversationEntropyReward', 
    'CollabLLMRewardCalculator',
    
    # Factory
    'RewardCalculatorFactory',
    'create_reward_calculator',
]

# Version info
__version__ = '1.0.0'

def get_dependency_status():
    """Get status of optional dependencies."""
    status = {}
    
    # Check evaluation metrics
    try:
        from evaluation.metrics.hit_checker import calculate_hit_rate_batch
        from evaluation.metrics.interactivity import evaluate_interactivity_batch_async_with_reasoning
        status['evaluation_metrics'] = True
    except ImportError:
        status['evaluation_metrics'] = False
    
    # Check bedrock
    try:
        from utils.bedrock_call import bedrock_call
        status['bedrock'] = True
    except ImportError:
        status['bedrock'] = False
    
    return status

def print_dependency_status():
    """Print status of optional dependencies."""
    status = get_dependency_status()
    print("Reward Calculator Dependencies:")
    print(f"  Evaluation Metrics: {'✓' if status['evaluation_metrics'] else '✗'}")
    print(f"  Bedrock:            {'✓' if status['bedrock'] else '✗'}")
    
    if not status['evaluation_metrics']:
        print("  Warning: CollabLLM reward calculator will raise ImportError without evaluation metrics")
    if not status['bedrock']:
        print("  Warning: Entropy reward calculator will raise ImportError without bedrock")

def validate_dependencies():
    """Validate that all dependencies are available. Raises ImportError if any are missing."""
    status = get_dependency_status()
    missing = [dep for dep, available in status.items() if not available]
    
    if missing:
        raise ImportError(
            f"Missing required dependencies: {', '.join(missing)}. "
            f"Please install the required packages or ensure the modules are available."
        )