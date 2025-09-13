"""
CollabLLM-based reward calculator for multi-turn conversation evaluation.
"""

"""
CollabLLM-based reward calculator for multi-turn conversation evaluation.
Uses the same metrics as the TurnWiseDPOExecutor: hit_rate, interactivity, and token_efficiency.
"""

from typing import List, Dict, Tuple, Optional, Any, Set
import asyncio
import tiktoken
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from .base_reward import BaseRewardCalculator

# Try to import evaluation dependencies
try:
    from evaluation.metrics.hit_checker import calculate_hit_rate_batch
    from evaluation.metrics.interactivity import evaluate_interactivity_batch_async_with_reasoning
    HAS_EVALUATION_METRICS = True
except ImportError:
    HAS_EVALUATION_METRICS = False

logger = logging.getLogger(__name__)

class CollabLLMRewardCalculator(BaseRewardCalculator):
    """
    CollabLLM-based reward calculator using hit_rate, interactivity, and token_efficiency metrics.
    Matches the implementation from TurnWiseDPOExecutor.
    """
    
    def __init__(
        self,
        weights: List[float] = None,
        interactivity_model: str = "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        max_workers: int = 3,
        max_eval_turns: int = 11,
        max_tokens: int = 512,
        encoding_name: str = 'cl100k_base',
        **kwargs
    ):
        super().__init__(**kwargs)
        
        if not HAS_EVALUATION_METRICS:
            raise ImportError(
                "Evaluation metrics dependencies not available. "
                "Please ensure evaluation.metrics.hit_checker and "
                "evaluation.metrics.interactivity are available."
            )
        
        # Configuration matching TurnWiseDPOExecutor
        # Ensure we have exactly 3 weights: [hit_rate, interactivity, token_efficiency]
        self.weights = weights or [1.0, 1.0, -0.1]
        if len(self.weights) != 3:
            logger.warning(f"CollabLLM expects 3 weights [hit_rate, interactivity, token_efficiency], got {len(self.weights)}. Using default.")
            self.weights = [1.0, 1.0, -0.1]
        
        self.interactivity_model = interactivity_model
        self.max_workers = max_workers
        self.max_eval_turns = max_eval_turns
        self.max_tokens = max_tokens
        self.encoding_name = encoding_name
        
        logger.info(f"CollabLLM reward calculator initialized:")
        logger.info(f"  Weights: {self.weights} [hit_rate, interactivity, token_efficiency]")
        logger.info(f"  Model: {self.interactivity_model}")
        logger.info(f"  Workers: {self.max_workers}, Max turns: {self.max_eval_turns}")
    
    def calculate_single_completion_reward(
        self, 
        conversation_history: List[Dict], 
        completion: str,
        ground_truth: str = "",
        **kwargs
    ) -> float:
        """Calculate reward for a single completion using CollabLLM metrics."""
        full_conversation = conversation_history + [{"role": "assistant", "content": completion}]
        
        try:
            if asyncio.get_event_loop().is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, 
                        self._evaluate_single_conversation_async(full_conversation, ground_truth)
                    )
                    score = future.result(timeout=60)
            else:
                score = asyncio.run(self._evaluate_conversation_async(full_conversation, ground_truth))
        except Exception as e:
            raise RuntimeError(f"Failed to evaluate conversation: {e}")
        
        return score
    
    async def _evaluate_single_conversation_async(self, conversation: List[Dict], ground_truth: str) -> float:
        """Async evaluation of a single conversation using the three metrics."""
        scores = await self.evaluate_metrics_with_reasoning(
            [conversation], 
            [{'ground_truth': ground_truth}], 
            {'hit_rate', 'interactivity', 'token_efficiency'}
        )
        
        score_dict = {
            'hit_rate': scores.get('hit_rate', [0.0])[0],
            'interactivity': scores.get('interactivity', [0.0])[0],
            'token_efficiency': scores.get('token_efficiency', [0.0])[0]
        }
        
        return self.calculate_combined_score(score_dict)
    
    async def evaluate_metrics_with_reasoning(self, conversations: List[List[Dict]], 
                              contexts: List[Dict], 
                              metrics: Set[str]) -> Dict[str, List[float]]:
        """Evaluate specified metrics for conversations."""
        results = {}
        
        if 'hit_rate' in metrics:
            conversation_data = [{
                'ground_truth': ctx['ground_truth'],
                'generated_conversation': conv
            } for conv, ctx in zip(conversations, contexts)]
            
            hit_results = calculate_hit_rate_batch(conversation_data)
            results['hit_rate'] = [score for score, _ in hit_results]
        
        if 'interactivity' in metrics:
            scores, reasonings = await evaluate_interactivity_batch_async_with_reasoning(
                conversations, 
                model_id=self.interactivity_model,
                max_workers=self.max_workers,
                max_turns=self.max_eval_turns
            )
            
            results['interactivity'] = scores
            results['interactivity_reasoning'] = reasonings

        if 'token_efficiency' in metrics:
            results['token_efficiency'] = self.evaluate_token_efficiency_parallel(conversations)
        
        return results

    def evaluate_token_efficiency_parallel(self, conversations: List[List[Dict]]) -> List[float]:
        """Evaluate token efficiency in parallel."""
        def calc_efficiency(conv):
            return self.calculate_token_efficiency(conv, self.max_tokens, self.encoding_name)
        
        scores = [None] * len(conversations)
        failed_indices = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {executor.submit(calc_efficiency, conv): idx 
                           for idx, conv in enumerate(conversations)}
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    scores[idx] = future.result()
                except Exception as e:
                    failed_indices.append(idx)
                    logger.error(f"Token efficiency error for conv {idx}: {e}")
        
        if failed_indices:
            raise RuntimeError(f"Token efficiency calculation failed for conversations at indices: {failed_indices}")
        
        return scores

    def calculate_token_efficiency(self, conversation: List[Dict[str, str]], 
                             max_tokens: int = 512,
                             encoding_name: str = "cl100k_base") -> float:
        """Calculate token efficiency score for the last assistant message."""
        encoding = tiktoken.get_encoding(encoding_name)
        
        last_assistant_message = None
        for message in reversed(conversation):
            if message.get('role') == 'assistant':
                last_assistant_message = message
                break
        
        if not last_assistant_message:
            raise ValueError("No assistant message found in conversation")
        
        content = last_assistant_message.get('content', '')
        tokens = encoding.encode(content)
        total_tokens = len(tokens)
        
        return total_tokens / max_tokens

    def calculate_combined_score(self, scores: Dict[str, float]) -> float:
        """Calculate combined score from metrics using configured weights."""
        required_metrics = ['hit_rate', 'interactivity', 'token_efficiency']
        present_metrics = [m for m in required_metrics if m in scores]
        
        if len(present_metrics) == 3:
            combined_score = (
                self.weights[0] * scores['hit_rate'] + 
                self.weights[1] * scores['interactivity'] + 
                self.weights[2] * scores['token_efficiency']
            )
            logger.debug(f"Combined score: {self.weights[0]}*{scores['hit_rate']:.3f} + {self.weights[1]}*{scores['interactivity']:.3f} + {self.weights[2]}*{scores['token_efficiency']:.3f} = {combined_score:.3f}")
            return combined_score
        
        total_score = 0.0
        total_weight = 0.0
        
        for i, metric in enumerate(required_metrics):
            if metric in scores:
                total_score += self.weights[i] * scores[metric]
                total_weight += abs(self.weights[i])
                logger.debug(f"Adding {metric}: {self.weights[i]} * {scores[metric]:.3f}")
        
        final_score = total_score / total_weight
        logger.debug(f"Partial score with {len(present_metrics)} metrics: {final_score:.3f}")
        return final_score
    
    def calculate_batch_completion_rewards(
        self,
        original_conversations: List[str],
        conversation_histories: List[str],
        completions: List[str],
        ground_truths: List[str] = None,
        **kwargs
    ) -> List[float]:
        """Optimized batch calculation for CollabLLM rewards."""
        if ground_truths is None:
            ground_truths = [""] * len(completions)
        
        full_conversations = []
        contexts = []
        
        for history, completion, gt in zip(conversation_histories, completions, ground_truths):
            if not isinstance(history, list):
                history = json.loads(history)
            full_conversation = history + [{"role": "assistant", "content": completion}]
            full_conversations.append(full_conversation)
            contexts.append({'ground_truth': gt})
        
        try:
            if asyncio.get_event_loop().is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, 
                        self._evaluate_conversations_batch_async(full_conversations, contexts)
                    )
                    scores = future.result(timeout=180)
            else:
                scores = asyncio.run(self._evaluate_conversations_batch_async(full_conversations, contexts))
            
            return scores
            
        except Exception as e:
            raise RuntimeError(f"Batch evaluation failed: {e}")
    
    async def _evaluate_conversations_batch_async(self, conversations: List[List[Dict]], contexts: List[Dict]) -> List[float]:
        """Async batch evaluation of conversations using the three metrics."""
        scores = await self.evaluate_metrics_with_reasoning(
            conversations, 
            contexts, 
            {'hit_rate', 'interactivity', 'token_efficiency'}
        )
        
        required_metrics = ['hit_rate', 'interactivity', 'token_efficiency']
        for metric in required_metrics:
            if metric not in scores:
                raise RuntimeError(f"Missing required metric: {metric}")
        
        combined_scores = []
        
        for i in range(len(conversations)):
            score_dict = {
                'hit_rate': scores['hit_rate'][i],
                'interactivity': scores['interactivity'][i],
                'token_efficiency': scores['token_efficiency'][i]
            }
            
            combined_score = self.calculate_combined_score(score_dict)
            combined_scores.append(combined_score)
        
        return combined_scores

