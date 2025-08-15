"""
Interactivity Calculator Module

This module provides functionality to evaluate the interactivity of conversations
using an LLM evaluator. 
"""

import json
import re
import asyncio
import ast
from typing import List, Dict, Tuple, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import your existing conversation generator
import os
import sys

# Automatically add the root project directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.bedrock_call import bedrock_call

# Interactivity evaluation prompt
INTERACTIVITY_PROMPT = '''You are a helpful and meticulous conversation evaluator. \
Your task is to evaluate the *interactivity* of the responses provided by an AI assistant \
to user questions in a given conversation:
<|The Start of the Conversation to be Evaluated|>
{chat_history}
<|The End of the Conversation to be Evaluated|>
You should assess the assistant's engagement, clarity, and ability to understand the user's needs. Evaluate the style of the conversation not the length. Longer conversation does not mean more interactivity. \
Give a float number between 0 and 1, where:
    1 = Highly interactive: The assistant is very engaging, asks all relevant questions, and significantly enhances understanding of user preference.
     - Example: The assistant thoroughly understands the user's question, asks for necessary clarifications, such as ""What aspects of Forrest Gump do you enjoy? Is it the storytelling, the character development, or the historical context?"
    0.5 = Moderately interactive: The assistant is engaging, asks some relevant questions, but can be substantially improved.
     - Example: The assistant asks some relevant questions about the user's inquiry but misses key details, and does not probe further for clarification, such as "How about The Pursuit of Happyness (2006)? It's A touching story of determination, courage, and love being more important than ability, just like Forrest Gump."
    0 = Low interactivity: The assistant shows low engagement, asks few relevant questions, and barely try to understand the user's needs.
     - Example: The assistant provides a vague or incomplete response without fully understanding the user's intent, such as "You should watch The Pursuit of Happyness (2006)" without asking any follow-up questions or providing detailed information.
Output format (JSON):
{{
    "thought": "<How interactive is the assistant?>",
    "interactivity": <score>
}}
Double check if the JSON object is formatted correctly. Ensure that all fields are present and properly structured. Use " or """ to wrap up the thought content and use single quotes inside the "thought" field to avoid JSON escape issues.
Your evaluation:
'''

def parse_and_format_conversation(conversation_source: Union[List[Dict[str, str]], str], 
                                max_turns: int = 10) -> str:
    """
    Parse conversation from any source format and format it for evaluation
    
    Args:
        conversation_source: Can be:
            - List of message dictionaries (already parsed)
            - JSON string representation of conversation
            - String that needs ast.literal_eval parsing
        max_turns: Maximum number of turns to include in evaluation
            
    Returns:
        Formatted conversation string ready for evaluation
    """
    try:
        # Step 1: Parse conversation from any source format
        conversation = []
        
        # If already a list, use as-is
        if isinstance(conversation_source, list):
            conversation = conversation_source
        
        # If string, try to parse it
        elif isinstance(conversation_source, str):
            # First try JSON parsing
            try:
                parsed = json.loads(conversation_source)
                if isinstance(parsed, list):
                    conversation = parsed
            except json.JSONDecodeError:
                pass
            
            # Then try ast.literal_eval (for Python literal strings)
            if not conversation:
                try:
                    parsed = ast.literal_eval(conversation_source)
                    if isinstance(parsed, list):
                        conversation = parsed
                except (ValueError, SyntaxError):
                    pass
        
        if not conversation:
            print(f"Could not parse conversation from: {type(conversation_source)}")
            return ""
        
        # Step 2: Apply length limiting using the unified approach
        if len(conversation) > max_turns:
            # Skip first few turns if conversation is too long (keeping most recent interactions)
            start_idx = max(0, len(conversation) - max_turns)
            conversation = conversation[start_idx:]
        
        # Step 3: Format conversation for evaluation
        chat_history = ""
        for turn in conversation:
            role = turn.get('role', '').capitalize()
            content = turn.get('content', '')
            # Clean up quotation marks
            content = content.replace('QUOTATION_MARK', '"')
            chat_history += f"{role}: {content}\n\n"
        
        return chat_history.strip()
    except Exception as e:
        print(f"Error parsing conversation: {e}")
        return ""

def _extract_with_regex(response: str) -> tuple[Optional[float], Optional[str]]:
    """Extract score and reasoning using regex patterns"""
    import re
    
    # Look for score
    score_patterns = [
        r'"interactivity":\s*([0-9]*\.?[0-9]+)',
        r'interactivity["\s]*:?\s*([0-9]*\.?[0-9]+)',
        r'score["\s]*:?\s*([0-9]*\.?[0-9]+)',
    ]
    
    score = None
    for pattern in score_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                if 0 <= score <= 1:
                    break
            except ValueError:
                continue
    
    # Look for reasoning/thought
    reasoning_patterns = [
        r'"thought":\s*"([^"]*)',
        r'thought["\s]*:?\s*["\']([^"\']*)',
        r'reasoning["\s]*:?\s*["\']([^"\']*)',
        r'explanation["\s]*:?\s*["\']([^"\']*)',
    ]
    
    reasoning = ""
    for pattern in reasoning_patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            reasoning = match.group(1).strip()
            break
    
    if not reasoning:
        # Use first 200 chars as reasoning if we can't find structured reasoning
        reasoning = response[:200].strip()
    
    return score, reasoning

def _try_parse_json_candidate(json_str: str) -> tuple[Optional[float], Optional[str]]:
    """Try to parse a JSON candidate string"""
    try:
        # Clean up the JSON string
        json_str = json_str.strip()
        
        # Remove common problematic characters
        json_str = json_str.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        
        # Fix common quote issues - more aggressive approach
        # Replace smart quotes with regular quotes
        json_str = json_str.replace('"', '"').replace('"', '"')
        json_str = json_str.replace(''', "'").replace(''', "'")
        
        # Try to fix unescaped quotes in the thought field
        # This is a more robust approach that handles nested quotes
        json_str = _fix_thought_field_quotes(json_str)
        
        # Try parsing
        result = json.loads(json_str)
        score = float(result.get('interactivity', 0))
        reasoning = result.get('thought', '')
        
        # Ensure score is in valid range
        score = max(0.0, min(1.0, score))
        return score, reasoning
        
    except json.JSONDecodeError as e:
        # If JSON parsing fails, try a more aggressive fix
        try:
            fixed_json = _aggressive_json_fix(json_str)
            result = json.loads(fixed_json)
            score = float(result.get('interactivity', 0))
            reasoning = result.get('thought', '')
            score = max(0.0, min(1.0, score))
            return score, reasoning
        except:
            pass
    except Exception:
        pass
    
    return None, None

def extract_interactivity_score_and_reasoning(response: str) -> tuple[Optional[float], Optional[str]]:
    """
    Extract both interactivity score and reasoning from LLM response with robust parsing
    
    Args:
        response: Raw response from the evaluator LLM
        
    Returns:
        Tuple of (score, reasoning) or (None, None) if extraction failed
    """
    try:
        # Clean the response first
        response = response.strip()
        
        # Strategy 1: Try multiple JSON extraction approaches
        json_candidates = []
        
        # Find all potential JSON blocks
        import re
        
        # Look for JSON-like patterns
        json_patterns = [
            r'\{[^{}]*"interactivity"[^{}]*\}',  # Simple single-level JSON
            r'\{.*?"interactivity".*?\}',         # More flexible JSON
            r'\{[\s\S]*?"interactivity"[\s\S]*?\}' # Multi-line JSON
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            json_candidates.extend(matches)
        
        # Also try the traditional approach
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        if start_idx != -1 and end_idx != -1:
            json_candidates.append(response[start_idx:end_idx])
        
        # Try to parse each candidate
        for json_str in json_candidates:
            score, reasoning = _try_parse_json_candidate(json_str)
            if score is not None:
                return score, reasoning
        
        # Strategy 2: Regex-based extraction as fallback
        score, reasoning = _extract_with_regex(response)
        if score is not None:
            return score, reasoning
        
        # Strategy 3: Last resort - look for any number between 0 and 1
        numbers = re.findall(r'\b(0\.[0-9]+|1\.0+|0|1)\b', response)
        for num_str in numbers:
            try:
                score = float(num_str)
                if 0 <= score <= 1:
                    reasoning = f"Extracted from: {response[:200]}..."
                    print(f"Using fallback score extraction: {score}")
                    return score, reasoning
            except ValueError:
                continue
        
        print(f"Could not extract score from response: {response[:300]}...")
        return None, None
        
    except Exception as e:
        print(f"Error extracting score and reasoning: {e}")
        print(f"Response preview: {response[:200]}...")
        return None, None

def evaluate_interactivity_single_with_reasoning(conversation_source: Union[List[Dict[str, str]], str], 
                                model_id: str = "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                                max_turns: int = 10) -> Optional[float]:
    """
    Evaluate interactivity for a single conversation from any source format
    
    Args:
        conversation_source: Can be List of message dicts or string representation
        model_id: Model ID for the evaluator
        max_turns: Maximum number of turns to include in evaluation
        
    Returns:
        Interactivity score (0-1) or None if evaluation failed
    """
    try:
        # Parse and format conversation in one step
        chat_history = parse_and_format_conversation(conversation_source, max_turns)
        if not chat_history:
            return None, None
        
        # Create the evaluation prompt
        prompt = INTERACTIVITY_PROMPT.format(chat_history=chat_history)
        
        # Prepare messages for the model
        messages = [{"role": "user", "content": prompt}]
        
        # Call the model
        if bedrock_call is None:
            print("Warning: bedrock_call not available, returning placeholder score")
            return 0.5, "Placeholder response - bedrock_call not available"  # Placeholder score
            
        response = bedrock_call(
            model=model_id,
            messages=messages,
            max_tokens=1000,
            temperature=0.1
        )
        
        if response is None:
            return None, None
        
        # Extract and return score
        return extract_interactivity_score_and_reasoning(response)
        
    except Exception as e:
        print(f"Error evaluating conversation interactivity: {e}")
        return None, None

def evaluate_interactivity_single(conversation_source: Union[List[Dict[str, str]], str], 
                                model_id: str = "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                                max_turns: int = 10) -> Optional[float]:
    """
    Backwards compatible function that only returns the score
    """
    score, _ = evaluate_interactivity_single_with_reasoning(conversation_source, model_id, max_turns)
    return score

def evaluate_interactivity_batch_with_reasoning(conversation_sources: List[Union[List[Dict[str, str]], str]], 
                                model_id: str = "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                                max_workers: int = 10,
                                max_turns: int = 10) -> List[Optional[float]]:
    """
    Evaluate interactivity for a batch of conversations in parallel
    
    Args:
        conversation_sources: List of conversations in any supported format
        model_id: Model ID for the evaluator
        max_workers: Number of parallel workers
        max_turns: Maximum number of turns to include in evaluation
        
    Returns:
        List of interactivity scores (same order as input)
    """
    scores = [None] * len(conversation_sources)
    reasonings = [None] * len(conversation_sources)
    
    def evaluate_single_wrapper(idx_conv_pair):
        idx, conversation_source = idx_conv_pair
        score, reasoning = evaluate_interactivity_single_with_reasoning(conversation_source, model_id, max_turns)
        return idx, score, reasoning
    
    # Process conversations in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(evaluate_single_wrapper, (idx, conv)): idx 
            for idx, conv in enumerate(conversation_sources)
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_index):
            idx, score, reasoning = future.result()
            scores[idx] = score
            reasonings[idx] = reasoning
            completed += 1
            
            if completed % 10 == 0:
                print(f"Completed {completed}/{len(conversation_sources)} interactivity evaluations")

    return scores, reasonings

def evaluate_interactivity_batch(conversation_sources: List[Union[List[Dict[str, str]], str]], 
                                model_id: str = "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                                max_workers: int = 10,
                                max_turns: int = 10) -> List[Optional[float]]:
    """
    Backwards compatible function that only returns scores
    """
    scores, _ = evaluate_interactivity_batch_with_reasoning(conversation_sources, model_id, max_workers, max_turns)
    return scores

async def evaluate_interactivity_batch_async_with_reasoning(conversation_sources: List[Union[List[Dict[str, str]], str]], 
                                                          model_id: str = "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                                                          max_workers: int = 10,
                                                          max_turns: int = 10) -> tuple[List[Optional[float]], List[Optional[str]]]:
    """
    Async version of batch interactivity evaluation with reasoning
    
    Returns:
        Tuple of (scores_list, reasoning_list)
    """
    loop = asyncio.get_event_loop()
    
    # Run the synchronous batch evaluation in a thread pool
    scores, reasonings = await loop.run_in_executor(
        None, 
        evaluate_interactivity_batch_with_reasoning, 
        conversation_sources, 
        model_id, 
        max_workers,
        max_turns
    )
    
    return scores, reasonings

async def evaluate_interactivity_batch_async(conversation_sources: List[Union[List[Dict[str, str]], str]], 
                                           model_id: str = "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                                           max_workers: int = 10,
                                           max_turns: int = 10) -> List[Optional[float]]:
    """
    Backwards compatible async function that only returns scores
    """
    scores, _ = await evaluate_interactivity_batch_async_with_reasoning(conversation_sources, model_id, max_workers, max_turns)
    return scores