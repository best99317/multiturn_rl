"""
Hit Rate Calculator Module

This module provides functionality to calculate hit rates for movie recommendation conversations.
It checks whether ground truth movies are mentioned in generated conversations using exact matching.
"""

import json
import re
from typing import List, Dict, Tuple, Any


def parse_ground_truth_movies(ground_truth: str) -> List[str]:
    """
    Parse ground truth that may contain multiple movies separated by commas.
    Extract clean movie titles from entries like "Movie1 (year), Movie2 (year)"
    
    Args:
        ground_truth (str): Ground truth string containing movie titles
        
    Returns:
        List[str]: List of cleaned movie titles
    """
    movies = []
    # Split by comma and clean each title
    movie_parts = ground_truth.split(',')
    
    for part in movie_parts:
        part = part.strip()
        # Remove year in parentheses like "(1984)"
        clean_title = re.sub(r'\s*\(\d{4}\)\s*', '', part).strip()
        if clean_title:
            movies.append(clean_title)
    
    return movies


def check_movie_mention_exact(ground_truth: str, conversation_text: str) -> Tuple[bool, str]:
    """
    Check if any movie from ground truth is mentioned using exact substring matching only.
    
    Args:
        ground_truth (str): Ground truth movie titles
        conversation_text (str): Text from the conversation to search in
        
    Returns:
        Tuple[bool, str]: (is_mentioned, match_type_description)
    """
    # Parse multiple movies from ground truth
    ground_truth_movies = parse_ground_truth_movies(ground_truth)
    
    conversation_lower = conversation_text.lower()
    
    # Check each movie in ground truth using exact substring match
    for movie in ground_truth_movies:
        movie_lower = movie.lower()
        
        # Exact substring match
        if movie_lower in conversation_lower:
            return True, f"exact_match ({movie})"
    
    return False, "no_match"


def calculate_hit_rate_single(ground_truth: str, generated_conversation: List[Dict[str, str]]) -> Tuple[float, str]:
    """
    Calculate hit rate for a single conversation.
    
    Args:
        ground_truth (str): Ground truth movie titles
        generated_conversation (List[Dict]): Generated conversation as list of message dicts
        
    Returns:
        Tuple[float, str]: (hit_rate_score, match_description)
    """
    # Convert conversation to a single string for searching (assistant messages only)
    conversation_text = ""
    for message in generated_conversation:
        if message.get('role') == 'assistant':
            conversation_text += message.get('content', '') + " "
    
    # Check if ground truth movie is mentioned
    is_mentioned, match_type = check_movie_mention_exact(ground_truth, conversation_text)
    
    # Return 1.0 if hit, 0.0 if miss
    hit_score = 1.0 if is_mentioned else 0.0
    
    return hit_score, match_type


def calculate_hit_rate_batch(conversations_data: List[Dict[str, Any]]) -> List[Tuple[float, str]]:
    """
    Calculate hit rates for a batch of conversations.
    
    Args:
        conversations_data (List[Dict]): List of conversation data dictionaries
                                       Each should have 'ground_truth' and 'generated_conversation'
        
    Returns:
        List[Tuple[float, str]]: List of (hit_rate_score, match_description) tuples
    """
    results = []
    
    for data in conversations_data:
        ground_truth = data.get('ground_truth', '')
        generated_conversation = data.get('generated_conversation', [])
        
        # Handle case where conversation might be None or invalid
        if not generated_conversation or not isinstance(generated_conversation, list):
            results.append((0.0, "invalid_conversation"))
            continue
            
        hit_score, match_type = calculate_hit_rate_single(ground_truth, generated_conversation)
        results.append((hit_score, match_type))
    
    return results


def calculate_hit_rate_from_json_string(ground_truth: str, conversation_json_str: str) -> Tuple[float, str]:
    """
    Calculate hit rate when conversation is provided as JSON string.
    
    Args:
        ground_truth (str): Ground truth movie titles
        conversation_json_str (str): JSON string representation of conversation
        
    Returns:
        Tuple[float, str]: (hit_rate_score, match_description)
    """
    try:
        conversation = json.loads(conversation_json_str)
        return calculate_hit_rate_single(ground_truth, conversation)
    except json.JSONDecodeError:
        return 0.0, "json_parse_error"
