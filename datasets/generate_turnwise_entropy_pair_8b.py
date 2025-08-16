import asyncio
import csv
import json
import logging
import pandas as pd
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import time
import nest_asyncio
from pathlib import Path
import copy
import math
from collections import defaultdict, Counter
import numpy as np
import re

# Enable nested event loops for Jupyter
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Import your existing conversation generator and evaluator
import sys
import torch
import os

# Automatically add the root project directory to sys.path
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from simulators.conversation_simulator import ConversationConfig, MultiTurnConversationGenerator

# Import the reward evaluator components
from utils.bedrock_call import bedrock_call
# from evaluation.metrics.hit_checker import calculate_hit_rate_batch
# from evaluation.metrics.interactivity import evaluate_interactivity_batch_async_with_reasoning
import tiktoken
from concurrent.futures import ThreadPoolExecutor, as_completed


class ConversationEntropyAnalyzer:
    def __init__(self, region_name='us-east-1', model_id='us.meta.llama3-1-8b-instruct-v1:0', num_samples=5, num_items=5):
        """
        Initialize the analyzer with Bedrock client
        
        Args:
            region_name: AWS region
            model_id: Bedrock model ID to use
        """
        self.model_id = model_id
        self._executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="entropy_analyzer")
        self.num_items = num_items
        self.num_samples = num_samples
        
    def format_conversation(self, conversation: List[Dict]) -> str:
        """
        Format conversation history up to a specific turn
        
        Args:
            conversation: List of conversation turns
            up_to_turn: Index of the last turn to include
            
        Returns:
            Formatted conversation string
        """
        formatted = ""
        for i, turn in enumerate(conversation):
            role = turn['role'].capitalize()
            content = turn['content'].replace('QUOTATION_MARK', '"')
            formatted += f"{role}: {content}\n"
        return formatted.strip()
    
    def call_llm(self, conversation_history: str) -> str:
        """
        Call the LLM on Bedrock with conversation history
        
        Args:
            conversation_history: Formatted conversation string
            
        Returns:
            LLM response
        """
        prompt = f"""Given the following conversation history:

                {conversation_history}

                Generate a list of the top {self.num_items} movies the user would like to watch based on this conversation. Format your response as a numbered list with no extra sentences:

                1. [First movie]
                2. [Second movie]
                3. [Third movie]

                Make sure each recommendation is unqiue and plausible given the conversation context."""
        
        try:
            # Create messages format for bedrock_call
            messages = [{"role": "user", "content": prompt}]
            
            response = bedrock_call(
                model=self.model_id,
                max_tokens=128,
                temperature=0.7,
                messages=messages,
                num_retries=30
            )
            return response if response else ""
            
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return ""
    
    def extract_responses(self, llm_output: str) -> List[str]:
        """
        Extract num_items recommendations from LLM output
        
        Args:
            llm_output: Raw LLM response
            
        Returns:
            List of extracted responses
        """
        if isinstance(llm_output, dict):
            llm_output = llm_output['generation']

        lines = llm_output.strip().split('\n')
        responses = []
        
        for line in lines:
            # Look for numbered list items
            match = re.match(r'^\d+\.\s*(.+)', line.strip())
            if match:
                responses.append(match.group(1).strip().lower())
        
        # If we didn't get exactly 5, try alternative parsing
        if len(responses) != self.num_items:
            # Try to split by numbers and clean up
            parts = re.split(r'\d+\.', llm_output)
            responses = [part.lower().strip() for part in parts if part.lower().strip()]
            
        return responses[:self.num_items]  # Take first 5 regardless
    
    def calculate_weighted_entropy(self, responses: List[List[str]]) -> float:
        """
        Computes position-weighted entropy over a list of recommendation lists.
        Each inner list is a ranked list of recommendations (e.g., top-k movies).
        """
        if not responses or not all(responses):
            return math.log2(self.num_items * self.num_samples)
        num_lists = len(responses)
        top_k = len(responses[0])
        # Position weights: w_r = 1 / log2(r + 1)
        weights = [1.0 / math.log2(r + 2) for r in range(top_k)]
        # Compute weighted counts
        weighted_counts = defaultdict(float)
        for rec_list in responses:
            for rank, item in enumerate(rec_list):
                weighted_counts[item] += weights[rank]
        total_weight = sum(weighted_counts.values())
        # Convert to probabilities
        probabilities = [count / total_weight for count in weighted_counts.values()]
        # Compute entropy
        entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
        return entropy

    def _single_sample(self, conversation_history: str) -> List[str]:
        """Single sampling call for parallel execution"""
        llm_output = self.call_llm(conversation_history)
        return self.extract_responses(llm_output)
    
    def sample_and_calculate_entropy(self, conversation_history: str) -> float:
        """
        Sample multiple times and calculate entropy based on response diversity (parallel)
        
        Args:
            conversation_history: Formatted conversation
            
        Returns:
            Entropy based on diversity across the samples
        """
        try:
            # Run sampling in parallel
            with ThreadPoolExecutor(max_workers=min(self.num_samples, 5)) as executor:
                futures = [
                    executor.submit(self._single_sample, conversation_history)
                    for _ in range(self.num_samples)
                ]
                
                all_responses = []
                for future in futures:
                    try:
                        response = future.result(timeout=60)  # 30 second timeout per call
                        if response:  # Only add non-empty responses
                            all_responses.append(response)
                    except Exception as e:
                        logger.warning(f"Sample failed: {e}")
                        
            if not all_responses:
                logger.warning("No successful samples obtained")
                return math.log2(self.num_items * self.num_samples)
            
            # Calculate entropy across all responses from all samples
            return self.calculate_weighted_entropy(all_responses)
            
        except Exception as e:
            logger.error(f"Error in parallel sampling: {e}")
            return math.log2(self.num_items * self.num_samples)

    def calculate_entropy_reduction(self, conversation: List[Dict], turn: int) -> tuple[float, float]:
        """
        Calculate entropy reduction metrics for a specific turn
        
        Args:
            conversation: Full conversation list
            turn: The turn to analyze (1-indexed)
            
        Returns:
            Tuple of (entropy_reduced_at_turn, expected_entropy_reduced)
        """
        try:
            user_turn_idx = turn * 2 - 1
            
            if user_turn_idx < 0 or user_turn_idx > len(conversation):
                logger.error(f"Invalid turn number {turn} for conversation of length {len(conversation)}")
                return 0.0, 0.0
            
            # Calculate entropy1: up to the user message only [U1, U2, ...]
            history1 = self.format_conversation(conversation[:user_turn_idx])
            entropy1 = self.sample_and_calculate_entropy(history1)
            
            #placeholder
            entropy2 = entropy1
            entropy3 = entropy1

            # Calculate entropy2: up to next user message [U1, A1, U2, ...]
            next_user_idx = turn * 2 + 1
            
            if next_user_idx < 0 or next_user_idx > len(conversation):
                logger.error(f"Invalid turn number {turn} for conversation of length {len(conversation)}")
                return 0.0, 0.0
            else:
                history2 = self.format_conversation(conversation[:next_user_idx])
                entropy2 = self.sample_and_calculate_entropy(history2)
                entropy_reduced_at_turn = entropy1 - entropy2
            
            # Calculate entropy3: full conversation
            if next_user_idx == len(conversation):
                # this is the last turn
                expected_entropy_reduced = entropy_reduced_at_turn
            else:
                histroy3 = self.format_conversation(conversation)
                entropy3 = self.sample_and_calculate_entropy(histroy3)
                expected_entropy_reduced = entropy1 - entropy3
            
            # If this is the last turn, entropy_reduced_at_turn = expected_entropy_reduced
            if next_user_idx is None:
                entropy_reduced_at_turn = expected_entropy_reduced
            
            logger.info(f"Turn {turn}: entropy1={entropy1:.3f}, entropy2={entropy2:.3f}, entropy3={entropy3:.3f}")
            logger.info(f"Turn {turn}: entropy_reduced_at_turn={entropy_reduced_at_turn:.3f}, expected_entropy_reduced={expected_entropy_reduced:.3f}")
            
            return entropy_reduced_at_turn, expected_entropy_reduced
            
        except Exception as e:
            logger.error(f"Error calculating entropy reduction for turn {turn}: {e}")
            return 0.0, 0.0

    def cleanup(self):
        """Explicitly clean up thread pool"""
        if hasattr(self, '_executor') and self._executor:
            logger.info("Shutting down entropy analyzer thread pool...")
            self._executor.shutdown(wait=True)
            self._executor = None


class TurnWiseDPOExecutor:
    """Execute turn-wise DPO generation with separate functions for each step"""
    
    def __init__(self, base_config: ConversationConfig, user_prompt_template_path: str, output_path: str,
                 reward_config: Dict[str, Any], terminal_signal: str = "[[TERMINATE CHAT]]"):
        self.base_config = base_config
        self.terminal_signal = terminal_signal
        self.reward_config = reward_config
        self.output_path = output_path
        
        # Load the user meta prompt template
        if user_prompt_template_path != "":
            with open(user_prompt_template_path, 'r') as f:
                self.user_prompt_template = f.read()
        
        # Initialize reward evaluator components
        self.weights = reward_config.get('weights', [1., -0.1])
        self.entropy_model = reward_config.get('entropy_model', 'us.meta.llama3-1-8b-instruct-v1:0')
        self.max_workers = reward_config.get('max_workers', 3)
        self.num_samples = reward_config.get('num_samples', 5)
        self.num_items = reward_config.get('num_items', 5)
        self.max_tokens = reward_config.get('max_tokens', 512)
        self.encoding_name = reward_config.get('encoding_name', 'cl100k_base')

        self.entropy_analyzer = ConversationEntropyAnalyzer(
            model_id=self.entropy_model,
            num_samples=self.num_samples,
            num_items=self.num_items
        )
        
        logger.info(f"🎯 TurnWise DPO Executor initialized: max_turns={self.base_config.max_total_turns}, weights={self.weights}")

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'entropy_analyzer') and self.entropy_analyzer:
            self.entropy_analyzer.cleanup()
    
    def load_csv_data(self, csv_path: str) -> pd.DataFrame:
        """Load and parse the CSV data"""
        df = pd.read_csv(csv_path)
        
        # Parse the conversation column with robust error handling
        def parse_conversation(conv_str):
            if pd.isna(conv_str) or conv_str is None:
                return []
            
            try:
                # If it's already a list, return it
                if isinstance(conv_str, list):
                    return conv_str
                
                # Convert to string if not already
                conv_str = str(conv_str).strip()
                
                # Handle empty strings
                if not conv_str or conv_str == 'nan':
                    return []
                
                # Try JSON parsing first (most common)
                if conv_str.startswith('[') or conv_str.startswith('{'):
                    try:
                        return json.loads(conv_str)
                    except json.JSONDecodeError:
                        pass
                
                # Try literal_eval for Python-style formatting
                try:
                    import ast
                    return ast.literal_eval(conv_str)
                except (ValueError, SyntaxError):
                    pass
                
                # Try fixing common JSON issues
                try:
                    # Replace single quotes with double quotes
                    fixed_str = conv_str.replace("'", '"')
                    return json.loads(fixed_str)
                except json.JSONDecodeError:
                    pass
                
                # Try fixing Python True/False/None to JSON format
                try:
                    fixed_str = conv_str.replace("True", "true").replace("False", "false").replace("None", "null")
                    fixed_str = fixed_str.replace("'", '"')
                    return json.loads(fixed_str)
                except json.JSONDecodeError:
                    pass
                
                # Last resort: try to extract content between brackets
                try:
                    import re
                    # Look for list-like structure
                    match = re.search(r'\[(.*)\]', conv_str, re.DOTALL)
                    if match:
                        content = '[' + match.group(1) + ']'
                        content = content.replace("'", '"').replace("True", "true").replace("False", "false").replace("None", "null")
                        return json.loads(content)
                except:
                    pass
                
                logger.warning(f"Could not parse conversation, using empty list. First 100 chars: {conv_str[:100]}")
                return []
                
            except Exception as e:
                logger.warning(f"Failed to parse conversation: {e}. First 100 chars: {str(conv_str)[:100]}")
                return []

        df['original_conversation_parsed'] = df['conversation'].apply(parse_conversation)
        
        # Filter out rows with empty conversations
        original_count = len(df)
        df = df[df['original_conversation_parsed'].apply(lambda x: len(x) > 0)]
        filtered_count = len(df)
        
        if filtered_count < original_count:
            logger.info(f"Filtered out {original_count - filtered_count} rows with empty/invalid conversations")
        
        return df

    # =================== FILE MANAGEMENT ===================
    def get_output_paths(self, dataset: str, turn: int):
        """Get all output paths for a specific turn"""
        base_dir = Path(self.output_path)
        base_dir.mkdir(parents=True, exist_ok=True)
        
        return {
            'turn_file': base_dir / f"turn_{turn}.csv",
            'states_file': base_dir / f"turn_{turn}_states.json"  # For continuation
        }

    def save_chunk_data(self, chunk_data: List[Dict], output_path: str, is_first_chunk: bool = False):
        """Save chunk data to CSV by appending"""
        csv_data = []
        for data_point in chunk_data:
            csv_row = {
                'id': data_point['id'],
                'original_id': data_point['original_id'],
                'turn_number': data_point['turn_number'],
                'ground_truth': data_point['ground_truth'],
                'original_conversation': json.dumps(data_point['original_conversation']),
                'prompt': json.dumps(data_point['prompt']),
                'assistant_response_chosen': data_point.get('assistant_response_chosen', ''),
                'assistant_response_rejected': data_point.get('assistant_response_rejected', ''),
                'chosen_conversation': json.dumps(data_point.get('chosen_conversation', [])),
                'rejected_conversation': json.dumps(data_point.get('rejected_conversation', [])),
                'chosen_score': json.dumps(data_point.get('chosen_score', {})),
                'rejected_score': json.dumps(data_point.get('rejected_score', {})),
                # Keep original format for backwards compatibility
                'assistant_response_1': data_point['assistant_response_1'],
                'assistant_response_2': data_point['assistant_response_2'],
                'response_1_conversation': json.dumps(data_point['response_1_conversation']),
                'response_2_conversation': json.dumps(data_point['response_2_conversation']),
                'response_1_score': json.dumps(data_point['response_1_score']),
                'response_2_score': json.dumps(data_point['response_2_score']),
                'response_1_reasoning': data_point.get('response_1_reasoning', ''),
                'response_2_reasoning': data_point.get('response_2_reasoning', '')
            }
            csv_data.append(csv_row)
        
        df_chunk = pd.DataFrame(csv_data)
        
        # Write header only for first chunk, append for subsequent chunks
        mode = 'w' if is_first_chunk else 'a'
        header = is_first_chunk
        
        df_chunk.to_csv(output_path, mode=mode, header=header, index=False)
        logger.info(f"💾 Chunk {'written' if is_first_chunk else 'appended'} to {output_path}")

    def save_conversation_states_chunk(self, chunk_states: List[Dict], output_path: str, start_idx: int, end_idx: int, total_size: int):
        """Save conversation states for a specific chunk, updating the full states file"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing states if file exists
        if Path(output_path).exists():
            with open(output_path, 'r') as f:
                all_states = json.load(f)
        else:
            # Create empty states array with the right size
            all_states = [None] * total_size
        
        # Update the chunk portion
        for i, chunk_state in enumerate(chunk_states):
            global_idx = start_idx + i
            if global_idx < total_size:
                all_states[global_idx] = chunk_state
        
        # Save updated states
        with open(output_path, 'w') as f:
            json.dump(all_states, f, indent=2)
        
        logger.info(f"💾 Chunk states saved to {output_path} (indices {start_idx}-{end_idx-1})")

    def load_conversation_states_chunk(self, input_path: str, start_idx: int, end_idx: int) -> List[Dict]:
        """Load conversation states for a specific chunk"""
        with open(input_path, 'r') as f:
            all_states = json.load(f)
        
        # Extract the chunk
        chunk_states = all_states[start_idx:end_idx]
        
        # Filter out None values (in case some chunks haven't been processed yet)
        chunk_states = [state for state in chunk_states if state is not None]
        
        logger.info(f"📂 Loaded chunk states from {input_path} (indices {start_idx}-{end_idx-1}, {len(chunk_states)} valid states)")
        return chunk_states

    # =================== STEP 1: Generate User Queries ===================
    async def turn1_step1_generate_initial_queries(self, df: pd.DataFrame, dataset: str, chunk_size: int = 30, start_chunk: int = 0) -> List[Dict]:
        """Turn 1, Step 1: Generate initial user queries chunk by chunk with state updates"""
        logger.info(f"🎬 TURN 1 - STEP 1: Generating initial user queries for {len(df)} examples (chunk-wise)")
        
        paths = self.get_output_paths(dataset, 1)
        total_chunks = (len(df) + chunk_size - 1) // chunk_size
        
        # Initialize states file if starting from scratch
        if start_chunk == 0:
            # Create empty states array
            empty_states = [None] * len(df)
            with open(paths['states_file'], 'w') as f:
                json.dump(empty_states, f)
            logger.info(f"🆕 Initialized states file for {len(df)} conversations")
        
        from simulators.user_simulator import UserSimulator
        
        for chunk_idx in range(start_chunk, total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(df))
            df_chunk = df.iloc[start_idx:end_idx]
            
            logger.info(f"📦 Processing chunk {chunk_idx + 1}/{total_chunks} (rows {start_idx}-{end_idx-1})")
            
            # Process each row in the chunk
            chunk_states = []
            for idx, row in df_chunk.iterrows():
                try:
                    # Create conversation text for context
                    conv_text = ""
                    for msg in row['original_conversation_parsed']:
                        role = msg['role']
                        content = msg['content']
                        if role == 'user':
                            conv_text += f"User: {content}\n"
                        elif role == 'assistant':
                            conv_text += f"Assistant: {content}\n"
                    
                    # Use the existing user_prompt_template
                    formatted_user_prompt = self.user_prompt_template.format(
                        conversation=conv_text.strip(),
                        ground_truth=row['ground_truth'],
                        terminal_signal=self.terminal_signal,
                        chat_history=''
                    )
                    
                    # Create temporary UserSimulator with the formatted prompt
                    temp_user_sim = UserSimulator(
                        user_meta_prompt=formatted_user_prompt,
                        **self.base_config.user_generation_kwargs
                    )
                    
                    # Generate the initial user query directly
                    user_query = await temp_user_sim.async_call([])
                    
                    if not user_query:
                        user_query = "I'm looking for movie recommendations."
                    
                    state = {
                        'id': row.get('dialog_id', idx),
                        'ground_truth': row['ground_truth'],
                        'original_conversation': row['original_conversation_parsed'],
                        'current_conversation': [{"role": "user", "content": user_query.strip()}],
                        'turn_number': 1,
                        'terminated': False
                    }
                    chunk_states.append(state)
                    
                except Exception as e:
                    logger.error(f"Error generating query for row {idx}: {e}")
                    state = {
                        'id': row.get('dialog_id', idx),
                        'ground_truth': row['ground_truth'],
                        'original_conversation': row['original_conversation_parsed'],
                        'current_conversation': [{"role": "user", "content": "I'm looking for movie recommendations."}],
                        'turn_number': 1,
                        'terminated': False
                    }
                    chunk_states.append(state)
            
            # 🔥 NEW: Save chunk states immediately
            self.save_conversation_states_chunk(chunk_states, paths['states_file'], start_idx, end_idx, len(df))
            logger.info(f"    ✅ Chunk {chunk_idx + 1} completed: {len(chunk_states)} queries generated and saved")

        logger.info(f"🎉 TURN 1 - STEP 1 COMPLETE: All chunks processed")
        return []
    

    # =================== STEP 2: Generate Assistant Responses ===================
    async def turn1_step2_generate_completions_and_conversations(self, dataset: str, chunk_size: int = 30, start_chunk: int = 0) -> List[Dict]:
        """Turn 1, Step 2: Generate assistant responses and conversations chunk by chunk"""
        logger.info(f"🤖 TURN 1 - STEP 2: Generating completions and conversations (chunk-wise)")
        
        paths = self.get_output_paths(dataset, 1)
        
        # Load total size from states file
        with open(paths['states_file'], 'r') as f:
            all_states = json.load(f)
        total_size = len(all_states)
        total_chunks = (total_size + chunk_size - 1) // chunk_size
        
        # Initialize AssistantSimulator
        if self.base_config.use_bedrock_assistant:
            from simulators.assistant_simulator import AssistantSimulator
            assistant_sim = AssistantSimulator(
                assistant_meta_prompt=self.base_config.assistant_meta_prompt,
                **self.base_config.assistant_generation_kwargs
            )
        else:
            from simulators.local_assistant_simulator import LoRAAssistantSimulator
            assistant_sim = LoRAAssistantSimulator(
                assistant_meta_prompt=self.base_config.assistant_meta_prompt,
                lora_model_path=self.base_config.local_model_path,
                base_model_path=self.base_config.base_model_path,
                num_gpus=torch.cuda.device_count() if torch.cuda.is_available() else 1,
                **self.base_config.assistant_generation_kwargs
            )

        for chunk_idx in range(start_chunk, total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, total_size)
            
            logger.info(f"📦 Processing chunk {chunk_idx + 1}/{total_chunks} (conversations {start_idx}-{end_idx-1})")
            
            # Load chunk states
            chunk_states = self.load_conversation_states_chunk(paths['states_file'], start_idx, end_idx)
            
            if not chunk_states:
                logger.warning(f"No states found for chunk {chunk_idx + 1}")
                continue
            
            # Step 2a: Generate assistant responses
            logger.info(f"    🤖 Generating assistant responses...")
            chunk_with_responses = await self._generate_assistant_responses_chunk(chunk_states, assistant_sim)
            
            # Step 2b: Generate conversations
            logger.info(f"    💬 Generating conversations...")
            chunk_with_conversations = await self._generate_conversations_chunk_with_simulator(chunk_with_responses)
            
            # 🔥 NEW: Save updated chunk states immediately
            self.save_conversation_states_chunk(chunk_with_conversations, paths['states_file'], start_idx, end_idx, total_size)
            logger.info(f"    ✅ Chunk {chunk_idx + 1} completed")
        
        logger.info(f"🎉 TURN 1 - STEP 2 COMPLETE: Generated responses and conversations")
        return []

    async def turn1_step3_evaluate_conversations(self, dataset: str, chunk_size: int = 30, start_chunk: int = 0) -> List[Dict]:
        """Turn 1, Step 3: Evaluate conversations and save turn data chunk by chunk"""
        logger.info(f"🏆 TURN 1 - STEP 3: Evaluating conversations (chunk-wise)")
        
        paths = self.get_output_paths(dataset, 1)
        
        # Load total size from states file
        with open(paths['states_file'], 'r') as f:
            all_states = json.load(f)
        total_size = len(all_states)
        total_chunks = (total_size + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(start_chunk, total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, total_size)
            
            logger.info(f"📦 Processing chunk {chunk_idx + 1}/{total_chunks} (conversations {start_idx}-{end_idx-1})")
            
            # Load chunk states
            chunk_states = self.load_conversation_states_chunk(paths['states_file'], start_idx, end_idx)
            
            if not chunk_states:
                logger.warning(f"No states found for chunk {chunk_idx + 1}")
                continue
            
            # Evaluate rewards
            logger.info(f"    🏆 Evaluating rewards...")
            chunk_with_rewards = await self._evaluate_rewards_chunk(chunk_states)
            
            # Extract turn data
            chunk_turn_data = self._extract_turn_data_with_chosen_rejected(chunk_with_rewards, 1)
            
            # Save chunk to CSV
            is_first_chunk = (chunk_idx == 0)
            if chunk_turn_data:
                self.save_chunk_data(chunk_turn_data, paths['turn_file'], is_first_chunk)
            
            # Update states with rewards for next turn
            for i, state in enumerate(chunk_with_rewards):
                turn_data_point = chunk_turn_data[i]
                
                state.update({
                    'chosen_conversation': turn_data_point['chosen_conversation'],
                    'chosen_score': turn_data_point['chosen_score'],
                    'response_1_score': turn_data_point['response_1_score'],
                    'response_2_score': turn_data_point['response_2_score']
                })
            
            # 🔥 NEW: Save updated chunk states immediately
            self.save_conversation_states_chunk(chunk_with_rewards, paths['states_file'], start_idx, end_idx, total_size)
            logger.info(f"    ✅ Chunk {chunk_idx + 1} completed: {len(chunk_turn_data)} data points saved")
        
        logger.info(f"🎉 TURN 1 - STEP 3 COMPLETE: All chunks evaluated and saved")
        return []

    # =================== HELPER FUNCTIONS ===================
    async def _generate_assistant_responses_chunk(self, chunk_states: List[Dict], assistant_sim) -> List[Dict]:
        """Generate assistant responses for a chunk of states"""
        updated_states = []
        for state in chunk_states:
            if state.get('terminated', False):
                logger.debug(f"State {state['id']}: Skipping response generation - conversation terminated")
                updated_states.append(state)
                continue
            
            try:
                current_conv = state['current_conversation']
                
                # Generate 2 different assistant responses in parallel
                response_tasks = []
                for resp_idx in range(2):
                    task = self._generate_assistant_response(current_conv, assistant_sim)
                    response_tasks.append(task)
                
                responses = await asyncio.gather(*response_tasks, return_exceptions=True)
                
                # Process results
                processed_responses = []
                for resp in responses:
                    if isinstance(resp, Exception):
                        logger.error(f"Error generating assistant response: {resp}")
                        processed_responses.append("I'd be happy to help you with movie recommendations!")
                    else:
                        processed_responses.append(resp)
                
                # Update state
                updated_state = state.copy()
                updated_state['assistant_response_1'] = processed_responses[0]
                updated_state['assistant_response_2'] = processed_responses[1]
                updated_states.append(updated_state)
                
            except Exception as e:
                logger.error(f"Error generating responses for {state['id']}: {e}")
                # Add fallback
                updated_state = state.copy()
                updated_state['assistant_response_1'] = "I'd be happy to help you with movie recommendations!"
                updated_state['assistant_response_2'] = "What kind of movies are you interested in?"
                updated_states.append(updated_state)
        
        return updated_states

    async def _generate_conversations_chunk_with_simulator(self, chunk_states: List[Dict]) -> List[Dict]:
        """Generate conversations using the conversation simulator to continue from existing conversation"""
        
        # 🔥 Filter out terminated states first
        active_states = [state for state in chunk_states if not state.get('terminated', False)]
        terminated_states = [state for state in chunk_states if state.get('terminated', False)]
        
        logger.debug(f"Conversation generation: {len(active_states)} active, {len(terminated_states)} terminated")
        
        if not active_states:
            logger.debug("No active states for conversation generation")
            return chunk_states

        # Prepare all conversation generation tasks for parallel execution
        all_tasks = []
        task_mapping = []  # (state_idx, resp_idx)
        
        for state_idx, state in enumerate(chunk_states):
            # Create user prompt context once per state
            custom_user_prompt = self._create_user_prompt_context(state)
            
            # Create tasks for both assistant responses in parallel
            for resp_idx, resp_key in enumerate(['assistant_response_1', 'assistant_response_2']):
                assistant_response = state[resp_key]
                
                # Create conversation with current context + assistant response
                conversation_start = state['current_conversation'] + [
                    {"role": "assistant", "content": assistant_response}
                ]
                
                # Create the conversation generation task using conversation simulator
                task = self._generate_conversation_with_simulator(
                    conversation_start=conversation_start,
                    custom_user_prompt=custom_user_prompt,
                    state_id=state['id'],
                    resp_idx=resp_idx,
                    current_turn=state.get('turn_number', 1)
                )
                
                all_tasks.append(task)
                task_mapping.append((state_idx, resp_idx))
        
        # Execute all conversation generation tasks in parallel
        logger.info(f"    🚀 Generating {len(all_tasks)} conversations in parallel...")
        results = await asyncio.gather(*all_tasks, return_exceptions=True)
        
        # Process results and assign back to states
        updated_states = []
        for state_idx, state in enumerate(chunk_states):
            updated_state = state.copy()
            
            # Find results for this state
            for task_idx, (s_idx, r_idx) in enumerate(task_mapping):
                if s_idx == state_idx:
                    result = results[task_idx]
                    
                    if isinstance(result, Exception):
                        logger.error(f"Error generating conversation for state {state['id']}, response {r_idx}: {result}")
                        # Fallback to just the assistant response
                        conversation_start = state['current_conversation'] + [
                            {"role": "assistant", "content": state[f'assistant_response_{r_idx + 1}']}
                        ]
                        conv_key = f'response_{r_idx + 1}_conversation'
                        updated_state[conv_key] = conversation_start
                    else:
                        # Store the successful result
                        conv_key = f'response_{r_idx + 1}_conversation'
                        updated_state[conv_key] = result
            
            updated_states.append(updated_state)
        
        return updated_states

    async def _generate_conversation_with_simulator(self, conversation_start: List[Dict], 
                                                   custom_user_prompt: str, 
                                                   state_id: str, resp_idx: int, current_turn: int = 1) -> List[Dict]:
        """Generate conversation continuation using the conversation simulator"""
        try:
            # Create custom config for this specific conversation
            custom_config = copy.deepcopy(self.base_config)
            custom_config.user_meta_prompt = custom_user_prompt
            
            # Create conversation generator
            generator = MultiTurnConversationGenerator(custom_config)

            # Calculate remaining turns based on max_total_turns and current progress
            total_messages_so_far = len(conversation_start)
            max_total_messages = self.base_config.max_total_turns
            remaining_messages = max(0, max_total_messages - total_messages_so_far)
            
            # Convert remaining messages to turns (each turn = user + assistant)
            remaining_turns = max(1, remaining_messages // 2)
            
            logger.debug(f"  🔄 Conversation {state_id}-{resp_idx}: {total_messages_so_far} messages so far, {remaining_turns} turns remaining")
            
            
            # Create a temporary conversation state for the generator
            temp_state = {
                'id': 0,
                'prompt': None,
                'chat_history': conversation_start.copy(),
                'user_sim': None,  # Will be created by the generator
                'completed': False,
                'turn_count': total_messages_so_far // 2
            }
            
            # Use the generator's internal methods to continue the conversation
            # Simulate the conversation state that the generator expects
            from simulators.user_simulator import UserSimulator
            user_sim = UserSimulator(
                user_meta_prompt=custom_user_prompt,
                **self.base_config.user_generation_kwargs
            )
            temp_state['user_sim'] = user_sim
            
            # Continue the conversation for a few more turns
            active_states = [temp_state]
            
            # Generate a few more turns (limit to avoid infinite conversations)
            for turn in range(remaining_turns):
                if temp_state['completed'] or len(temp_state['chat_history']) >= max_total_messages:
                    break
                    
                await generator._process_user_turn_batch(active_states)
                
                if temp_state['completed'] or len(temp_state['chat_history']) >= max_total_messages:
                    break
                    
                await generator._process_assistant_turn_batch(active_states)
                
                if (len(temp_state['chat_history']) > 0 and 
                    generator._should_terminate_conversation(temp_state['chat_history'][-1].get('content', ''))):
                    temp_state['completed'] = True
                    break
            
            return temp_state['chat_history']
            
        except Exception as e:
            logger.error(f"Error in _generate_conversation_with_simulator for {state_id}, response {resp_idx}: {e}")
            # Return just the conversation start as fallback
            return conversation_start

    def _should_terminate_conversation(self, message: str) -> bool:
        """Check if conversation should be terminated"""
        termination_phrases = ["that's all I need", "goodbye", "thank you"]
        message_lower = message.lower()
        
        if "TERMINATE" in message or "[[TERMINATE CHAT]]" in message:
            return True
            
        return any(phrase in message_lower for phrase in termination_phrases)

    async def _evaluate_rewards_chunk(self, chunk_states: List[Dict]) -> List[Dict]:
        """Evaluate rewards for conversations in a chunk"""
        active_states = [state for state in chunk_states if not state.get('terminated', False)]
        terminated_states = [state for state in chunk_states if state.get('terminated', False)]
        
        logger.debug(f"Evaluation: {len(active_states)} active states, {len(terminated_states)} terminated states")
        
        if not active_states:
            logger.debug("No active states to evaluate")
            return chunk_states

        # Prepare conversations for evaluation
        conversations = []
        contexts = []
        mapping = []  # (state_idx, response_idx)
        
        for state_idx, state in enumerate(chunk_states):
            for resp_idx in range(2):
                conv_key = f'response_{resp_idx + 1}_conversation'
                if conv_key in state and state[conv_key]:
                    conversations.append(state[conv_key])
                    contexts.append({
                        'ground_truth': state['ground_truth'],
                        'turn_number': state.get('turn_number', 1),
                        'original_conversation': state.get('original_conversation', [])
                    })
                    mapping.append((state_idx, resp_idx))
        
        if not conversations:
            return chunk_states
        
        # Evaluate metrics
        scores = await self.evaluate_metrics(conversations, contexts, {'entropy_reduced', 'token_efficiency'})
        
        # Assign scores back to states
        updated_states = []
        for state_idx, state in enumerate(chunk_states):
            updated_state = state.copy()
            
            # Find scores for this state's conversations
            for conv_idx, (s_idx, r_idx) in enumerate(mapping):
                if s_idx == state_idx:
                    # Extract scores for this conversation
                    conv_scores = {}
                    conv_reasoning = {}
                    
                    for metric, score_list in scores.items():
                        conv_scores[metric] = score_list[conv_idx] if conv_idx < len(score_list) else 0.0
                    
                    # Calculate combined score
                    conv_scores['combined'] = self.calculate_combined_score(conv_scores)
                    
                    # Store scores
                    score_key = f'response_{r_idx + 1}_score'
                    updated_state[score_key] = conv_scores
            
            updated_states.append(updated_state)
        
        return updated_states

    async def _generate_user_response_for_entropy(self, conversation: List[Dict], context: Dict) -> Optional[str]:
        """Generate user response for entropy calculation when conversation ends with assistant message"""
        try:
            state = {
                'id': 'entropy_eval',
                'ground_truth': context.get('ground_truth', ''),
                'original_conversation': context.get('original_conversation', [])
            }
            
            # Create user prompt context
            custom_user_prompt = self._create_user_prompt_context(state)
            
            from simulators.user_simulator import UserSimulator
            user_sim = UserSimulator(
                user_meta_prompt=custom_user_prompt,
                **self.base_config.user_generation_kwargs
            )
            
            user_response = await user_sim.async_call(conversation)
            return user_response.strip() if user_response else None
            
        except Exception as e:
            logger.error(f"Error generating user response for entropy: {e}")
            return None

    async def evaluate_metrics(self, conversations: List[List[Dict]], 
                              contexts: List[Dict], 
                              metrics: Set[str]) -> Dict[str, List[float]]:
        """Evaluate specified metrics for conversations"""
        results = {}
        
        # 🔥 CHANGED: Replace hit_rate and interactivity with entropy_reduced
        if 'entropy_reduced' in metrics:
            turn_entropy_scores = []
            conv_entropy_scores = []
            
            async def process_single_conversation(conv, ctx, index):
                try:
                    # Check if last message is assistant, generate user response if needed
                    conversation_for_entropy = conv.copy()
                    if conversation_for_entropy and conversation_for_entropy[-1].get('role') == 'assistant':
                        # Need to generate user response to complete the turn
                        user_response = await self._generate_user_response_for_entropy(conversation_for_entropy, ctx)
                        if user_response:
                            conversation_for_entropy.append({"role": "user", "content": user_response})

                    current_turn = ctx.get('turn_number', 1)
                    
                    if current_turn < 1:
                        return math.log2(self.num_items * self.num_samples), math.log2(self.num_items * self.num_samples)
                    
                    # Run entropy calculation in thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    turn_entropy_reduced, conv_entropy_reduced = await loop.run_in_executor(
                        None,  # Use default executor
                        self.entropy_analyzer.calculate_entropy_reduction,
                        conv,
                        current_turn
                    )

                    return turn_entropy_reduced, conv_entropy_reduced
                    
                except Exception as e:
                    logger.error(f"Error calculating entropy for conversation {i}: {e}")
                    return math.log2(self.num_items * self.num_samples), math.log2(self.num_items * self.num_samples)

            entropy_tasks = [
                process_single_conversation(conv, ctx, i) 
                for i, (conv, ctx) in enumerate(zip(conversations, contexts))
            ]
            entropy_results = await asyncio.gather(*entropy_tasks, return_exceptions=True)
            
            # 🔥 CHANGED: Separate turn and conversation entropy scores
            turn_entropy_scores = []
            conv_entropy_scores = []
            
            for i, result in enumerate(entropy_results):
                if isinstance(result, Exception):
                    logger.error(f"Entropy calculation failed for conversation {i}: {result}")
                    turn_entropy_scores.append(math.log2(self.num_items * self.num_samples))
                    conv_entropy_scores.append(math.log2(self.num_items * self.num_samples))
                else:
                    turn_entropy_reduced, conv_entropy_reduced = result
                    turn_entropy_scores.append(turn_entropy_reduced)
                    conv_entropy_scores.append(conv_entropy_reduced)
            
            # 🔥 CHANGED: Store both entropy metrics
            results['turn_entropy_reduced'] = turn_entropy_scores
            results['conv_entropy_reduced'] = conv_entropy_scores

        if 'token_efficiency' in metrics:
            results['token_efficiency'] = self.evaluate_token_efficiency_parallel(conversations)
        
        return results

    def evaluate_token_efficiency_parallel(self, conversations: List[List[Dict]]) -> List[float]:
        """Evaluate token efficiency in parallel"""
        def calc_efficiency(conv):
            return self.calculate_token_efficiency(conv, self.max_tokens, self.encoding_name)
        
        scores = [0.0] * len(conversations)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {executor.submit(calc_efficiency, conv): idx 
                           for idx, conv in enumerate(conversations)}
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    scores[idx] = future.result()
                except Exception as e:
                    logger.warning(f"Token efficiency error for conv {idx}: {e}")
                    scores[idx] = 0.0
        
        return scores

    def calculate_token_efficiency(self, conversation: List[Dict[str, str]], 
                             max_tokens: int = 512,
                             encoding_name: str = "cl100k_base") -> float:
        """Calculate token efficiency score for just the completion (last assistant message)"""
        try:
            encoding = tiktoken.get_encoding(encoding_name)
            
            # Find the last assistant message (the completion we're evaluating)
            last_assistant_message = None
            for message in reversed(conversation):
                if message.get('role') == 'assistant':
                    last_assistant_message = message
                    break
            
            if not last_assistant_message:
                return 0.0
            
            # Count tokens only for this specific completion
            content = last_assistant_message.get('content', '')
            tokens = encoding.encode(content)
            total_tokens = len(tokens)
            
            return total_tokens / max_tokens
            
        except Exception as e:
            logger.warning(f"Error counting tokens: {e}")
            return 0.0

    def calculate_combined_score(self, scores: Dict[str, float]) -> float:
        """Calculate combined score from available metrics"""
        available_metrics = ['turn_entropy_reduced', 'token_efficiency']
        present_metrics = [m for m in available_metrics if m in scores]
        
        if not present_metrics:
            return 0.0
        
        # If we have all two metrics, use configurable weights
        if 'turn_entropy_reduced' in scores and 'token_efficiency' in scores:
            return (self.weights[0] * scores['turn_entropy_reduced'] + 
                   self.weights[1] * scores['token_efficiency'])
        
        if 'turn_entropy_reduced' in scores:
            return scores['turn_entropy_reduced']
        
        # Fallback: equal weighting of available metrics
        return sum(scores[m] for m in present_metrics) / len(present_metrics)

    def _extract_turn_data_with_chosen_rejected(self, states_with_rewards: List[Dict], turn: int) -> List[Dict]:
        """Extract data for this turn with chosen/rejected labels based on scores"""
        
        turn_data = []
        
        for state in states_with_rewards:
            if state.get('terminated', False):
                logger.debug(f"State {state['id']}: Skipping CSV extraction - conversation terminated")
                continue
            

            # Create prompt (conversation up to this turn's user query)
            if turn == 1:
                # Turn 1: prompt is just the initial user query
                prompt = state['current_conversation'].copy()
            
            else:
                chosen_path_length = (turn-1) * 2
                chosen_conversation = state.get('chosen_conversation', [])
                
                if isinstance(chosen_conversation, str):
                    try:
                        chosen_conversation = json.loads(chosen_conversation)
                    except:
                        chosen_conversation = []
                
                # Get the chosen path (previous turns)
                chosen_path = chosen_conversation[:chosen_path_length] if len(chosen_conversation) >= chosen_path_length else chosen_conversation
                
                # Get the current user query (last message in current_conversation)
                current_conv = state.get('current_conversation', [])
                current_user_query = current_conv[-1] if len(current_conv) > 0 and current_conv[-1].get('role') == 'user' else None
                
                if current_user_query:
                    prompt = chosen_path + [current_user_query]
                else:
                    # Fallback to current_conversation
                    prompt = current_conv.copy()
            
            # Get scores for both responses
            score_1 = state.get('response_1_score', {}).get('combined', 0.0)
            score_2 = state.get('response_2_score', {}).get('combined', 0.0)
            
            # Determine chosen and rejected based on scores
            if score_1 >= score_2:
                chosen_idx, rejected_idx = 1, 2
            else:
                chosen_idx, rejected_idx = 2, 1
            
            data_point = {
                'id': f"{state['id']}_turn_{turn}",
                'original_id': state['id'],
                'turn_number': turn,
                'ground_truth': state['ground_truth'],
                'original_conversation': state['original_conversation'],
                'prompt': prompt,
                
                # Chosen/Rejected format
                'assistant_response_chosen': state.get(f'assistant_response_{chosen_idx}', ''),
                'assistant_response_rejected': state.get(f'assistant_response_{rejected_idx}', ''),
                'chosen_conversation': state.get(f'response_{chosen_idx}_conversation', []),
                'rejected_conversation': state.get(f'response_{rejected_idx}_conversation', []),
                'chosen_score': state.get(f'response_{chosen_idx}_score', {}),
                'rejected_score': state.get(f'response_{rejected_idx}_score', {}),
                'chosen_reasoning': state.get(f'response_{chosen_idx}_reasoning', ''),
                'rejected_reasoning': state.get(f'response_{rejected_idx}_reasoning', ''),
                
                # Original format for backwards compatibility
                'assistant_response_1': state.get('assistant_response_1', ''),
                'assistant_response_2': state.get('assistant_response_2', ''),
                'response_1_conversation': state.get('response_1_conversation', []),
                'response_2_conversation': state.get('response_2_conversation', []),
                'response_1_score': state.get('response_1_score', {}),
                'response_2_score': state.get('response_2_score', {}),
                'response_1_reasoning': state.get('response_1_reasoning', ''),
                'response_2_reasoning': state.get('response_2_reasoning', '')
            }
            
            turn_data.append(data_point)
        
        return turn_data


    # =================== CHUNK-WISE PROCESSING METHODS ===================

    async def _generate_next_turn_user_queries(self, states: List[Dict], turn: int) -> List[Dict]:
        """Generate user queries for next turn based on chosen conversation"""
        updated_states = []
        
        for state in states:
            try:
                # Get the chosen conversation from previous turn
                chosen_path = state.get('chosen_conversation', state.get('current_conversation', []))[:(turn-1)*2+1]
                
                if not chosen_path or len(chosen_path) == 0:
                    logger.warning(f"State {state['id']}: No chosen path found, using current_conversation")
                    chosen_path = state.get('current_conversation', [])

                # Check if the chosen conversation should be terminated
                if len(chosen_path) > 0:
                    last_message = chosen_path[-1].get('content', '')
                    if self._should_terminate_conversation(last_message):
                        # Mark as terminated
                        updated_state = state.copy()
                        updated_state['terminated'] = True
                        updated_states.append(updated_state)
                        continue
                
                chosen_path = chosen_path[:(turn-1)*2]
                # Create user prompt context
                custom_user_prompt = self._create_user_prompt_context(state)
                
                # Generate next user query using the chosen conversation as history
                user_query = await self._generate_user_response_with_context(chosen_path, custom_user_prompt)
                
                if not user_query or self._should_terminate_conversation(user_query):
                    # Mark as terminated
                    updated_state = state.copy()
                    updated_state['terminated'] = True
                    updated_states.append(updated_state)
                    continue
                
                # Update conversation with new user query
                # IMPORTANT: Use chosen conversation + new user query as the current conversation
                new_current_conv = chosen_path + [{"role": "user", "content": user_query}]
                current_turn = state.get('turn_number', 1) + 1
                expected_length = current_turn * 2 - 1 

                updated_state = state.copy()
                updated_state['current_conversation'] = new_current_conv
                updated_state['turn_number'] = current_turn
                updated_states.append(updated_state)
                
            except Exception as e:
                logger.error(f"Error generating user query for {state['id']}: {e}")
                # Mark as terminated on error
                updated_state = state.copy()
                updated_state['terminated'] = True
                updated_states.append(updated_state)
        
        return updated_states

    async def _generate_next_turn_assistant_responses(self, states: List[Dict], chunk_size: int) -> List[Dict]:
        """Generate assistant responses for next turn"""
        # Initialize assistant simulator
        if self.base_config.use_bedrock_assistant:
            from simulators.assistant_simulator import AssistantSimulator
            assistant_sim = AssistantSimulator(
                assistant_meta_prompt=self.base_config.assistant_meta_prompt,
                **self.base_config.assistant_generation_kwargs
            )
        else:
            from simulators.local_assistant_simulator import LoRAAssistantSimulator
            assistant_sim = LoRAAssistantSimulator(
                assistant_meta_prompt=self.base_config.assistant_meta_prompt,
                lora_model_path=self.base_config.local_model_path,
                base_model_path=self.base_config.base_model_path,
                num_gpus=torch.cuda.device_count() if torch.cuda.is_available() else 1,
                **self.base_config.assistant_generation_kwargs
            )
        
        updated_states = []
        total_chunks = (len(states) + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(states))
            chunk_states = states[start_idx:end_idx]
            
            # Generate responses for this chunk
            chunk_with_responses = await self._generate_assistant_responses_chunk(chunk_states, assistant_sim)
            updated_states.extend(chunk_with_responses)
        
        return updated_states

    async def _generate_next_turn_conversations(self, states: List[Dict], chunk_size: int) -> List[Dict]:
        """Generate conversations for next turn"""
        updated_states = []
        total_chunks = (len(states) + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(states))
            chunk_states = states[start_idx:end_idx]
            
            # Generate conversations for this chunk using the simulator
            chunk_with_conversations = await self._generate_conversations_chunk_with_simulator(chunk_states)
            updated_states.extend(chunk_with_conversations)
        
        return updated_states

    # =================== SHARED HELPER FUNCTIONS ===================
    def _create_user_prompt_context(self, state: Dict) -> str:
        """Create user prompt context from original conversation"""
        conv_text = ""
        for msg in state['original_conversation']:
            role = msg['role']
            content = msg['content']
            if role == 'user':
                conv_text += f"User: {content}\n"
            elif role == 'assistant':
                conv_text += f"Assistant: {content}\n"
        
        # Fill in template for user simulation
        return self.user_prompt_template.format(
            conversation=conv_text.strip(),
            ground_truth=state['ground_truth'],
            terminal_signal=self.terminal_signal,
            chat_history="{chat_history}"
        )

    async def _generate_assistant_response(self, conversation_context: List[Dict], assistant_sim) -> str:
        """Generate a single assistant response"""
        try:
            if self.base_config.use_bedrock_assistant:
                response = await assistant_sim.async_call(conversation_context)
            else:
                response = assistant_sim.generate_response(conversation_context)
            
            return response.strip() if response else "I'd be happy to help you with movie recommendations!"
        except Exception as e:
            logger.error(f"Error generating assistant response: {e}")
            return "I'd be happy to help you with movie recommendations!"

    async def _generate_user_response_with_context(self, conversation_context: List[Dict], custom_user_prompt: str) -> Optional[str]:
        """Generate user response with custom prompt context"""
        try:
            from simulators.user_simulator import UserSimulator
            user_sim = UserSimulator(
                user_meta_prompt=custom_user_prompt,
                **self.base_config.user_generation_kwargs
            )
            
            user_response = await user_sim.async_call(conversation_context)
            return user_response.strip() if user_response else None
        except Exception as e:
            logger.error(f"Error generating user response: {e}")
            return None


    # =================== INTERFACE METHODS FOR COMPATIBILITY ===================
    async def turn_step1_generate_user_response_and_completions(self, turn: int, dataset: str, chunk_size: int = 30, start_chunk: int = 0):
        """Turn N Step 1: Generate user response and new assistant completions"""
        logger.info(f"👤🤖 TURN {turn} - STEP 1: Generating user responses and completions")
        
        # Load states from previous turn
        prev_paths = self.get_output_paths(dataset, turn - 1)
        with open(prev_paths['states_file'], 'r') as f:
            all_prev_states = json.load(f)
        
        # Filter out terminated conversations to get active indices
        active_indices = []
        for i, state in enumerate(all_prev_states):
            if state and not state.get('terminated', False):
                active_indices.append(i)
        
        if not active_indices:
            logger.info("🏁 All conversations terminated")
            current_paths = self.get_output_paths(dataset, turn)
            with open(current_paths['states_file'], 'w') as f:
                json.dump([], f)
            return []
        
        logger.info(f"📊 {len(active_indices)} active conversations out of {len(all_prev_states)} total")
        
        # Process active conversations in chunks
        total_chunks = (len(active_indices) + chunk_size - 1) // chunk_size
        current_paths = self.get_output_paths(dataset, turn)
        
        # Initialize new states file for this turn if starting from scratch
        if start_chunk == 0:
            new_states = [None] * len(all_prev_states)
            with open(current_paths['states_file'], 'w') as f:
                json.dump(new_states, f)
        
        for chunk_idx in range(start_chunk, total_chunks):
            start_active_idx = chunk_idx * chunk_size
            end_active_idx = min((chunk_idx + 1) * chunk_size, len(active_indices))
            
            # Get actual indices in the original states array
            chunk_indices = active_indices[start_active_idx:end_active_idx]
            
            logger.info(f"📦 Processing chunk {chunk_idx + 1}/{total_chunks} (active conversations {start_active_idx}-{end_active_idx-1})")
            
            # Load chunk states by indices
            chunk_states = []
            for idx in chunk_indices:
                if all_prev_states[idx]:
                    chunk_states.append(all_prev_states[idx])
            
            if not chunk_states:
                logger.warning(f"No valid states found for chunk {chunk_idx + 1}")
                continue
            
            # 🔥 LEVERAGE EXISTING METHOD: Generate user queries
            logger.info(f"    👤 Generating user queries...")
            states_with_queries = await self._generate_next_turn_user_queries(chunk_states, turn)
            
            # Filter out newly terminated states
            active_states = [s for s in states_with_queries if not s.get('terminated', False)]
            
            if not active_states:
                logger.info(f"    🏁 All conversations in chunk terminated")
                # Save terminated states back to the file
                states_to_save = states_with_queries
                self._save_chunk_states_by_indices(states_to_save, chunk_indices, current_paths['states_file'], len(all_prev_states))
                continue
            
            # 🔥 LEVERAGE EXISTING METHOD: Generate assistant responses
            logger.info(f"    🤖 Generating assistant responses...")
            states_with_responses = await self._generate_next_turn_assistant_responses(active_states, chunk_size)
            
            # Combine active and terminated states for saving
            final_states_for_chunk = []
            active_idx = 0
            for state in states_with_queries:
                if state.get('terminated', False):
                    final_states_for_chunk.append(state)
                else:
                    if active_idx < len(states_with_responses):
                        final_states_for_chunk.append(states_with_responses[active_idx])
                        active_idx += 1
                    else:
                        final_states_for_chunk.append(state)
            
            # Save updated chunk states
            self._save_chunk_states_by_indices(final_states_for_chunk, chunk_indices, current_paths['states_file'], len(all_prev_states))
            
            logger.info(f"    ✅ Chunk {chunk_idx + 1} completed")
        
        logger.info(f"🎉 TURN {turn} - STEP 1 COMPLETE: All chunks processed")
        return []

    def _save_chunk_states_by_indices(self, chunk_states: List[Dict], chunk_indices: List[int], states_file_path: str, total_size: int):
        """Helper method to save chunk states at specific indices"""
        # Load existing states
        if Path(states_file_path).exists():
            with open(states_file_path, 'r') as f:
                all_states = json.load(f)
        else:
            all_states = [None] * total_size
        
        # Update states at specific indices
        for i, state in enumerate(chunk_states):
            if i < len(chunk_indices):
                all_states[chunk_indices[i]] = state
        
        # Save back to file
        with open(states_file_path, 'w') as f:
            json.dump(all_states, f, indent=2)
        
        indices_str = f"{min(chunk_indices)}-{max(chunk_indices)}" if chunk_indices else "none"
        logger.debug(f"💾 Saved chunk states at indices {indices_str}")

    async def turn_step2_generate_conversations(self, turn: int, dataset: str, chunk_size: int = 30, start_chunk: int = 0):
        """Turn N Step 2: Generate conversations chunk by chunk"""
        logger.info(f"💬 TURN {turn} - STEP 2: Generating conversations (chunk-wise)")
        
        paths = self.get_output_paths(dataset, turn)
        if not paths['states_file'].exists():
            logger.error(f"❌ States file not found: {paths['states_file']}")
            return []
        
        # Load states and get active indices
        with open(paths['states_file'], 'r') as f:
            all_states = json.load(f)
        
        active_indices = []
        for i, state in enumerate(all_states):
            if state and not state.get('terminated', False):
                active_indices.append(i)
        
        if not active_indices:
            logger.info("🏁 No active conversations to process")
            return []
        
        total_chunks = (len(active_indices) + chunk_size - 1) // chunk_size
        logger.info(f"📊 Processing {len(active_indices)} active conversations in {total_chunks} chunks")
        
        for chunk_idx in range(start_chunk, total_chunks):
            start_active_idx = chunk_idx * chunk_size
            end_active_idx = min((chunk_idx + 1) * chunk_size, len(active_indices))
            
            chunk_indices = active_indices[start_active_idx:end_active_idx]
            
            logger.info(f"📦 Processing chunk {chunk_idx + 1}/{total_chunks} (active conversations {start_active_idx}-{end_active_idx-1})")
            
            # Load chunk states
            chunk_states = [all_states[idx] for idx in chunk_indices if all_states[idx]]
            
            if not chunk_states:
                logger.warning(f"No valid states found for chunk {chunk_idx + 1}")
                continue
            
            # 🔥 LEVERAGE EXISTING METHOD: Generate conversations
            logger.info(f"    💬 Generating conversations...")
            chunk_with_conversations = await self._generate_next_turn_conversations(chunk_states, len(chunk_states))
            
            # Save updated states
            self._save_chunk_states_by_indices(chunk_with_conversations, chunk_indices, paths['states_file'], len(all_states))
            
            logger.info(f"    ✅ Chunk {chunk_idx + 1} completed")
        
        logger.info(f"🎉 TURN {turn} - STEP 2 COMPLETE: All conversations generated")
        return []

    async def turn_step3_evaluate_conversations(self, turn: int, dataset: str, chunk_size: int = 30, start_chunk: int = 0):
        logger.info(f"🏆 TURN {turn} - STEP 3: Evaluating conversations (chunk-wise)")
    
        paths = self.get_output_paths(dataset, turn)
        if not paths['states_file'].exists():
            logger.error(f"❌ States file not found: {paths['states_file']}")
            return []
        
        # Load states and get active indices
        with open(paths['states_file'], 'r') as f:
            all_states = json.load(f)
        
        active_indices = []
        for i, state in enumerate(all_states):
            if state and not state.get('terminated', False):
                active_indices.append(i)
        
        if not active_indices:
            logger.info("🏁 No conversations to evaluate")
            return []
        
        total_chunks = (len(active_indices) + chunk_size - 1) // chunk_size
        logger.info(f"📊 Evaluating {len(active_indices)} active conversations in {total_chunks} chunks")
        
        for chunk_idx in range(start_chunk, total_chunks):
            start_active_idx = chunk_idx * chunk_size
            end_active_idx = min((chunk_idx + 1) * chunk_size, len(active_indices))
            
            chunk_indices = active_indices[start_active_idx:end_active_idx]
            
            logger.info(f"📦 Processing chunk {chunk_idx + 1}/{total_chunks} (active conversations {start_active_idx}-{end_active_idx-1})")
            
            # Load chunk states
            chunk_states = [all_states[idx] for idx in chunk_indices if all_states[idx]]
            
            if not chunk_states:
                logger.warning(f"No valid states found for chunk {chunk_idx + 1}")
                continue
            
            # 🔥 EVALUATE MANUALLY
            logger.info(f"    🏆 Evaluating rewards...")
            chunk_with_rewards = await self._evaluate_rewards_chunk(chunk_states)
            
            # Extract turn data
            chunk_turn_data = self._extract_turn_data_with_chosen_rejected(chunk_with_rewards, turn)
            
            # Save chunk to CSV
            is_first_chunk = (chunk_idx == start_chunk)  # Use start_chunk instead of 0 for resumability
            if chunk_turn_data:
                self.save_chunk_data(chunk_turn_data, paths['turn_file'], is_first_chunk)
            
            # Update states with chosen conversation for next turn
            for i, state in enumerate(chunk_with_rewards):
                turn_data_point = chunk_turn_data[i]
                
                state.update({
                    'chosen_conversation': turn_data_point['chosen_conversation'],
                    'chosen_score': turn_data_point['chosen_score'],
                    'response_1_score': turn_data_point['response_1_score'],
                    'response_2_score': turn_data_point['response_2_score']
                })
            
            # 🔥 FIX: Use chunk-wise state saving
            self._save_chunk_states_by_indices(chunk_with_rewards, chunk_indices, paths['states_file'], len(all_states))
            
            logger.info(f"    ✅ Chunk {chunk_idx + 1} completed: {len(chunk_turn_data)} data points saved")
        
        logger.info(f"🎉 TURN {turn} - STEP 3 COMPLETE: All conversations evaluated and saved")
        return []

    def combine_all_turns(self, dataset: str, max_turns: int = 4) -> pd.DataFrame:
        """Combine all turn data into a single DataFrame grouped by original_id"""
        logger.info(f"🔗 Combining data from {max_turns} turns and grouping by original_id")
        
        all_dataframes = []
        
        # Load each turn's CSV
        for turn in range(1, max_turns + 1):
            turn_file = Path(f"{self.output_path}/turn_{turn}.csv")
            
            if turn_file.exists():
                try:
                    df_turn = pd.read_csv(turn_file)
                    all_dataframes.append(df_turn)
                    logger.info(f"  ✅ Turn {turn}: {len(df_turn)} rows loaded")
                except Exception as e:
                    logger.warning(f"  ⚠️  Could not load Turn {turn}: {e}")
            else:
                logger.warning(f"  ⚠️  Turn {turn} file not found: {turn_file}")
        
        if not all_dataframes:
            logger.error("❌ No turn data found to combine")
            return pd.DataFrame()
        
        # Concatenate all turn data
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        logger.info(f"📊 Combined {len(combined_df)} total rows from {len(all_dataframes)} turns")
        
        # 🔥 KEY FIX: Sort by original_id first, then by turn_number
        # This groups all turns for the same conversation together
        combined_df_sorted = combined_df.sort_values(['original_id', 'turn_number']).reset_index(drop=True)
        logger.info(f"📊 Sorted {len(combined_df_sorted)} rows by original_id and turn_number")
        
        # Analyze trajectories
        trajectory_stats = []
        unique_ids = combined_df_sorted['original_id'].unique()
        logger.info(f"📊 Processing {len(unique_ids)} unique conversation trajectories")
        
        # Add trajectory length info
        trajectory_lengths = {}
        for original_id in unique_ids:
            conversation_turns = combined_df_sorted[combined_df_sorted['original_id'] == original_id]
            num_turns = len(conversation_turns)
            trajectory_lengths[original_id] = num_turns
            trajectory_stats.append(num_turns)
        
        # Add trajectory info to DataFrame
        combined_df_sorted['num_turns_in_trajectory'] = combined_df_sorted['original_id'].map(trajectory_lengths)
        
        # Save combined dataset (SORTED BY ID)
        output_path = Path(f"{self.output_path}/combined_all_turns.csv")
        combined_df_sorted.to_csv(output_path, index=False)
        logger.info(f"💾 Combined dataset saved: {output_path} ({len(combined_df_sorted)} total rows)")
        
        # Create and save trajectory summary
        trajectory_summary = []
        for original_id in unique_ids:
            conversation_turns = combined_df_sorted[combined_df_sorted['original_id'] == original_id]
            conversation_turns = conversation_turns.sort_values('turn_number')
            
            summary = {
                'original_id': original_id,
                'ground_truth': conversation_turns.iloc[0]['ground_truth'],
                'original_conversation': conversation_turns.iloc[0]['original_conversation'],
                'num_turns': len(conversation_turns),
                'turns': conversation_turns['turn_number'].tolist()
            }
            trajectory_summary.append(summary)
        
        trajectory_path = Path(f"{self.output_path}/trajectory_summary.json")
        with open(trajectory_path, 'w') as f:
            json.dump(trajectory_summary, f, indent=2, default=str)
        logger.info(f"💾 Trajectory summary saved: {trajectory_path} ({len(trajectory_summary)} trajectories)")
        
        # Print statistics
        logger.info(f"\n📈 TRAJECTORY STATISTICS:")
        logger.info(f"  🔢 Total unique conversations: {len(unique_ids)}")
        logger.info(f"  🔢 Total preference pairs: {len(combined_df_sorted)}")
        logger.info(f"  📊 Average turns per conversation: {sum(trajectory_stats) / len(trajectory_stats):.2f}")
        logger.info(f"  📊 Min turns: {min(trajectory_stats)}")
        logger.info(f"  📊 Max turns: {max(trajectory_stats)}")
        
        # Show turn distribution
        from collections import Counter
        turn_distribution = Counter(trajectory_stats)
        logger.info(f"  📊 Turn distribution:")
        for turns, count in sorted(turn_distribution.items()):
            logger.info(f"    {turns} turns: {count} conversations ({count/len(unique_ids)*100:.1f}%)")
        
        # 🔥 Show sample of the grouped structure
        logger.info(f"\n📋 SAMPLE GROUPED STRUCTURE:")
        sample_ids = list(unique_ids)[:3]
        for sample_id in sample_ids:
            sample_rows = combined_df_sorted[combined_df_sorted['original_id'] == sample_id]
            logger.info(f"  💬 Conversation {sample_id}: {len(sample_rows)} turns")
            for _, row in sample_rows.iterrows():
                chosen_response = row.get('assistant_response_chosen', '')
                logger.info(f"    📝 Turn {row['turn_number']}: '{chosen_response[:50]}...'")
        
        return combined_df_sorted

# =================== CONFIGURATION ===================
class TurnWiseConfig:
    def __init__(self):
        # Dataset and model configuration
        self.dataset = "inspired"
        self.local_model_path_name = "test_epoch5_seed2"
        self.alg = "vanilla"
        self.chunk_size = 5
        self.max_turns = 5
        
        # File paths
        self.csv_input_path = f"../datasets/{self.dataset}/multiturn_form/train.csv"
        self.user_prompt_template_path = "../prompts/test_user_prompt.txt"
        self.output_path = f'{self.dataset}/DPO_entropy_turnwise'
        
        # Model paths
        self.local_model_path = f"/home/sagemaker-user/csbai/multiturn_rl/outputs/{self.alg}/{self.dataset}/{self.local_model_path_name}"
        self.base_model_path = "meta-llama/Llama-3.2-1B-Instruct"
        
        # Generation kwargs
        self.assistant_generation_kwargs = {
            "temperature": 0.8,
            "max_tokens": 512,
            "model": "us.meta.llama3-1-8b-instruct-v1:0",
            "num_retries": 50
        }
        
        self.user_generation_kwargs = {
            "model": "us.anthropic.claude-sonnet-4-20250514-v1:0",
            "temperature": 0.8,
            "max_tokens": 256,
            "num_retries": 50
        }
        
        # Reward evaluation config
        self.reward_config = {
            "weights": [1., -0.1],
            "entropy_model": "us.meta.llama3-1-8b-instruct-v1:0",
            "num_samples": 5,
            "num_items": 5,
            "max_workers": 50,
            "max_tokens": 512,
            "encoding_name": "cl100k_base"
        }
        
        # Other settings
        self.max_total_turns = 10
        self.max_gen_workers = 20
        self.enable_batching = True
        self.use_bedrock_assistant = (self.alg == "vanilla")


# =================== EXECUTION FUNCTIONS ===================

async def execute_turn_1(config: TurnWiseConfig = None):
    """Execute Turn 1: Initial queries -> Responses -> Conversations -> Evaluation"""
    if config is None:
        config = TurnWiseConfig()
    
    logger.info("🚀 EXECUTING TURN 1")
    
    # Create conversation config
    conv_config = ConversationConfig(
        assistant_meta_prompt="You are a helpful movie recommendation assistant. Provide personalized movie suggestions based on user preferences and engage in natural conversation about movies.",
        user_meta_prompt="",  # Will be filled from template
        max_total_turns=config.max_turns,
        max_gen_workers=config.max_gen_workers,
        local_model_path=config.local_model_path,
        base_model_path=config.base_model_path,
        assistant_generation_kwargs=config.assistant_generation_kwargs,
        user_generation_kwargs=config.user_generation_kwargs,
        enable_batching=config.enable_batching,
        use_bedrock_assistant=config.use_bedrock_assistant
    )
    
    # Initialize executor
    executor = TurnWiseDPOExecutor(
        base_config=conv_config,
        user_prompt_template_path=config.user_prompt_template_path,
        reward_config=config.reward_config,
        output_path=config.output_path
    )
    
    # Load data
    df = executor.load_csv_data(config.csv_input_path)
    # df = df.head(5)  # For testing - remove for full dataset
    
    start_time = time.time()
    
    # Step 1: Generate initial user queries
    await executor.turn1_step1_generate_initial_queries(df, config.dataset, config.chunk_size, 0)
    step1_time = time.time()
    logger.info(f"⏱️  Turn 1 Step 1 completed in {step1_time - start_time:.2f} seconds")
    
    # Step 2: Generate assistant responses and conversations
    await executor.turn1_step2_generate_completions_and_conversations(config.dataset, config.chunk_size, 0)
    step2_time = time.time()
    logger.info(f"⏱️  Turn 1 Step 2 completed in {step2_time - step1_time:.2f} seconds")
    
    # Step 3: Evaluate conversations
    turn_data = await executor.turn1_step3_evaluate_conversations(config.dataset, config.chunk_size, 0)
    step3_time = time.time()
    logger.info(f"⏱️  Turn 1 Step 3 completed in {step3_time - step2_time:.2f} seconds")
    
    total_time = step3_time - start_time
    logger.info(f"🎉 TURN 1 COMPLETE in {total_time:.2f} seconds: {len(turn_data)} data points")
    
    return turn_data

async def execute_turn_n(turn: int, config: TurnWiseConfig = None):
    """Execute Turn N (2-5): User response -> Responses -> Conversations -> Evaluation"""
    if config is None:
        config = TurnWiseConfig()
    
    logger.info(f"🚀 EXECUTING TURN {turn}")
    
    # Create conversation config
    conv_config = ConversationConfig(
        assistant_meta_prompt="You are a helpful movie recommendation assistant. Provide personalized movie suggestions based on user preferences and engage in natural conversation about movies.",
        user_meta_prompt="",  # Will be filled from template
        max_total_turns=config.max_turns,
        max_gen_workers=config.max_gen_workers,
        local_model_path=config.local_model_path,
        base_model_path=config.base_model_path,
        assistant_generation_kwargs=config.assistant_generation_kwargs,
        user_generation_kwargs=config.user_generation_kwargs,
        enable_batching=config.enable_batching,
        use_bedrock_assistant=config.use_bedrock_assistant
    )
    
    # Initialize executor
    executor = TurnWiseDPOExecutor(
        base_config=conv_config,
        user_prompt_template_path=config.user_prompt_template_path,
        output_path=config.output_path,
        reward_config=config.reward_config
    )
    
    start_time = time.time()
    
    # Step 1: Choose better completion, generate user response, and new assistant responses
    await executor.turn_step1_generate_user_response_and_completions(turn, config.dataset, config.chunk_size, 0)
    step1_time = time.time()
    logger.info(f"⏱️  Turn {turn} Step 1 completed in {step1_time - start_time:.2f} seconds")
    
    # Step 2: Generate conversations
    await executor.turn_step2_generate_conversations(turn, config.dataset, config.chunk_size, 0)
    step2_time = time.time()
    logger.info(f"⏱️  Turn {turn} Step 2 completed in {step2_time - step1_time:.2f} seconds")
    
    # Step 3: Evaluate conversations
    turn_data = await executor.turn_step3_evaluate_conversations(turn, config.dataset, config.chunk_size, 0)
    step3_time = time.time()
    logger.info(f"⏱️  Turn {turn} Step 3 completed in {step3_time - step2_time:.2f} seconds")
    
    total_time = step3_time - start_time
    logger.info(f"🎉 TURN {turn} COMPLETE in {total_time:.2f} seconds: {len(turn_data)} data points")
    
    return 


# =================== USAGE EXAMPLES ===================

def create_config(dataset="inspired", model="llama3-2-1b-instruct", alg="vanilla", local_model_path_name="test_epoch5_seed2", 
                 max_turns=5, chunk_size=30, **kwargs):
    """Create custom configuration"""
    config = TurnWiseConfig()
    config.dataset = dataset
    config.model = model
    config.alg = alg
    config.local_model_path_name = local_model_path_name
    config.max_turns = max_turns
    config.chunk_size = chunk_size
    
    # Update additional parameters
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Recalculate paths
    config.csv_input_path = f"../datasets/{config.dataset}/multiturn_form/train.csv"
    config.local_model_path = f"/home/sagemaker-user/csbai/multiturn_rl/outputs/{config.alg}/{config.dataset}/{config.local_model_path_name}"
    config.use_bedrock_assistant = (config.alg == "vanilla")
    config.output_path = f'{dataset}/DPO_entropy_turnwise/{model}/'
    
    return config

custom_config = create_config(
    dataset="redial",
    model="llama3-1-8b-instruct",
    max_turns=10,  # Smaller chunks for testing
    chunk_size=30
)

async def main():
    turn1_result = await execute_turn_1(custom_config)

    print("🚀 Starting Turn 2...")
    turn2_result = await execute_turn_n(2, custom_config)

    print("🚀 Starting Turn 3...")
    turn3_result = await execute_turn_n(3, custom_config)

    print("🚀 Starting Turn 4...")
    turn4_result = await execute_turn_n(4, custom_config)

    print("🚀 Starting Turn 4...")
    turn5_result = await execute_turn_n(5, custom_config)

if __name__ == "__main__":
        asyncio.run(main())