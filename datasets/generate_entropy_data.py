import pandas as pd
import json
import asyncio
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import Counter, defaultdict
import re
import time
import math
from concurrent.futures import ThreadPoolExecutor
import sys
import os

# Automatically add the root project directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from simulators.user_simulator import UserSimulator
from utils.bedrock_call import bedrock_call

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

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
    
    def calculate_entropy(self, responses: List[List[str]]) -> float:
        """
        Calculate entropy based on response similarity/diversity across the 5 samples
        
        Args:
            responses: List of response strings (should be 5 responses)
            
        Returns:
            Entropy value based on response diversity
        """
        all_responses = []
        for response in responses:
            all_responses.extend(response)

        if not all_responses or len(all_responses) < 2:
            return 0.0
        
        
        # Count frequency of each unique response
        response_counts = Counter(all_responses)
        total_responses = len(all_responses)
        
        # Calculate entropy based on response distribution
        entropy = 0.0
        for count in response_counts.values():
            probability = count / total_responses
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy

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

    def calculate_entropy_reduction(self, conversation: List[Dict], turn: int) -> Tuple[float, float]:
        """
        Calculate entropy reduction metrics for a specific turn
        
        Args:
            conversation: Full conversation list
            turnr: The turn to analyze (1-indexed)
            
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

class CSVUserResponseGenerator:
    def __init__(self, 
                user_meta_prompt: str, 
                terminal_signal: str = "[[TERMINATE CHAT]]", 
                user_model: str = "us.anthropic.claude-sonnet-4-20250514-v1:0", 
                assistant_model: str = "us.meta.llama3-2-1b-instruct-v1:0", 
                num_samples: int = 5, num_items: int = 5, **llm_kwargs):
        """
        Initialize the CSV User Response Generator
        
        Args:
            user_meta_prompt: The meta prompt for user simulation
            user_model: The model to use for user generation
            **llm_kwargs: Additional LLM parameters
        """

        self.user_prompt_template = user_meta_prompt
        
        self.terminal_signal = terminal_signal

        self.user_simulator = UserSimulator(
            user_meta_prompt=self.user_prompt_template,
            model=user_model,
            **llm_kwargs
        )

        self.entropy_analyzer = ConversationEntropyAnalyzer(model_id=assistant_model, num_samples=num_samples, num_items=num_items)

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'entropy_analyzer') and self.entropy_analyzer:
            self.entropy_analyzer.cleanup()
        if hasattr(self, 'user_simulator') and hasattr(self.user_simulator, '__del__'):
            # UserSimulator has its own __del__ method, let it handle cleanup naturally
            pass

    def load_csv_data(self, csv_path: pd.DataFrame) -> pd.DataFrame:
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
                
                if isinstance(conv_str, str):
                    return conv_str.strip()
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

        # Parse both conversation columns
        if 'response_1_conversation' in df.columns:
            df['response_1_conversation_parsed'] = df['response_1_conversation'].apply(parse_conversation)
        if 'response_2_conversation' in df.columns:
            df['response_2_conversation_parsed'] = df['response_2_conversation'].apply(parse_conversation)
        if 'original_conversation' in df.columns:
            df['original_conversation_parsed'] = df['original_conversation'].apply(parse_conversation)
        
        # Filter out rows with empty conversations
        original_count = len(df)
        df = df[df['original_conversation_parsed'].apply(lambda x: len(x) > 0)]
        filtered_count = len(df)
        
        if filtered_count < original_count:
            logger.info(f"Filtered out {original_count - filtered_count} rows with empty/invalid conversations")
            
        return df

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

    def _last_message_is_assistant(self, conversation: List[Dict[str, str]]) -> bool:
        """
        Check if the last message in conversation is from assistant
        
        Args:
            conversation: List of message dictionaries
            
        Returns:
            True if last message is from assistant
        """
        if not conversation:
            return False
        return conversation[-1].get('role') == 'assistant'

    async def _generate_user_response(self, conversation: List[Dict[str, str]], state: Dict) -> Optional[str]:
        """
        Generate a user response given the conversation history
        
        Args:
            conversation: List of message dictionaries
            
        Returns:
            Generated user response or None if generation fails
        """
        try:
            # Use the UserSimulator to generate response
            custom_user_prompt = self._create_user_prompt_context(state)
            self.user_simulator.user_meta_prompt = custom_user_prompt
            response = await self.user_simulator.async_call(conversation)
            
            if response:
                return response.strip()
            else:
                logger.info("Error: Generated response is None")
                return None
                
        except Exception as e:
            logger.error(f"Error generating user response: {e}")
            return None

    async def _calculate_entropy_for_conversation(self, conversation_data: List[Dict], conversation_name: str, turn_number: int, state_id: str) -> Tuple[float, float]:
        """
        Calculate entropy metrics for a single conversation (async wrapper)
        
        Args:
            conversation_data: The conversation to analyze
            conversation_name: Name for logging (e.g., "response_1_conversation")
            turn_number: Turn number to analyze
            state_id: State ID for logging
            
        Returns:
            Tuple of (entropy_reduced, expected_entropy_reduced)
        """
        if not conversation_data:
            return 0.0, 0.0
            
        try:
            logger.info(f"Calculating entropy metrics for {conversation_name} (ID: {state_id}, Turn: {turn_number})")
            
            # Run entropy calculation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            entropy_reduced, expected_entropy_reduced = await loop.run_in_executor(
                None,  # Use default executor
                self.entropy_analyzer.calculate_entropy_reduction,
                conversation_data,
                turn_number
            )
            
            return entropy_reduced, expected_entropy_reduced
            
        except Exception as e:
            logger.error(f"Error calculating entropy for {conversation_name} (ID: {state_id}): {e}")
            return 0.0, 0.0

    async def process_csv_row(self, row: pd.Series) -> Dict:
        """
        Process a single CSV row and generate user responses if needed
        
        Args:
            row: A pandas Series representing a CSV row
            
        Returns:
            Dictionary with original data plus generated responses
        """
        result = row.to_dict()

        turn_number = row.get('turn_number', 1)
        state_id = row.get('id', 'unknown')
        original_conversation_parsed = row.get('original_conversation_parsed', [])
        if isinstance(original_conversation_parsed, str):
            try:
                original_conversation_parsed = json.loads(original_conversation_parsed)
            except Exception as e:
                logger.warning(f"Failed to load original_conversation_parsed for {state_id}: {e}. Wrong Format, First 100 chars: {str(original_conversation_parsed)[:100]}")

        state = {
            'id': state_id,
            'ground_truth': row.get('ground_truth', ''),
            'original_conversation': original_conversation_parsed  # Fallback
        }
        
        # Process response_1_conversation using parsed data
        conv1 = row.get('response_1_conversation_parsed', [])
        if isinstance(conv1, str):
            try:
                conv1 = json.loads(conv1)
            except Exception as e:
                logger.warning(f"Failed to load conversation 1 for {state_id}: {e}. Wrong Format, First 100 chars: {str(conv1)[:100]}")
            
        if conv1 and self._last_message_is_assistant(conv1):
            logger.info(f"Generating user response for response_1_conversation (ID: {row.get('id', 'unknown')})")
            user_response_1 = await self._generate_user_response(conv1, state)
            if user_response_1:
                # Add user response to conversation
                updated_conv1 = conv1 + [{"role": "user", "content": user_response_1}]
                result['response_1_conversation_extended'] = json.dumps(updated_conv1)
                conv1_for_entropy = updated_conv1
            else:
                result['response_1_conversation_extended'] = json.dumps(conv1)
                conv1_for_entropy = conv1
        else:
            result['response_1_conversation_extended'] = row.get('response_1_conversation', '')
            conv1_for_entropy = conv1
        
        # Process response_2_conversation using parsed data
        conv2 = row.get('response_2_conversation_parsed', [])
        if isinstance(conv2, str):
            try:
                conv2 = json.loads(conv2)
            except Exception as e:
                logger.warning(f"Failed to load conversation 2 for {state_id}: {e}. Wrong Format, First 100 chars: {str(conv2)[:100]}")
        if conv2 and self._last_message_is_assistant(conv2):
            logger.info(f"Generating user response for response_2_conversation (ID: {row.get('id', 'unknown')})")
            user_response_2 = await self._generate_user_response(conv2, state)
            if user_response_2:
                # Add user response to conversation
                updated_conv2 = conv2 + [{"role": "user", "content": user_response_2}]
                result['response_2_conversation_extended'] = json.dumps(updated_conv2)
                conv2_for_entropy = updated_conv2
            else:
                result['response_2_conversation_extended'] = json.dumps(conv2)
                conv2_for_entropy = conv2
        else:
            result['response_2_conversation_extended'] = row.get('response_2_conversation', '')
            conv2_for_entropy = conv2

        # Calculate entropy metrics for both conversations in parallel
        entropy_tasks = []
        
        # Add entropy calculation tasks
        if conv1_for_entropy:
            entropy_tasks.append(
                self._calculate_entropy_for_conversation(
                    conv1_for_entropy, "response_1_conversation", turn_number, state_id
                )
            )
        else:
            # Create a dummy coroutine that returns (0.0, 0.0)
            async def dummy_entropy():
                return 0.0, 0.0
            entropy_tasks.append(dummy_entropy())
            
        if conv2_for_entropy:
            entropy_tasks.append(
                self._calculate_entropy_for_conversation(
                    conv2_for_entropy, "response_2_conversation", turn_number, state_id
                )
            )
        else:
            # Create a dummy coroutine that returns (0.0, 0.0)
            async def dummy_entropy():
                return 0.0, 0.0
            entropy_tasks.append(dummy_entropy())
        
        # Run entropy calculations in parallel
        entropy_results = await asyncio.gather(*entropy_tasks, return_exceptions=True)
        
        # Process results
        if len(entropy_results) >= 1 and not isinstance(entropy_results[0], Exception):
            entropy_reduced_1, expected_entropy_reduced_1 = entropy_results[0]
        else:
            entropy_reduced_1, expected_entropy_reduced_1 = 0.0, 0.0
            if isinstance(entropy_results[0], Exception):
                logger.error(f"Entropy calculation failed for conv1: {entropy_results[0]}")
        
        if len(entropy_results) >= 2 and not isinstance(entropy_results[1], Exception):
            entropy_reduced_2, expected_entropy_reduced_2 = entropy_results[1]
        else:
            entropy_reduced_2, expected_entropy_reduced_2 = 0.0, 0.0
            if len(entropy_results) >= 2 and isinstance(entropy_results[1], Exception):
                logger.error(f"Entropy calculation failed for conv2: {entropy_results[1]}")
        
        # Add entropy results to output
        result['response_1_turn_entropy_reduced'] = entropy_reduced_1
        result['response_2_turn_entropy_reduced'] = entropy_reduced_2
        result['response_1_conv_entropy_reduced'] = expected_entropy_reduced_1
        result['response_2_conv_entropy_reduced'] = expected_entropy_reduced_2
        
        return result

    async def process_csv_file(self, input_file: str, output_file: str, max_concurrent: int = 10) -> None:
        """
        Process entire CSV file and generate user responses
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output CSV file
            max_concurrent: Maximum number of concurrent processing tasks
        """
        try:
            # Read and parse CSV file using your existing method
            logger.info(f"Reading CSV file: {input_file}")
            df = self.load_csv_data(input_file)
            logger.info(f"Loaded {len(df)} rows")
            
            # Process rows in batches to control concurrency
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def process_row_with_semaphore(row):
                async with semaphore:
                    return await self.process_csv_row(row)
            
            # Create tasks for all rows
            tasks = [process_row_with_semaphore(row) for _, row in df.iterrows()]
            
            # Process all tasks
            logger.info("Processing rows...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            output_df = pd.DataFrame(results)

            # Remove specified columns before saving
            columns_to_remove = [
                'chosen_conversation',
                'rejected_conversation', 
                'chosen_score',
                'rejected_score',
                'chosen_reasoning',
                'rejected_reasoning',
                'response_1_reasoning',
                'response_2_reasoning',
                'response_1_conversation_parsed',
                'response_2_conversation_parsed',
                'original_conversation_parsed',
                'assistant_response_chosen',
                'assistant_response_rejected'
            ]
            
            # Only remove columns that actually exist in the DataFrame
            existing_columns_to_remove = [col for col in columns_to_remove if col in output_df.columns]
            if existing_columns_to_remove:
                logger.info(f"Removing columns: {existing_columns_to_remove}")
                output_df = output_df.drop(columns=existing_columns_to_remove)
            
            # Save to CSV
            logger.info(f"Saving results to: {output_file}")
            output_df.to_csv(output_file, index=False)
            logger.info("Processing complete!")
            
        except Exception as e:
            logger.error(f"Error processing CSV file: {e}")
            raise
    
    async def reprocess_failed_entropy_rows(self, csv_file_path: str, max_concurrent: int = 10) -> None:
        """
        Reprocess rows with failed entropy calculations (0.0 values)
        
        Args:
            csv_file_path: Path to the CSV file to reprocess
            max_concurrent: Maximum number of concurrent processing tasks
        """
        try:
            logger.info(f"Reading CSV file for entropy reprocessing: {csv_file_path}")
            df = pd.read_csv(csv_file_path)
            logger.info(f"Loaded {len(df)} rows")
            # Parse conversations if not already parsed
            # if 'response_1_conversation_parsed' not in df.columns:
            df = self.load_csv_data(csv_file_path)  # This will add parsed columns
            
            # Find rows where any entropy calculation failed (has 0.0 values)
            entropy_columns = [
                'response_1_turn_entropy_reduced',
                'response_2_turn_entropy_reduced', 
                'response_1_conv_entropy_reduced',
                'response_2_conv_entropy_reduced',
            ]
            
            # Check which columns actually exist
            existing_entropy_cols = [col for col in entropy_columns if col in df.columns]
            if not existing_entropy_cols:
                logger.warning("No entropy columns found in CSV. Nothing to reprocess.")
                return
            
            # Find rows that need reprocessing (any entropy column is 0.0)
            mask = df[existing_entropy_cols].eq(0.0).any(axis=1)
            failed_indices = df[mask].index.tolist()
            
            if len(failed_indices) == 0:
                logger.info("No failed entropy calculations found. Nothing to reprocess.")
                return
                
            logger.info(f"Found {len(failed_indices)} rows with failed entropy calculations")
            
            
            # Process failed rows using existing process_csv_row method
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def reprocess_single_row(idx):
                async with semaphore:
                    try:
                        # Use existing process_csv_row method
                        row = df.iloc[idx]
                        updated_row_data = await self.process_csv_row(row)
                        return idx, updated_row_data
                    except Exception as e:
                        logger.error(f"Error reprocessing row {idx}: {e}")
                        return idx, None
            
            # Create tasks for all failed rows
            tasks = [reprocess_single_row(idx) for idx in failed_indices]
            
            # Process all tasks
            logger.info(f"Reprocessing entropy for {len(tasks)} rows...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update the DataFrame with new results
            updated_count = 0
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Exception during reprocessing: {result}")
                    continue
                
                idx, updated_row_data = result
                if updated_row_data is None:
                    continue
                
                # Update only the entropy columns in the original dataframe
                for col in existing_entropy_cols:
                    if col in updated_row_data:
                        df.at[idx, col] = updated_row_data[col]
                        
                updated_count += 1
            
            # Remove specified columns before saving
            columns_to_remove = [
                'response_1_conversation_parsed',
                'response_2_conversation_parsed',
                'original_conversation_parsed'
            ]
            
            # Only remove columns that actually exist in the DataFrame
            existing_columns_to_remove = [col for col in columns_to_remove if col in df.columns]
            if existing_columns_to_remove:
                logger.info(f"Removing columns: {existing_columns_to_remove}")
                df = df.drop(columns=existing_columns_to_remove)

            # Save the updated CSV
            logger.info(f"Saving updated CSV with {updated_count} reprocessed rows...")
            df.to_csv(csv_file_path, index=False)
            logger.info(f"Successfully reprocessed and saved {updated_count} rows to {csv_file_path}")
            
        except Exception as e:
            logger.error(f"Error during entropy reprocessing: {e}")
            raise

async def main(dataset: str):
    alg = "vanilla"
    model = "llama3-2-1b-instruct"
    
    # File paths
    csv_input_path = f"/home/sagemaker-user/csbai/multiturn_rl/datasets/{dataset}/DPO_turnwise/{model}/combined_all_turns.csv"
    user_prompt_template_path = "../prompts/test_user_prompt.txt"
    csv_output_path = f"/home/sagemaker-user/csbai/multiturn_rl/datasets/{dataset}/DPO_entropy/{model}/combined_all_turns.csv"
    
    # Load user prompt template
    try:
        with open(user_prompt_template_path, 'r') as f:
            user_meta_prompt = f.read().strip()
    except FileNotFoundError:
        logger.warning(f"User prompt template not found at {user_prompt_template_path}, using default")
        user_meta_prompt = """
        You are a user looking for movie recommendations. Respond naturally based on the conversation context.
        
        Conversation History:
        {chat_history}
        
        Provide only the user's next message. Do not include any role labels or formatting.
        If you think the conversation should end naturally, respond with "[[TERMINATE CHAT]]".
        """
    
    # Initialize the generator
    generator = CSVUserResponseGenerator(
        user_meta_prompt=user_meta_prompt,
        user_model="us.anthropic.claude-sonnet-4-20250514-v1:0",
        assistant_model="us.meta.llama3-2-1b-instruct-v1:0",
        temperature=0.8,
        max_tokens=256,
        num_samples=10,
        num_items=3
    )
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
    
    # Process the CSV file
    await generator.process_csv_file(
        input_file=csv_input_path,
        output_file=csv_output_path,
        max_concurrent=50 # Adjust based on your rate limits
    )

async def reprocess_failed_entropy(dataset: str):
    """Reprocess rows with failed entropy calculations"""
    alg = "vanilla"
    model = "llama3-2-1b-instruct"
    user_prompt_template_path = "../prompts/test_user_prompt.txt"
    csv_output_path = f"/home/sagemaker-user/csbai/multiturn_rl/datasets/{dataset}/DPO_entropy/{model}/combined_all_turns.csv"
    
    logger.info("Reprocessing failed entropy calculations...")

    # Load user prompt template
    try:
        with open(user_prompt_template_path, 'r') as f:
            user_meta_prompt = f.read().strip()
    except FileNotFoundError:
        logger.warning(f"User prompt template not found at {user_prompt_template_path}, using default")
        user_meta_prompt = """
        You are a user looking for movie recommendations. Respond naturally based on the conversation context.
        
        Conversation History:
        {chat_history}
        
        Provide only the user's next message. Do not include any role labels or formatting.
        If you think the conversation should end naturally, respond with "[[TERMINATE CHAT]]".
        """
    
    generator = CSVUserResponseGenerator(
            user_meta_prompt=user_meta_prompt,
            user_model="us.anthropic.claude-sonnet-4-20250514-v1:0",
            assistant_model="us.meta.llama3-2-1b-instruct-v1:0",
            temperature=0.8,
            max_tokens=256,
            num_samples=10,
            num_items=3
        )

    try:
        # Reprocess failed entropy calculations
        await generator.reprocess_failed_entropy_rows(
            csv_file_path=csv_output_path,
            max_concurrent=1  # Lower concurrency for reprocessing
        )
    except Exception as e:
        logger.error(f"Error in entropy reprocessing: {e}")
        raise
    finally:
        generator.cleanup()
        logger.info("Reprocessing cleanup completed")

def process_entropy_csv(input_file, output_file):
        """
        Process CSV file to choose responses based on entropy score and create new columns.
        
        Args:
            input_file (str): Path to input CSV file
            output_file (str): Path to output CSV file
        """
        
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        def safe_eval_dict(dict_str):
            """Safely evaluate dictionary string to actual dictionary"""
            if pd.isna(dict_str):
                return {}
            try:
                # Try parsing as JSON first
                if isinstance(dict_str, str):
                    return json.loads(dict_str.replace("'", '"'))
                else:
                    return dict_str
            except:
                try:
                    # Fallback to ast.literal_eval
                    return ast.literal_eval(dict_str)
                except:
                    return {}
        
        def calculate_entropy_score(conv_entropy_reduced, token_efficiency):
            """Calculate entropy score = conv_entropy_reduced - 0.1 * token_efficiency"""
            if pd.isna(conv_entropy_reduced) or pd.isna(token_efficiency):
                return float('-inf')  # Return very low score for missing data
            return conv_entropy_reduced - 0.1 * token_efficiency
        
        def create_entropy_score_dict(conv_entropy_reduced, token_efficiency, entropy_score):
            """Create entropy score dictionary in the required format"""
            return {
                "conv_entropy_reduced": float(conv_entropy_reduced) if not pd.isna(conv_entropy_reduced) else 0.0,
                "token_efficiency": float(token_efficiency) if not pd.isna(token_efficiency) else 0.0,
                "combined": float(entropy_score) if not pd.isna(entropy_score) else 0.0
            }
        
        # Initialize new columns
        df['assistant_response_chosen'] = ''
        df['assistant_response_rejected'] = ''
        df['chosen_conversation'] = ''
        df['rejected_conversation'] = ''
        df['chosen_entropy_score'] = ''
        df['rejected_entropy_score'] = ''
        
        # Process each row
        for idx, row in df.iterrows():
            # Parse score dictionaries
            score_1 = safe_eval_dict(row['response_1_score'])
            score_2 = safe_eval_dict(row['response_2_score'])
            
            # Get token efficiencies
            token_eff_1 = score_1.get('token_efficiency', 0.0) if score_1 else 0.0
            token_eff_2 = score_2.get('token_efficiency', 0.0) if score_2 else 0.0
            
            # Get conv entropy reduced values
            conv_entropy_1 = row['response_1_conv_entropy_reduced']
            conv_entropy_2 = row['response_2_conv_entropy_reduced']
            
            # Calculate entropy scores
            entropy_score_1 = calculate_entropy_score(conv_entropy_1, token_eff_1)
            entropy_score_2 = calculate_entropy_score(conv_entropy_2, token_eff_2)
            
            # Choose based on higher entropy score
            if entropy_score_1 >= entropy_score_2:
                # Response 1 is chosen
                df.at[idx, 'assistant_response_chosen'] = row['assistant_response_1']
                df.at[idx, 'assistant_response_rejected'] = row['assistant_response_2']
                df.at[idx, 'chosen_conversation'] = row['response_1_conversation_extended']
                df.at[idx, 'rejected_conversation'] = row['response_2_conversation_extended']
                
                chosen_score_dict = create_entropy_score_dict(conv_entropy_1, token_eff_1, entropy_score_1)
                rejected_score_dict = create_entropy_score_dict(conv_entropy_2, token_eff_2, entropy_score_2)
            else:
                # Response 2 is chosen
                df.at[idx, 'assistant_response_chosen'] = row['assistant_response_2']
                df.at[idx, 'assistant_response_rejected'] = row['assistant_response_1']
                df.at[idx, 'chosen_conversation'] = row['response_2_conversation_extended']
                df.at[idx, 'rejected_conversation'] = row['response_1_conversation_extended']
                
                chosen_score_dict = create_entropy_score_dict(conv_entropy_2, token_eff_2, entropy_score_2)
                rejected_score_dict = create_entropy_score_dict(conv_entropy_1, token_eff_1, entropy_score_1)
            
            # Convert dictionaries to JSON strings
            df.at[idx, 'chosen_entropy_score'] = json.dumps(chosen_score_dict)
            df.at[idx, 'rejected_entropy_score'] = json.dumps(rejected_score_dict)
            
            # Print progress for large files
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1} rows...")
        
        # Save the processed DataFrame
        df.to_csv(output_file, index=False)
        print(f"Processing complete! Output saved to: {output_file}")
        
        # Print some statistics
        print(f"\nProcessing Statistics:")
        print(f"Total rows processed: {len(df)}")
        print(f"Columns in output file: {len(df.columns)}")
        
        return df

# Run the script
if __name__ == "__main__":
    dataset = "redial"
    asyncio.run(main(dataset))
    asyncio.run(reprocess_failed_entropy(dataset))
    # Replace with your actual file paths
    
    model = "llama3-2-1b-instruct"
    for i in range(50):
        print("dataset: ", dataset)
    input_file = f"/home/sagemaker-user/csbai/multiturn_rl/datasets/{dataset}/DPO_entropy/{model}/combined_all_turns.csv"
    output_file = f"/home/sagemaker-user/csbai/multiturn_rl/datasets/{dataset}/DPO_entropy/{model}/DPO_entropy_reduced.csv"

    try:
        processed_df = process_entropy_csv(input_file, output_file)
        
        # Display first few rows to verify
        print("\nFirst 3 rows of new columns:")
        new_columns = ['assistant_response_chosen', 'assistant_response_rejected', 
                        'chosen_conversation', 'rejected_conversation', 
                        'chosen_entropy_score', 'rejected_entropy_score']
        
        for col in new_columns:
            if col in processed_df.columns:
                print(f"\n{col}:")
                print(processed_df[col].head(3).to_string())
        
    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_file}'")
        print("Please make sure the file exists and update the file path.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    