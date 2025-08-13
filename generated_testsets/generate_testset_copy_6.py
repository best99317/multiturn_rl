import asyncio
import csv
import json
import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
import nest_asyncio
from pathlib import Path
import ast

# Enable nested event loops for Jupyter
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Import your existing conversation generator
import sys
sys.path.append('/home/sagemaker-user/csbai/multiturn_rl')
from simulators.conversation_simulator import ConversationConfig, MultiTurnConversationGenerator

class MovieConversationGenerator:
    """Generator for movie recommendation conversations based on CSV data"""
    
    def __init__(self, base_config: ConversationConfig, user_prompt_template_path: str, terminal_signal: str = "[[TERMINATE CHAT]]"):
        self.base_config = base_config
        self.terminal_signal = terminal_signal
        
        # Load the user meta prompt template
        with open(user_prompt_template_path, 'r') as f:
            self.user_prompt_template = f.read()
    
    def load_csv_data(self, csv_path: str) -> pd.DataFrame:
        """Load and parse the CSV data"""
        df = pd.read_csv(csv_path)
        
        # Parse the conversation column (assuming it's stored as string representation of list)
        def parse_conversation(conv_str):
            try:
                # Handle the conversation string - it might be a JSON string or Python literal
                if isinstance(conv_str, str):
                    return ast.literal_eval(conv_str)
                return conv_str
            except (ValueError, SyntaxError) as e:
                logger.warning(f"Failed to parse conversation: {e}")
                return []
        
        df['conversation_parsed'] = df['conversation'].apply(parse_conversation)
        return df
    
    def create_custom_config(self, conversation: List[Dict], ground_truth: str) -> ConversationConfig:
        """Create a custom config with conversation-specific user meta prompt"""
        # Convert conversation to a readable format
        conv_text = ""
        for msg in conversation:
            role = msg['role']
            content = msg['content']
            if role == 'user':
                conv_text += f"User: {content}\n"
            elif role == 'assistant':
                conv_text += f"Assistant: {content}\n"
        
        # Fill in the template with the conversation data
        custom_user_prompt = self.user_prompt_template.format(
            conversation=conv_text.strip(),
            ground_truth=ground_truth,
            terminal_signal=self.terminal_signal,
            chat_history="{chat_history}"  # Keep this placeholder for UserSimulator
        )
        
        # Create a new config with the custom prompt
        custom_config = ConversationConfig(
            assistant_meta_prompt=self.base_config.assistant_meta_prompt,
            user_meta_prompt=custom_user_prompt,
            max_total_turns=self.base_config.max_total_turns,
            max_gen_workers=self.base_config.max_gen_workers,
            local_model_path=self.base_config.local_model_path,
            base_model_path=self.base_config.base_model_path,
            assistant_generation_kwargs=self.base_config.assistant_generation_kwargs,
            user_generation_kwargs=self.base_config.user_generation_kwargs,
            enable_batching=self.base_config.enable_batching
        )
        
        return custom_config
    
    async def generate_single_movie_conversation(self, dialog_id: str, conversation: List[Dict], ground_truth: str) -> Optional[Dict]:
        """Generate a single conversation based on the movie data"""
        try:
            logger.info(f"Generating conversation for dialog_id: {dialog_id}")
            
            # Create custom config for this specific conversation
            custom_config = self.create_custom_config(conversation, ground_truth)
            
            # Create a new generator with the custom config
            generator = MultiTurnConversationGenerator(custom_config)
            
            # Use a simple initial prompt since all context is in user_meta_prompt
            initial_prompt = "I'm looking for a movie recommendation."
            
            # Generate the conversation
            generated_conv = await generator.generate_single_conversation(initial_prompt)
            
            if generated_conv:
                result = {
                    'dialog_id': dialog_id,
                    'ground_truth': ground_truth,
                    'original_conversation': conversation,
                    'generated_conversation': generated_conv,
                    'status': 'success'
                }
                logger.info(f"Successfully generated conversation for {dialog_id}")
                return result
            else:
                logger.warning(f"Failed to generate conversation for {dialog_id}")
                return {
                    'dialog_id': dialog_id,
                    'ground_truth': ground_truth,
                    'original_conversation': conversation,
                    'generated_conversation': None,
                    'status': 'failed'
                }
                
        except Exception as e:
            logger.error(f"Error generating conversation for {dialog_id}: {str(e)}")
            return {
                'dialog_id': dialog_id,
                'ground_truth': ground_truth,
                'original_conversation': conversation,
                'generated_conversation': None,
                'status': 'error',
                'error': str(e)
            }
    
    async def generate_conversations_batch(self, df: pd.DataFrame, init_prompts: List=None) -> List[Dict]:
        """Generate conversations for all rows in the dataframe"""
        
        total_rows = len(df)
        logger.info(f"Starting batch generation for {total_rows} conversations")
        
        # Create prompts with meta prompt
        batch_configs = []
        if not init_prompts:
            init_prompts = [None] * total_rows
        
       
        for idx, row in df.iterrows():
            # Create custom prompt for this conversation
            conv_text = ""
            for msg in row['conversation_parsed']:
                role = msg['role']
                content = msg['content']
                if role == 'user':
                    conv_text += f"User: {content}\n"
                elif role == 'assistant':
                    conv_text += f"Assistant: {content}\n"
            
            # Fill in the template
            custom_user_prompt = self.user_prompt_template.format(
                conversation=conv_text.strip(),
                ground_truth=row['ground_truth'],
                terminal_signal=self.terminal_signal,
                chat_history="{chat_history}"  # Keep this placeholder for UserSimulator
            )
            
            # Create config for this conversation
            config = {
                'user_meta_prompt': custom_user_prompt,
                'user_generation_kwargs': self.base_config.user_generation_kwargs,
                'initial_prompt': "I'm looking for a movie recommendation.",
                # Add metadata
                'dialog_id': row['dialog_id'],
                'ground_truth': row['ground_truth'],
                'original_conversation': row['conversation_parsed']
            }
            
            batch_configs.append(config)
        
        logger.info(f"📝 Prepared {len(batch_configs)} conversation configs")

        # Create a SINGLE generator for batch processing
        logger.info("🔧 Creating single generator instance (like working code)...")
        generator = MultiTurnConversationGenerator(self.base_config)
        
        # Generate all conversations using the enhanced batch method
        start_time = time.time()
        generated_conversations = await generator.generate_conversations_batch(
            prompts=init_prompts,
            conv_num=total_rows,
            batch_configs=batch_configs
        )
        end_time = time.time()
        
        logger.info(f"Generation completed in {end_time - start_time:.2f} seconds")

        results = []
        for i, (config, generated_conv) in enumerate(zip(batch_configs, generated_conversations)):
            if generated_conv and len(generated_conv) > 0:
                result = {
                    'dialog_id': config['dialog_id'],
                    'ground_truth': config['ground_truth'],
                    'original_conversation': config['original_conversation'],
                    'generated_conversation': generated_conv,
                    'status': 'success'
                }
            else:
                result = {
                    'dialog_id': config['dialog_id'],
                    'ground_truth': config['ground_truth'],
                    'original_conversation': config['original_conversation'],
                    'generated_conversation': None,
                    'status': 'failed'
                }
            results.append(result)
        
        successful = sum(1 for r in results if r['status'] == 'success')
        logger.info(f"🎉 Batch generation complete: {successful}/{total_rows} successful")
        
        return results
    
    def save_results_to_csv(self, results: List[Dict], output_path: str):
        """Save results to CSV file"""
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare data for CSV
        csv_data = []
        for result in results:
            # Convert generated conversation to string for CSV storage
            generated_conv_str = json.dumps(result['generated_conversation']) if result['generated_conversation'] else None
            original_conv_str = json.dumps(result['original_conversation'])
            
            csv_row = {
                'dialog_id': result['dialog_id'],
                'ground_truth': result['ground_truth'],
                'original_conversation': original_conv_str,
                'generated_conversation': generated_conv_str,
                'status': result['status']
            }
            
            # Add error information if present
            if 'error' in result:
                csv_row['error'] = result['error']
            
            csv_data.append(csv_row)
        
        # Write to CSV
        df_output = pd.DataFrame(csv_data)
        df_output.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
        
        # Print summary
        status_counts = df_output['status'].value_counts()
        logger.info(f"Summary: {status_counts.to_dict()}")


async def main():
    """Main function to run the movie conversation generation"""
    
    dataset = "redial"
    local_model_path_name = "test_epoch5_seed2_turn_entropy_lr2e-6/checkpoint-508"
    alg = "DPO"
    model = "llama3-2-1b-instruct"
    # File paths
    csv_input_path = "../datasets/"+dataset+"/multiturn_form/test.csv"
    user_prompt_template_path = "../prompts/test_user_prompt.txt"
    csv_output_path = f"multiturn_test/{alg}/{model}/{dataset}/generated_movie_conversations_turn_entropy_lr2e-6_1.csv"
    
    # Create base configuration
    config = ConversationConfig(
        assistant_meta_prompt="You are a helpful movie recommendation assistant. Provide personalized movie suggestions based on user preferences and engage in natural conversation about movies. Recommend at most one movie at a time.",
        # This will be overridden for each conversation with custom context
        user_meta_prompt="You are a user looking for movie recommendations. Respond naturally based on the conversation context.",
        max_total_turns=10,
        max_gen_workers=20,
        local_model_path=f"/home/sagemaker-user/csbai/multiturn_rl/outputs/{alg}/{dataset}/{model}/{local_model_path_name}",
        base_model_path="meta-llama/Llama-3.2-1B-Instruct",
        assistant_generation_kwargs={
            "temperature": 0.1,
            "max_tokens": 512,
            "model": "us.meta.llama3-2-1b-instruct-v1:0",
            "num_retries": 50
        },
        user_generation_kwargs={
            "model": "us.anthropic.claude-sonnet-4-20250514-v1:0",
            "temperature": 0.8,
            "max_tokens": 256,
            "num_retries": 50
        },
        enable_batching=True,
        use_bedrock_assistant= (alg == "vanilla")
    )
    
    # Initialize generator with template path
    movie_generator = MovieConversationGenerator(
        base_config=config,
        user_prompt_template_path=user_prompt_template_path,
        terminal_signal="[[TERMINATE CHAT]]"
    )
    
    # Load data
    logger.info("Loading CSV data...")
    df = movie_generator.load_csv_data(csv_input_path)
    logger.info(f"Loaded {len(df)} conversations from CSV")
    
    # Generate conversations
    logger.info("Starting conversation generation...")
    start_time = time.time()
    
    results = await movie_generator.generate_conversations_batch(df)
    
    end_time = time.time()
    logger.info(f"Total generation time: {end_time - start_time:.2f} seconds")
    
    # Save results
    logger.info("Saving results...")
    movie_generator.save_results_to_csv(results, csv_output_path)
    
    logger.info("Process complete!")
    
    return results

# Run main process
if __name__ == "__main__":
    print("🚀 Starting main generation process...")
    results = asyncio.run(main())
