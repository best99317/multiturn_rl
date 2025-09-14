#!/bin/bash
export HUGGING_FACE_HUB_TOKEN=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
export WANDB_API_KEY=f5ab2278d3c710ad96e4b6c662e2dbc002fb5eaf
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 WANDB__SERVICE_WAIT=300 torchrun --master_port=56800 --nnodes=1 --nproc_per_node=8 -m finetuning.ppo \
    --dataset_repo datasets/inspired/sft_generated/llama3-2-1b-instruct/turn_entropy_top801.csv \
    --output_dir outputs/PPO/inspired/llama3-2-1b-instruct/test_epoch2_seed2_turn_entropy \
    --seed 2 \
    --model_name outputs/sft_generated/inspired/llama3-2-1b-instruct/test_epoch5_seed2_turn_entropy_lr2e-5 \
    --base_model_name meta-llama/Llama-3.2-1B-Instruct \
    --batch_size 2 \
    --mini_batch_size 1 \
    --save_total_limit 5 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 2 \
    --learning_rate 2e-6 \
    --logging_steps 10 \
    --wandb_entity best99317-purdue-university \
    --wandb_project Multiturn_RL \
    --user_meta_prompt prompts/test_user_prompt.txt \
    --reward_type entropy \
    --use_4bit \
    # --gpu_memory_utilization 0.6 \
    # --model_name meta-llama/Llama-3.2-1B-Instruct \
    # --dataset_repo datasets/inspired/DPO_turnwise/llama3-2-1b-instruct/combined_all_turns.csv \
    # --ref_model_name meta-llama/Llama-3.2-1B-Instruct \
    # --use_vllm \