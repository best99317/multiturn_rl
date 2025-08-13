#!/bin/bash
export HUGGING_FACE_HUB_TOKEN=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
export WANDB_API_KEY=f5ab2278d3c710ad96e4b6c662e2dbc002fb5eaf
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 WANDB__SERVICE_WAIT=300 torchrun --master_port=56800 --nnodes=1 --nproc_per_node=8 -m finetuning.dpo \
    --dataset_repo datasets/redial/DPO_turnwise/llama3-2-1b-instruct/combined_all_turns.csv \
    --output_dir outputs/DPO/redial/llama3-2-1b-instruct/test_epoch1_seed2_collab_lr2e-6 \
    --seed 2 \
    --model_name outputs/sft_generated/redial/llama3-2-1b-instruct/test_epoch5_seed2_collab_lr2e-5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --save_total_limit 10 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --learning_rate 2e-6 \
    --eval_steps 20 \
    --logging_steps 1 \
    --wandb_entity best99317-purdue-university \
    --wandb_project Multiturn_RL \
    --use_lora  \
    --eval_ratio 0.02 \
    # --use_4bit
    # --model_name meta-llama/Llama-3.2-1B-Instruct \
    # --dataset_repo datasets/inspired/DPO_turnwise/llama3-2-1b-instruct/combined_all_turns.csv \