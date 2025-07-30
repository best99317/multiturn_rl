#!/bin/bash
export HUGGING_FACE_HUB_TOKEN=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
export WANDB_API_KEY=f5ab2278d3c710ad96e4b6c662e2dbc002fb5eaf
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 WANDB__SERVICE_WAIT=300 torchrun --master_port=56800 --nnodes=1 --nproc_per_node=8 -m finetuning.sft \
    --dataset_repo datasets/redial/sft_generated/llama3-2-1b-instruct/train_sft_generated1.csv \
    --output_dir outputs/sft_generated_thd1/redial/llama3-2-1b-instruct/test_epoch5_seed2 \
    --seed 2 \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 3 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 5 \
    --learning_rate 2e-5 \
    --eval_steps 1 \
    --logging_steps 1 \
    --wandb_entity best99317-purdue-university \
    --wandb_project Multiturn_RL \
    --use_lora  \
    # --use_4bit