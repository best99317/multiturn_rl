#!/usr/bin/env python3
from __future__ import annotations

import sys
import os

# Automatically add the root project directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import argparse, json
from typing import Tuple, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import torch.distributed as dist
from peft import PeftConfig, PeftModel, LoraConfig, get_peft_model
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk, Features, Value, Sequence
from trl import DPOConfig, DPOTrainer
import wandb
import ast
import random
from utils.fixseed import fix_all_seeds


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Parameter-free multiturn DPO trainer")

    # Data / paths
    p.add_argument("--dataset_repo", type=str, required=True)
    p.add_argument("--eval_ratio",   type=float, default=0.1)
    p.add_argument("--output_dir",   type=str, required=True)
    p.add_argument("--resume_ckpt_dir", type=str, default=None)

    # Base / adapter models
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--peft_r",     type=int,   default=32)
    p.add_argument("--peft_alpha", type=int,   default=16)
    p.add_argument("--peft_dropout", type=float, default=0.1)
    p.add_argument("--target_modules",
                   type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # Optim & schedule
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--warmup_ratio", type=float, default=0)        
    p.add_argument("--logging_steps", type=int, default=1)           
    p.add_argument("--max_prompt_length", type=int, default=4096) 
    p.add_argument("--max_new_tokens", type=int, default=2048) 
    # p.add_argument("--minimum_gap", type=float, default=0.02) 

    # Precision / hardware
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_lora", action="store_true", default=False)
    p.add_argument("--use_4bit", action="store_true", default=False)

    # Tracking
    p.add_argument("--wandb_project", type=str)
    p.add_argument("--wandb_entity",  type=str)
    p.add_argument("--push_to_hub",   action="store_true")
    p.add_argument("--hf_org",        type=str)

    # Optional JSON/YAML override
    p.add_argument("--config_file", type=str)

    args = p.parse_args()
    if args.config_file:
        with open(args.config_file) as f:
            override = json.load(f) if args.config_file.endswith(".json") else \
                       __import__("yaml").safe_load(f)
        for k, v in override.items():
            setattr(args, k, v)
    return args

def _uniform_split(
    full_ds: Dataset,
    *,
    eval_ratio: float,
    n_eval: Optional[int],
    seed: int,
) -> DatasetDict:
    k = n_eval if n_eval is not None else int(eval_ratio * len(full_ds))
    k = min(k, len(full_ds))
    eval_idx = set(random.sample(range(len(full_ds)), k=k))
    train_idx = [i for i in range(len(full_ds)) if i not in eval_idx]

    return DatasetDict(
        {
            "train": full_ds.select(train_idx),
            "eval": full_ds.select(sorted(eval_idx)),
        }
    )

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def load_model_and_tokenizer(
    model_name: str,
    bnb_cfg: Optional[BitsAndBytesConfig],
    lora_cfg: Optional[LoraConfig],
    device: str = "cuda",
    is_eval: bool = False,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    try:
        pc = PeftConfig.from_pretrained(model_name)
        base = AutoModelForCausalLM.from_pretrained(
            pc.base_model_name_or_path,
            device_map={"": device},
            quantization_config=bnb_cfg,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_name, is_trainable=not is_eval)
        tok = AutoTokenizer.from_pretrained(pc.base_model_name_or_path, trust_remote_code=True)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": device},
            quantization_config=bnb_cfg,
            trust_remote_code=True,
        )
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if lora_cfg:
            model = get_peft_model(model, lora_cfg)

    tok.padding_side, tok.pad_token = ("left" if is_eval else "right"), tok.eos_token
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,}/{total:,} ({trainable/total:.2%})")
    return model, tok


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    fix_all_seeds(args.seed)

    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(backend='nccl', init_method=None)
    torch.cuda.set_device(local_rank)
    dist.barrier()


    def parse_conversations_batch(examples):
        """Parse conversation column in batches"""
        
        # More conservative Unicode cleaning - only remove surrogates
        def clean_unicode_conservative(obj):
            if isinstance(obj, str):
                # Only remove surrogate characters, keep everything else
                return ''.join(c for c in obj if not (0xD800 <= ord(c) <= 0xDFFF))
            elif isinstance(obj, list):
                return [clean_unicode_conservative(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: clean_unicode_conservative(v) for k, v in obj.items()}
            else:
                return obj
        
        parsed_prompts = []
        chosen_responses = []
        rejected_responses = []
        
        for i in range(len(examples['prompt'])):
            try:
                # Parse the prompt (conversation history)
                prompt = examples['prompt'][i]
                if isinstance(prompt, str):
                    parsed_prompt = ast.literal_eval(prompt)
                else:
                    parsed_prompt = prompt
                
                # Clean Unicode
                cleaned_prompt = clean_unicode_conservative(parsed_prompt)
                parsed_prompts.append(cleaned_prompt)
                
                # Get chosen and rejected responses
                chosen = examples['assistant_response_chosen'][i]
                rejected = examples['assistant_response_rejected'][i]
                
                # Clean responses
                chosen_cleaned = clean_unicode_conservative(chosen) if isinstance(chosen, str) else str(chosen)
                rejected_cleaned = clean_unicode_conservative(rejected) if isinstance(rejected, str) else str(rejected)
                
                chosen_responses.append(chosen_cleaned)
                rejected_responses.append(rejected_cleaned)
                
            except (ValueError, SyntaxError) as e:
                print(f"Warning: Could not parse prompt at index {i}: {e}")
                # Fallback to treating as string
                parsed_prompts.append([{"role": "user", "content": str(examples['prompt'][i])}])
                chosen_responses.append(str(examples['assistant_response_chosen'][i]))
                rejected_responses.append(str(examples['assistant_response_rejected'][i]))
        
        return {
            'prompt': parsed_prompts,
            'chosen': chosen_responses,
            'rejected': rejected_responses
        }

    ds = load_dataset("csv", data_files=args.dataset_repo, encoding="utf-8", encoding_errors="ignore")
    
    # Transform the data to DPO format
    ds = ds.map(parse_conversations_batch, batched=True, remove_columns=ds['train'].column_names)
    
    # Create dataset with proper format for DPO
    dpo_dataset = Dataset.from_dict({
        "prompt": ds['train']['prompt'],
        "chosen": ds['train']['chosen'], 
        "rejected": ds['train']['rejected']
    })
    
    # Split into train/eval
    ds = _uniform_split(dpo_dataset, eval_ratio=args.eval_ratio, seed=args.seed, n_eval=None)

    # Bits-and-bytes
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if args.use_4bit else None

    # LoRA
    lora_cfg = LoraConfig(
        r=args.peft_r,
        lora_alpha=args.peft_alpha,
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights="gaussian",
        target_modules=args.target_modules.split(","),
    ) if args.use_lora else None
    
    # Load model
    model, tok = load_model_and_tokenizer(
        args.model_name,
        bnb_cfg=bnb_cfg,
        lora_cfg=lora_cfg,
        device=args.device,
        is_eval=False,
    )

    # DeepSpeed zero
    ds_cfg = {
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": True,
            "reduce_bucket_size": "auto",
            "contiguous_gradients": True,
            "offload_optimizer": {"device": "none"},
            "offload_param": {"device": "none"},
        },
        "gradient_clipping": "auto",
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "steps_per_print": 200,
    }

    # Trainer config
    train_args = DPOConfig(
        beta=0.1,
        loss_type="sigmoid",
        max_grad_norm=1.0,
        optim="adamw_torch",
        report_to="wandb" if args.wandb_project else "none",
        do_eval=True,
        eval_steps=args.eval_steps, 
        save_strategy='epoch',
        eval_strategy="steps",
        gradient_checkpointing=True,  
        lr_scheduler_type="cosine",
        metric_for_best_model="eval_loss",
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        max_length=args.max_new_tokens, 
        max_prompt_length=args.max_prompt_length, 
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        run_name=args.output_dir,
        output_dir=args.output_dir,
        deepspeed=ds_cfg, 
        fp16=not torch.cuda.is_bf16_supported(), 
        bf16=torch.cuda.is_bf16_supported(),
    )

    # W&B
    if args.wandb_project and os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.output_dir.replace("/", "_"),
            config=train_args.to_dict(),
            save_code=True,
            job_type="train",
        )
    
    def process(row):
        """Process each row to format for DPO training"""
        # Apply chat template to the prompt (conversation history)
        reference = tok.apply_chat_template(
            row["prompt"] + [{'role': 'assistant', 'content': row["chosen"]}], 
            tokenize=False
        )
        
        # Format prompt with generation template
        row["prompt"] = tok.apply_chat_template(
            row["prompt"], 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Add EOS token to responses
        row["chosen"] = row["chosen"].strip() + tok.eos_token
        row["rejected"] = row["rejected"].strip() + tok.eos_token
        
        # Verify the formatting is correct
        assert row["prompt"] + row["chosen"] == reference
        return row

    # Apply processing to both train and eval datasets
    ds["train"] = ds["train"].map(process, load_from_cache_file=False)
    ds["eval"] = ds["eval"].map(process, load_from_cache_file=False)

    trainer = DPOTrainer(
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        processing_class=tok,
        peft_config=lora_cfg,
        args=train_args,
    )
    trainer.train(resume_from_checkpoint=args.resume_ckpt_dir)

    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

    if args.push_to_hub and args.hf_org:
        repo = f"offline_dpo-{args.dataset_repo.replace('/', '_')}"
        trainer.model.push_to_hub(f"{args.hf_org}/{repo}", private=True)
        tok.push_to_hub(f"{args.hf_org}/{repo}", private=True)

    if args.wandb_project:
        wandb.finish()

if __name__ == "__main__":
    main()