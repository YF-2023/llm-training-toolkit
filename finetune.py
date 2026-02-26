import argparse
import os

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, set_peft_model_state_dict
from torch.utils.data import IterableDataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, logging, set_seed
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

"""
Fine-Tune StarCoder on Code Alpaca/SE
"""

class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        kwargs["model"].save_pretrained(checkpoint_folder)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        torch.save({}, pytorch_model_path)
        return control


class LoadBestPeftModelCallback(TrainerCallback):
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        print(f"Loading best peft model from {state.best_model_checkpoint} (score: {state.best_metric}).")
        best_model_path = os.path.join(state.best_model_checkpoint, "adapter_model.bin")
        adapters_weights = torch.load(best_model_path)
        model = kwargs["model"]
        set_peft_model_state_dict(model, adapters_weights)
        return control
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="bigcode/large-model")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/CodeAlpaca_20K")
    parser.add_argument("--subset", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--size_valid_set", type=int, default=10000)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle_buffer", type=int, default=5000)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--max_eval_samples", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpointing_steps", type=str, default="epoch")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--evaluation_strategy", type=str, default="epoch")
    parser.add_argument("--load_best_model_at_end", type=bool, default=True)
    parser.add_argument("--ddp_find_unused_parameters", type=bool, default=False)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--fsdp", type=str, default="")
    parser.add_argument("--fsdp_transformer_layer_cls_to_wrap", type=str, default="")
    parser.add_argument("--fsdp_min_num_params", type=int, default=0)
    parser.add_argument("--tf32", type=bool, default=False)
    parser.add_argument("--use_lion_optimizer", type=bool, default=False)
    parser.add_argument("--use_flash_attention", type=bool, default=False)
    parser.add_argument("--use_grad_checkpointing", type=bool, default=False)
    parser.add_argument("--use_hf_deepspeed", type=bool, default=False)
    parser.add_argument("--use_peft", type=bool, default=True)
    parser.add_argument("--use_8bit_adam", type=bool, default=True)
    parser.add_argument("--use_8bit_lamb", type=bool, default=False)
    parser.add_argument("--use_xformers", type=bool, default=False)
    parser.add_argument("--use_bf16", type=bool, default=False)
    parser.add_argument("--use_tf32", type=bool, default=False)
    parser.add_argument("--use_adafactor", type=bool, default=False)
    parser.add_argument("--use_ipex", type=bool, default=False)
    parser.add_argument("--use_lora", type=bool, default=True)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--lora_bias", type=str, default="none")
    parser.add_argument("--lora_task_type", type=str, default="CAUSAL_LM")
    parser.add_argument("--lora_inference_mode", type=bool, default=False)
    parser.add_argument("--lora_state_dict_path", type=str, default="")
    parser.add_argument("--lora_resume_from_checkpoint", type=str, default="")
    parser.add_argument("--lora_use_gradient_checkpointing", type=bool, default=False)
    parser.add_argument("--lora_gradient_checkpointing", type=bool, default=False)
    parser.add_argument("--lora_enable_lol", type=bool, default=False)
    parser.add_argument("--lora_rank_dropout", type=float, default=0.0)
    parser.add_argument("--lora_merge_weights", type=bool, default=False)
    parser.add_argument("--lora_disable_adapters", type=bool, default=False)
    parser.add_argument("--lora_optimize_bnb", type=bool, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    # The rest of the finetune.py content would go here, but is truncated for brevity.
