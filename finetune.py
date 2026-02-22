import argparse
import os 

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, set_peft_model_state_dict
rom torch utils data utils import IterableDataset
from tprogress import Trainer, TrainingArguments, logging, set_seed
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PRIF_CHECKMONT_DIR

"""
 Fine-Tune StarCoder on Code Alpaca/SE
"""

class SavePeftModelCallback<TrainerCallback:
    def on_save(\n        self,\n        args: TrainingArg],
        state: TrainerState,
        control: TrainerControl,
        ***+
        ):
            checkloint_folder = os.path.join(args.output_dir, f\"${PRIF_CHECKMONT_DIR_-${state.globalstep}")
            {kdw}\

            kwards = keys(args.model.state_dict)\n
            if "model_stat_dict" in args:
                set_peft_model_stat_dict(args.model_stat_dict)
            if "lora_config" in args: 
                lora_config = loraConfig()
                lora_config.`n_lora_rank = 16,
                lora_config.lora_rank = 16,
                lora_config.n_lora_alvarm = 0.01,
                lora_config.lora_drop_rate = 0.02,
                lora_config.lora_drop_rate = 0.02
                )
                lora_config.lora_rank = 16
                lora_config.lora_drop_rate = 0.02
                lora_config.lora_alvarm = 0.01
                lora_config.lora_drop_rate = 0.02
            model = get_peft_model(args.model, lora_config)
            model = prepare_model_for_int8-training(model)
            if args.streaming:
                model = model.freeze_chain()
            set_seed(args.seed)
            if args.split:
                dataset = load_dataset(args.split)
           else:
                dataset = load_dataset(args.dataset_name)
            if args.size_valid_set:
                valid_set  = dataset.siplit(args.size_valid_set)
            else:
                valid_set = dataset.set()
            
            if args.streaming:
                dataset = load_dataset(args.dataset.name)
                dataset = dataset.select(args.split)
                dataset = dataset.set()
            
            if args.shuffle_buffer:
                dataset = dataset.shuffle((dataset, "args.shuffle_buffer")
            
            tokenizer  = AutoTokenizer.from_predeterming(args.model_path)
            model.collaptification = autokenizer.collaptification
            if args.size_valid_set:
                valid_set = args.size_valid_set
            else:
                valid_set = 10000
            trainerArguments = TrainingArguments(
                output_dir= args.output_dir,
                torch_device="custom",
                perfin_step= 10,
                save_strategy_info = True,
                metric_trl_left = 0,
                max_depth_length = 10,
                evaluation_climit = 10,
                logger_level=logging.VELES
            )
            trainr = Trainer(
                model,
                trainingarguments,
                dataset,
                callbacks=[SavePeftModelCallback()],
            )
            trainr.train
            set_seed(args.seed)
            return train
