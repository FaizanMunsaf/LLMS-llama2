# -*- coding: utf-8 -*-
"""
Islamic ChatGpt
"""

# ======================================================
# load the dataset
# ======================================================

from datasets import load_dataset

# ======================================================
# Use Pytorch and Transformers for Training our Models
# ======================================================
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer

# PEFT use for enhance our performance
from peft import LoraConfig, get_peft_model

# Training Arguments
from transformers import TrainingArguments


#dataset_name = "timdettmers/openassistant-guanaco" ###Human ,.,,,,,, ###Assistant

dataset_name = 'AlexanderDoria/novel17_test' #french novels
dataset = load_dataset(dataset_name, split="train")

# ==================================================
# 4bit Training for give us the better Results
# ==================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# ====================================
# Model For Pretraining 
# ====================================
model_name = "tiiuae/falcon-7b"  # This model from Hugging Face

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True
)
model.config.use_cache = False

#=========================
# Load Tokenizer also
#=========================
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token




# =============================================================================================================
# Alpha Lora PEFT to update our performance level divided the load on GPU and CPU as well for train our model
# =============================================================================================================
lora_alpha = 16
lora_dropout = 0.1
lora_r = 64

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM"
)

# ==========================
# Loading the Trainer
# ==========================
from transformers import TrainingArguments

output_dir = "./model"
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
optim = "paged_adamw_32bit"
save_steps = 100
logging_steps = 10
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 100
warmup_ratio = 0.03
lr_scheduler_type = "constant"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
)


# ===============================================
# Finally pass the all parameters to the trainer
# ===============================================
from trl import SFTTrainer

max_seq_length = 512

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)


# Norm use for upscale the layer
for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)
        
        
        
# ==============================
# Train the model now
# ==============================
trainer.train()


model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
model_to_save.save_pretrained("model")