import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
import torch

# ===========================
# 基础配置
# ===========================
MODEL_NAME = "./Qwen1.5-1.8B"  # 本地模型路径
DATA_PATH = "dataset/final_dataset.jsonl"
OUTPUT_DIR = "model_result/qwen_lora_sft"
MAX_LENGTH = 2048

# ===========================
# LoRA 配置
# ===========================
LORA_R = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

# ===========================
# tokenizer
# ===========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


# ===========================
# 数据处理
# ===========================
def format_example(example):
    if example.get("input"):
        prompt = f"问题：{example['instruction']}\n附加信息：{example['input']}\n回答："
    else:
        prompt = f"问题：{example['instruction']}\n回答："
    answer = example["output"]
    text = prompt + answer
    tokenized = tokenizer(text, truncation=True, max_length=MAX_LENGTH, padding="max_length")

    # label 处理，只计算 answer 部分
    labels = tokenized["input_ids"].copy()
    prompt_len = len(tokenizer(prompt, truncation=True, max_length=MAX_LENGTH)["input_ids"])
    labels[:prompt_len] = [-100] * prompt_len
    tokenized["labels"] = labels
    return tokenized


dataset = load_dataset("json", data_files=DATA_PATH, split="train")
dataset = dataset.map(format_example)

# ===========================
# 加载模型 + LoRA 注入
# ===========================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ===========================
# 训练参数（快速测试）
# ===========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    max_steps=5,  # 小测试，快速完成
    learning_rate=2e-4,
    logging_steps=1,
    save_steps=5,
    fp16=torch.cuda.is_available(),  # GPU 可用就用 fp16
    optim="paged_adamw_32bit",
    report_to="none"
)

# ===========================
# Data collator
# ===========================
data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, padding=True, return_tensors="pt"
)

# ===========================
# Trainer
# ===========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

# ===========================
# 开始训练
# ===========================
if __name__ == "__main__":
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    print("训练完成，LoRA 权重已保存到：", OUTPUT_DIR)
