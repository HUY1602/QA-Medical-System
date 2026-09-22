import os
import json
import torch
import wandb
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig


# Cấu hình đường dẫn 
os.environ["WANDB_PROJECT"] = "MedQA-Qwen-SFT" 
TRAIN_DATA_PATH = "/data2/cmdir/home/ioit107/mqhuy/medModel/MedQA-USMLE/questions/US/train.jsonl" 
OUTPUT_DIR = "/data2/cmdir/home/ioit107/mqhuy/medModel/medqa-qwen-sft"

# 1. CHUẨN BỊ DỮ LIỆU
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

PROMPT_TEMPLATE = """A conversation between User and Assistant.

User:
{question}

Options:
{options}

Assistant:
<answer>{answer}</answer>
"""

def prepare_sft_dataset(path):
    raw_data = load_jsonl(path)
    formatted_data = []
    for example in raw_data:
        options_text = "\n".join([f"{k}. {v}" for k, v in example.get("options", {}).items()])
        gold_answer = example.get("answer_idx", example.get("answer", "")).strip().upper()
        
        full_text = PROMPT_TEMPLATE.format(
            question=example["question"], 
            options=options_text, 
            answer=gold_answer
        )
        
        formatted_data.append({"text": full_text})
        
    return Dataset.from_list(formatted_data)

print("Đang chuẩn bị dữ liệu SFT...")
train_dataset = prepare_sft_dataset(TRAIN_DATA_PATH)
print(f"Tổng số mẫu Train: {len(train_dataset)}")

# 2. Load model VÀ TOKENIZER
model_name = "/data2/cmdir/home/ioit107/vnu/models/Qwen/Qwen3-8B"

print("Đang tải Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" 

print("Đang tải Model lên GPU A100...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16, 
    device_map="auto",
    trust_remote_code=True
)

# 3. CẤU hình LoRA & SFT TRAINER
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# Sử dụng SFTConfig thay vì TrainingArguments
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=2e-5, 
    per_device_train_batch_size=4,     
    gradient_accumulation_steps=2,     
    num_train_epochs=3,       
    
    save_strategy="steps",             
    save_steps=100,                    
    save_total_limit=3,                
    
    logging_steps=10,                  
    report_to="wandb",                 
    run_name="qwen-sft-baseline-no-val",

    dataset_text_field="text", 
    max_length=1024,  
)

# Trainer cho SFT
trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    peft_config=peft_config,
    # tokenizer=tokenizer,
    args=training_args,
)
# 4. CHẠY HUẤN LUYỆN
print("Bắt đầu quy trình huấn luyện SFT...")

last_checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    checkpoints = [os.path.join(OUTPUT_DIR, d) for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint")]
    if checkpoints:
        last_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
        print(f"Tìm thấy Checkpoint: {last_checkpoint}. Sẽ tiếp tục huấn luyện từ đây!")

if last_checkpoint:
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    trainer.train()

print("Huấn luyện hoàn tất, lưu mô hình...")
trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
wandb.finish()