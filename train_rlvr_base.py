import os
import json
import re
import torch
import wandb
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig

# CẤU hình WANDB
os.environ["WANDB_PROJECT"] = "MedQA-Qwen-GRPO-e2" 

# CẤU hình đường dẫn trên Server 
TRAIN_DATA_PATH = "/data2/cmdir/home/ioit107/mqhuy/medModel/MedQA-USMLE/questions/US/train.jsonl" 
OUTPUT_DIR = "/data2/cmdir/home/ioit107/mqhuy/medModel/medqa-qwen-grpo-e2"

# 1. CHUẨN BỊ DỮ LIỆU
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

PROMPT_TEMPLATE = """A conversation between User and Assistant.
The assistant thinks internally and then answers.

The reasoning must be inside <think></think>
The final answer must be inside <answer></answer>

User:
{question}

Options:
{options}

Assistant:
"""

def prepare_dataset(path):
    raw_data = load_jsonl(path)
    formatted_data = []
    for example in raw_data:
        options_text = "\n".join(
            [f"{k}. {v}" for k, v in example.get("options", {}).items()]
            )
        
        prompt = PROMPT_TEMPLATE.format(
            question=example["question"], 
            options=options_text
            )
        
        formatted_data.append({
            "prompt": prompt,
            "answer": example.get("answer_idx", example.get("answer", ""))
        })
    return Dataset.from_list(formatted_data)

print("Đang chuẩn bị dữ liệu...")

train_dataset = prepare_dataset(TRAIN_DATA_PATH)
print(f"Tổng số mẫu Train: {len(train_dataset)}") # Đã xóa in thông tin tập Val

# 2. LOAD MODEL VÀ TOKENIZER
model_name = "/data2/cmdir/home/ioit107/vnu/models/Qwen/Qwen3-8B"

print("Đang tải Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
    )

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" 

print("Đang tải Model lên GPU A100...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    trust_remote_code=True
)


# 3. HÀM REWARD (HÀM THƯỞNG KÉP)
def mcqa_reward_func(prompts, completions, answer, **kwargs):
    rewards = []
    for response, gold in zip(completions, answer):
        think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
        
        if not think_match or not answer_match:
            rewards.append(-1.0) 
            continue
            
        think_content = think_match.group(1).strip()
        if len(think_content) < 20: 
            rewards.append(-0.5)
            continue
            
        model_answer = answer_match.group(1).strip().upper()
        if model_answer == str(gold).strip().upper():
            rewards.append(1.0) 
        else:
            rewards.append(0.0) 
            
    return rewards

# 4. CẤU HÌNH LORA & GRPO TRAINER
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=5e-6,
    per_device_train_batch_size=4,     
    gradient_accumulation_steps=2,     
    num_train_epochs=1,                
    num_generations=4,                 
    
    max_prompt_length=1024,      
    max_completion_length=512,   
    
    # CẤU HÌNH CHECKPOINT
    save_strategy="steps",             
    save_steps=100,                    
    save_total_limit=3,                
    
    logging_steps=10,                  
    report_to="wandb",                 
    run_name="qwen-grpo-a100-run2"     
)

trainer = GRPOTrainer(
    model=base_model,
    reward_funcs=[mcqa_reward_func],
    args=training_args,
    train_dataset=train_dataset,
    peft_config=peft_config,
    processing_class=tokenizer, 
)

# 5. KHỞI CHẠY HUẤN LUYỆN
print("Bắt đầu quy trình huấn luyện RLVR trên A100 (Không dùng tập Validation)...")

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

print("Huấn luyện hoàn tất! Đang lưu mô hình cuối...")
trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))

wandb.finish() 

print(f"Model đã được lưu an toàn tại: {OUTPUT_DIR}/final_model")