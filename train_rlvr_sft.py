import os
import json
import re
import random
import torch
import wandb
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig

# WANDB
os.environ["WANDB_PROJECT"] = "MedQA-Qwen-GRPO-e3-baseSFT"
# WANDB_API_KEY is provided via environment.

# PATH
TRAIN_DATA_PATH = "/data2/cmdir/home/ioit107/mqhuy/medModel/MedQA-USMLE/questions/US/train.jsonl"
OUTPUT_DIR = "/data2/cmdir/home/ioit107/mqhuy/medModel/medqa-qwen-grpo-e3-baseSFT"

# LOAD DATA
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# Prompt mạnh hơn 
PROMPT_TEMPLATE = """A conversation between User and Assistant.

You MUST strictly follow this format:

<think>
Step-by-step reasoning here
</think>
<answer>
ONE LETTER ONLY (A/B/C/D/E)
</answer>

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

print("Loading dataset...")
train_dataset = prepare_dataset(TRAIN_DATA_PATH)
print("Train size:", len(train_dataset))

# LOAD MODEL
model_name = "/data2/cmdir/home/ioit107/mqhuy/medModel/medqa-qwen-sft/final_model"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# REWARD FUNCTION
def mcqa_reward_func(prompts, completions, answer, **kwargs):
    rewards = []

    for response, gold in zip(completions, answer):
        reward = 0.0

        # FORMAT 
        think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)

        if think_match:
            reward += 0.2
        else:
            reward -= 0.2

        if answer_match:
            reward += 0.2
        else:
            reward -= 0.2

        # THINK LENGTH 
        if think_match:
            think_content = think_match.group(1).strip()
            length = len(think_content)

            if length > 50:
                reward += 0.05
            elif length > 20:
                reward += 0.1
            else:
                reward -= 0.05

        # ANSWER
        if answer_match:
            model_answer = answer_match.group(1).strip().upper()
            gold_answer = str(gold).strip().upper()

            if model_answer == gold_answer:
                reward += 1

        # CLIP 
        reward = max(min(reward, 1.0), -1.0)

        rewards.append(reward)

        # DEBUG RANDOM
        if random.random() < 0.001:
            print("\n--- DEBUG SAMPLE ---")
            print("RESPONSE:", response[:300])
            print("REWARD:", reward)

    return rewards

# LORA
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# TRAIN CONFIG
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=3e-6,   # giảm LR để ổn định
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=1,

    num_generations=8,    # tăng group size

    max_prompt_length=1024,
    max_completion_length=512,

    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,

    logging_steps=10,
    report_to="wandb",
    run_name="qwen-grpo-stable-run",

    # nếu version TRL có:
    # kl_coeff=0.05
)

# TRAINER
trainer = GRPOTrainer(
    model=base_model,
    reward_funcs=[mcqa_reward_func],
    args=training_args,
    train_dataset=train_dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
)

# TRAIN
print("Start training...")

last_checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    checkpoints = [
        os.path.join(OUTPUT_DIR, d)
        for d in os.listdir(OUTPUT_DIR)
        if d.startswith("checkpoint")
    ]
    if checkpoints:
        last_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
        print("Resume from:", last_checkpoint)

if last_checkpoint:
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    trainer.train()

# SAVE
print("Saving model...")
trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))

wandb.finish()

print("Done!")