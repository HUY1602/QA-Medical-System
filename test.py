import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. CẤU HÌNH

# Folder LoRA đã train
LORA_PATH = r"C:\Users\Admin\Documents\AnalysisDataKLTN\checkpoint-700-e2"

# Folder chứa Qwen3-8B đã tải về
BASE_MODEL = r"C:\Users\Admin\Documents\AnalysisDataKLTN\Qwen3-8B"

# 2. LOAD TOKENIZER

print("Đang tải tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

# 3. LOAD BASE MODEL

print("Đang tải model...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

# 4. LOAD LORA

print("Đang ghép LoRA...")

model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH
)

model.eval()

print("Model đã sẵn sàng!\n")

# 5. TEMPLATE PROMPT

PROMPT_TEMPLATE = """A conversation between User and Assistant.
The assistant thinks internally and then answers.

The reasoning must be inside <think></think>
The final answer must be inside <answer></answer>

User:
{question}

Options:
A. {A}
B. {B}
C. {C}
D. {D}
E. {E}

Assistant:
"""

# 6. NHẬP CÂU HỎI

question = input("Nhập câu hỏi:\n")

print("\nNhập các đáp án:\n")

option_A = input("Nhập đáp án A: ")
option_B = input("Nhập đáp án B: ")
option_C = input("Nhập đáp án C: ")
option_D = input("Nhập đáp án D: ")
option_E = input("Nhập đáp án E: ")

# 7. TẠO PROMPT

prompt = PROMPT_TEMPLATE.format(
    question=question,
    A=option_A,
    B=option_B,
    C=option_C,
    D=option_D,
    E=option_E
)

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

# 8. GENERATE

print("\nĐang suy luận...\n")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(
    outputs[0][inputs.input_ids.shape[1]:],
    skip_special_tokens=True
)

# 9. TÁCH THINK

think_match = re.search(
    r"<think>(.*?)</think>",
    response,
    re.DOTALL
)

if think_match:
    think_content = think_match.group(1).strip()
else:
    think_content = "Không tìm thấy <think>"

# 10. TÁCH ANSWER

answer_match = re.search(
    r"<answer>\s*(.*?)\s*</answer>",
    response,
    re.DOTALL
)

if answer_match:
    model_answer = answer_match.group(1).strip().upper()
else:
    model_answer = "LỖI_ĐỊNH_DẠNG"

# 11. HIỂN THỊ KẾT QUẢ

print("=" * 60)
print("KẾT QUẢ SUY LUẬN")
print("=" * 60)

print("\n[SUY LUẬN]")
print(think_content)

print("\n[ĐÁP ÁN MODEL]")
print(model_answer)

print("\n[FULL RESPONSE]")
print(response)

print("=" * 60)