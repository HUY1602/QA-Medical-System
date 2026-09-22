import json
import re
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# 1. CẤU HÌNH ĐƯỜNG DẪN
BASE_MODEL = "/data2/cmdir/home/ioit107/vnu/models/Qwen/Qwen3-8B" 
LORA_PATH = "/data2/cmdir/home/ioit107/mqhuy/medModel/medqa-qwen-grpo-e3/final_model"
TEST_FILE_PATH = "/data2/cmdir/home/ioit107/mqhuy/medModel/MMLU-Pro-Health/test.jsonl" 
OUTPUT_RESULT_PATH = "medqa_test_mmlu_e3_final.jsonl"

# 2. TẢI MODEL VÀ TOKENIZER LÊN GPU
print("Đang tải Tokenizer từ mô hình gốc...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

print("Đang tải Base Model lên GPU...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    trust_remote_code=True
)

print(f"Đang ghép trọng số LoRA từ {LORA_PATH}...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

# 3. CHUẨN BỊ DỮ LIỆU TEST
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

print(f"\nĐang tải dữ liệu từ {TEST_FILE_PATH}...")
test_data = load_jsonl(TEST_FILE_PATH)
print(f"Tổng số câu hỏi cần test: {len(test_data)}\n")

correct_count = 0
valid_think_count = 0  
empty_think_count = 0  
missing_think_count = 0 
results = []

# 4. CHẠY VÒNG LẶP ĐÁNH GIÁ
for item in tqdm(test_data, desc="Đang đánh giá mô hình"):
    # 1. Lấy thông tin từ cấu trúc data mới
    question = item.get("extra_info", {}).get("question", "Unknown Question")
    gold_answer = item.get("extra_info", {}).get("correct_option", "").strip().upper()
    
    # Nếu không tìm thấy trong extra_info, thử tìm trong reward_model
    if not gold_answer:
        gold_answer = item.get("reward_model", {}).get("ground_truth", "").strip().upper()

    # 2. Xây dựng Prompt sử dụng Chat Template chuẩn của Qwen
    # Lấy nguyên mảng "prompt" gồm role system và user từ data
    conversation = item.get("prompt", [])
    
    prompt = tokenizer.apply_chat_template(
        conversation, 
        tokenize=False, 
        add_generation_prompt=True # Yêu cầu mô hình bắt đầu trả lời (role: assistant)
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 3. Sinh câu trả lời
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024, 
            temperature=0.1,    
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Cắt bỏ phần prompt đầu vào, chỉ lấy phần chữ do mô hình sinh ra
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    # KIỂM TRA PHẦN SUY LUẬN <think> 
    
    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    think_content = ""
    has_valid_reasoning = False

    if think_match:
        think_content = think_match.group(1).strip()
        if len(think_content) >= 20: 
            valid_think_count += 1
            has_valid_reasoning = True
        else:
            empty_think_count += 1
    else:
        missing_think_count += 1

    # KIỂM TRA ĐÁP ÁN <answer> 
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
    if answer_match:
        
        model_answer = answer_match.group(1).strip()[0:1].upper()
    else:
        model_answer = "LỖI_ĐỊNH_DẠNG"

    is_correct = (model_answer == gold_answer)
    if is_correct:
        correct_count += 1

    # Lưu log chi tiết
    results.append({
        "question": question,
        "gold_answer": gold_answer,
        "model_answer": model_answer,
        "is_correct": is_correct,
        "has_valid_reasoning": has_valid_reasoning,
        "think_length": len(think_content),
        "think_content": think_content,
        "full_response": response
    })

# 5. TỔNG KẾT KẾT QUẢ
accuracy = (correct_count / len(test_data)) * 100
print("BÁO CÁO KẾT QUẢ KIỂM THỬ MMLU (TEST REPORT)")
print(f"Tổng số câu hỏi    : {len(test_data)}")
print(f"Số câu trả lời đúng: {correct_count}")
print(f"Độ chính xác (Acc) : {accuracy:.2f}%")
print("-" * 55)
print("THỐNG KÊ HOẠT ĐỘNG SUY LUẬN (<think>):")
print(f"- CÓ suy luận chi tiết   : {valid_think_count} câu")
print(f"- Thẻ <think> rỗng/ngắn  : {empty_think_count} câu (Lỗi lười biếng)")
print(f"- Thiếu thẻ <think>      : {missing_think_count} câu (Lỗi định dạng)")

with open(OUTPUT_RESULT_PATH, "w", encoding="utf-8") as f:
    for res in results:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")

print(f"Chi tiết từng câu hỏi đã được lưu tại: {OUTPUT_RESULT_PATH}")