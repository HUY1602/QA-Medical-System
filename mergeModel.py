import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# 1. ĐƯỜNG DẪN
BASE_MODEL_PATH = "/data2/cmdir/home/ioit107/vnu/models/Qwen/Qwen3-8B" 
LORA_PATH = "/data2/cmdir/home/ioit107/mqhuy/medModel/medqa-qwen-sft/final_model"
MERGED_OUTPUT_DIR = "/data2/cmdir/home/ioit107/mqhuy/medModel/medqa-qwen-merged"

# Tạo thư mục đầu ra nếu chưa có
os.makedirs(MERGED_OUTPUT_DIR, exist_ok=True)

# 2. LOAD BASE MODEL VÀ TOKENIZER
print("1. Đang tải Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

print("2. Đang tải Base Model (Mô hình gốc)...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16, 
    device_map="cpu", 
    trust_remote_code=True
)

# 3. LOAD LORA VÀ (MERGE)
print("3. Đang lắp ráp bản vá LoRA vào mô hình gốc...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)

print("4. Đang tiến hành HỢP THỂ (Merge and Unload)... Quá trình này có thể mất vài phút.")
#  LoRA vào Base Model và gỡ bỏ cấu trúc Peft
merged_model = model.merge_and_unload()

# 4. LƯU MÔ HÌNH HOÀN CHỈNH
print(f"5. Đang xuất mô hình hoàn chỉnh ra ổ cứng tại: {MERGED_OUTPUT_DIR}...")
# Lưu toàn bộ trọng số mới
merged_model.save_pretrained(
    MERGED_OUTPUT_DIR, 
    safe_serialization=True, 
    max_shard_size="5GB"     # Chia nhỏ file ra mỗi cục 5GB để dễ di chuyển/upload
)

# lưu Tokenizer để sau này dùng
tokenizer.save_pretrained(MERGED_OUTPUT_DIR)

print("QUÁ TRÌNH HỢP THỂ HOÀN TẤT!")
print(f"Bây giờ bạn có thể trỏ model_name thẳng vào '{MERGED_OUTPUT_DIR}' để sử dụng.")