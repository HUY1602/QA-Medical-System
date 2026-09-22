# QA Medical System

Dự án huấn luyện và đánh giá mô hình ngôn ngữ cho bài toán hỏi đáp trắc nghiệm y khoa. Dự án hỗ trợ:

- Fine-tuning bằng **Supervised Fine-Tuning (SFT)** với LoRA.
- Huấn luyện **RLVR/GRPO** với hàm reward kiểm tra định dạng suy luận và đáp án.
- Đánh giá trên tập **MedQA-USMLE** và **MMLU-Pro-Health**.

## Yêu cầu hệ thống

- Linux hoặc môi trường có thể chạy PyTorch CUDA.
- GPU NVIDIA có hỗ trợ `bfloat16` (các script hiện được cấu hình cho GPU A100).
- Python 3.10 trở lên.
- Đủ dung lượng để tải mô hình Qwen và lưu checkpoint.
- Tài khoản [Weights & Biases](https://wandb.ai/) nếu muốn theo dõi quá trình huấn luyện.

> Các script hiện đang sử dụng đường dẫn tuyệt đối trên máy chủ nghiên cứu. Hãy sửa các biến `TRAIN_DATA_PATH`, `TEST_FILE_PATH`, `BASE_MODEL`, `LORA_PATH`, `OUTPUT_DIR` và `model_name` cho phù hợp với máy của bạn trước khi chạy.

## Cài đặt

Clone repository và tạo môi trường ảo:

```bash
git clone https://github.com/HUY1602/QA-Medical-System.git
cd QA-Medical-System

python -m venv .venv
source .venv/bin/activate
```

Trên Windows PowerShell, kích hoạt môi trường bằng:

```powershell
.\.venv\Scripts\Activate.ps1
```

Cài đặt PyTorch phù hợp với phiên bản CUDA của máy. Ví dụ:

```bash
pip install torch
```

Sau đó cài đặt các thư viện còn lại:

```bash
pip install -r requirements.txt
```

Đăng nhập Weights & Biases nếu sử dụng logging:

```bash
wandb login
```

## Chuẩn bị dữ liệu và mô hình

Các tập dữ liệu và trọng số mô hình không được lưu trong repository vì kích thước lớn. Cần chuẩn bị:

```text
MedQA-USMLE/
└── questions/US/
    ├── train.jsonl
    └── test.jsonl

MMLU-Pro-Health/
└── test.jsonl
```

Mỗi mẫu MedQA cần có trường `question`, `options` và đáp án trong `answer_idx` hoặc `answer`. Script MMLU đọc câu hỏi từ `prompt` và đáp án từ `extra_info.correct_option` hoặc `reward_model.ground_truth`.

Tải mô hình Qwen tương thích với cấu hình trong các script, hoặc thay `model_name`/`BASE_MODEL` bằng đường dẫn mô hình local của bạn. Với mô hình private trên Hugging Face, đăng nhập trước:

```bash
huggingface-cli login
```

## Huấn luyện SFT

Mở [train_SFTmodel.py](./train_SFTmodel.py) và cập nhật các đường dẫn:

```python
TRAIN_DATA_PATH = "/path/to/MedQA-USMLE/questions/US/train.jsonl"
OUTPUT_DIR = "/path/to/output/medqa-qwen-sft"
model_name = "/path/to/Qwen3-8B"
```

Chạy:

```bash
python train_SFTmodel.py
```

Mô hình LoRA cuối cùng được lưu trong thư mục `OUTPUT_DIR/final_model`. Nếu thư mục output có checkpoint, script sẽ tự động tiếp tục từ checkpoint có số bước lớn nhất.

## Huấn luyện RLVR/GRPO từ mô hình gốc

Sửa các đường dẫn trong [train_rlvr_base.py](./train_rlvr_base.py), sau đó chạy:

```bash
python train_rlvr_base.py
```

Hàm reward ưu tiên:

1. Có thẻ `<think>...</think>`.
2. Phần suy luận có độ dài tối thiểu.
3. Có thẻ `<answer>...</answer>`.
4. Đáp án cuối trùng với đáp án đúng.

## Huấn luyện RLVR/GRPO từ mô hình SFT

Sau khi có mô hình SFT, sửa `model_name` trong [train_rlvr_sft.py](./train_rlvr_sft.py) thành thư mục `final_model` của SFT:

```python
model_name = "/path/to/medqa-qwen-sft/final_model"
```

Cập nhật `TRAIN_DATA_PATH` và `OUTPUT_DIR`, rồi chạy:

```bash
python train_rlvr_sft.py
```

## Gộp LoRA vào mô hình gốc

Nếu cần một mô hình hoàn chỉnh thay vì adapter LoRA, sửa `BASE_MODEL_PATH`, `LORA_PATH` và `MERGED_OUTPUT_DIR` trong [mergeModel.py](./mergeModel.py), sau đó chạy:

```bash
python mergeModel.py
```

Mô hình sau khi merge và tokenizer sẽ được lưu tại `MERGED_OUTPUT_DIR`. Quá trình này cần nhiều RAM/ổ đĩa vì mô hình được tải trên CPU và có thể được chia thành các shard tối đa 5 GB.

## Đánh giá mô hình

### MedQA-USMLE

Sửa `BASE_MODEL`, `LORA_PATH` và `TEST_FILE_PATH` trong [test_medqa.py](./test_medqa.py), rồi chạy:

```bash
python test_medqa.py
```

Script in accuracy, thống kê thẻ `<think>` và lưu kết quả từng câu hỏi vào file JSONL được chỉ định bởi `OUTPUT_RESULT_PATH`.

### MMLU-Pro-Health

Sửa các đường dẫn trong [test_mmlu.py](./test_mmlu.py), rồi chạy:

```bash
python test_mmlu.py
```

Kết quả được in ra terminal và lưu vào file JSONL ở `OUTPUT_RESULT_PATH`.

## Phân tích kết quả

Các file sau dùng để phân tích dữ liệu và trực quan hóa phân phối:

- [analysis_data.py](./analysis_data.py)
- `answer_distribution.png`
- `num_options_distribution.png`
- `prompt_token_distribution.png`
- `question_token_distribution.png`

Chạy script phân tích sau khi đặt các file dữ liệu đầu vào đúng vị trí mà script yêu cầu:

```bash
python analysis_data.py
```

## Cấu trúc chính

```text
.
├── train_SFTmodel.py       # Huấn luyện SFT + LoRA
├── train_rlvr_base.py      # Huấn luyện GRPO từ mô hình gốc
├── train_rlvr_sft.py       # Huấn luyện GRPO từ mô hình SFT
├── mergeModel.py           # Merge adapter LoRA vào base model
├── test_medqa.py           # Đánh giá trên MedQA-USMLE
├── test_mmlu.py            # Đánh giá trên MMLU-Pro-Health
├── analysis_data.py        # Phân tích dữ liệu
├── requirements.txt        # Danh sách thư viện Python
└── .gitignore              # Loại trừ dataset, checkpoint và model artifact
```

## Lưu ý

- Không commit dataset, checkpoint, API key hoặc trọng số mô hình vào repository.
- Các file `.jsonl`, `.csv`, `.pt`, `.pth`, `.safetensors` và thư mục checkpoint đã được loại khỏi Git qua [.gitignore](./.gitignore).
- Các tham số batch size, `max_prompt_length`, `max_completion_length` và số lượng generation có thể cần giảm nếu GPU không đủ bộ nhớ.
- Đảm bảo phiên bản `transformers`, `trl` và `peft` tương thích với API được sử dụng trong các script.
