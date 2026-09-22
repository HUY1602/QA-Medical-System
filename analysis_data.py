import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from transformers import AutoTokenizer

# CONFIG
DATA_PATH = r"C:\Users\Admin\Documents\AnalysisDataKLTN\MedQA-USMLE\questions\US\train.jsonl"

MODEL_PATH = "Qwen/Qwen2.5-0.5B"

# LOAD DATA
print("Loading dataset...")

samples = []

with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        samples.append(json.loads(line))

print(f"Total samples: {len(samples)}")

# LOAD TOKENIZER
print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

# STORAGE
question_lengths_char = []
question_lengths_token = []

options_lengths_char = []
options_lengths_token = []

prompt_lengths_token = []

num_options_list = []

answer_distribution = Counter()

# PROMPT TEMPLATE
PROMPT_TEMPLATE = """A conversation between User and Assistant.

User:
{question}

Options:
{options}

Assistant:
"""

# ANALYSIS LOOP
print("Analyzing dataset...")

for sample in tqdm(samples):

    question = sample["question"]
    options = sample["options"]
    answer = sample.get("answer_idx", "")

    # Question length
    question_lengths_char.append(len(question))

    q_tokens = tokenizer.encode(question)
    question_lengths_token.append(len(q_tokens))

    # Options
    options_text = "\n".join(
        [f"{k}. {v}" for k, v in options.items()]
    )

    options_lengths_char.append(len(options_text))

    opt_tokens = tokenizer.encode(options_text)
    options_lengths_token.append(len(opt_tokens))

    # Full prompt
    prompt = PROMPT_TEMPLATE.format(
        question=question,
        options=options_text
    )

    prompt_tokens = tokenizer.encode(prompt)
    prompt_lengths_token.append(len(prompt_tokens))

    # Number of options
    num_options_list.append(len(options))

    # Answer distribution
    answer_distribution[answer] += 1

# DATAFRAME
stats_df = pd.DataFrame({
    "question_char_length": question_lengths_char,
    "question_token_length": question_lengths_token,
    "options_char_length": options_lengths_char,
    "options_token_length": options_lengths_token,
    "prompt_token_length": prompt_lengths_token,
    "num_options": num_options_list
})

# BASIC STATISTICS
print("\n================ DATASET STATISTICS ================")

print(f"Total samples: {len(samples)}")

print("\nQuestion Token Length Statistics")
print(stats_df["question_token_length"].describe())

print("\nOptions Token Length Statistics")
print(stats_df["options_token_length"].describe())

print("\nPrompt Token Length Statistics")
print(stats_df["prompt_token_length"].describe())

print("\nNumber of Options")
print(pd.Series(num_options_list).value_counts())

print("\nAnswer Distribution")
print(answer_distribution)

# SAVE CSV
stats_df.to_csv("dataset_detailed_stats.csv", index=False)

# VISUALIZATION

# Prompt token distribution
plt.figure(figsize=(10, 6))
plt.hist(prompt_lengths_token, bins=50)
plt.xlabel("Prompt Length (Tokens)")
plt.ylabel("Frequency")
plt.title("Distribution of Prompt Token Lengths")
plt.savefig("prompt_token_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

# Question token distribution
plt.figure(figsize=(10, 6))
plt.hist(question_lengths_token, bins=50)
plt.xlabel("Question Length (Tokens)")
plt.ylabel("Frequency")
plt.title("Distribution of Question Token Lengths")
plt.savefig("question_token_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

# Answer distribution
plt.figure(figsize=(8, 5))
plt.bar(answer_distribution.keys(), answer_distribution.values())
plt.xlabel("Answer")
plt.ylabel("Count")
plt.title("Answer Distribution")
plt.savefig("answer_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

# Number of options distribution
option_counter = Counter(num_options_list)

plt.figure(figsize=(8, 5))
plt.bar(option_counter.keys(), option_counter.values())
plt.xlabel("Number of Options")
plt.ylabel("Count")
plt.title("Distribution of Number of Options")
plt.savefig("num_options_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nAnalysis completed successfully!")