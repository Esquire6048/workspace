import os
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# 参数设置
MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
SAVE_DIR = "teacher_cache_slim"
NUM_SAMPLES = 10_000
MAX_LENGTH = 1024
SELECTED_HIDDEN_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28]

# 创建保存目录
os.makedirs(SAVE_DIR, exist_ok=True)

# 加载 tokenizer 和教师模型
print("🔧 Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# 加载数据集
print("📚 Loading CNN/DailyMail dataset...")
dataset = load_dataset("cnn_dailymail", "3.0.0")["train"].select(range(NUM_SAMPLES))

# 格式化输入
def format_sample(example):
    return {
        "id": example["id"],
        "input": f"[INST] Summarize the following article:\n{example['article']} [/INST]",
        "output": example["highlights"]
    }

dataset = dataset.map(format_sample)

# 逐样本保存（sample-wise）
print(f"🚀 Saving only logits + 8 hidden_states for {NUM_SAMPLES} samples...")
for idx, example in enumerate(tqdm(dataset)):
    save_path = os.path.join(SAVE_DIR, f"teacher_{idx}.pt")
    if os.path.exists(save_path):
        continue

    input_text = example["input"]
    input_id = example["id"]

    try:
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True,
                           max_length=MAX_LENGTH, padding=False).to(model.device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                output_attentions=False,  # 关闭注意力节省内存
                return_dict=True
            )

        selected_hidden = [
            outputs.hidden_states[i].cpu()
            for i in SELECTED_HIDDEN_LAYERS
        ]

        sample_output = {
            "id": input_id,
            "logits": outputs.logits.cpu(),
            "hidden_states": selected_hidden
        }

        torch.save(sample_output, save_path)

    except Exception as e:
        print(f"❌ Error at sample {idx}: {e}")
        continue
