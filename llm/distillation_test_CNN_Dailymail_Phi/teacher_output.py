import os
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# 参数设置
MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
SAVE_DIR = "teacher_cache"
SAVE_PATH = os.path.join(SAVE_DIR, "teacher_outputs_10k.pt")
NUM_SAMPLES = 10_000
MAX_LENGTH = 1024  # 截断最大长度

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

# 加载数据集前 10k 条
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

# 生成教师输出
cached_outputs = []
print("🚀 Generating teacher outputs...")
for example in tqdm(dataset):
    input_text = example["input"]
    input_id = example["id"]

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True,
                       max_length=MAX_LENGTH, padding=False).to(model.device)

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True
        )

    cached_outputs.append({
        "id": input_id,
        "logits": outputs.logits.cpu(),
        "hidden_states": [layer.cpu() for layer in outputs.hidden_states],  # 33层
        "attentions": [attn.cpu() for attn in outputs.attentions]  # 32层
    })

# 保存到 pt 文件
torch.save(cached_outputs, SAVE_PATH)
print(f"✅ Saved teacher outputs to: {SAVE_PATH}")
