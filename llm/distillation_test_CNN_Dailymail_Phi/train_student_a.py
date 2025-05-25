import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments
from distill_dataset import DistillDataset
from distill_loss import distill_loss_fn
from distill_collator import DistillDataCollator
from student_model import StudentModelForCausalLM

# === 加载 tokenizer 和学生模型 ===
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-4k-instruct", trust_remote_code=True)
model = StudentModelForCausalLM.from_pretrained("student_model_A")

# === 加载前 10000 个样本 ===
raw_dataset = load_dataset("cnn_dailymail", "3.0.0")["train"].select(range(10000))

# === 格式化输入输出结构 ===
def format_sample(example):
    return {
        "id": example["id"],
        "input": f"[INST] Summarize the following article:\n{example['article']} [/INST]",
        "output": example["highlights"]
    }

dataset = raw_dataset.map(format_sample)

# === 构建 Dataset ===
train_dataset = DistillDataset(
    data=dataset,
    tokenizer=tokenizer,
    teacher_dir="teacher_cache",
    index_map=list(range(10000))
)

# === 自定义 Trainer（蒸馏 loss） ===
class DistillTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        device = next(model.parameters()).device
        outputs = model(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            labels=inputs["labels"].to(device)
        )
        loss = distill_loss_fn(
            outputs.logits,
            outputs.hidden_states[-8:],
            inputs["teacher_logits"].to(device),
            inputs["teacher_hidden"].to(device),
            inputs["labels"].to(device),
            alpha=1.0, beta=1.0, gamma=1.0
        )
        return (loss, outputs) if return_outputs else loss

# === 训练配置 ===
args = TrainingArguments(
    output_dir="checkpoints_student_A",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_steps=50,
    save_steps=1000,
    save_total_limit=2,
    report_to="none"
)

# === 启动训练 ===
trainer = DistillTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=DistillDataCollator()
)

trainer.train()
