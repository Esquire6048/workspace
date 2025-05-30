from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments
from transformers import BertForSequenceClassification
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F

# 加载训练好的教师模型
teacher = BertForSequenceClassification.from_pretrained("./models/bert")

# 加载学生模型
student = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# 加载分词器
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# 加载IMDb的数据集
dataset = load_dataset("imdb")

# 定义分词函数
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

# 批量处理整个数据集并设置格式为PyTorch Tensor
dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

print("🔍 Sample keys in training batch:", dataset["train"][0].keys())

class DistillTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        device = inputs["input_ids"].device
        teacher.to(device)
        labels = inputs.pop("labels")
        outputs_student = model(**inputs)
        with torch.no_grad():
            outputs_teacher = teacher(**inputs)

        loss_ce = F.cross_entropy(outputs_student.logits, labels)
        loss_kl = F.kl_div(
            F.log_softmax(outputs_student.logits / 2.0, dim=-1),
            F.softmax(outputs_teacher.logits / 2.0, dim=-1),
            reduction="batchmean"
        ) * (2.0 ** 2)
        loss = 0.5 * loss_ce + 0.5 * loss_kl
        return (loss, outputs_student) if return_outputs else loss

training_args = TrainingArguments(
    output_dir="./checkpoints/distilbert-distilled",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    logging_dir="./logs",
    logging_steps=500,
    save_total_limit=2,
    fp16=True if torch.cuda.is_available() else False
)

trainer = DistillTrainer(
    model=student,
    args=training_args,
    train_dataset=dataset["train"].shuffle(seed=42),
    eval_dataset=dataset["test"],
)

trainer.train()
student.save_pretrained("./models/distilbert-distilled")
tokenizer.save_pretrained("./models/distilbert-distilled")