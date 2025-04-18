from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# 加载模型和分词器，使用了BERT模型，任务是二分类
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# 加载IMDb的数据集
dataset = load_dataset("imdb")

# 定义分词函数
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

# 批量处理整个数据集并设置格式为PyTorch Tensor
dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./checkpoints/distilbert", # 输出目录
    evaluation_strategy="epoch", # 每轮评估
    save_strategy="epoch", # 每轮保存
    per_device_train_batch_size=8, # 每张 GPU 每个 step 训练的样本数
    per_device_eval_batch_size=8, # 每张 GPU 每个 step 验证的样本数
    num_train_epochs=2, # 训练轮数
    logging_dir="./logs", # 日志目录
    logging_steps=500, # 训练500步记录一次日志
    save_total_limit=2, # checkpoint数量上限
    fp16=True if torch.cuda.is_available() else False #如果有 GPU，就使用混合精度训练（FP16）
)

# 定义训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"].shuffle(seed=42), # 打乱顺序
    eval_dataset=dataset["test"] # 不应该打乱顺序
)
# 启动训练并保存模型
trainer.train()
model.save_pretrained("./models/distilbert")
tokenizer.save_pretrained("./models/distilbert")