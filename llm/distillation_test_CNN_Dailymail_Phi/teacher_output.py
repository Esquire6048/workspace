from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch


# 加载数据集
dataset = load_dataset("cnn_dailymail", "3.0.0")
train_data = dataset["train"].select(range(10000))
val_data = dataset["validation"].select(range(10000))

print("🌰Train data sample", train_data[0])


# 格式化数据集
def format_sample(example):
    return {
        "input": f"[INST] Summarize the following article:\n{example['article']} [/INST]",
        "output": example["highlights"]
    }

train_data = train_data.map(format_sample)
val_data = val_data.map(format_sample)


# 加载 tokenizer 和模型
model_name = "microsoft/phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",           # 自动放入GPU
    torch_dtype="auto",          # 以最低显存需求加载
    trust_remote_code=True
)
model.eval()



def get_teacher_outputs(input_text, tokenizer, model):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True,return_dict=True)
    
    return {
        "logits": outputs.logits,
        "hidden_states": outputs.hidden_states
    }

# # 加载模型和分词器，使用了BERT模型，任务是二分类
# model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
# tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# # 加载IMDb的数据集
# dataset = load_dataset("cnn_dailymail", "3.0.0")
# train_data = dataset["train"].shuffle(seed = 47)
# val_data = dataset["validation"]


# # 定义分词函数
# def tokenize(batch):
#     return tokenizer(batch["text"], padding=True, truncation=True)

# # 批量处理整个数据集并设置格式为PyTorch Tensor
# dataset = dataset.map(tokenize, batched=True)
# dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# # 定义训练参数
# training_args = TrainingArguments(
#     output_dir="./checkpoints/bert", # 输出目录
#     evaluation_strategy="epoch", # 每轮评估
#     save_strategy="epoch", # 每轮保存
#     per_device_train_batch_size=8, # 每张 GPU 每个 step 训练的样本数
#     per_device_eval_batch_size=8, # 每张 GPU 每个 step 验证的样本数
#     num_train_epochs=2, # 训练轮数
#     logging_dir="./logs", # 日志目录
#     logging_steps=500, # 训练500步记录一次日志
#     save_total_limit=2, # checkpoint数量上限
#     fp16=True if torch.cuda.is_available() else False #如果有 GPU，就使用混合精度训练（FP16）
# )

# # 定于训练器
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset["train"].shuffle(seed=42), # 打乱顺序
#     eval_dataset=dataset["test"] # 不应该打乱顺序
# )
# # 启动训练并保存模型
# trainer.train()
# model.save_pretrained("./models/bert")
# tokenizer.save_pretrained("./models/bert")