## 库导入

```python
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizerFast
from datasets import load_dataset
import torch
```

## 加载模型和分词器，使用了BERT模型，任务是二分类

```python
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
```

## 加载IMDb的数据集

```python
dataset = load_dataset("imdb")
```

## 定义分词函数

```python
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)
```

|操作 | 描述|
| :--- | :--- |
|tokenizer(...) | 将字符串变成 token ID 和 attention mask 等|
|batch["text"] | 是一批文本（batch 是一个 dict）|
|padding=True | 自动将不同长度的文本 padding 成相同长度（默认 pad 到 batch 中最长）|
|truncation=True | 文本太长时自动截断（默认上限为模型的最大输入，如 BERT 是 512）|

<details>
<summary>分词函数具体示例</summary>

```python
tokenizer("this is a test.")
```

Token | ID | Attension mask
:--- | :--- | :---
[CLS] | 101 | 1
this | 2023 | 1
is | 2003 | 1
a | 1037 | 1
test | 3231 | 1
. | 1012 | 1
[SEP] | 102 | 1

</details>

另外，Hugging Face Datasets 会自动保留没被修改的字段

<details>
<summary>具体示例</summary>

假设我们有这样的样本：

```json
{
  "text": "This movie sucks.",
  "label": 0
}
```

分词函数处理完会返回：

```json
{
  "input_ids": [...],
  "attention_mask": [...]
}
```

而最终样本会合并成：

```json
{
  "text": "This movie sucks.",
  "label": 0,
  "input_ids": [...],
  "attention_mask": [...]
}
```

</details>

所以 label 会自动留在里面，除非你显式删除它。

## 批量处理整个数据集并设置格式为PyTorch Tensor

```python
dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
```

map 是 HuggingFace Datasets 提供的方法，用于对整个数据集逐条（或批量）应用一个函数。

batched=True 表示每次处理的是一个“batch”，而不是单条数据。

整个数据集被“添加”了新的字段，比如 input_ids, attention_mask

set_format把数据从 Python dict 变成 PyTorch Tensor 格式。

只保留指定的字段：input_ids, attention_mask, label

## 定义训练参数

```python
training_args = TrainingArguments(
    output_dir="./bert-imdb", # 输出目录
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
```

Epoch 是模型完整遍历一遍训练/验证集的过程

<details>
<summary>具体例子</summary>

假设训练数据有 2,000 条样本，batch size 是 8：

一次训练会处理：2000 / 8 = 250 个 batch（也叫 step）

当模型完成这 250 个 step，就叫做 1 个 epoch（轮）

</details>

“每张 GPU 各处理一个 batch”这一过程，称为一个 step

## 定义训练器

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"].shuffle(seed=42), # 打乱顺序
    eval_dataset=dataset["test"] # 不应该打乱顺序
)
```

训练集shuffle可以提升模型收敛速度和泛化能力

验证集不应该shuffle，因为评估不会反向传播，顺序对最终结果没有影响，也不像训练会造成过拟合。而且保留顺序有助于结果对齐 / 误差分析

## 执行训练并保存模型和分词器

```python
trainer.train()
model.save_pretrained("./bert-teacher")
tokenizer.save_pretrained("./bert-teacher")
```

模型保存的是“怎么理解 token ID”

tokenizer保存的是“token ID 是谁”

两者缺一不可

但这里，实际没有对tokenizer做任何修改，只是一种好习惯，因为在其他场合可能有区别