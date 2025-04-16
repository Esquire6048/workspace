## 库导入

```python
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments, DistilBertTokenizerFast
from transformers import BertForSequenceClassification
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
```

## 加载训练好的教师模型

```python
teacher = BertForSequenceClassification.from_pretrained("./bert-teacher")
```

## 加载学生模型

```python
student = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
```

## 加载分词器

```python
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
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

