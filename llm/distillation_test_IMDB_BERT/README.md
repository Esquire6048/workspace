# LLM Knowledge Distillation Starter

## 目录结构
- `teacher_finetune.py`: 训练 BERT 教师模型
- `distill_train.py`: 用教师模型蒸馏训练 DistilBERT 学生模型

## 使用方法

### 1. 教师模型训练
```bash
python teacher_finetune.py
```

### 2. 学生模型蒸馏训练
```bash
python distill_train.py
```
