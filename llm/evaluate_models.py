from transformers import (
    BertForSequenceClassification,
    DistilBertForSequenceClassification,
    BertTokenizer,
    DistilBertTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
import torch
import time

def tokenize_function(example):
    return tokenizer(example["text"], padding=True, truncation=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.tensor(logits).argmax(dim=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
    }

# ========== Load Dataset ==========
dataset = load_dataset("imdb")
dataset = dataset["test"].shuffle(seed=42).select(range(1000))  # reduce eval time

# ========== Define Eval Function ==========
def evaluate_model(model, tokenizer, model_name="model"):
    encoded_dataset = dataset.map(lambda x: tokenizer(x["text"], padding=True, truncation=True), batched=True)
    encoded_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    args = TrainingArguments(
        output_dir="./eval-temp",
        per_device_eval_batch_size=16,
        dataloader_drop_last=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=encoded_dataset,
        compute_metrics=compute_metrics
    )

    start_time = time.time()
    metrics = trainer.evaluate()
    end_time = time.time()

    metrics["inference_time"] = round(end_time - start_time, 2)
    print(f"\n📊 Evaluation for: {model_name}")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    return metrics

# ========== Evaluate Teacher ==========
teacher_tokenizer = BertTokenizer.from_pretrained("./bert-teacher")
teacher_model = BertForSequenceClassification.from_pretrained("./bert-teacher")
metrics_teacher = evaluate_model(teacher_model, teacher_tokenizer, "Teacher Model (BERT)")

# ========== Evaluate Student (Before Distillation) ==========
raw_student_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
raw_student_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
metrics_student_before = evaluate_model(raw_student_model, raw_student_tokenizer, "Student Model (Before Distillation)")

# ========== Evaluate Student (After Distillation) ==========
trained_student_tokenizer = DistilBertTokenizer.from_pretrained("./distilbert-student")
trained_student_model = DistilBertForSequenceClassification.from_pretrained("./distilbert-student")
metrics_student_after = evaluate_model(trained_student_model, trained_student_tokenizer, "Student Model (After Distillation)")