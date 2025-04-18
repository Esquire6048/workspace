from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
import torch
import time

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.tensor(logits).argmax(dim=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
    }

def prepare_dataset(tokenizer, num_samples=1000):
    dataset = load_dataset("imdb")["test"].shuffle(seed=42).select(range(num_samples))
    encoded = dataset.map(lambda x: tokenizer(x["text"], padding=True, truncation=True), batched=True)
    encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return encoded

def evaluate_model(model, tokenizer, name):
    dataset = prepare_dataset(tokenizer)
    args = TrainingArguments(
        output_dir=f"./eval-temp-{name.replace(' ', '_')}",
        per_device_eval_batch_size=16,
        dataloader_drop_last=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=dataset,
        compute_metrics=compute_metrics,
    )
    print(f"\n📊 Evaluating: {name}")
    start = time.time()
    metrics = trainer.evaluate()
    end = time.time()
    metrics["inference_time_sec"] = round(end - start, 2)
    for k, v in metrics.items():
        print(f"{k}: {v}")
    return metrics

# 1. Student model (trained only on ground truth, no distillation)
student_finetuned = DistilBertForSequenceClassification.from_pretrained("./models/distilbert")
student_tokenizer = DistilBertTokenizer.from_pretrained("./models/distilbert")
evaluate_model(student_finetuned, student_tokenizer, "Student (Fine-tuned DistilBERT)")

# 2. Teacher model (fully trained)
teacher_model = BertForSequenceClassification.from_pretrained("./models/bert")
teacher_tokenizer = BertTokenizer.from_pretrained("./models/bert")
evaluate_model(teacher_model, teacher_tokenizer, "Teacher (Fine-tuned BERT)")

# 3. Student model distilled from teacher
student_distilled = DistilBertForSequenceClassification.from_pretrained("./models/distilbert-distilled")
student_distilled_tokenizer = DistilBertTokenizer.from_pretrained("./models/distilbert-distilled")
evaluate_model(student_distilled, student_distilled_tokenizer, "Student (Distilled from Teacher)")