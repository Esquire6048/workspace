import matplotlib.pyplot as plt
import json
import os

# === Manually input evaluation results ===
results = {
    "Student (Supervised Only)": {"accuracy": 0.914, "f1": 0.912, "inference_time_sec": 1.83},
    "Teacher (Fine-tuned BERT)": {"accuracy": 0.918, "f1": 0.916, "inference_time_sec": 3.49},
    "Student (Distilled from Teacher)": {"accuracy": 0.924, "f1": 0.921, "inference_time_sec": 1.68}
}

# === Bar Chart: Accuracy and F1 ===
model_names = list(results.keys())
accuracy_scores = [results[m]["accuracy"] for m in model_names]
f1_scores = [results[m]["f1"] for m in model_names]

x = range(len(model_names))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar([i - width/2 for i in x], accuracy_scores, width=width, label="Accuracy")
plt.bar([i + width/2 for i in x], f1_scores, width=width, label="F1 Score")
plt.xticks(ticks=x, labels=model_names, rotation=15)
plt.ylabel("Score")
plt.ylim(0.9, 0.93)  # ✅ 缩小纵轴范围，放大分数差异
plt.title("Model Accuracy and F1 Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("bar_comparison.png")
plt.show()

# === Loss Curve from Trainer logs ===
# This part requires that training logs are stored in Trainer's log_history.json

checkpoints = {
    "Student (Supervised Only)": "./distilbert-imdb/checkpoint-6250",
    "Teacher (Fine-tuned BERT)": "./distilbert-imdb-no-teacher/checkpoint-6250",
    "Student (Distilled from Teacher)": "./bert-imdb/checkpoint-6250"
}

plt.figure(figsize=(10, 6))

# Plot each loss curve
for label, path in checkpoints.items():
    log_file = os.path.join(path, "trainer_state.json")
    if not os.path.exists(log_file):
        print(f"⚠️ Log file not found for {label}: {log_file}")
        continue

    with open(log_file, "r") as f:
        data = json.load(f)
        loss_logs = [(entry["epoch"], entry["loss"]) for entry in data["log_history"] if "loss" in entry and "epoch" in entry]

    if loss_logs:
        epochs, losses = zip(*loss_logs)
        plt.plot(epochs, losses, marker='o', label=label)
    else:
        print(f"⚠️ No loss data found in: {label}")

# Plot formatting
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Comparison")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("multi_loss_comparison.png")
plt.show()