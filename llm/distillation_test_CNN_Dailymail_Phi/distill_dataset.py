import os
import torch
from torch.utils.data import Dataset

class DistillDataset(Dataset):
    def __init__(self, data, tokenizer, teacher_dir, index_map, max_input_len=1024, max_target_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.teacher_dir = teacher_dir
        self.index_map = index_map
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        actual_idx = self.index_map[idx]
        sample = self.data[idx]

        inputs = self.tokenizer(sample["input"], max_length=self.max_input_len,
                                padding="max_length", truncation=True, return_tensors="pt")
        labels = self.tokenizer(sample["output"], max_length=self.max_target_len,
                                padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
        labels[labels == self.tokenizer.pad_token_id] = -100

        teacher_path = os.path.join(self.teacher_dir, f"teacher_{actual_idx}.pt")
        try:
            data = torch.load(teacher_path)
            logits = data["logits"].squeeze(0)
            hidden_states = torch.stack(data["hidden_states"]).squeeze(1)
        except Exception as e:
            print(f"❌ teacher_{actual_idx}.pt 无法加载或缺失字段: {e}")
            return {
                "input_ids": inputs["input_ids"][0],
                "attention_mask": inputs["attention_mask"][0],
                "labels": labels,
                "teacher_logits": None,
                "teacher_hidden": None
            }

        return {
            "input_ids": inputs["input_ids"][0],
            "attention_mask": inputs["attention_mask"][0],
            "labels": labels,
            "teacher_logits": logits,
            "teacher_hidden": hidden_states
        }
