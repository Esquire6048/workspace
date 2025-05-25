import torch

class DistillDataCollator:
    def __call__(self, features):
        # 丢弃 teacher_logits 为 None 的样本
        features = [
            f for f in features
            if isinstance(f.get("teacher_logits", None), torch.Tensor)
        ]
        if len(features) == 0:
            raise ValueError("❌ 本 batch 全部样本无效（teacher_logits 缺失）")

        batch = {}
        for key in ["input_ids", "attention_mask", "labels"]:
            batch[key] = torch.stack([f[key] for f in features])
        batch["teacher_logits"] = torch.stack([f["teacher_logits"] for f in features])
        batch["teacher_hidden"] = torch.stack([f["teacher_hidden"] for f in features])

        return batch
