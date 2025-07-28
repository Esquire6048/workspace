import torch.nn.functional as F

def distill_loss_fn(student_logits, student_hidden, teacher_logits, teacher_hidden, labels,
                    alpha=1.0, beta=1.0, gamma=1.0, temperature=2.0):
    # CrossEntropy Loss
    loss_ce = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)),
                              labels.view(-1), ignore_index=-100)

    # Logit distillation loss (KL Divergence)
    s_logit = F.log_softmax(student_logits / temperature, dim=-1)
    t_logit = F.softmax(teacher_logits / temperature, dim=-1)
    loss_kl = F.kl_div(s_logit, t_logit, reduction="batchmean") * (temperature ** 2)

    # Hidden MSE loss (逐层)
    loss_hidden = 0.0
    for s, t in zip(student_hidden, teacher_hidden):
        loss_hidden += F.mse_loss(s, t)

    return alpha * loss_ce + beta * loss_kl + gamma * loss_hidden
