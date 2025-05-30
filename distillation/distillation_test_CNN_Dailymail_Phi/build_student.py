import os
import torch
from student_model import StudentConfig, StudentModelForCausalLM
from transformers import AutoModelForCausalLM

# 加载教师模型（Phi-3）
teacher = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-3-mini-4k-instruct",
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True
)

# 生成 Student A 配置
config_a = StudentConfig(
    num_hidden_layers=8,
    hidden_size=1536,
    intermediate_size=4096,
    num_attention_heads=8,
    num_key_value_heads=8,
    vocab_size=teacher.config.vocab_size,
    max_position_embeddings=1024,
    attention_dropout=0.0,
    _attn_implementation="eager",
    rope_theta=10000.0
)


model_a = StudentModelForCausalLM(config_a)

os.makedirs("student_model_A", exist_ok=True)
model_a.save_pretrained("student_model_A")
config_a.save_pretrained("student_model_A")

# 生成 Student B 配置
config_b = StudentConfig(
    num_hidden_layers=4,
    hidden_size=1024,
    intermediate_size=2048,
    num_attention_heads=4,
    num_key_value_heads=4,
    vocab_size=teacher.config.vocab_size,
    max_position_embeddings=1024,
    attention_dropout=0.0,
    _attn_implementation="eager",
    rope_theta=10000.0
)

model_b = StudentModelForCausalLM(config_b)

# 注意：完全随机初始化，无需复制权重
os.makedirs("student_model_B", exist_ok=True)
model_b.save_pretrained("student_model_B")
config_b.save_pretrained("student_model_B")
