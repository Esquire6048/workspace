import torch
import torch.nn as nn
from transformers import PreTrainedModel, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer  # 使用 Phi-3 类似结构
from transformers.modeling_outputs import CausalLMOutputWithPast

class StudentConfig(LlamaConfig):
    model_type = "student"

    def __init__(self,
                 vocab_size=50280,
                 hidden_size=1536,
                 intermediate_size=4096,
                 num_hidden_layers=8,
                 num_attention_heads=8,
                 max_position_embeddings=1024,
                 initializer_range=0.02,
                 rms_norm_eps=1e-5,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps


class StudentModelForCausalLM(PreTrainedModel):
    config_class = StudentConfig

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :]  # [B, 1, 1, S]

        hidden_states = self.embed_tokens(input_ids)
        all_hidden_states = []

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids
            )[0]
            all_hidden_states.append(hidden_states)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            hidden_states=all_hidden_states  # 用于蒸馏
        )


