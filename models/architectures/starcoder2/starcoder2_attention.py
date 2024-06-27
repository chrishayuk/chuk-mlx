import mlx.nn as nn
import mlx.core as mx
from models.model_config import ModelConfig

class StarCoder2Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size

        use_bias = getattr(config, 'attention_bias', False)
        attention_dropout = getattr(config, 'attention_dropout', 0.1)

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=use_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=use_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=use_bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=use_bias)

        self.attn_dropout = nn.Dropout(attention_dropout)
        self.scale_attn_weights = True

    def _project_and_reshape(self, hidden_states):
        batch_size, seq_length, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        return q, k, v

    def _compute_attention(self, q, k, v, attention_mask):
        attn_weights = mx.matmul(q, k.transpose(0, 1, 3, 2))
        
        if self.scale_attn_weights:
            attn_weights = attn_weights / (self.head_dim ** 0.5)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = mx.softmax(attn_weights, axis=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = mx.matmul(attn_weights, v)
        return attn_output

    def _merge_heads(self, attn_output):
        batch_size, _, seq_length, _ = attn_output.shape
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_length, self.hidden_size)
        return attn_output

    def __call__(self, hidden_states, attention_mask=None, cache=None):
        q, k, v = self._project_and_reshape(hidden_states)
        attn_output = self._compute_attention(q, k, v, attention_mask)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.o_proj(attn_output)

        return attn_output, None  # Return None for cache