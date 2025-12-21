class MLXAdapter:
    def create_embedding_layer(self, vocab_size, hidden_size):
        import mlx.nn as nn

        return nn.Embedding(vocab_size, hidden_size)

    def create_causal_mask(self, seq_length, dtype):
        import mlx.nn as nn

        return nn.MultiHeadAttention.create_additive_causal_mask(seq_length).astype(dtype)

    def create_norm_layer(self, hidden_size, eps):
        import mlx.nn as nn

        return nn.RMSNorm(hidden_size, eps=eps)
