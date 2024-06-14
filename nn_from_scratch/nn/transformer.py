from typing import Any
import numpy as np
from .linear import Linear
from .funcs import softmax

class MultiHeadAttention:
    def __init__(self, nheads, d_model):
        assert (d_model%nheads==0), "Embedding dimension must be divisible by number of attention heads"
        self.nheads = nheads
        self.d_k = int(d_model/nheads)
        self.d_model = d_model
        self.q_w = Linear([d_model, d_model])
        self.k_w = Linear([d_model, d_model])
        self.v_w = Linear([d_model, d_model])
        self.o_w = Linear([d_model, d_model])
    def scaled_dot_product_attention_mine(self, q, k, v):
            scale_factor = 1/np.sqrt(q.shape[-1])
            attn_scores = q@k.transpose(-2, -1) * scale_factor
            attn_probs = softmax(attn_scores, axis=-1)
            output = attn_probs@v
            return output
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.shape()
        return x.view(batch_size, seq_length, self.nheads, self.d_k).transpose(1, 2)
    def forward(self, Q, K, V):
        batch_size, seq_length, _ = Q.shape()
        Q = self.q(self.split_heads(Q))
        K = self.q(self.split_heads(K))
        V = self.q(self.split_heads(V))
        output = self.scaled_dot_product_attention(Q,K,V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        output = self.o(output)
        return output
    def __call__(self, q, k, v):
         return self.forward(q, k, v)

class Transformer:
    def __init__(self, nheads, d_model):
        pass