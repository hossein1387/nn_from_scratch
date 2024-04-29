import numpy as np
from .linear import Linear
from .funcs import softmax

class MultiHeadAttention:

    def __init__(self, nheads, d_model):
        assert (d_model%nheads==0), "Embedding dimension must be divisible by number of attention heads"
        self.nheads = nheads
        self.d_k = int(d_model/nheads)
        self.d_model = d_model

        self.q = Linear([d_model, d_model])
        self.k = Linear([d_model, d_model])
        self.v = Linear([d_model, d_model])
        self.o = Linear([d_model, d_model])

    def scaled_dot_product_attention(self, q, k, v):
        attn_scores = q@k.transpose()/np.sqrt(self.d_k)
        attn_probs = softmax(attn_scores)
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

class Transformer:
    def __init__(self, nheads, d_model):
        pass