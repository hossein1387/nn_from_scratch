from .linear import Linear
from .funcs import scaled_dot_product


class TransformerBlock:

    def __init__(self, nheads, ):
        q = Linear(nheads)
        k = Linear(nheads)
        o = Linear(nheads)

    def attention(self, q, k, v):
        pass

    def forward(self, x):
        pass
    