import numpy as np

class Function:
  def __init__(self, requires_grad=False):
    self.requires_grad = requires_grad

  def forward(self, *args, **kwargs): raise NotImplementedError(f"forward not implemented for {type(self)}")
  def backward(self, *args, **kwargs): raise RuntimeError(f"backward not implemented for {type(self)}")

class softmax(Function):
    def forward(self, x, axis=-1):
        t1 = x - np.max(x, axis=axis, keepdims=True)
        t2 = np.exp(t1)
        return (t2)/(np.sum(t2, axis=axis, keepdims=True))
