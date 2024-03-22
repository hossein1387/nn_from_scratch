import numpy as np

class Conv2d:
    def __init__(self, shape, stride=(1,1), padding=(0,0), weights=[]):
        self.stride = stride
        self.padding = padding
        self.shape = shape
        assert len(shape)==4, "Expecting shape of 4 dims but got {}".format(len(shape))
        if np.any(weights):
            self.weights = weights
        else:
            self.weights = np.random.rand(*shape)
    
    def forward(self, x):
        b, cin, w, h = x.shape
        cout, cin, k, k = self.shape
        wout = int((w-k+self.padding[0])/self.stride[0])+1
        hout = int((h-k+self.padding[1])/self.stride[1])+1
        out = np.zeros((b, cout, wout, hout))
        for batch in range(b):
            for co in range(cout):
                for wo in range(wout):
                    for ho in range(hout):
                        tmp = 0
                        for ci in range(cin):
                            w_offset = wo*self.stride[0]
                            h_offset = ho*self.stride[1]
                            tmp += (x[batch, ci, w_offset:w_offset+k, h_offset:h_offset+k]*self.weights[co, ci, :, :]).sum()
                        out[batch, co, wo, ho] = tmp
        return out
