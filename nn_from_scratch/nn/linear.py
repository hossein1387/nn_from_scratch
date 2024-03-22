import numpy as np

class Linear:
    def __init__(self, shape, weights=[], tiled=False, tile_sz = 4):
        if np.any(weights):
            self.weights = weights
        else:
            self.weights = np.random.rand(*shape)
        self.tiled = tiled
        self.tile_sz = tile_sz

    def matmul_tiled(self, in1, y1, x1, in2, y2, x2):
        in4_1 = in1[y1:y1 + self.tile_sz, x1:x1 + self.tile_sz]
        in4_2 = in2[y2:y2 + self.tile_sz, x2:x2 + self.tile_sz]
        out4 = np.matmul(in4_1, in4_2)
        return out4
        
    def forward(self, x):
        # import ipdb as pdb; pdb.set_trace()
        w, h = x.shape
        m, n = self.weights.transpose().shape
        assert h==m, "Size mismatch, x.shape [{},{}], weight.shape [{},{}]".format(w,h,m,n)
        self.weights = self.weights.transpose()
        out = np.zeros((w, n))
        if self.tiled:
            for i in range(0, w, self.tile_sz):
                for j in range(0, n, self.tile_sz):
                    tmp = np.zeros((self.tile_sz, self.tile_sz))
                    for k in range(0, m, self.tile_sz):
                        tmp += self.matmul_tiled(x, i, k, self.weights, k, j)
                    out[i:i+self.tile_sz, j:j+self.tile_sz] = tmp
        else:
            for i in range(w):
                for j in range(n):
                    tmp = 0
                    for k in range(m):
                        tmp += x[i, k] * self.weights[k, j]
                    out[i,j] = tmp
        return out
