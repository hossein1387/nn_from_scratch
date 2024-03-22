#!/usr/bin/env python

# if you'd like to use the line profiler
try:
  import line_profiler
  prof = line_profiler.LineProfiler()
  import builtins
  builtins.__dict__['profile'] = prof
  # add @profile decorator to probe
except ImportError:
  prof = None

import torch
import time
import cProfile
import pstats
import unittest
import numpy as np
from nn_from_scratch.nn import Conv2d

def profile_conv(bs, cout, cin, k, cnt=10):
  img = np.zeros((bs, 3, 28, 28))
  conv2d = Conv2d((cout, cin, k, k), padding=(0,0))
  fpt = 0.0
  for i in range(cnt):
    et0 = time.time()
    _ = conv2d.forward(img)
    et1 = time.time()
    fpt += (et1-et0)
  return fpt/cnt

class TestConvSpeed(unittest.TestCase):
  def test_forward_3x3(self):
    # warmup
    profile_conv(1, 64, 3, 3, cnt=1)

    # profile
    pr = cProfile.Profile(timer=lambda: int(time.time()*1e9), timeunit=1e-6)
    pr.enable()
    fpt = profile_conv(1, 64, 3, 3, cnt=1)
    pr.disable()
    ps = pstats.Stats(pr)
    ps.strip_dirs()
    ps.sort_stats('cumtime')
    ps.print_stats(0.3)

    if prof is not None:
      prof.print_stats()

    print("forward pass:  %.3f ms" % (fpt*1000))


class TestConvOutput(unittest.TestCase):
    def test_conv2d_output(self):
        stride = (2,2)
        padding= (0,0)
        ci = 3
        co = 64
        w = 28
        h = 28
        k = 3
        b = 1
        x = np.random.rand(b, ci, w, h)
        kernel = np.random.rand(co, ci, k, k)
        conv2d = Conv2d((co, ci, k, k), stride=stride, padding=padding, weights=kernel)
        torch_out = torch.conv2d(torch.from_numpy(x), torch.from_numpy(kernel), stride=stride, padding=padding).numpy()
        out_mine = conv2d.forward(x)
        # import ipdb as pdb; pdb.set_trace()
        try:
            np.testing.assert_allclose(out_mine, torch_out, atol=1e-5)
            print("PASS")
        except AssertionError as e:
            print(e)
    

if __name__ == '__main__':
  unittest.main()
