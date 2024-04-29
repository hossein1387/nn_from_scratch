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
from nn_from_scratch.nn import Linear

class TestLinearTiledOutput(unittest.TestCase):
    def test_linear_output(self):
        in_features, out_features, inter_mediate = 2048, 1024, 50
        x = np.random.rand(in_features, inter_mediate)
        weights = np.random.rand(out_features, inter_mediate)
        linear = Linear((inter_mediate, out_features), weights=weights, tile_sz=4)
        torch_out = torch.nn.functional.linear(torch.from_numpy(x), torch.from_numpy(weights)).numpy()
        out_mine = linear.forward(x)
        # import ipdb as pdb; pdb.set_trace()
        try:
            np.testing.assert_allclose(out_mine, torch_out, atol=1e-5)
            print("PASS")
        except AssertionError as e:
            print(e)
    
class TestLinearOutput(unittest.TestCase):
    def test_linear_output(self):
        in_features, out_features, inter_mediate = 2048, 1024, 50
        x = np.random.rand(in_features, inter_mediate)
        weights = np.random.rand(out_features, inter_mediate)
        linear = Linear((inter_mediate, out_features), weights=weights)
        torch_out = torch.nn.functional.linear(torch.from_numpy(x), torch.from_numpy(weights)).numpy()
        out_mine = linear.forward(x)
        # import ipdb as pdb; pdb.set_trace()
        try:
            np.testing.assert_allclose(out_mine, torch_out, atol=1e-5)
            print("PASS")
        except AssertionError as e:
            print(e)

if __name__ == '__main__':
  unittest.main()
