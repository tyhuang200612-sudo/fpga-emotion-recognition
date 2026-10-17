"""CPU-only synthetic checks. No FER2013 training images or GPU."""
import tempfile
import unittest
from pathlib import Path
import torch
import numpy as np
from src.model_c import TinyCNNC
from tools.quantize_c_int8 import fuse,verify_fused,make_params,infer_int,int_conv,write_export


class Int8CTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.model=TinyCNNC().eval()
        self.fused=fuse(self.model)
        self.params=make_params(self.fused,[0.04,0.04,0.04,0.04])

    def test_bn_fusion(self):
        self.assertLess(verify_fused(self.model,self.fused),1e-4)

    def test_integer_output_shapes_and_order(self):
        x=torch.rand(1,1,48,48)
        scores,gold=infer_int(x,self.params,collect=True)
        self.assertEqual(scores.shape,(1,3))
        self.assertEqual(len(gold),5)
        self.assertEqual(gold[0].dtype,np.dtype("uint8"))
        self.assertEqual(gold[-1].dtype,np.dtype("int32"))
        self.assertEqual(gold[0].shape,(1,16,48,48))

    def test_conv_matches_naive_integer_mac(self):
        # Direct exact integer formula independent of fast unfold/MM path
        p=self.params[0]; y=torch.zeros((1,1,48,48),dtype=torch.int64)
        for row in range(5):
            for col in range(5):
                y[0,0,row,col]=row*15+col
        fast=int_conv(y,p)
        w=p["w"][0,0].to(torch.int64)
        for yy in range(5):
            for xx in range(5):
                direct=int(p["b"][0])
                for ky in range(3):
                    for kx in range(3):
                        iy=yy+ky-1;ix=xx+kx-1
                        if 0<=iy<48 and 0<=ix<48:
                            direct+=int(w[ky,kx])*int(y[0,0,iy,ix])
                self.assertEqual(int(fast[0,0,yy,xx]),direct)

    def test_layer_weight_count(self):
        self.assertEqual(sum(p["w"].numel() for p in self.params),60240)


if __name__=="__main__":unittest.main()
