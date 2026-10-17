"""Experimental C5 spatial-head models. Frozen original C is left unchanged.

The proposed model changes ONLY the terminal adaptive average pooling
from 1x1 to 2x2 and the 3-class fully connected head. The four 3x3
convolution layers retain exact original channels 16/32/64/64.

Weights / MACs below are architectural theoretical figures, NOT verified RTL,
post-BN-fold INT8 accuracy, FPGA BRAM, timing closure or on-board throughput.
"""
from __future__ import annotations
import torch
from torch import nn
from src.model_c import TinyCNNC, CLASS_NAMES_C, C_CHANNELS, resource_estimates

class TinyCNNCSpatial2x2(nn.Module):
    def __init__(self):
        super().__init__()
        # Reuse exactly the same convolutional class, channels, strides, BN and ReLU.
        template=TinyCNNC()
        self.features=template.features
        self.gap=nn.AdaptiveAvgPool2d((2, 2))
        self.fc=nn.Linear(64*2*2,len(CLASS_NAMES_C))

    def forward(self,x:torch.Tensor)->torch.Tensor:
        if x.ndim!=4 or x.shape[1:]!=(1,48,48):
            raise ValueError("TinyCNNCSpatial2x2 expects [N,1,48,48]")
        return self.fc(self.gap(self.features(x)).flatten(1))

def initialize_spatial_from_c(source:TinyCNNC,target:TinyCNNCSpatial2x2)->None:
    """Make C5 output EXACTLY the C baseline's scores at initialization.

    2x2 adaptive mean of a 12x12 feature map has four equal 6x6 partitions.
    The average of four quadrant means is the global mean. Copying each
    original FC weight into all 4 positions divided by four therefore
    reproduces 1x1 logits (up to float rounding), isolating spatial head
    training from arbitrary random new-head initialization.
    """
    if not isinstance(source,TinyCNNC) or not isinstance(target,TinyCNNCSpatial2x2):
        raise TypeError("Expected original C and C5 2x2 models")
    target.features.load_state_dict(source.features.state_dict(),strict=True)
    with torch.no_grad():
        target.fc.weight.copy_(source.fc.weight.repeat_interleave(4,dim=1)/4)
        target.fc.bias.copy_(source.fc.bias)

def theoretical_resources():
    c=resource_estimates()
    conv_weights=c["weight_count"]-64*3
    new_fc=64*4*len(CLASS_NAMES_C)
    extras=new_fc-64*len(CLASS_NAMES_C)
    base_pytorch=sum(p.numel() for p in TinyCNNC().parameters())
    experimental_pytorch=sum(p.numel() for p in TinyCNNCSpatial2x2().parameters())
    assert conv_weights+new_fc==60816
    assert experimental_pytorch-base_pytorch==extras==576
    return {
        "base_conv_weights":conv_weights,
        "gap1_int8_weights_without_bn_bias":c["weight_count"],
        "gap2_int8_weights_without_bn_bias":conv_weights+new_fc,
        "extra_classifier_weights":extras,
        "gap1_trainable_parameters_including_bn_bias":base_pytorch,
        "gap2_trainable_parameters_including_bn_bias":experimental_pytorch,
        "extra_fc_macs":extras,
        "gap1_macs_per_frame_excluding_pool":c["macs_per_frame"],
        "gap2_macs_per_frame_excluding_pool":c["macs_per_frame"]+extras,
        "post_pool_activation_values_gap1":64,
        "post_pool_activation_values_gap2":256,
        "spatial_map_before_pooling":[64,12,12],
        "input_shape":[1,48,48],
        "pooling_arithmetic_not_in_mac_count":True,
        "requires_bn_fusion":True,
        "int8_performance_unverified":True,
        "fpga_resource_timing_unverified":True,
        "deployment":False,
    }
