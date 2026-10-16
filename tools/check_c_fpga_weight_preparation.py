"""Minimal deployment sanity checks for TinyCNN-C weight-exported assets.

Note: this is NOT full FPGA hardware inference. No bitstream is produced.
"""
from __future__ import annotations
import json
from pathlib import Path

def check_export(root: Path) -> dict:
    meta=json.loads((root/'model_arch.json').read_text(encoding='utf-8'))
    if meta.get('NOT_BITSTREAM_READY') is not True:
        raise ValueError('Weight-only exports must not claim ready bitstream')
    assert meta['classes']==['neutral','happy','sad']
    assert meta['input_shape']==[1,48,48]
    assert len(meta['layers'])==5
    assert meta['int8_weight_bytes']==60240
    total=0
    for i,layer in enumerate(meta['layers'],1):
        b=(root/f'weight_layer{i:02d}.bin').read_bytes()
        lines=(root/f'weight_layer{i:02d}.mem').read_text(encoding='ascii').splitlines()
        if len(b)!=layer['signed_int8_weight_count'] or len(lines)!=len(b):
            raise AssertionError('Weight count mismatch: '+str(i))
        if any(int(t,16)!=a for t,a in zip(lines,b)):
            raise AssertionError('HEX/BIN mismatch: '+str(i))
        total+=len(b)
    if total!=60240:raise AssertionError('Unexpected C weight size')
    return {'layers':5,'weight_bytes':total,'source_sha256':meta['source_checkpoint_sha256']}
