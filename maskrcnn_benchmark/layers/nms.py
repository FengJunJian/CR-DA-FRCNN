# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from ._utils import _C
#from maskrcnn_benchmark import _C
# from maskrcnn_benchmark import _C
#
# nms = _C.nms
from torchvision.ops import nms as ops_nms
nms = ops_nms
# nms.__doc__ = """
# This function performs Non-maximum suppresion"""
