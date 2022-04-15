# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN,SW_DA_RCNN


_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN,"SW_DA_RCNN":SW_DA_RCNN}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
