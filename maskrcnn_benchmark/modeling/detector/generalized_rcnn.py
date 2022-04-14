# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list
from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from ..da_heads.da_heads import build_da_heads
from maskrcnn_benchmark.modeling.poolers import Pooler
# import torchvision
# torchvision.models.detection.FasterRCNN
# modify according to torchvision.models.detection.FasterRCNN(GeneralizedRCNN)

class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        self.cfg=cfg
        self.backbone = build_backbone(cfg)#feature extractor  backbone/backbone.py
        self.rpn = build_rpn(cfg) # rpn/rpn.py
        self.roi_heads = build_roi_heads(cfg) #roi_heads/roi_heads.py
        self.da_heads = build_da_heads(cfg)
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION

        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        self.pooler = Pooler(
            output_size=(resolution, resolution),
            scales=cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES,
            sampling_ratio=sampling_ratio,
        )

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:#model forward
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)#[1,3,608,1088]
        features = self.backbone(images.tensors)#features [1,1024,38,68]
        proposals, proposal_losses = self.rpn(images, features, targets)#proposals：[BoxList(num_boxes=1000, image_width=1066, image_height=600, mode=xyxy)]
        da_losses = {}
        if self.roi_heads:
            x, result, detector_losses, da_ins_feas, da_ins_labels = self.roi_heads(features, proposals,
                                                                                    targets)#x:[1000, 2048, 7, 7], da_ins_feas:[1000, 2048, 7, 7]
            if self.da_heads:
                da_losses = self.da_heads(features, da_ins_feas, da_ins_labels, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses.update(da_losses)
            return losses

        return result

    def featureTarget(self,images, targets=None):
        """
        Aiming to obtain the feature vectors of proposals.
        For rpn, it generates the features of anchors.(using Pooler with resolution as same as ROI)
        For ROI, it generates the features of proposals.
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image

        Returns:
            proposals (list[BoxList]):the output from the rpn. The proposals contains the fields of features and other like `scores`, `labels` and `mask` (for Mask R-CNN models)
            rois (list[BoxList]):the output from the ROI. The format of rois is as same as the proposals. (optional)
        """
        if self.training and targets is None:  # model forward
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)  # [1,3,608,1088]
        features = self.backbone(images.tensors)  # features [1,1024,38,68] extract the features of the backbone
        ###################################################
        # las=list(self.backbone.body.children())
        # fss=[]
        # x1 = self.backbone.body.stem(images.tensors)
        # x2 = self.backbone.body.layer1(x1)
        # x3 = self.backbone.body.layer2(x2)
        # x30=self.backbone.body.layer2[0](x2)#down 2x
        # x30d=self.backbone.body.layer2[0].downsample(x2)#down 2x
        # x31c1 = self.backbone.body.layer2[1].conv1(x30d)  # down 2x
        # x=images.tensors
        # fss.append(x.shape)
        # for la in las:
        #     x=la(x)
        #     fss.append(x.shape)
        ######################################################
        # if self.roi_heads:
        #self.box(features, proposals, targets)
        boxFeatures = self.roi_heads.box.featureROI(features, targets)

        # proposals, proposal_losses,targetFeatures = self.rpn.featureRPN(images, features,
        #                                       targets)  # proposals：[BoxList(num_boxes=1000, image_width=1066, image_height=600, mode=xyxy)]
        #model.backbone.body.layer1,layer2,layer3
        boxes = []
        #boxFeatures = None
        if targets:
            # for feature in features:
            assert len(features) == 1, 'using the FPN'  # TODO fix for FPN
            #_, C, H, W = features[0].shape
            #boxes = [target.resize((W, H)) for target in targets]  # (width, height)
            boxes=targets.copy()

            #boxFeatures = self.pooler(features, boxes)  # target feature
        for i,t in enumerate(boxes):
            if len(t)>0:
                t.add_field('featureROI',boxFeatures[0])# only for batch==1
        return boxes

    def featurePredict(self,images, targets=None):
        """
        Not yet finished!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        Aiming to obtain the feature vectors of proposals.
        For rpn, it generates the features of anchors.(using Pooler with resolution as same as ROI)
        For ROI, it generates the features of proposals.
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            proposals (list[BoxList]):the output from the rpn. The proposals contains the fields of features and other like `scores`, `labels` and `mask` (for Mask R-CNN models)
            rois (list[BoxList]):the output from the ROI. The format of rois is as same as the proposals. (optional)
        """
        if self.training and targets is None:  # model forward
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)  # [1,3,608,1088]
        features = self.backbone(images.tensors)  # features [1,1024,38,68] extract the features of the backbone
        proposals, proposal_losses,anchorsFeatures = self.rpn.featureRPN(images, features,
                                              targets)  # proposals：[BoxList(num_boxes=1000, image_width=1066, image_height=600, mode=xyxy)]
        # proposals.addfield('anchorF',anchorsFeatures)
        # da_losses = {}
        if self.roi_heads:
            x, rois, detector_losses, da_ins_feas, da_ins_labels,proposalsFeatures = \
                self.roi_heads.featureROI(features, proposals, targets)
            rois.addfield('anchorF', anchorsFeatures)
            return proposals,rois
            # if self.da_heads:
            #     da_losses = self.da_heads(features, da_ins_feas, da_ins_labels, targets)
        else:
            return proposals, []
