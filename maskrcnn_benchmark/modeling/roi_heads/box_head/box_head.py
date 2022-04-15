# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator
from maskrcnn_benchmark.modeling.utils import cat

class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg)
        self.predictor = make_roi_box_predictor(cfg)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():# no grad computation
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)#ROIPooling
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)
            return x, result, {}, x, None

        loss_classifier, loss_box_reg, _ = self.loss_evaluator(
            [class_logits], [box_regression]
        )

        if self.training:
            with torch.no_grad():
                da_proposals = self.loss_evaluator.subsample_for_da(proposals, targets)
        da_ins_labels = cat([proposal.get_field("domain_labels") for proposal in da_proposals], dim=0).bool()
        da_ins_feas = self.feature_extractor(features, da_proposals)# features of instances
        # class_logits, box_regression = self.predictor(da_ins_feas)
        # _, _, da_ins_labels = self.loss_evaluator(
        #     [class_logits], [box_regression]
        # )
        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
            da_ins_feas,
            da_ins_labels
        )

    def featureROI(self, features, targets):
        """
        Aiming to extract the ROI features
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        featuresRoi = self.feature_extractor(features, targets)
        return featuresRoi
        #(
            # x,
            # proposals,
            # dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
            # da_ins_feas,
            # da_ins_labels
        # )


class SW_ROIBoxHead(ROIBoxHead):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg)
        self.predictor = make_roi_box_predictor(cfg)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)
        self.lc = cfg.MODEL.SW.LC
        self.gc = cfg.MODEL.SW.GC
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)

    def forward_predict(self, avgpooled_feat, proposals, feat_local=None,feat_global=None,targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        # if self.training:
        #     # Faster R-CNN subsamples during training the proposals with a fixed
        #     # positive / negative ratio
        #     with torch.no_grad():# no grad computation
        #         proposals = self.loss_evaluator.subsample(proposals, targets)
        #
        # # extract features that will be fed to the final classifier. The
        # # feature_extractor generally corresponds to the pooler + heads
        # pooled_feat = self.feature_extractor(features, proposals)#ROIPooling
        batch_size=len(proposals)
        #SW-DA-FRCNN
        if self.lc:
            feat_pixel = torch.repeat_interleave(feat_local.view(batch_size, -1),torch.tensor(avgpooled_feat.size(0)/batch_size,dtype=int,device=feat_local.device),dim=0)
            avgpooled_feat = torch.cat((feat_pixel, avgpooled_feat), 1)
            #if self.da_use_contex:
            # instance_pooled_feat = torch.cat(
            #     (feat_pixel.detach(), pooled_feat), 1
            # )
        if self.gc:
            feat = feat_global.view(batch_size, -1).repeat(avgpooled_feat.size(0), 1)
            avgpooled_feat = torch.cat((feat, avgpooled_feat), 1)
            # if self.da_use_contex:
            # instance_pooled_feat = torch.cat(
            #     (feat.detach(), instance_pooled_feat), 1
            # )

        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(avgpooled_feat)

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)
            return avgpooled_feat, result, {}, avgpooled_feat, None,None


        loss_classifier, loss_box_reg, _ = self.loss_evaluator(
            [class_logits], [box_regression]
        )

        if self.training:
            with torch.no_grad():
                da_proposals = self.loss_evaluator.subsample_for_da(proposals, targets)
        da_ins_labels = cat([proposal.get_field("domain_labels") for proposal in da_proposals], dim=0).bool()
        da_ins_feas = self.feature_extractor(avgpooled_feat, da_proposals)# features of instances
        # class_logits, box_regression = self.predictor(da_ins_feas)
        # _, _, da_ins_labels = self.loss_evaluator(
        #     [class_logits], [box_regression]
        # )
        return (
            avgpooled_feat,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
            da_ins_feas,
            da_ins_labels,
            class_logits
        )
    def forward_pooled_feat(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():# no grad computation
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        avgpooled_feat = self.feature_extractor(features, proposals)#ROIPooling

        return (
            avgpooled_feat,
            proposals,
        )
    def featureROI(self, features, targets):
        """
        Aiming to extract the ROI features
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        # extract features of proposals. The
        # feature_extractor generally corresponds to the pooler
        featuresRoi = self.feature_extractor(features, targets)

        return featuresRoi


def build_roi_box_head(cfg):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    if cfg.MODEL.SW.SW_ROIBoxHead:
        return SW_ROIBoxHead(cfg)
    else:
        return ROIBoxHead(cfg)
