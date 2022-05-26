# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import print_function
import torch
import torch.nn.functional as F
from torch import nn
from maskrcnn_benchmark.layers import GradientScalarLayer
# from maskrcnn_benchmark.layers.gradient_scalar_layer import gradient_scalar
from maskrcnn_benchmark.layers.sigmoid_focal_loss import SigmoidFocalLoss
from .loss import make_da_heads_loss_evaluator

class DAImgHead(nn.Module):
    """
    Adds a simple Image-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            USE_FPN (boolean): whether FPN feature extractor is used
        """
        super(DAImgHead, self).__init__()

        self.conv1_da = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1)
        self.conv2_da = nn.Conv2d(512, 1, kernel_size=1, stride=1)

        for l in [self.conv1_da, self.conv2_da]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        img_features = []
        for feature in x:
            t = F.relu(self.conv1_da(feature))
            img_features.append(self.conv2_da(t))
        return img_features


class DAInsHead(nn.Module):
    """
    Adds a simple Instance-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(DAInsHead, self).__init__()
        self.fc1_da = nn.Linear(in_channels, 1024)
        self.fc2_da = nn.Linear(1024, 1024)
        self.fc3_da = nn.Linear(1024, 1)

        for l in [self.fc1_da, self.fc2_da]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
        nn.init.normal_(self.fc3_da.weight, std=0.05)
        nn.init.constant_(self.fc3_da.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc1_da(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.fc2_da(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.fc3_da(x)
        return x


class DomainAdaptationModule(torch.nn.Module):
    """
    Module for Domain Adaptation Component. Takes feature maps from the backbone and instance
    feature vectors, domain labels and proposals. Works for both FPN and non-FPN.
    """

    def __init__(self, cfg):
        super(DomainAdaptationModule, self).__init__()

        self.cfg = cfg.clone()

        stage_index = 4
        stage2_relative_factor = 2 ** (stage_index - 1)
        res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        num_ins_inputs = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM if cfg.MODEL.BACKBONE.CONV_BODY.startswith('V') else res2_out_channels * stage2_relative_factor
        
        self.resnet_backbone = cfg.MODEL.BACKBONE.CONV_BODY.startswith('R')
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        
        self.img_weight = cfg.MODEL.DA_HEADS.DA_IMG_LOSS_WEIGHT
        self.ins_weight = cfg.MODEL.DA_HEADS.DA_INS_LOSS_WEIGHT
        self.cst_weight = cfg.MODEL.DA_HEADS.DA_CST_LOSS_WEIGHT

        self.grl_img = GradientScalarLayer(-1.0*self.cfg.MODEL.DA_HEADS.DA_IMG_GRL_WEIGHT)#GRL
        self.grl_ins = GradientScalarLayer(-1.0*self.cfg.MODEL.DA_HEADS.DA_INS_GRL_WEIGHT)
        self.grl_img_consist = GradientScalarLayer(1.0*self.cfg.MODEL.DA_HEADS.DA_IMG_GRL_WEIGHT)
        self.grl_ins_consist = GradientScalarLayer(1.0*self.cfg.MODEL.DA_HEADS.DA_INS_GRL_WEIGHT)
        
        in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS

        self.imghead = DAImgHead(in_channels)
        self.inshead = DAInsHead(num_ins_inputs)
        self.loss_evaluator = make_da_heads_loss_evaluator(cfg)

    def forward(self, img_features, da_ins_feature, da_ins_labels, targets=None):
        """
        Arguments:
            img_features (list[Tensor]): features computed from the images that are
                used for computing the predictions.
            da_ins_feature (Tensor): instance-level feature vectors
            da_ins_labels (Tensor): domain labels for instance-level feature vectors
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        if self.resnet_backbone:
            da_ins_feature = self.avgpool(da_ins_feature)#da_ins_feature:[1000, 2048, 1, 1]
        da_ins_feature = da_ins_feature.view(da_ins_feature.size(0), -1)#[1000, 2048]

        img_grl_fea = [self.grl_img(fea) for fea in img_features]
        ins_grl_fea = self.grl_ins(da_ins_feature)
        img_grl_consist_fea = [self.grl_img_consist(fea) for fea in img_features]
        ins_grl_consist_fea = self.grl_ins_consist(da_ins_feature)

        da_img_features = self.imghead(img_grl_fea)
        da_ins_features = self.inshead(ins_grl_fea)
        da_img_consist_features = self.imghead(img_grl_consist_fea)
        da_ins_consist_features = self.inshead(ins_grl_consist_fea)
        da_img_consist_features = [fea.sigmoid() for fea in da_img_consist_features]
        da_ins_consist_features = da_ins_consist_features.sigmoid()
        if self.training:
            da_img_loss, da_ins_loss, da_consistency_loss = self.loss_evaluator(
                da_img_features, da_ins_features, da_img_consist_features, da_ins_consist_features, da_ins_labels, targets
            )
            losses = {}
            if self.img_weight > 0:
                losses["loss_da_image"] = self.img_weight * da_img_loss
            if self.ins_weight > 0:
                losses["loss_da_instance"] = self.ins_weight * da_ins_loss
            if self.cst_weight > 0:
                losses["loss_da_consistency"] = self.cst_weight * da_consistency_loss
            return losses
        return {}

class SWDomainModule(torch.nn.Module):
    """
        Module for Domain Adaptation Component. Takes feature maps from the backbone and instance
        feature vectors, domain labels and proposals. Works for both FPN and non-FPN.
        """
    def __init__(self, cfg):
        super(SWDomainModule, self).__init__()

        self.cfg = cfg.clone()

        stage_index = 4
        stage2_relative_factor = 2 ** (stage_index - 1)
        res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        num_ins_inputs = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM if cfg.MODEL.BACKBONE.CONV_BODY.startswith(
            'V') else res2_out_channels * stage2_relative_factor #num_ins_inputs=2048
        self.resnet_backbone = cfg.MODEL.BACKBONE.CONV_BODY.startswith('R')
        ########################################################################################
        #SW-DA-FRCNN
        self.lc=cfg.MODEL.SW.LC
        self.gc=cfg.MODEL.SW.GC

        if self.lc:
            num_ins_inputs += 128
        if self.gc:
            num_ins_inputs += 128

        self.netD_pixel = netD_pixel(cfg, context=self.lc)  # 局部特征对齐
        self.netD = netD(cfg, context=self.gc)  # 全局特征对齐
        self.AdaAvg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        ########################################################################################
        #self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)

        self.grl_ICR_CCR = GradientScalarLayer(-1.0 * self.cfg.MODEL.ICR_CCR.DA_GRL_WEIGHT)# GRL
        self.grl_SW=GradientScalarLayer(-1.0 * self.cfg.MODEL.SW.DA_GRL_WEIGHT)# GRL
        #in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        #self.imghead = DAImgHead(in_channels)
        ########################################################################################
        #ICR & CCR
        #Image-level categorical regularization (ICR)
        self.conv_ICR = nn.Conv2d(cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES - 1,
                                  1, 1, 0)
        nn.init.normal_(self.conv_ICR.weight, 0.0, 0.01)
        nn.init.constant_(self.conv_ICR.bias, 0.0,)
        ##categorical consistency regularization (CCR)
        self.insHead = DAInsHead(num_ins_inputs)
        ########################################################################################
        # loss
        self.sigmoid_focal_loss = SigmoidFocalLoss(
            cfg.MODEL.RETINANET.LOSS_GAMMA,
            cfg.MODEL.RETINANET.LOSS_ALPHA
        )
        #self.loss_evaluator = make_da_heads_loss_evaluator(cfg)

    def forwardCR(self, img_features2, flattenpooled_feat, class_logits,da_ins_labels, targets=None):
        """
        Arguments:
            img_features (list[Tensor]): features computed from the images that are
                used for computing the predictions.
            da_ins_feature (Tensor): instance-level feature vectors
            da_ins_labels (Tensor): domain labels for instance-level feature vectors
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        ############################################################################################
        img_features2=img_features2[0]
        #batch_size=img_features2.size(0)
        ICR_feat = self.AdaAvg_pool(img_features2)
        ICR_feat = self.conv_ICR(ICR_feat).squeeze(-1).squeeze(-1)
        # cls_feat = self.conv_lst(self.bn1(self.avg_pool(base_feat))).squeeze(-1).squeeze(-1)

        # ICR_CCR_InsFeatures = [self.grl_ICR_CCR(fea) for fea in flattenpooled_feat]
        ICR_CCR_InsFeatures = self.grl_ICR_CCR(flattenpooled_feat)

        da_ins_features = self.insHead(ICR_CCR_InsFeatures)

        if self.training:
            domain_labels = torch.tensor([t.get_field('is_source').any() for t in targets]).to(self.cfg.MODEL.DEVICE)
            source_mask=domain_labels#source domain label:1
            source_inds=torch.where(source_mask==1)[0]
            #target_inds = torch.where(source_mask == 0)[0]

            im_cls_lb = torch.zeros((source_inds.size(0), self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES -1),
                                    dtype=ICR_feat.dtype,device=ICR_feat.device)
            for i, s in enumerate(source_inds):
                gt_classes = targets[s].get_field('labels')
                im_cls_lb[i][gt_classes - 1] = 1

            # Image-level categorical regularization (ICR) loss
            ICR_loss = nn.BCEWithLogitsLoss()(ICR_feat[source_inds] , im_cls_lb)#would be useless for two class detection

            # compute categorical consistency regularization (CCR) loss for instances
            cls_prob = torch.softmax(class_logits, 1)
            cls_pre_label = cls_prob.argmax(1).detach()
            cls_feat_sig = torch.sigmoid(ICR_feat).detach()
            source_ins_inds=torch.where(da_ins_labels)[0]
            target_ins_inds = torch.where(torch.logical_not(da_ins_labels))[0]
            source_weight=[1.0]*len(source_ins_inds)
            source_target_weight = []
            source_target_weight.extend(source_weight)
            for j,target_ins_ind in enumerate(target_ins_inds):
                label_i = cls_pre_label[target_ins_ind].item()
                if label_i > 0:
                    diff_value = torch.exp(
                        torch.abs(cls_feat_sig[label_i - 1] - cls_prob[target_ins_ind][label_i])
                    ).item()#compute the weight
                    source_target_weight.append(diff_value)
                else:
                    source_target_weight.append(1.0)
            #F.binary_cross_entropy()
            instance_loss = nn.BCELoss(
                weight=torch.Tensor(source_target_weight).view(-1, 1).cuda()
            )
            # else:
            #     instance_loss = nn.BCELoss()
            da_ins_features_sigmoid = torch.sigmoid(da_ins_features)
            CCR_loss = instance_loss(da_ins_features_sigmoid, da_ins_labels.float().view(da_ins_features.size(0),-1).to(da_ins_features_sigmoid.device))
            # nn.BCELoss()(da_ins_features_sigmoid, da_ins_labels.view(da_ins_features.size(0),-1).float().to(self.cfg.MODEL.DEVICE))
            # F.binary_cross_entropy_with_logits(da_ins_features_sigmoid, da_ins_labels.view(da_ins_features.size(0),-1).float().to(self.cfg.MODEL.DEVICE))
            losses = {}
            losses["ICR_loss"]=ICR_loss*self.cfg.MODEL.ICR_CCR.ICR_WEIGHT
            # if self.img_weight > 0:
            #     losses["loss_da_image"] = self.img_weight * da_img_loss
            #if self.ins_weight > 0:
            losses["CCR_loss"] =  CCR_loss*self.cfg.MODEL.ICR_CCR.CCR_WEIGHT
            # if self.cst_weight > 0:
            #     losses["loss_da_consistency"] = self.cst_weight * da_consistency_loss
            return losses
        return {}

    def forwardSW(self, img_features1, img_features2,targets):
        """
        Arguments:
            img_features (list[Tensor]): features computed from the images that are
                used for computing the predictions.
            da_ins_feature (Tensor): instance-level feature vectors
            da_ins_labels (Tensor): domain labels for instance-level feature vectors
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        ############################################################################################
        # ICR_feat = self.AdaAvg_pool(img_features2)
        # ICR_feat = self.conv_ICR(ICR_feat).squeeze(-1).squeeze(-1)
        # cls_feat = self.conv_lst(self.bn1(self.avg_pool(base_feat))).squeeze(-1).squeeze(-1)

        ############################################################################################
        batch=img_features1.shape[0]


        SW_netD_pixelInputFeatures = self.grl_SW(img_features1)
        feat_pixel=None
        if self.lc:
            d_pixel, _ = self.netD_pixel(SW_netD_pixelInputFeatures)#局部特征对齐
            # print(d_pixel)
            # if not target:
            #if True:
            _, feat_pixel = self.netD_pixel(img_features1.detach())
        else:
            d_pixel = self.netD_pixel(SW_netD_pixelInputFeatures)#局部特征对齐
        img_features2=torch.cat(img_features2, 0)
        SW_netDInputFeatures = self.grl_SW(img_features2)
        #SW_netDInputFeatures=torch.cat(SW_netDInputFeatures,0)
        feat=None
        if self.gc:
            domain_p, _ = self.netD(SW_netDInputFeatures)#全局特征对齐

            _, feat = self.netD(img_features2.detach())
        else:
            domain_p = self.netD(SW_netDInputFeatures)#全局特征对齐

        if self.training:
            domain_labels = torch.tensor([t.get_field('is_source').any() for t in targets]).to(self.cfg.MODEL.DEVICE)
            # SW loss
            pixeldomain_labels=domain_labels.view(batch,1,1,-1)
            pixeldomain_labels=pixeldomain_labels.expand(d_pixel.size())#.to(self.cfg.MODEL.DEVICE)
            #SLloss=0.5*torch.where(pixeldomain_labels,torch.mean(d_pixel ** 2),torch.mean((1 - d_pixel) ** 2))
            SLloss = torch.where(pixeldomain_labels, d_pixel ** 2, (1 - d_pixel) ** 2)

            #SLloss = 0.5 *(torch.mean(d_pixel ** 2))
                           #0.5 * torch.mean((1 - out_d_pixel) ** 2)) # local Strong loss

            domain_p_target=torch.zeros(domain_p.size(),dtype=torch.int64,device=self.cfg.MODEL.DEVICE)
            domain_p_target.scatter_(1,domain_labels.view(batch,1).long(),1)#one-hot for domain label
            WGloss = self.sigmoid_focal_loss(domain_p, domain_p_target.float())  # global Weak loss

            ####
            SWlosses = {}
            SWlosses["SLloss"] = SLloss.mean()*self.cfg.MODEL.SW.SW_WEIGHT
            SWlosses["WGloss"] = WGloss*self.cfg.MODEL.SW.SW_WEIGHT
            return d_pixel,feat_pixel,domain_p,feat,SWlosses
        return d_pixel,feat_pixel,domain_p,feat, {}
        ############################################################################################

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )

class netD_pixel(nn.Module):
    def __init__(self, cfg,context=False):
        super(netD_pixel, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.context = context
        self.cfg=cfg
        self._init_weights()

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
        weight initalizer: truncated normal and random normal.
        """
            # x is a parameter
            if truncated:
                nn.init.trunc_normal_(m.weight,mean,stddev)
                # m.weight.normal_().fmod_(2).mul_(stddev).add_(
                #     mean
                # )  # not a perfect approximation
            else:
                nn.init.normal_(m.weight,mean, stddev)

                # m.bias.data.zero_()

        normal_init(self.conv1, 0, 0.01)
        normal_init(self.conv2, 0, 0.01)
        normal_init(self.conv3, 0, 0.01)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.context:
            feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
            x = self.conv3(x)
            return torch.sigmoid(x), feat
        else:
            x = self.conv3(x)
            return torch.sigmoid(x)


class netD(nn.Module):
    def __init__(self, cfg,context=False):
        super(netD, self).__init__()
        self.conv1 = conv3x3(1024, 512, stride=2)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = conv3x3(512, 128, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128, 128, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128, 2,bias=False)
        self.context = context
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.cfg=cfg
        self._init_weights()

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
        weight initalizer: truncated normal and random normal.
        """
            # x is a parameter
            if truncated:
                nn.init.trunc_normal_(m.weight,mean,stddev)
                # m.weight.normal_().fmod_(2).mul_(stddev).add_(
                #     mean
                # )  # not a perfect approximation
            else:
                nn.init.normal_(m.weight,mean, stddev)

        normal_init(self.conv1, 0, 0.01)
        normal_init(self.conv2, 0, 0.01)
        normal_init(self.conv3, 0, 0.01)
        normal_init(self.fc, 0, 0.01)

    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.conv1(x))), training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x))), training=self.training)
        x = F.dropout(F.relu(self.bn3(self.conv3(x))), training=self.training)
        x = F.avg_pool2d(x, (x.size(2), x.size(3)))
        x = x.view(-1, 128)
        if self.context:
            feat = x
        x = self.fc(x)
        if self.context:
            return x, feat
        else:
            return x


def build_da_heads(cfg):
    if cfg.MODEL.SW_ON:
        return SWDomainModule(cfg)
    elif cfg.MODEL.DOMAIN_ADAPTATION_ON:
        return DomainAdaptationModule(cfg)
    return []
