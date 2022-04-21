# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.structures.bounding_box import BoxList
from . import transforms as T
import albumentations as A
import cv2

def build_transforms(cfg, is_train=True):

    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
        other_flip_prob=0.1
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_prob = 0.0
        other_flip_prob=0.0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )

    Talbu = A.Compose([
        #A.Resize(int(H / 2), int(W / 2)),
        A.Blur(p=other_flip_prob),
        A.RandomFog(p=other_flip_prob),
        A.RandomRain(p=other_flip_prob),
        A.ShiftScaleRotate(shift_limit=0, rotate_limit=0, scale_limit=0.6, border_mode=cv2.BORDER_CONSTANT,p=flip_prob),
        # A.Downscale(always_apply=True)#下采样
        # A.Cutout(8)
        # A.RandomFog(p=1.0),#雾True霾
        # A.RandomRain(p=1.0)#下雨
        # A.RandomShadow(p=1.0)#阴影
        # A.RandomScale(p=1.0)
        # A.RandomSunFlare(p=1.0)
    ], bbox_params=A.BboxParams("pascal_voc", label_fields=['class_labels']), )#mode:xyxy

    Ttorch = T.Compose(
        [
            T.Resize(min_size, max_size),
            T.RandomHorizontalFlip(flip_prob),
            T.ToTensor(),
            normalize_transform,
            #T.Edge_T()
        ]
    )

    return MulTransform(Talbu,Ttorch)

class MulTransform(object):
    def __init__(self,Talbu=None,Ttorch=None):
        self.Talbu=Talbu
        self.Ttorch=Ttorch
        self.mode="xyxy"
    def __call__(self, image, target):
        if self.Talbu:
            albuformat = "pascal_voc"
            if target.mode()=="xyxy":
                albuformat="pascal_voc"
            elif target.mode()=="xywh":
                albuformat = "coco"
            extra_fields=target.extra_fields()
            original_bbox_p = self.Talbu.processors['bboxes'].params._to_dict()
            original_bbox_p.update({"format":albuformat,"label_fields": list(extra_fields.keys())})
            self.Talbu.processors["bboxes"] = A.BboxProcessor(A.BboxParams(**original_bbox_p))
            albuDict={}
            albuDict["image"]=image
            albuDict["bboxes"]=target.bbox
            albuDict.update(extra_fields)
            #for k,v in target.extra_fields.items():

            Tout=self.Talbu(**albuDict)
            image=Tout["image"]
            bboxes=Tout["bboxes"]
            target = BoxList(bboxes, target.size, mode=target.mode())
            for k in extra_fields.keys():
                target.add_field(k,Tout[k])
        if self.Ttorch:
            image, target = self.Ttorch(image, target)

        return image,target



# def build_transforms_edge(cfg, is_train=True):
#
#     if is_train:
#         min_size = cfg.INPUT.MIN_SIZE_TRAIN
#         max_size = cfg.INPUT.MAX_SIZE_TRAIN
#         flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
#     else:
#         min_size = cfg.INPUT.MIN_SIZE_TEST
#         max_size = cfg.INPUT.MAX_SIZE_TEST
#         flip_prob = 0
#
#     # to_bgr255 = cfg.INPUT.TO_BGR255
#     # normalize_transform = T.Normalize(
#     #     mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
#     # )
#
#
#     transform = T.Compose(
#         [
#             T.Resize(min_size, max_size),
#             T.RandomHorizontalFlip(flip_prob),
#             T.Edge_T(),
#             T.ToTensor(),
#             #normalize_transform,
#
#         ]
#     )
#     return transform