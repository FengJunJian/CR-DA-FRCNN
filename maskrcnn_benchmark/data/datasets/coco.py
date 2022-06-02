# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints
from maskrcnn_benchmark.data.transforms.preprocessing import horizon_detect
from maskrcnn_benchmark.config import cfg
import numpy as np
import os
import cv2
from tqdm import tqdm
min_keypoints_per_image = 10

def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False

class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None, is_source= True,is_pseudo=False,
    predictionPath=None):
        from pycocotools.coco import COCO
        # ann_file = "E:\SeaShips_SMD/ship_test_SMD_cocostyle.json"
        # predictionPath = "E:\DA1\SW\logSSToSMDship\inference20000\ship_test_SMD_cocostyle/predictions.pth"
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)
        self.pseudo_threshold=cfg.DATASETS.PSEUDO_THRESHOLD#0.1#0.1#0.9\0.8\0.7 adative learning cfg
        # print('######################################################')
        # print('COCODataset:',self.pseudo_threshold)
        # print('######################################################')
        # filter images without detection annotations
        if is_pseudo:
            assert predictionPath
            self.pseudo_label=torch.load(predictionPath)

        if remove_images_without_annotations:
            ids = []
            print('{}: remove images without annotations......'.format('Pseudo' if is_pseudo else 'Source/Target'))
            for img_id in tqdm(self.ids):
                if is_pseudo:
                    boxes=self.pseudo_label[img_id - 1]
                    if len(boxes)>0:
                        scores=boxes.get_field('scores')
                        if isinstance(scores,torch.Tensor):
                            scores=scores.numpy()
                        inds=np.where(scores>self.pseudo_threshold)[0]
                        if len(inds) == 0:
                            continue
                        boxes=boxes[inds]#阈值筛选
                        path = self.coco.loadImgs(img_id)[0]['file_name']
                        img = cv2.imread(os.path.join(self.root, path))
                        ymeans = boxes.bbox[:, 1::2].mean(dim=1)  # ymean
                        horizonLineT, horizonLineB, horizonLine = horizon_detect(img)
                        yinds = torch.where(ymeans - horizonLineT >= 0)[0]
                        # target = target[yinds]
                        if len(yinds)>0:
                            self.pseudo_label[img_id - 1]=boxes[yinds]#水平线筛选
                            ids.append(img_id)
                else:
                    ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                    anno = self.coco.loadAnns(ann_ids)
                    if has_valid_annotation(anno):
                        ids.append(img_id)
            self.ids = ids
            print('{}: remaining ids is {}'.format('Pseudo' if is_pseudo else 'Source/Target',len(self.ids)))

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms
        self.is_source = is_source
        self.is_pseudo=is_pseudo

    def __getitem__(self, idx):#index
        img, anno = super(COCODataset, self).__getitem__(idx)
        #indices = [index] + random.choices(self.indices, k=3) # mixup
        if self.is_pseudo:
            target=self.pseudo_label[self.ids[idx] - 1]
            target=target.resize(img.size).convert('xyxy')
            classes=target.get_field('labels')
            pseudo_flag=torch.ones_like(classes, dtype=torch.uint8)
            target.add_field("is_pseudo", pseudo_flag)  # add pseudo flag

            # ymeans = (dets[:, 3] + dets[:, 1]) / 2
            # yinds = np.where(ymeans - horizonLineT >= 0)[0]
            # dets = dets[yinds, :]
            # masks = [obj["segmentation"] for obj in anno]
            # masks = SegmentationMask(masks, img.size)
            # target.add_field("masks", masks)
        else:
            # filter crowd annotations
            # TODO might be better to add an extra field
            anno = [obj for obj in anno if obj["iscrowd"] == 0]

            boxes = [obj["bbox"] for obj in anno]
            boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
            target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

            classes = [obj["category_id"] for obj in anno]
            classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
            classes = torch.tensor(classes)
            target.add_field("labels", classes)

            masks = [obj["segmentation"] for obj in anno]
            masks = SegmentationMask(masks, img.size)
            target.add_field("masks", masks)
            pseudo_flag = torch.zeros_like(classes, dtype=torch.uint8)
            target.add_field("is_pseudo", pseudo_flag)  # add pseudo flag
        domain_labels = torch.ones_like(classes, dtype=torch.uint8) if self.is_source else torch.zeros_like(classes, dtype=torch.uint8)#source label:1,target label:0
        target.add_field("is_source", domain_labels)

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)
        #try:
        target = target.clip_to_image(remove_empty=True)
        # except:
        #     img_id = self.ids[idx]
        #     path = self.coco.loadImgs(img_id)[0]['file_name']
        #     print(idx, target,path)
        #     raise ValueError(path)
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
