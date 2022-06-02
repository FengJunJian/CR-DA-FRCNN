import argparse
import torch
#from torchsummary import summary
import os
import re
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.modeling.detector import build_detection_model

import logging
from tqdm import tqdm
# import time

def extractor(model,
            data_loader,
            dataset_name,
            device="cuda",
            output_folder=None,flag_fg=False):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    cpu_device = torch.device("cpu")
    logger = logging.getLogger("tools.featureExtractor.extractor")
    dataset = data_loader.dataset
    logger.info("Start extract the target features on {} dataset({} images).".format(dataset_name, len(dataset)))
    #start_time = time.time()
    model.eval()
    boxfeature_dict = {}
    if flag_fg:
        FeaMapFolder = os.path.join(output_folder, 'feaMapfg')
    else:
        FeaMapFolder = os.path.join(output_folder, 'feaMap')
    if not os.path.exists(FeaMapFolder):
        os.mkdir(FeaMapFolder)
    for i, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch#images:(1,3,608,1088), targets:(600,1066)
        images = images.to(device)
        targets=[target.to(device) for target in targets]
        file_names = []
        for image_id in image_ids:
            file_names.append(dataset.coco.loadImgs(dataset.ids[image_id])[0]['file_name'])
        with torch.no_grad():  # no compute the gradient
            if flag_fg:
                roifeatures=model.featureTarget_fb(images, targets=targets,saveFolder=FeaMapFolder,imgnames=file_names)
                try:
                    #outputFeatures=[target.get_field('featureROI').to(cpu_device) for target in targets]
                    boxfeature_dict.update({img_id: result for img_id, result in zip(image_ids, roifeatures)})
                except KeyError as e:
                    print('Error ',image_ids,)
            else:
                roifeatures = model.featureTarget(images,targets)
                #outputFeatures=[(p,r) for p,r in zip(proposals,rois)]
                try:
                    outputFeatures=[roifeature.get_field('featureROI').to(cpu_device) for roifeature in roifeatures]
                    boxfeature_dict.update({img_id: result for img_id, result in zip(image_ids, outputFeatures)})
                except KeyError as e:
                    print('Error ',image_ids,)
        #     output = [o.to(cpu_device) for o in output]
        # results_dict.update(
        #     {img_id: result for img_id, result in zip(image_ids, output)}
        # )
    if flag_fg:
        torch.save(boxfeature_dict,os.path.join(output_folder,'targetFeaturesFG.pth'))
    else:
        torch.save(boxfeature_dict, os.path.join(output_folder, 'targetFeatures.pth'))
    # torch.save(boxfeature_dict,os.path.join(output_folder,'targetFeatures.pth'))
    return boxfeature_dict

def featureExtractor(cfg, model, comment=''):
    regxint = re.findall(r'\d+', cfg.MODEL.WEIGHT)
    numstr = ''
    if len(regxint) > 0:
        numstr = str(int(regxint[-1]))

    torch.cuda.empty_cache()  #
    # iou_types = ("bbox",)
    # if cfg.MODEL.MASK_ON:
    #     iou_types = iou_types + ("segm",)
    # if cfg.MODEL.KEYPOINT_ON:
    #     iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "extract"+comment+numstr, dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder

    data_loaders_val = make_data_loader(cfg, is_train=False)
    results=[]
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        result=extractor(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            device=cfg.MODEL.DEVICE,
            output_folder=output_folder,
            flag_fg=True
        )
        #results.append(result)
        #torch.save()
    return results

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="../configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD00.yaml",
        # "../configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD03.yaml",#
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument('--comment',default='feature',type=str,help='comment of the name of the folder')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=['OUTPUT_DIR', 'logFRCNN0'
                 ],
        nargs=argparse.REMAINDER, type=str,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    output_dir = cfg.OUTPUT_DIR
    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)
    checkpointer = DetectronCheckpointer(cfg, model)
    _ = checkpointer.load(os.path.join(output_dir,cfg.MODEL.WEIGHT))
    featureExtractor(cfg, model, comment=args.comment)#'feature'