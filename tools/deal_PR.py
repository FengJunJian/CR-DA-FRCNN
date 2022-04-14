'''
Generate the Precision & Recall etc. data as .pkl format
'''
import argparse
import os
import json
from tools.voc_ap import voc_ap
import numpy as np
import cv2
import time
import pickle as plk
#import scipy.io as sio

from maskrcnn_benchmark.config import cfg
def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD12.yaml",# test_e2e_da_faster_rcnn_R_50_C4_cityscapes_to_foggy_cityscapes_car
        # default= "configs/da_ship/ship_e2e_da_faster_rcnn_R_50_C4_SMD2SeaShips.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=['MODEL.DOMAIN_ADAPTATION_ON',False,
                 'OUTPUT_DIR','log12'],
        # default= ['MODEL.DOMAIN_ADAPTATION_ON',False,
        #          'DATASETS.TEST',('ship_test_SeaShips_cocostyle',),'OUTPUT_DIR','logSMD2SS'],#'DATASETS.TEST',('visual_cocostyle',)
        #ship_test_SeaShips_cocostyle
        nargs=argparse.REMAINDER,type=str
    )

    args = parser.parse_args()

    #num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    #distributed = num_gpus > 1
    #print(args.opts)
    #time.sleep(1)
    #print('aaa')

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # cfg.freeze()
    # filename=os.path.splitext(os.path.basename(args.config_file))[0]
    # label_flag=filename[-2]
    # unlabel_flag=filename[-1]

    model_dir = cfg.OUTPUT_DIR
    save_dir = os.path.join(model_dir, 'inference')
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if save_dir:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(save_dir, dataset_name)
            #os.mkdir(output_folder)
            output_folders[idx] = output_folder

    for output_folder in output_folders:
        # T = len(p.iouThrs)
        # R = len(p.recThrs)
        # K = len(p.catIds) if p.useCats else 1
        # A = len(p.areaRng)
        # M = len(p.maxDets)
        # precision = -np.ones((T, R, K, A, M))  # -1 for the precision of absent categories
        # recall = -np.ones((T, K, A, M))
        # scores = -np.ones((T, R, K, A, M))
        # T:10 iouThrs    - [0.5:.05:0.95]
        # R:101 recThrs    - [0:.01:1]
        # K:number of categories
        # A:4, object area ranges,[[0, 10000000000.0], [0, 1024], [1024, 9216], [9216, 10000000000.0]]->[all,small,medium,large]
        # M:3 thresholds on max detections per image, [1 10 100]
        with open(os.path.join(output_folder, "coco_PR_all.pkl"), 'rb') as f:
            PR=plk.load(f)
        print('done!')
        #sio.savemat(os.path.join(output_folder, "coco_PR_all.mat"),mdict={'a':PR['total']})
        # p_a1 = PR['total']['precision'][0, :, 0, 0, 2]  # (T, R, K, A, M)  (10, 101, 15, 4, 3)
        s_a1=PR['scores'][0,:,0,0,2]
        # r_a1 = PR['recall'][0, 0, 0, 2]  # (T, K, A, M)
        r_a1 = np.arange(0.0, 1.01, 0.01)
        p_a1 = PR['total']['precision'][:, :, :, 1, 2]
        #p_a1=np.mean(p_a1, (0,2))
        ap=[]
        for i in range(10):
            for j in range(15):
                ap.append(voc_ap(r_a1,p_a1[i,:,j]))
        np.mean(ap)


if __name__ == "__main__":
    main()

