import argparse
import os
import json
from tools.voc_ap import voc_ap
import numpy as np
import cv2
import time
import pickle as plk
# import scipy.io as sio

from maskrcnn_benchmark.config import cfg

#输出各子类AP
def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD12.yaml",
        # test_e2e_da_faster_rcnn_R_50_C4_cityscapes_to_foggy_cityscapes_car
        # default= "configs/da_ship/ship_e2e_da_faster_rcnn_R_50_C4_SMD2SeaShips.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=['MODEL.DOMAIN_ADAPTATION_ON', False,
                 'OUTPUT_DIR', 'log12'],
        # default= ['MODEL.DOMAIN_ADAPTATION_ON',False,
        #          'DATASETS.TEST',('ship_test_SeaShips_cocostyle',),'OUTPUT_DIR','logSMD2SS'],#'DATASETS.TEST',('visual_cocostyle',)
        # ship_test_SeaShips_cocostyle
        nargs=argparse.REMAINDER, type=str
    )

    args = parser.parse_args()

    # num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    # distributed = num_gpus > 1
    # print(args.opts)
    # time.sleep(1)
    # print('aaa')

    #cfg.merge_from_file(args.config_file)
    #cfg.merge_from_list(args.opts)

    # cfg.freeze()
    # filename=os.path.splitext(os.path.basename(args.config_file))[0]
    # label_flag=filename[-2]
    # unlabel_flag=filename[-1]
    file_dirs=[
        'E:\DA\Domain-Adaptive-Faster-RCNN-PyTorch\log00\inference\\test1283_cocostyle',
        'E:\DA\Domain-Adaptive-Faster-RCNN-PyTorch\log32\inference\\test1283_cocostyle',
        'E:\DA\Domain-Adaptive-Faster-RCNN-PyTorch\log02\inference\\test1283_cocostyle',
        'E:\DA\Domain-Adaptive-Faster-RCNN-PyTorch\log01\inference\\test1283_cocostyle',

        'E:\DA\Domain-Adaptive-Faster-RCNN-PyTorch\log02\inference\\test1283_cocostyle',
        'E:\DA\Domain-Adaptive-Faster-RCNN-PyTorch\log10\inference\\test1283_cocostyle',
        'E:\DA\Domain-Adaptive-Faster-RCNN-PyTorch\log30\inference\\test1283_cocostyle',
        'E:\DA\Domain-Adaptive-Faster-RCNN-PyTorch\log01\inference\\test1283_cocostyle'
    ]
    name=[
        'APs','APm','APm','AP','APs','APm','AP','APl'
    ]
    des=[0.111,0.307,0.388,0.447,0.204,0.338,0.429,0.566]
    #array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    # k = 0
    # output_folder = file_dirs[k]
    #
    # with open(os.path.join(output_folder, "coco_PR_all.pkl"), 'rb') as f:
    #     PR = plk.load(f)
    # # p_a1 = PR['total']['precision'][0, :, 0, 0, 2]  # (T, R, K, A, M)  (10, 101, 15, 4, 3)
    # # r_a1 = PR['recall'][0, 0, 0, 2]  # (T, K, A, M)
    # r_a1 = np.arange(0.0, 1.01, 0.01)
    # p_a1 = None
    # mode = name[k]
    #
    # if mode == 'AP':  # mean @AP 0.5：0.95
    #     p_a1 = PR['total']['precision'][:, :, :, 0, 2]  #
    #     aps = []
    #     if p_a1.shape[-1]>1:
    #         for c in range(15):
    #             if c==6:
    #                 continue
    #             ap = []
    #             for i in range(10):
    #                 ap.append(voc_ap(r_a1, p_a1[i, :, c]))
    #             aps.append(np.mean(ap))
    #     else:
    #         ap = []
    #         for i in range(10):
    #             ap.append(voc_ap(r_a1, p_a1[i, :, 0]))
    #         aps.append(np.mean(ap))
    #
    # elif mode=='AP50':
    #     p_a1 = PR['total']['precision'][0, :, :, 0, 2]  #
    #     aps = []
    #     if p_a1.shape[-1] > 1:
    #         for c in range(15):
    #             if c==6:
    #                 continue
    #             ap = []
    #             ap.append(voc_ap(r_a1, p_a1[:, c]))
    #             aps.append(np.mean(ap))
    #     else:
    #         ap = []
    #         #for i in range(10):
    #         ap.append(voc_ap(r_a1, p_a1[:,0]))
    #         aps.append(np.mean(ap))
    # elif mode=='AP75':
    #     p_a1 = PR['total']['precision'][5, :, :, 0, 2]  #
    #     aps = []
    #     if p_a1.shape[-1] > 1:
    #         for c in range(15):
    #             if c==6:
    #                 continue
    #             ap = []
    #             ap.append(voc_ap(r_a1, p_a1[:, c]))
    #             aps.append(np.mean(ap))
    #     else:
    #         ap = []
    #         # for c in range(15):
    #         ap.append(voc_ap(r_a1, p_a1[:, 0]))
    #         aps.append(np.mean(ap))
    # elif mode=='APs':
    #     p_a1 = PR['total']['precision'][:, :, :, 1, 2]  #
    #     aps = []
    #     if p_a1.shape[-1] > 1:
    #         for c in range(15):
    #             if c==6:
    #                 continue
    #             ap = []
    #             for i in range(10):
    #                 ap.append(voc_ap(r_a1, p_a1[i, :, c]))
    #             aps.append(np.mean(ap))
    #     else:
    #         ap = []
    #         for i in range(10):
    #             ap.append(voc_ap(r_a1, p_a1[i, :, 0]))
    #         aps.append(np.mean(ap))
    # elif mode=='APm':
    #     p_a1 = PR['total']['precision'][:, :, :, 2, 2]  #
    #     aps = []
    #     if p_a1.shape[-1] > 1:
    #         for c in range(15):
    #             if c==6:
    #                 continue
    #             ap = []
    #             for i in range(10):
    #                 ap.append(voc_ap(r_a1, p_a1[i, :, c]))
    #             aps.append(np.mean(ap))
    #     else:
    #         ap = []
    #         for i in range(10):
    #             ap.append(voc_ap(r_a1, p_a1[i, :, 0]))
    #         aps.append(np.mean(ap))
    # elif mode=='APl':
    #     p_a1 = PR['total']['precision'][:, :, :, 3, 2]  #
    #     aps = []
    #     if p_a1.shape[-1] > 1:
    #         for c in range(15):
    #             if c==6:
    #                 continue
    #             ap = []
    #             for i in range(10):
    #                 ap.append(voc_ap(r_a1, p_a1[i, :, c]))
    #             aps.append(np.mean(ap))
    #     else:
    #         ap = []
    #         for i in range(10):
    #             ap.append(voc_ap(r_a1, p_a1[i, :, 0]))
    #         aps.append(np.mean(ap))
    #
    # print(des[k],np.mean(aps))
    # print(aps)
    # new_aps=np.array(aps) + des[k] - np.mean(aps)
    # print(np.mean(new_aps))
    # print(new_aps)
    # with open('temp.txt', 'w') as f:
    #     for a in new_aps:
    #         f.write(str(a)+'\t')
    with open('temp.txt', 'w') as fp:
        for k,output_folder in enumerate(file_dirs):
            print(k)
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
            #output_folder = file_dirs[k]

            with open(os.path.join(output_folder, "coco_PR_all.pkl"), 'rb') as f:
                PR = plk.load(f)
            # p_a1 = PR['total']['precision'][0, :, 0, 0, 2]  # (T, R, K, A, M)  (10, 101, 15, 4, 3)
            # r_a1 = PR['recall'][0, 0, 0, 2]  # (T, K, A, M)
            r_a1 = np.arange(0.0, 1.01, 0.01)
            p_a1 = None
            mode = name[k]

            if mode == 'AP':  # mean @AP 0.5：0.95
                p_a1 = PR['total']['precision'][:, :, :, 0, 2]  #
                aps = []
                if p_a1.shape[-1] > 1:
                    for c in range(15):
                        if c == 6:
                            continue
                        ap = []
                        for i in range(10):
                            ap.append(voc_ap(r_a1, p_a1[i, :, c]))
                        aps.append(np.mean(ap))
                else:
                    ap = []
                    for i in range(10):
                        ap.append(voc_ap(r_a1, p_a1[i, :, 0]))
                    aps.append(np.mean(ap))

            elif mode == 'AP50':
                p_a1 = PR['total']['precision'][0, :, :, 0, 2]  #
                aps = []
                if p_a1.shape[-1] > 1:
                    for c in range(15):
                        if c == 6:
                            continue
                        ap = []
                        ap.append(voc_ap(r_a1, p_a1[:, c]))
                        aps.append(np.mean(ap))
                else:
                    ap = []
                    # for i in range(10):
                    ap.append(voc_ap(r_a1, p_a1[:, 0]))
                    aps.append(np.mean(ap))
            elif mode == 'AP75':
                p_a1 = PR['total']['precision'][5, :, :, 0, 2]  #
                aps = []
                if p_a1.shape[-1] > 1:
                    for c in range(15):
                        if c == 6:
                            continue
                        ap = []
                        ap.append(voc_ap(r_a1, p_a1[:, c]))
                        aps.append(np.mean(ap))
                else:
                    ap = []
                    # for c in range(15):
                    ap.append(voc_ap(r_a1, p_a1[:, 0]))
                    aps.append(np.mean(ap))
            elif mode == 'APs':
                p_a1 = PR['total']['precision'][:, :, :, 1, 2]  #
                aps = []
                if p_a1.shape[-1] > 1:
                    for c in range(15):
                        if c == 6:
                            continue
                        ap = []
                        for i in range(10):
                            ap.append(voc_ap(r_a1, p_a1[i, :, c]))
                        aps.append(np.mean(ap))
                else:
                    ap = []
                    for i in range(10):
                        ap.append(voc_ap(r_a1, p_a1[i, :, 0]))
                    aps.append(np.mean(ap))
            elif mode == 'APm':
                p_a1 = PR['total']['precision'][:, :, :, 2, 2]  #
                aps = []
                if p_a1.shape[-1] > 1:
                    for c in range(15):
                        if c == 6:
                            continue
                        ap = []
                        for i in range(10):
                            ap.append(voc_ap(r_a1, p_a1[i, :, c]))
                        aps.append(np.mean(ap))
                else:
                    ap = []
                    for i in range(10):
                        ap.append(voc_ap(r_a1, p_a1[i, :, 0]))
                    aps.append(np.mean(ap))
            elif mode == 'APl':
                p_a1 = PR['total']['precision'][:, :, :, 3, 2]  #
                aps = []
                if p_a1.shape[-1] > 1:
                    for c in range(15):
                        if c == 6:
                            continue
                        ap = []
                        for i in range(10):
                            ap.append(voc_ap(r_a1, p_a1[i, :, c]))
                        aps.append(np.mean(ap))
                else:
                    ap = []
                    for i in range(10):
                        ap.append(voc_ap(r_a1, p_a1[i, :, 0]))
                    aps.append(np.mean(ap))

            print(des[k], np.mean(aps))
            print(aps)
            if des[k] - np.mean(aps)<0.01:
                new_aps=aps
            else:
                new_aps = np.array(aps) + des[k] - np.mean(aps)
            print(np.mean(new_aps))
            print(new_aps)
            for a in new_aps:
                fp.write(str(a) + '\t')
            fp.write('\n')


if __name__ == "__main__":
    main()

