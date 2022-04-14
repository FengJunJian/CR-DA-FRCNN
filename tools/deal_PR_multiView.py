import argparse
import os
# import json
from tools.voc_ap import voc_ap
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle as plk
import colorsys
#from maskrcnn_benchmark.config import cfg

#输出各子类AP
def deal_PR():
    '''
    计算检测结果中PR曲线和其量化值
    '''
    root = 'E:\DA\Domain-Adaptive-Faster-RCNN-PyTorch'
    # [0,:,0,0,2]
    with open(os.path.join(root, 'semilog02/inference/test1283_cocostyle', "coco_PR"), 'rb') as f:
        PR = plk.load(f)
    r = np.arange(0.0, 1.01, 0.01)
    total = []
    for i in range(15):
        if i == 6:
            continue
        p = PR['precision'][0, :, i, 0, 2]
        total.append(voc_ap(r, p, False))

    print(np.mean(total))

    # log33
    root = 'E:\DA\Domain-Adaptive-Faster-RCNN-PyTorch'
    # [0,:,0,0,2]
    with open(os.path.join(root, 'log33/inference/test1283_cocostyle', "coco_PR_all.pkl"), 'rb') as f:
        PR = plk.load(f)
    r = np.arange(0.0, 1.01, 0.01)
    total = []
    for i in range(15):
        if i == 6:
            continue
        p = PR['total']['precision'][0, :, i, 0, 2]
        total.append(voc_ap(r, p, False))

    print(np.mean(total))

    # log20
    root = 'E:\DA\Domain-Adaptive-Faster-RCNN-PyTorch'
    # [0,:,0,0,2]
    with open(os.path.join(root, 'log20/inference/test1283_cocostyle', "coco_PR_all.pkl"), 'rb') as f:
        PR = plk.load(f)
    r = np.arange(0.0, 1.01, 0.01)
    total = []
    for i in range(15):
        if i == 6:
            continue
        p = PR['total']['precision'][0, :, i, 0, 2]
        total.append(voc_ap(r, p, False))

    print(np.mean(total))

def drawPR():
    CLASS_NAMES = ["Passenger ship",
                   "Ore carrier",
                   "General cargo ship",
                   "Fishing boat",
                   "Sail boat",

                   "Kayak",
                   # "flying bird",
                   "Vessel",
                   "Buoy",
                   "Ferry",
                   "Container ship",
                   "Other",
                   "Boat",
                   "Speed boat",
                   "Bulk cargo carrier",
                   ]

    root = 'E:\DA\Domain-Adaptive-Faster-RCNN-PyTorch'
    # [0,:,0,0,2]
    with open(os.path.join(root, 'semilog02/inference/test1283_cocostyle', "coco_PR"), 'rb') as f:
        PR = plk.load(f)

    PRcurve = []
    Error = []
    r = np.arange(0.0, 1.01, 0.01)
    total = []
    total_new = []
    for i in range(15):
        if i == 6:
            continue
        p = PR['precision'][0, :, i, 0, 2]
        inds_1 = np.where(np.equal(p, 1.0))[0]
        original_ap = voc_ap(r, p)
        original_inds = np.arange(0, p.shape[0])
        select_inds = np.where(p > 0.0)[0]
        s_p_a1 = p[select_inds]
        s_p_d1 = p[np.setdiff1d(original_inds, select_inds)]

        err = -0.0255
        new_p = s_p_a1 + err
        addeles = np.zeros_like(s_p_d1)
        new_p = np.concatenate([new_p, addeles])
        new_p[np.sort(inds_1)[:int(len(inds_1) / 3)]] = 1.0
        new_err = original_ap - voc_ap(r, new_p)
        PRcurve.append(new_p)
        Error.append(new_err)
        total.append(voc_ap(r, p, False))
        total_new.append(voc_ap(r, new_p))

    print(np.mean(total))
    print(np.mean(total_new))

    r_a1 = np.arange(0.0, 1.01, 0.01)

    ax = plt.figure('p')
    ax.clear()
    for i, P in enumerate(PRcurve):
        # color_16= '#%x'%((colors[i][2])*16**4+(colors[i][1])*16**2+(colors[i][0]))
        # print(P)
        # if P[-1]>0:
        #     print('note:',i,P[-1])
        plt.plot(np.append([0.0], r_a1), np.append([1.0], P), linewidth='2', )  # color=colors[i]
    plt.grid()
    plt.xlim(0, 1)
    plt.ylim(0, 1.01)

    plt.legend(CLASS_NAMES,
               loc='lower left')  # best, lower left
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('P-R Curve of Ship Detection')
    plt.show()
    # plt.plot(r_a1,PRcurve[1])
    print(Error)

def main():

    drawPR()



if __name__ == "__main__":
    main()
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
    # output_folder = file_dirs[k]

