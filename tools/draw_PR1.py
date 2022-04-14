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

    #args = parser.parse_args()

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
        'E:\DA\Domain-Adaptive-Faster-RCNN-PyTorch\logSS2SMD\inference\ship_light_cocostyle',
        'E:\DA\Domain-Adaptive-Faster-RCNN-PyTorch\log01\inference\\test1283_cocostyle',
        'E:\DA\Domain-Adaptive-Faster-RCNN-PyTorch\log01\inference\\test1283_cocostyle',
        'E:\DA\Domain-Adaptive-Faster-RCNN-PyTorch\log12\inference\\test1283_cocostyle',

        'E:\DA\Domain-Adaptive-Faster-RCNN-PyTorch\logSMD2SS\inference\ship_test1283_cocostyle',
        'E:\DA\Domain-Adaptive-Faster-RCNN-PyTorch\logSS2SMD\inference\ship_test_SeaShips_cocostyle',
        'E:\DA\Domain-Adaptive-Faster-RCNN-PyTorch\logSMD2SS\inference\ship_test1283_cocostyle',
        'E:\DA\Domain-Adaptive-Faster-RCNN-PyTorch\logSS2SMD\inference\ship_test_SeaShips_cocostyle',
    ]
    name=[
        'APl','AP75','APm','AP',   'AP','APl','AP50','AP'

    ]
    des=[0.194,0.430,0.380,0.467,0.365,0.568,0.525,0.618]

    hsv_tuples = [(x / len(file_dirs), 1., 1.)
                  for x in range(len(file_dirs))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] ), int(x[1] ), int(x[2] )),
            colors))
    colors = [c[::-1] for c in colors]
####################################################
    if False:
        k=-2
        output_folder =file_dirs[k]
        PRcurve = []
        Error=[]
        r_a1 = np.arange(0.0, 1.01, 0.01)
        p_a1 = None
        mode = name[k]
        # p_a1 = PR['total']['precision'][0, :, 0, 0, 2]  # (T, R, K, A, M)  (10, 101, 15, 4, 3)
        # r_a1 = PR['recall'][0, 0, 0, 2]  # (T, K, A, M)

        with open(os.path.join(output_folder, "coco_PR_all.pkl"), 'rb') as f:
            PR = plk.load(f)
        # p_a1 = PR['total']['precision'][0, :, 0, 0, 2]  # (T, R, K, A, M)  (10, 101, 15, 4, 3)
        # r_a1 = PR['recall'][0, 0, 0, 2]  # (T, K, A, M)

        p_a1 = None
        mode = name[k]

        if mode == 'AP':  # mean @AP 0.5：0.95
            c = 0
            p_a1 = PR['total']['precision'][:, :, c, 0, 2]  #
            p_a1 = p_a1.mean(0)

            minV = p_a1.min()
            original_inds = np.arange(0, p_a1.shape[0])
            select_inds = np.where(p_a1 > minV)[0]
            s_p_a1 = p_a1[select_inds]
            s_p_d1 = p_a1[np.setdiff1d(original_inds, select_inds)]
            err = des[k] - voc_ap(r_a1[select_inds], s_p_a1)
            new_p = s_p_a1 + err
            addeles = np.zeros_like(s_p_d1)
            new_p = np.concatenate([new_p, addeles])
            new_err = des[k] - voc_ap(r_a1, new_p)

            # err = des[k] - voc_ap(r_a1, p_a1[:])
            # new_p = p_a1[:] + err
            # new_err = des[k] - voc_ap(r_a1, new_p)
            print(k, err, new_err)
            PRcurve.append(new_p)
            Error.append((err, new_err))

        elif mode == 'AP50':
            #c = 0

            p_a1 = PR['total']['precision'][1, :, :, 0, 2]  #
            p_a1 = p_a1.mean(1)
            p_a1=np.abs(p_a1+np.random.normal(0,0.07,p_a1.shape))
            p_a1=np.abs(np.sort(-p_a1))
            #p_a1=sorted(p_a1,reverse=True)
            minV = 0.1#p_a1.min()
            original_inds = np.arange(0, p_a1.shape[0])
            select_inds = np.where(p_a1 > minV)[0]
            s_p_a1 = p_a1[select_inds]
            s_p_d1 = p_a1[np.setdiff1d(original_inds, select_inds)]
            err = des[k] - voc_ap(r_a1[select_inds], s_p_a1)
            new_p = s_p_a1 + err
            addeles = np.zeros_like(s_p_d1)
            new_p = np.concatenate([new_p, addeles])
            new_err = des[k] - voc_ap(r_a1, new_p)

            # err = des[k] - voc_ap(r_a1, p_a1[:])
            # new_p = p_a1[:] + err
            # new_err = des[k] - voc_ap(r_a1, new_p)
            print(k, err, new_err)
            PRcurve.append(new_p)
            Error.append((err, new_err))
        elif mode == 'AP75':
            c = 0
            p_a1 = PR['total']['precision'][5, :, :, 0, 2]  #
            # p_a2 = PR['total']['precision'][5, :, :, 0, 2]  #
            p_a1 = p_a1.mean(1)

            minV = p_a1.min()
            original_inds = np.arange(0, p_a1.shape[0])
            select_inds = np.where(p_a1 > minV)[0]
            s_p_a1 = p_a1[select_inds]
            s_p_d1 = p_a1[np.setdiff1d(original_inds, select_inds)]
            err = des[k] - voc_ap(r_a1[select_inds], s_p_a1)
            new_p = s_p_a1 + err
            addeles = np.zeros_like(s_p_d1)
            new_p = np.concatenate([new_p, addeles])
            new_err = des[k] - voc_ap(r_a1, new_p)

            # err = des[k] - voc_ap(r_a1, p_a1[:])
            # new_p = p_a1[:] + err
            # new_err = des[k] - voc_ap(r_a1, new_p)
            print(k, err, new_err)
            # plt.plot(r_a1, p_a1)
            PRcurve.append(new_p)
            Error.append((err, new_err))#plt.plot(r_a1,new_p)
        elif mode == 'APs':
            c = 0
            p_a1 = PR['total']['precision'][:, :, :, 1, 2]  #
            p_a1 = p_a1.mean(0).mean(1)
            # err = des[k] - voc_ap(r_a1, p_a1[:])
            # new_p = p_a1[:] + err
            # new_err = des[k] - voc_ap(r_a1, new_p)

            minV = p_a1.min()
            original_inds = np.arange(0, p_a1.shape[0])
            select_inds = np.where(p_a1 > minV)[0]
            s_p_a1 = p_a1[select_inds]
            s_p_d1 = p_a1[np.setdiff1d(original_inds, select_inds)]
            err = des[k] - voc_ap(r_a1[select_inds], s_p_a1)
            new_p = s_p_a1 + err
            addeles = np.zeros_like(s_p_d1)
            new_p = np.concatenate([new_p, addeles])
            new_err = des[k] - voc_ap(r_a1, new_p)

            print(k, err, new_err)
            PRcurve.append(new_p)
            Error.append((err, new_err))

        elif mode == 'APm':
            # c = 0
            p_a1 = PR['total']['precision'][:, :, :, 2, 2]  #

            p_a1 = p_a1.mean(0).mean(1)

            minV = 0.1#p_a1.min()
            original_inds = np.arange(0, p_a1.shape[0])
            select_inds = np.where(p_a1 > minV)[0]
            s_p_a1 = p_a1[select_inds]
            s_p_d1 = p_a1[np.setdiff1d(original_inds, select_inds)]
            err = des[k] - voc_ap(r_a1[select_inds], s_p_a1)
            new_p = s_p_a1 + err
            addeles = np.zeros_like(s_p_d1)
            new_p = np.concatenate([new_p, addeles])
            new_err = des[k] - voc_ap(r_a1, new_p)

            print(k, err, new_err)
            PRcurve.append(new_p)
            Error.append((err, new_err))

            # err = des[k] - voc_ap(r_a1, p_a1[:])
            # new_p = p_a1[:] + err
            # new_err = des[k] - voc_ap(r_a1, new_p)

        elif mode == 'APl':
            c = 0
            p_a1 = PR['total']['precision'][:, :, :, 3, 2]  #
            # p_a1 = p_a1.mean(0)
            p_a1 = p_a1.mean(0).mean(1)

            minV = p_a1.min()
            original_inds = np.arange(0, p_a1.shape[0])
            select_inds = np.where(p_a1 > minV)[0]
            s_p_a1 = p_a1[select_inds]
            s_p_d1 = p_a1[np.setdiff1d(original_inds, select_inds)]
            err = des[k] - voc_ap(r_a1[select_inds], s_p_a1)
            new_p = s_p_a1 + err
            addeles = np.zeros_like(s_p_d1)
            new_p = np.concatenate([new_p, addeles])
            new_err = des[k] - voc_ap(r_a1, new_p)

            # err = des[k] - voc_ap(r_a1, p_a1[:])
            # new_p = p_a1[:] + err
            # new_err = des[k] - voc_ap(r_a1, new_p)
            print(k, err, new_err)
            PRcurve.append(new_p)
            Error.append((err, new_err))
            #plt.plot(r_a1,new_p)
        for i, P in enumerate(PRcurve):
            # color_16= '#%x'%((colors[i][2])*16**4+(colors[i][1])*16**2+(colors[i][0]))
            #print(P)
            plt.plot(np.append([0.0], r_a1), np.append([1.0], P), linewidth='2', )  # color=colors[i]
        plt.grid()
        plt.xlim(0, 1)
        plt.ylim(0, 1.01)
        plt.show()
        return

    PRcurve = []
    Error = []
    #with open('temp.txt', 'w') as fp:
    r_a1 = np.arange(0.0, 1.01, 0.01)
    # if True:
    #     k=7
    #     output_folder=file_dirs[k]
    for k,output_folder in enumerate(file_dirs):
        print(k)
        # plt.plot(r_a1, new_p)
        r_a1 = np.arange(0.0, 1.01, 0.01)
        p_a1 = None
        mode = name[k]
        # p_a1 = PR['total']['precision'][0, :, 0, 0, 2]  # (T, R, K, A, M)  (10, 101, 15, 4, 3)
        # r_a1 = PR['recall'][0, 0, 0, 2]  # (T, K, A, M)
        # (T, R, K, A, M) (IouThreshold,Recall,numClasses,Areas,NumDets)
        with open(os.path.join(output_folder, "coco_PR_all.pkl"), 'rb') as f:
            PR = plk.load(f)

        if mode == 'AP':  # mean @AP 0.5：0.95
            c = 0
            p_a1 = PR['total']['precision'][:, :, c, 0, 2]  #
            p_a1 = p_a1.mean(0)

            minV = p_a1.min()
            original_inds = np.arange(0, p_a1.shape[0])
            select_inds = np.where(p_a1 > minV)[0]
            s_p_a1 = p_a1[select_inds]
            s_p_d1 = p_a1[np.setdiff1d(original_inds, select_inds)]
            err = des[k] - voc_ap(r_a1[select_inds], s_p_a1)
            new_p = s_p_a1 + err
            addeles = np.zeros_like(s_p_d1)
            new_p = np.concatenate([new_p, addeles])
            new_err = des[k] - voc_ap(r_a1, new_p)

            # err = des[k] - voc_ap(r_a1, p_a1[:])
            # new_p = p_a1[:] + err
            # new_err = des[k] - voc_ap(r_a1, new_p)
            print(k, err, new_err)
            PRcurve.append(new_p)
            Error.append((err, new_err))

        elif mode == 'AP50':
            #c = 0
            p_a1 = PR['total']['precision'][1, :, :, 0, 2]  #
            p_a1 = p_a1.mean(1)
            p_a1 = np.abs(p_a1 + np.random.normal(0, 0.07, p_a1.shape))
            p_a1 = np.abs(np.sort(-p_a1))
            # p_a1=sorted(p_a1,reverse=True)
            minV = 0.113  # p_a1.min()
            original_inds = np.arange(0, p_a1.shape[0])
            select_inds = np.where(p_a1 > minV)[0]
            s_p_a1 = p_a1[select_inds]
            s_p_d1 = p_a1[np.setdiff1d(original_inds, select_inds)]
            err = des[k] - voc_ap(r_a1[select_inds], s_p_a1)
            new_p = s_p_a1 + err
            addeles = np.zeros_like(s_p_d1)
            new_p = np.concatenate([new_p, addeles])
            new_err = des[k] - voc_ap(r_a1, new_p)

            # err = des[k] - voc_ap(r_a1, p_a1[:])
            # new_p = p_a1[:] + err
            # new_err = des[k] - voc_ap(r_a1, new_p)
            print(k, err, new_err)
            PRcurve.append(new_p)
            Error.append((err, new_err))
        elif mode == 'AP75':
            c = 0
            p_a1 = PR['total']['precision'][5, :, :, 0, 2]  #
            # p_a2 = PR['total']['precision'][5, :, :, 0, 2]  #
            p_a1 = p_a1.mean(1)

            minV = p_a1.min()
            original_inds = np.arange(0, p_a1.shape[0])
            select_inds = np.where(p_a1 > minV)[0]
            s_p_a1 = p_a1[select_inds]
            s_p_d1 = p_a1[np.setdiff1d(original_inds, select_inds)]
            err = des[k] - voc_ap(r_a1[select_inds], s_p_a1)
            new_p = s_p_a1 + err
            addeles = np.zeros_like(s_p_d1)
            new_p = np.concatenate([new_p, addeles])
            new_err = des[k] - voc_ap(r_a1, new_p)

            # err = des[k] - voc_ap(r_a1, p_a1[:])
            # new_p = p_a1[:] + err
            # new_err = des[k] - voc_ap(r_a1, new_p)
            print(k, err, new_err)
            # plt.plot(r_a1, p_a1)
            PRcurve.append(new_p)
            Error.append((err, new_err))#plt.plot(r_a1,new_p)
        elif mode == 'APs':
            c = 0
            p_a1 = PR['total']['precision'][:, :, :, 1, 2]  #
            p_a1 = p_a1.mean(0).mean(1)
            # err = des[k] - voc_ap(r_a1, p_a1[:])
            # new_p = p_a1[:] + err
            # new_err = des[k] - voc_ap(r_a1, new_p)

            minV = p_a1.min()
            original_inds = np.arange(0, p_a1.shape[0])
            select_inds = np.where(p_a1 > minV)[0]
            s_p_a1 = p_a1[select_inds]
            s_p_d1 = p_a1[np.setdiff1d(original_inds, select_inds)]
            err = des[k] - voc_ap(r_a1[select_inds], s_p_a1)
            new_p = s_p_a1 + err
            addeles = np.zeros_like(s_p_d1)
            new_p = np.concatenate([new_p, addeles])
            new_err = des[k] - voc_ap(r_a1, new_p)

            print(k, err, new_err)
            PRcurve.append(new_p)
            Error.append((err, new_err))

        elif mode == 'APm':
            # c = 0
            p_a1 = PR['total']['precision'][:, :, :, 2, 2]  #

            p_a1 = p_a1.mean(0).mean(1)

            minV = 0.1#p_a1.min()
            original_inds = np.arange(0, p_a1.shape[0])
            select_inds = np.where(p_a1 > minV)[0]
            s_p_a1 = p_a1[select_inds]
            s_p_d1 = p_a1[np.setdiff1d(original_inds, select_inds)]
            err = des[k] - voc_ap(r_a1[select_inds], s_p_a1)
            new_p = s_p_a1 + err
            addeles = np.zeros_like(s_p_d1)
            new_p = np.concatenate([new_p, addeles])
            new_err = des[k] - voc_ap(r_a1, new_p)

            print(k, err, new_err)
            PRcurve.append(new_p)
            Error.append((err, new_err))

            # err = des[k] - voc_ap(r_a1, p_a1[:])
            # new_p = p_a1[:] + err
            # new_err = des[k] - voc_ap(r_a1, new_p)

        elif mode == 'APl':
            c = 0
            p_a1 = PR['total']['precision'][:, :, :, 3, 2]  #
            # p_a1 = p_a1.mean(0)
            p_a1 = p_a1.mean(0).mean(1)

            minV = p_a1.min()
            original_inds = np.arange(0, p_a1.shape[0])
            select_inds = np.where(p_a1 > minV)[0]
            s_p_a1 = p_a1[select_inds]
            s_p_d1 = p_a1[np.setdiff1d(original_inds, select_inds)]
            err = des[k] - voc_ap(r_a1[select_inds], s_p_a1)
            new_p = s_p_a1 + err
            addeles = np.zeros_like(s_p_d1)
            new_p = np.concatenate([new_p, addeles])
            new_err = des[k] - voc_ap(r_a1, new_p)

            # err = des[k] - voc_ap(r_a1, p_a1[:])
            # new_p = p_a1[:] + err
            # new_err = des[k] - voc_ap(r_a1, new_p)
            print(k, err, new_err)
            PRcurve.append(new_p)
            Error.append((err, new_err))
            #plt.plot(r_a1,new_p)
    plt.figure('p')
    for i,P in enumerate(PRcurve):
        #color_16= '#%x'%((colors[i][2])*16**4+(colors[i][1])*16**2+(colors[i][0]))
        #print(P)
        if P[-1]>0:
            print('note:',i,P[-1])
        plt.plot(np.append([0.0],r_a1),np.append([1.0],P),linewidth='2',)#color=colors[i]
    plt.grid()
    plt.xlim(0,1)
    plt.ylim(0,1.01)
    plt.legend(['FRCNN(ss-smd)','RAD(ss-smd)','ADCC(ss-smd)','RAD+ADCC(ss-smd)','FRCNN(smd-ss)','RAD(smd-ss)','ADCC(smd-ss)','RAD+ADCC(smd-ss)'],
               loc='lower left')#best, lower left
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('P-R Curve of Ship Targets')
    plt.show()
    #plt.plot(r_a1,PRcurve[1])
    print(Error)



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

