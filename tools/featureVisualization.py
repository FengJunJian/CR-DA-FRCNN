from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import torch
from tqdm import tqdm
import colorsys
import argparse
def ncolors(num_color):
    #import seaborn
    colors = []
    if False:
        cs=list(seaborn.xkcd_rgb.values())
        inv=1#int(len(cs)/num_color)
        for i in range(num_color):
            ind=i*inv
            r = int(cs[ind][1:3], 16)
            g = int(cs[ind][3:5], 16)
            b = int(cs[ind][5:7], 16)
            colors.append((r,g,b))
    else:
        hsv_tuples = [(x / num_color, 1.0, 1.0)
                      for x in range(num_color)]
        # hsv_tuples = [(x / num_color, 1.0, 1.0)
        #               for x in range(num_color)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                colors))
        colors = [c[::-1] for c in colors]

    return colors

def t_sne_projection(x,y=None,dims=2):
    #sns.set(color_codes=True)
    #sns.set(rc={'figure.figsize': (11.7, 8.27)})
    #palette = sns.color_palette("bright", 80)
    tsne = TSNE(n_components=dims)
    x_embedded=tsne.fit_transform(x,y)  # 进行数据降维,降成两维
    # a=tsne.fit_transform(data_zs) #a是一个array,a相当于下面的tsne_embedding_
    # plt.figure()
    # sns.scatterplot(x_embedded[:, 0], x_embedded[:, 1], hue=y, legend='full', )#palette=palette
    #plt.show()
    return x_embedded,tsne#y

def pca_projection(x,y=None,dims=2):
    pca = PCA(n_components=dims)
    x_embedded=pca.fit_transform(x,y)  # 进行数据降维,降成两维
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_)
    return x_embedded,pca#y

def drawDiffClass(x,y,colors):
    for i in set(y):
        indices = np.where(
            np.isin(np.array(y), i)
        )[0]
        plt.scatter(x[indices, 0], x[indices, 1], c=colors[i], marker='o',
                    edgecolors='k')  # sns.color_palette(palettes[0])

    plt.xticks([])
    plt.yticks([])

def Embedding_feature(folder,featureFiles,CLASS_NAMES=None,update_save_file=False,comment=''):#练习调试
    '''
        根据GT来生成聚类
        保存降维后的特征x，gt，预测标签

    '''
    saveDir = folder#os.path.join(folder, 'gt')

    if CLASS_NAMES:
        NUM = len(CLASS_NAMES)  # - 1
    colors = np.array(((255,0,0),(0,0,255)))/255.0#np.array(ncolors(NUM)) / 255.0

    if not os.path.exists(saveDir):
        os.mkdir(saveDir)

    features=[]
    y=np.empty((0,),np.int32)
    for i in range(len(featureFiles)):
        file = os.path.join(folder, featureFiles[i])  # 'proj_numpy_ship.npz'
        data = torch.load(file)  # type: dict k:img_id, v:feature (1024,7,7)
        for k,v in tqdm(data.items()):
            feature=v.numpy().reshape(-1)
            y=np.concatenate([y,(i,)],axis=0)
            features.append(feature)

    features=np.array(features)
    indices={}
    for i in set(y):
        indice = np.where(
            np.isin(np.array(y), i)
        )[0]
        indices.update({i:indice})
    #################################################total
    print('processing tsne')
    x_e,tsne=t_sne_projection(features, y=y, dims=2)
    drawDiffClass(x_e, y, colors)
    plt.savefig(os.path.join(saveDir, 'tsne%s.png' % (comment)))
    print('save tsne%s.png' % (comment))
    print('processing tsne')
    x_ep,pca=pca_projection(features,  dims=2)
    #pca.transform(features)
    plt.figure()
    drawDiffClass(x_ep, y, colors)
    plt.savefig(os.path.join(saveDir, 'pca%s.png' % (comment)))
    print('save pca%s.png' % (comment))

    #################################################single SMD->SS
    try:
        x_ep0, pca = pca_projection(features[indices[0]], dims=2)
        x_ep1=pca.transform(features[indices[1]])
        x_eT=np.concatenate([x_ep0,x_ep1],axis=0)
        yT=np.concatenate([y[indices[0]],y[indices[1]]])
        plt.figure()
        drawDiffClass(x_eT, yT, colors)
        saveName='pca%s.png' % ('SS-SMD')
        plt.savefig(os.path.join(saveDir, saveName))
        print(saveName)

        x_e1, pca = pca_projection(features[indices[1]], dims=2)
        x_e0=pca.transform(features[indices[0]])
        x_eT = np.concatenate([x_e0, x_e1], axis=0)
        yT = np.concatenate([y[indices[0]], y[indices[1]]])
        plt.figure()
        drawDiffClass(x_eT, yT, colors)
        saveName = 'pca%s.png' % ('SMD-SS')
        plt.savefig(os.path.join(saveDir, saveName))
        print(saveName)
    except:
        print('single test set')

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--mode", type=str,default=None,help='mode to visual the feature:S1 and S2 are set for single domain training set, S3 is set for two domain training set'
    )
    parser.add_argument('--comment', default='feature', type=str, help='comment of the name of the folder')

    args = parser.parse_args()
    #saveFile = 'embedding.npz'
    if not args.mode:
        raise ValueError("Please enter the mode! and the mode is None!")
        sys.exit(1)
    if args.mode=='S1':#single domain training set
        featureFiles=['test_SeaShips_cocostyle/targetFeatures.pth','test_SMD_cocostyle/targetFeatures.pth']
        folders = ['E:/DA/logSMD/extractfeature/','E:/DA/logSMD0/extractfeature/','E:/DA/logSMD1/extractfeature/',
                   'E:/DA/logSMD2/extractfeature/','E:/DA/logSMD3/extractfeature/','E:/DA/logSMD4/extractfeature/',
                   'E:/DA/logSS/extractfeature/','E:/DA/logSS0/extractfeature/','E:/DA/logSS1/extractfeature/',
                   'E:/DA/logSS2/extractfeature/','E:/DA/logSS3/extractfeature/','E:/DA/logSS4/extractfeature/',
                   ]

    elif args.mode=='S2': #single domain training set for 2 CLASSES (as same as mode S1)
        featureFiles = ['ship_test_SeaShips_cocostyle/targetFeatures.pth',
                        'ship_test_SMD_cocostyle/targetFeatures.pth']
        folders = ['E:/DA/logSMDship/extractfeature/', 'E:/DA/logSMD0ship/extractfeature/',
                   'E:/DA/logSMD1ship/extractfeature/', 'E:/DA/logSMD2ship/extractfeature/',
                   'E:/DA/logSMD3ship/extractfeature/', 'E:/DA/logSMD4ship/extractfeature/',
                   'E:/DA/logSSship/extractfeature/', 'E:/DA/logSS0ship/extractfeature/',
                   'E:/DA/logSS1ship/extractfeature/', 'E:/DA/logSS2ship/extractfeature/',
                   'E:/DA/logSS3ship/extractfeature/', 'E:/DA/logSS4ship/extractfeature/',
                   'E:/DA/logSS_SMDship/extractfeature/',
                   ]
    elif args.mode=='S3':# two domain training set
        featureFiles = ['ship_test_SeaShips_cocostyle/targetFeatures.pth',
                        'ship_test_SMD_cocostyle/targetFeatures.pth']
        #['ship_test_SS_SMD_cocostyle/targetFeatures.pth' ]
        folders = ['../logSMDToSSship/extractfeature','../logSSToSMDship/extractfeature']
                   #'E:/DA/logSMDToSSship/extractfeature/',
                   # 'E:/DA/logSSToSMDship/extractfeature/']

    for folder in folders:
        print(folder)
        Embedding_feature(folder, featureFiles,  update_save_file=False, comment='')

def main_t():
    colors = np.array(((255, 0, 0), (0, 0, 255))) / 255.0  # np.array(ncolors(NUM)) / 255.0
    plt.scatter(np.array([0]),np.array([1]),c=colors[0],marker='o',
                edgecolors='k')
    plt.scatter(np.array([ -3,]), np.array([2, ]), c=(0,0,1.0),marker='o',
                edgecolors='k')
    # plt.scatter(x[indices, 0], x[indices, 1], c=(1.0,0,0), marker='o',
    #             edgecolors='k')  # sns.color_palette(palettes[0])
if __name__=='__main__':
    # main_t()
    main()