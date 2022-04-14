import os
import re
from glob import glob
def featureExtractor_visualization_D2D_ship():
    batfile='featureExtractor_visualization_D2D_ship.bat'#featureExtractor_D2D_ship
    VisualMode='S2'
    comment = 'ship'  # ship
    commandstr=lambda config,log,testset:'python tools/featureExtractor.py --config-file configs/da_ship/{config}.yaml OUTPUT_DIR ../{log} MODEL.WEIGHT ../{log}/model_0020000.pth DATASETS.TEST (\'{testset}\',)\n'.format(config=config,log=log,testset=testset)#MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 7

    commandstrV = lambda mode: 'python tools/featureVisualization.py --mode {mode}\n'.format(mode=mode)  # MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 7
    # 'python tools/featureExtractor.py --config-file configs/da_ship/{config}.yaml OUTPUT_DIR ../{log} MODEL.WEIGHT ../{log}/model_0020000.pth DATASETS.TEST (\'{testset}\',) MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 7'.format(config='da_faster_rcnn_R_50_C4_SS_ship',log='logSSship',testset='ship_test_SeaShips_cocostyle')
    logs=['logSMD','logSMD0','logSMD1','logSMD2','logSMD3','logSMD4',
         'logSS','logSS0','logSS1','logSS2','logSS3','logSS4','logSS_SMD']

    testsets=['ship_test_SeaShips_cocostyle','ship_test_SMD_cocostyle']
    with open(batfile,'w') as f:
        for testset in testsets:
            f.write('rem {}########\n'.format(testset))
            for log in logs:
                writestr=commandstr('faster_rcnn_R_50_C4_SS_ship',log=log+comment,testset=testset)
                f.write(writestr)
            f.write('\n')

        f.write('rem #####################Visualization##################\n')
        f.write(commandstrV(VisualMode))
        #f.write(commandstrV('S3'))

def featureExtractor_visualization_da_ship():
    batfile='featureExtractor_visualization_da_ship.bat'#featureExtractor_D2D_ship
    VisualMode='S3'
    comment = 'ship'  # ship
    commandstr=lambda config,log,testset:'python tools/featureExtractor.py --config-file configs/da_ship/{config}.yaml OUTPUT_DIR ../{log} MODEL.WEIGHT ../{log}/model_0020000.pth DATASETS.TEST (\'{testset}\',)\n'.format(config=config,log=log,testset=testset)

    commandstrV = lambda mode: 'python tools/featureVisualization.py --mode {mode}\n'.format(mode=mode)  # MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 7
    # 'python tools/featureExtractor.py --config-file configs/da_ship/{config}.yaml OUTPUT_DIR ../{log} MODEL.WEIGHT ../{log}/model_0020000.pth DATASETS.TEST (\'{testset}\',) MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 7'.format(config='da_faster_rcnn_R_50_C4_SS_ship',log='logSSship',testset='ship_test_SeaShips_cocostyle')
    logs=['logSMDToSS','logSSToSMD']

    testsets=['ship_test_SeaShips_cocostyle','ship_test_SMD_cocostyle']
    with open(batfile,'w') as f:
        for testset in testsets:
            f.write('rem {}########\n'.format(testset))
            for log in logs:
                writestr=commandstr('da_faster_rcnn_R_50_C4_SS2SMD_ship',log=log+comment,testset=testset)
                f.write(writestr)
            f.write('\n')

        f.write('rem #####################Visualization##################\n')
        f.write(commandstrV(VisualMode))

def test_net_D2D_ship():
    batfile='test_net_D2D_ship.bat'
    commandstr=lambda config,log,testset:'python tools/test_net.py --config-file configs/da_ship/{config}.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../{log} DATASETS.TEST (\'{testset}\',) \n'.format(config=config,log=log,testset=testset)
    dacommandstr = lambda config, log,testset: 'python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml --flagVisual True --checkpoint model_0020000.pth OUTPUT_DIR ../{log} DATASETS.TEST (\'{testset}\',) \n'.format(
        config=config, log=log, testset=testset)
    # 'python tools/featureExtractor.py --config-file configs/da_ship/{config}.yaml OUTPUT_DIR ../{log} MODEL.WEIGHT ../{log}/model_0020000.pth DATASETS.TEST (\'{testset}\',) MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 7'.format(config='da_faster_rcnn_R_50_C4_SS_ship',log='logSSship',testset='ship_test_SeaShips_cocostyle')
    logs=['logSMD','logSMD0','logSMD1','logSMD2','logSMD3','logSMD4',
         'logSS','logSS0','logSS1','logSS2','logSS3','logSS4',]
    comment='ship'
    testsets=['ship_test_SeaShips_cocostyle','ship_test_SMD_cocostyle']
    with open(batfile,'w') as f:
        for testset in testsets:
            f.write('rem {}########\n'.format(testset))
            for log in logs:
                writestr=commandstr('faster_rcnn_R_50_C4_SS_ship',log=log+comment,testset=testset)
                f.write(writestr)
            f.write('\n')
    # --config-file ../configs/da_ship/da_faster_rcnn_R_50_C4_SS_ship.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../../logSMD0ship DATASETS.TRAIN('ship_train_SeaShips0_cocostyle',)

def test_net_D2D_dashipV():
    batfile='test_net_D2D_dashipV.bat'
    # commandstr=lambda config,log,testset:'python tools/test_net.py --config-file configs/da_ship/{config}.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../{log} DATASETS.TEST (\'{testset}\',) \n'.format(config=config,log=log,testset=testset)
    commandstr = lambda config, log,testset: 'python tools/test_net.py --config-file configs/da_ship/{config}.yaml --flagVisual True --checkpoint model_0020000.pth OUTPUT_DIR ../{log} DATASETS.TEST (\'{testset}\',) \n'.format(
        config=config, log=log, testset=testset)

    logs=['logSSToSMD','logSMDToSS']
    comment='ship'
    testsets=['ship_test_SeaShips_cocostyle','ship_test_SMD_cocostyle']
    with open(batfile,'w') as f:
        for testset in testsets:
            f.write('rem {}########\n'.format(testset))
            for log in logs:
                writestr=commandstr('da_faster_rcnn_R_50_C4_SS2SMD_ship',log=log+comment,testset=testset)
                f.write(writestr)
            f.write('\n')
    # --config-file ../configs/da_ship/da_faster_rcnn_R_50_C4_SS_ship.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../../logSMD0ship DATASETS.TRAIN('ship_train_SeaShips0_cocostyle',)

def test_net_allcheckpoints():
    batfile = 'test_net_allcheckpoints.bat'

    commandstr = lambda config, log, checkpoints ,testset: 'python tools/test_net.py --config-file configs/da_ship/{config}.yaml --checkpoint {checkpoints} OUTPUT_DIR ../{log} DATASETS.TEST (\'{testset}\',) \n'.format(config=config, log=log, checkpoints=checkpoints, testset=testset)

    logs = ['logSSToSMDship', 'logSMDToSSship']
    testsets = ['ship_test_SeaShips_cocostyle', 'ship_test_SMD_cocostyle']

    with open(batfile, 'w',encoding='utf-8') as f:
        for testset in testsets:
            f.write('rem {}########\n'.format(testset))
            for log in logs:
                checkpoints=glob(os.path.join('..',log,'model_0*.pth'))
                checkpoints.sort()
                checkpoints=[os.path.basename(cp) for cp in checkpoints]
                chstr='('
                for cp in checkpoints:
                    chstr+='\'{}\','.format(cp)
                chstr+=')'
                writestr = commandstr('da_faster_rcnn_R_50_C4_SS2SMD_ship', log=log, checkpoints=chstr, testset=testset)
                # for cp in checkpoints:
                #     checkpoint=os.path.basename(cp)
                #     writestr = commandstr('da_faster_rcnn_R_50_C4_SS2SMD_ship', log=log, checkpoints=checkpoints,testset=testset)
                f.write(writestr)
            f.write('\n')
    # --config-file ../configs/da_ship/da_faster_rcnn_R_50_C4_SS_ship.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../../logSMD0ship DATASETS.TRAIN('ship_train_SeaShips0_cocostyle',)

def train_net_da_ship():
    batfile='train_net_da_ship.bat'
    commandstr=lambda config,log,sourceset,targetset,testset:'python tools/train_net.py --config-file configs/da_ship/{config}.yaml OUTPUT_DIR ../{log} DATASETS.TRAIN (\'{sourceset}\',) DATASETS.SOURCE_TRAIN (\'{sourceset}\',) DATASETS.TARGET_TRAIN (\'{targetset}\',) DATASETS.TEST (\'{testset}\',) SOLVER.MAX_ITER 20000\n'.format(config=config,log=log,sourceset=sourceset,targetset=targetset,testset=testset)
    # python tools/train_net.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SMD.yaml OUTPUT_DIR ../logSMD DATASETS.TRAIN ('train_SMD_cocostyle',) MODEL.DOMAIN_ADAPTATION_ON False
    # SMDlogs=['SMD','SMD0','SMD1','SMD2','SMD3',]#SMD
    # SMDsets = ['ship_train_SMD_cocostyle', 'ship_train_SMD_cocostyle0', 'ship_train_SMD_cocostyle1',
    #            'ship_train_SMD_cocostyle2', 'ship_train_SMD_cocostyle3', ]  # SMD
    #
    # SSlogs=['SS','SS0','SS1','SS2','SS3',]#SS
    # SSsets= ['ship_train_SeaShips_cocostyle', 'ship_train_SeaShips_cocostyle0', 'ship_train_SeaShips_cocostyle1',
    #           'ship_train_SeaShips_cocostyle2', 'ship_train_SeaShips_cocostyle3', ]  # SS

    SMDlogs = ['SMD', 'SMD0',  'SMD3', ]  # SMD
    SMDsets = ['ship_train_SMD_cocostyle', 'ship_train_SMD0_cocostyle',  'ship_train_SMD3_cocostyle', ]  # SMD

    SSlogs = ['SS', 'SS0', 'SS3', ]  # SS
    SSsets = ['ship_train_SeaShips_cocostyle', 'ship_train_SeaShips0_cocostyle', 'ship_train_SeaShips3_cocostyle', ]  # SS

    comment='ship'
    testsets=['ship_test_SeaShips_cocostyle','ship_test_SMD_cocostyle']
    with open(batfile,'w') as f:
        #SMD2SS#####################################################################
        f.write('rem {}########\n'.format('da_faster_rcnn_R_50_C4_SMD2SS'))
        for i,sset in enumerate(SMDsets):
            slog=SMDlogs[i]
            for j,tset in enumerate(SSsets):
                tlog=SSlogs[j]
                writestr=commandstr(config='da_faster_rcnn_R_50_C4_SMD2SS_ship',log='log%sTo%s%s'%(slog,tlog,comment),sourceset=sset,targetset=tset,testset=testsets[0])
                f.write(writestr)
            f.write('\n')
        f.write('\n')
        # SS2SMD#####################################################################
        f.write('rem {}########\n'.format('da_faster_rcnn_R_50_C4_SS2SMD'))#
        for i,sset in enumerate(SSsets):
            slog=SSlogs[i]
            for j,tset in enumerate(SMDsets):
                tlog=SMDlogs[j]
                writestr=commandstr(config='da_faster_rcnn_R_50_C4_SS2SMD_ship',log='log%sTo%s%s'%(slog,tlog,comment),sourceset=sset,targetset=tset,testset=testsets[1])
                f.write(writestr)
            f.write('\n')
    #--config-file ../configs/da_ship/da_faster_rcnn_R_50_C4_SMD2SS_ship.yaml OUTPUT_DIR ../../SMDToSSship DATASETS.TRAIN ('ship_train_SMD_cocostyle',) DATASETS.SOURCE_TRAIN ('ship_train_SMD_cocostyle',) DATASETS.TARGET_TRAIN ('ship_train_SeaShips_cocostyle',)

if __name__=='__main__':
    #test_net_allcheckpoints()
    # train_net_da_ship()
    # test_net_D2D_ship()
    test_net_D2D_dashipV()
    # featureExtractor_visualization_D2D_ship()
    # featureExtractor_visualization_da_ship()
    # featureExtractor_visualization_D2D_ship()

