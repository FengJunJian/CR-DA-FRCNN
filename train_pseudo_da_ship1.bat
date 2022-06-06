rem Init domain adaptation
python tools/train_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS_ship.yaml OUTPUT_DIR ../SW_1/logSMDToSSship MODEL.WEIGHT ../logSMDship/model_0020000.pth SOLVER.MAX_ITER 15000
python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS_ship.yaml --checkpoint model_0015000.pth OUTPUT_DIR ../SW_1/logSMDToSSship DATASETS.TEST ('ship_train_SeaShips_cocostyle',)

rem one......
python tools/train_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS_ship.yaml OUTPUT_DIR ../SWA1_1/logSMDToSSship DATASETS.PSEUDO True DATASETS.PSEUDO_TRAIN ship_train_SeaShips_cocostyle DATASETS.PSEUDO_PATH ../SW_1/logSMDToSSship/inference15000/ MODEL.WEIGHT ../SW_1/logSMDToSSship/model_0015000.pth SOLVER.MAX_ITER 15000 DATASETS.PSEUDO_THRESHOLD 0.5
python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS_ship.yaml --checkpoint model_0015000.pth OUTPUT_DIR ../SWA1_1/logSMDToSSship DATASETS.TEST ('ship_train_SeaShips_cocostyle',)

python tools/train_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS_ship.yaml OUTPUT_DIR ../SWA2_1/logSMDToSSship DATASETS.PSEUDO True DATASETS.PSEUDO_TRAIN ship_train_SeaShips_cocostyle DATASETS.PSEUDO_PATH ../SWA1_1/logSMDToSSship/inference15000/ MODEL.WEIGHT ../SWA1_1/logSMDToSSship/model_0015000.pth SOLVER.MAX_ITER 15000 DATASETS.PSEUDO_THRESHOLD 0.5
python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS_ship.yaml --checkpoint model_0015000.pth OUTPUT_DIR ../SWA2_1/logSMDToSSship DATASETS.TEST ('ship_train_SeaShips_cocostyle',)
rem SWagain
python tools/train_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS_ship.yaml OUTPUT_DIR ../SWA3_1/logSMDToSSship DATASETS.PSEUDO True DATASETS.PSEUDO_TRAIN ship_train_SeaShips_cocostyle DATASETS.PSEUDO_PATH ../SWA2_1/logSMDToSSship/inference15000/ MODEL.WEIGHT ../SWA2_1/logSMDToSSship/model_0015000.pth SOLVER.MAX_ITER 15000 DATASETS.PSEUDO_THRESHOLD 0.5
python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS_ship.yaml --checkpoint model_0015000.pth OUTPUT_DIR ../SWA3_1/logSMDToSSship DATASETS.TEST ('ship_train_SeaShips_cocostyle',)

rem python tools/train_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS_ship.yaml OUTPUT_DIR ../SWA31/logSMDToSSship DATASETS.PSEUDO True DATASETS.PSEUDO_TRAIN ship_train_SeaShips_cocostyle DATASETS.PSEUDO_PATH ../SWA21/logSMDToSSship/inference15000/ MODEL.WEIGHT ../SWA21/logSMDToSSship/model_0015000.pth SOLVER.MAX_ITER 15000 DATASETS.PSEUDO_THRESHOLD 0.5
rem python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS_ship.yaml --flagVisual True --checkpoint model_0015000.pth OUTPUT_DIR ../SWA3/logSMDToSSship DATASETS.TEST ('ship_train_SeaShips_cocostyle',)
python tools/train_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml OUTPUT_DIR ../SW_1/logSSToSMDship MODEL.WEIGHT ../logSSship/model_0020000.pth SOLVER.MAX_ITER 15000
python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml --checkpoint model_0015000.pth OUTPUT_DIR ../SW_1/logSSToSMDship DATASETS.TEST ('ship_train_SMD_cocostyle',)

rem one......
python tools/train_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml OUTPUT_DIR ../SWA1_1/logSSToSMDship DATASETS.PSEUDO True DATASETS.PSEUDO_TRAIN ship_train_SMD_cocostyle DATASETS.PSEUDO_PATH ../SW_1/logSSToSMDship/inference15000/ MODEL.WEIGHT ../SW_1/logSSToSMDship/model_0015000.pth SOLVER.MAX_ITER 15000 DATASETS.PSEUDO_THRESHOLD 0.5
python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml --checkpoint model_0015000.pth OUTPUT_DIR ../SWA1_1/logSSToSMDship DATASETS.TEST ('ship_train_SMD_cocostyle',)

python tools/train_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml OUTPUT_DIR ../SWA2_1/logSSToSMDship DATASETS.PSEUDO True DATASETS.PSEUDO_TRAIN ship_train_SMD_cocostyle DATASETS.PSEUDO_PATH ../SWA1_1/logSSToSMDship/inference15000/ MODEL.WEIGHT ../SWA1_1/logSSToSMDship/model_0015000.pth SOLVER.MAX_ITER 15000 DATASETS.PSEUDO_THRESHOLD 0.5
python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml --checkpoint model_0015000.pth OUTPUT_DIR ../SWA2_1/logSSToSMDship DATASETS.TEST ('ship_train_SMD_cocostyle',)
rem SWagain
python tools/train_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml OUTPUT_DIR ../SWA3_1/logSSToSMDship DATASETS.PSEUDO True DATASETS.PSEUDO_TRAIN ship_train_SMD_cocostyle DATASETS.PSEUDO_PATH ../SWA2_1/logSSToSMDship/inference15000/ MODEL.WEIGHT ../SWA2_1/logSSToSMDship/model_0015000.pth SOLVER.MAX_ITER 15000 DATASETS.PSEUDO_THRESHOLD 0.5
python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml --checkpoint model_0015000.pth OUTPUT_DIR ../SWA3_1/logSSToSMDship DATASETS.TEST ('ship_train_SMD_cocostyle',)
