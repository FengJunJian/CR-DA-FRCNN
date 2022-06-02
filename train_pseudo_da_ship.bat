

rem python tools/train_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml OUTPUT_DIR ../SWA1/logSSToSMDship DATASETS.PSEUDO True DATASETS.SOURCE_TRAIN ('ship_train_SeaShips_cocostyle',) DATASETS.PSEUDO_TRAIN ship_train_SMD_cocostyle DATASETS.TARGET_TRAIN ('ship_train_SMD_cocostyle',) DATASETS.TEST ('ship_test_SMD_cocostyle',) DATASETS.PSEUDO_PATH ../SW/logSSToSMDship/inference20000/ MODEL.WEIGHT ../SW/logSSToSMDship/model_0020000.pth

rem python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml --flagVisual True --checkpoint model_0020000.pth OUTPUT_DIR ../SWA1/logSSToSMDship DATASETS.TEST ('ship_train_SMD_cocostyle',)
rem --flagVisual=False
rem python tools/train_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml OUTPUT_DIR ../SWA2/logSSToSMDship DATASETS.PSEUDO True DATASETS.SOURCE_TRAIN ('ship_train_SeaShips_cocostyle',) DATASETS.PSEUDO_TRAIN ship_train_SMD_cocostyle DATASETS.TARGET_TRAIN ('ship_train_SMD_cocostyle',) DATASETS.TEST ('ship_test_SMD_cocostyle',) DATASETS.PSEUDO_PATH ../SWA1/logSSToSMDship/inference20000/ MODEL.WEIGHT ../SWA1/logSSToSMDship/model_0020000.pth

rem python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml --flagVisual True --checkpoint model_0020000.pth OUTPUT_DIR ../SWA2/logSSToSMDship DATASETS.TEST ('ship_train_SMD_cocostyle',)
rem --flagVisual=False
rem python tools/train_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml OUTPUT_DIR ../SWA3/logSSToSMDship DATASETS.PSEUDO True DATASETS.SOURCE_TRAIN ('ship_train_SeaShips_cocostyle',) DATASETS.PSEUDO_TRAIN ship_train_SMD_cocostyle DATASETS.TARGET_TRAIN ('ship_train_SMD_cocostyle',) DATASETS.TEST ('ship_test_SMD_cocostyle',) DATASETS.PSEUDO_PATH ../SWA2/logSSToSMDship/inference20000/ MODEL.WEIGHT ../SWA2/logSSToSMDship/model_0020000.pth


rem python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS_ship.yaml --flagVisual True --checkpoint model_0020000.pth OUTPUT_DIR ../SWA1/logSMDToSSship DATASETS.TEST ('ship_train_SMD_cocostyle',)
rem python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS_ship.yaml --flagVisual True --checkpoint model_0020000.pth OUTPUT_DIR ../SW/logSMDToSSship DATASETS.TEST ('ship_train_SeaShips_cocostyle',)
rem --flagVisual=False
rem python tools/train_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS_ship.yaml OUTPUT_DIR ../SWA1/logSMDToSSship DATASETS.PSEUDO True DATASETS.PSEUDO_TRAIN ship_train_SeaShips_cocostyle DATASETS.PSEUDO_PATH ../SW/logSMDToSSship/inference20000/ MODEL.WEIGHT ../SW/logSMDToSSship/model_0020000.pth SOLVER.MAX_ITER 15000
rem python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS_ship.yaml --flagVisual True --checkpoint model_0015000.pth OUTPUT_DIR ../SWA1/logSMDToSSship DATASETS.TEST ('ship_train_SeaShips_cocostyle',)
rem --flagVisual=False

rem python tools/train_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS_ship.yaml OUTPUT_DIR ../SWA2/logSMDToSSship DATASETS.PSEUDO True DATASETS.PSEUDO_TRAIN ship_train_SeaShips_cocostyle DATASETS.PSEUDO_PATH ../SWA1/logSMDToSSship/inference15000/ MODEL.WEIGHT ../SWA1/logSMDToSSship/model_0015000.pth SOLVER.MAX_ITER 15000
rem python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS_ship.yaml --flagVisual True --checkpoint model_0015000.pth OUTPUT_DIR ../SWA2/logSMDToSSship DATASETS.TEST ('ship_train_SeaShips_cocostyle',)

rem python tools/train_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS_ship.yaml OUTPUT_DIR ../SWA3/logSMDToSSship DATASETS.PSEUDO True DATASETS.PSEUDO_TRAIN ship_train_SeaShips_cocostyle DATASETS.PSEUDO_PATH ../SWA2/logSMDToSSship/inference15000/ MODEL.WEIGHT ../SWA2/logSMDToSSship/model_0015000.pth SOLVER.MAX_ITER 15000
rem python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS_ship.yaml --flagVisual True --checkpoint model_0015000.pth OUTPUT_DIR ../SWA3/logSMDToSSship DATASETS.TEST ('ship_train_SeaShips_cocostyle',)
rem 伪标注
python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS_ship.yaml --flagVisual True --checkpoint model_0015000.pth OUTPUT_DIR ../SWA1/logSMDToSSship DATASETS.TEST ('ship_train_SeaShips_cocostyle',)
rem SWP
python tools/train_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS_ship.yaml OUTPUT_DIR ../SWP2/logSMDToSSship DATASETS.PSEUDO True DATASETS.PSEUDO_TRAIN ship_train_SeaShips_cocostyle DATASETS.PSEUDO_PATH ../SWA1/logSMDToSSship/inference15000/ MODEL.WEIGHT ../SW/logSMDToSSship/model_0020000.pth SOLVER.MAX_ITER 15000
python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS_ship.yaml --flagVisual True --checkpoint model_0015000.pth OUTPUT_DIR ../SWP2/logSMDToSSship DATASETS.TEST ('ship_train_SeaShips_cocostyle',)

python tools/train_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS_ship.yaml OUTPUT_DIR ../SWP3/logSMDToSSship DATASETS.PSEUDO True DATASETS.PSEUDO_TRAIN ship_train_SeaShips_cocostyle DATASETS.PSEUDO_PATH ../SWP2/logSMDToSSship/inference15000/ MODEL.WEIGHT ../SW/logSMDToSSship/model_0020000.pth SOLVER.MAX_ITER 15000
rem python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS_ship.yaml --flagVisual True --checkpoint model_0015000.pth OUTPUT_DIR ../SWP3/logSMDToSSship DATASETS.TEST ('ship_train_SeaShips_cocostyle',)
rem SWagain
python tools/train_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS_ship.yaml OUTPUT_DIR ../SWA21/logSMDToSSship DATASETS.PSEUDO True DATASETS.PSEUDO_TRAIN ship_train_SeaShips_cocostyle DATASETS.PSEUDO_PATH ../SWA1/logSMDToSSship/inference15000/ MODEL.WEIGHT ../SWA1/logSMDToSSship/model_0015000.pth SOLVER.MAX_ITER 15000
python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS_ship.yaml --flagVisual True --checkpoint model_0015000.pth OUTPUT_DIR ../SWA21/logSMDToSSship DATASETS.TEST ('ship_train_SeaShips_cocostyle',)

python tools/train_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS_ship.yaml OUTPUT_DIR ../SWA31/logSMDToSSship DATASETS.PSEUDO True DATASETS.PSEUDO_TRAIN ship_train_SeaShips_cocostyle DATASETS.PSEUDO_PATH ../SWA21/logSMDToSSship/inference15000/ MODEL.WEIGHT ../SWA21/logSMDToSSship/model_0015000.pth SOLVER.MAX_ITER 15000
rem python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS_ship.yaml --flagVisual True --checkpoint model_0015000.pth OUTPUT_DIR ../SWA3/logSMDToSSship DATASETS.TEST ('ship_train_SeaShips_cocostyle',)
