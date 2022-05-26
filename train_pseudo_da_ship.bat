

rem python tools/train_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml OUTPUT_DIR ../SWA1/logSSToSMDship DATASETS.PSEUDO True DATASETS.SOURCE_TRAIN ('ship_train_SeaShips_cocostyle',) DATASETS.PSEUDO_TRAIN ship_train_SMD_cocostyle DATASETS.TARGET_TRAIN ('ship_train_SMD_cocostyle',) DATASETS.TEST ('ship_test_SMD_cocostyle',) DATASETS.PSEUDO_PATH ../SW/logSSToSMDship/inference20000/ MODEL.WEIGHT ../SW/logSSToSMDship/model_0020000.pth

python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml --flagVisual True --checkpoint model_0020000.pth OUTPUT_DIR ../SWA1/logSSToSMDship DATASETS.TEST ('ship_train_SMD_cocostyle',)
rem --flagVisual=False
python tools/train_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml OUTPUT_DIR ../SWA2/logSSToSMDship DATASETS.PSEUDO True DATASETS.SOURCE_TRAIN ('ship_train_SeaShips_cocostyle',) DATASETS.PSEUDO_TRAIN ship_train_SMD_cocostyle DATASETS.TARGET_TRAIN ('ship_train_SMD_cocostyle',) DATASETS.TEST ('ship_test_SMD_cocostyle',) DATASETS.PSEUDO_PATH ../SWA1/logSSToSMDship/inference20000/ MODEL.WEIGHT ../SWA1/logSSToSMDship/model_0020000.pth

python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml --flagVisual True --checkpoint model_0020000.pth OUTPUT_DIR ../SWA2/logSSToSMDship DATASETS.TEST ('ship_train_SMD_cocostyle',)
rem --flagVisual=False
python tools/train_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml OUTPUT_DIR ../SWA3/logSSToSMDship DATASETS.PSEUDO True DATASETS.SOURCE_TRAIN ('ship_train_SeaShips_cocostyle',) DATASETS.PSEUDO_TRAIN ship_train_SMD_cocostyle DATASETS.TARGET_TRAIN ('ship_train_SMD_cocostyle',) DATASETS.TEST ('ship_test_SMD_cocostyle',) DATASETS.PSEUDO_PATH ../SWA2/logSSToSMDship/inference20000/ MODEL.WEIGHT ../SWA2/logSSToSMDship/model_0020000.pth
