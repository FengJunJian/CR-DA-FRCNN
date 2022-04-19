rem da_faster_rcnn_R_50_C4_SMD2SS########
rem python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS_ship.yaml OUTPUT_DIR ../SW12/logSMDToSSship DATASETS.TRAIN ('ship_train_SMD_cocostyle',) DATASETS.SOURCE_TRAIN ('ship_train_SMD_cocostyle',) DATASETS.TARGET_TRAIN ('ship_train_SeaShips_cocostyle',) DATASETS.TEST ('ship_test_SeaShips_cocostyle',) SOLVER.MAX_ITER 100000
rem python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS_ship.yaml --flagVisual True --checkpoint model_0020000.pth OUTPUT_DIR ../SW1/logSMDToSSship DATASETS.TEST ('ship_test_SMD_cocostyle',)
python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS_ship.yaml --flagVisual True --checkpoint model_0020000.pth OUTPUT_DIR ../SW/logSMDToSSship DATASETS.TEST ('ship_test_SeaShips_cocostyle',)


rem da_faster_rcnn_R_50_C4_SS2SMD########
rem python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml OUTPUT_DIR ../SW12/logSSToSMDship DATASETS.TRAIN ('ship_train_SeaShips_cocostyle',) DATASETS.SOURCE_TRAIN ('ship_train_SeaShips_cocostyle',) DATASETS.TARGET_TRAIN ('ship_train_SMD_cocostyle',) DATASETS.TEST ('ship_test_SMD_cocostyle',) SOLVER.MAX_ITER 100000
rem python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml --flagVisual True --checkpoint model_0020000.pth OUTPUT_DIR ../SW1/logSSToSMDship DATASETS.TEST ('ship_test_SeaShips_cocostyle',)
python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml --flagVisual True --checkpoint model_0020000.pth OUTPUT_DIR ../SW/logSSToSMDship DATASETS.TEST ('ship_test_SMD_cocostyle',)

rem python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS.yaml --flagVisual True --checkpoint model_0005000.pth OUTPUT_DIR ../SW4/logSMDToSS
rem python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS.yaml --flagVisual True --checkpoint model_0005000.pth OUTPUT_DIR ../SW4/logSMDToSS DATASETS.TEST ('trainsfer_test_SMD_cocostyle',)

