rem original images LG=False, GC=False, useful
rem da_faster_rcnn_R_50_C4_SMD2SS########
rem python tools/train_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS_ship.yaml OUTPUT_DIR ../SW/logSMDToSSship DATASETS.TRAIN ('ship_train_SMD_cocostyle',) DATASETS.SOURCE_TRAIN ('ship_train_SMD_cocostyle',) DATASETS.TARGET_TRAIN ('ship_train_SeaShips_cocostyle',) DATASETS.TEST ('ship_test_SeaShips_cocostyle',) SOLVER.MAX_ITER 20000

rem da_faster_rcnn_R_50_C4_SS2SMD########
rem python tools/train_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml OUTPUT_DIR ../SW/logSSToSMDship DATASETS.TRAIN ('ship_train_SeaShips_cocostyle',) DATASETS.SOURCE_TRAIN ('ship_train_SeaShips_cocostyle',) DATASETS.TARGET_TRAIN ('ship_train_SMD_cocostyle',) DATASETS.TEST ('ship_test_SMD_cocostyle',) SOLVER.MAX_ITER 20000

rem original images pseudo label
rem da_faster_rcnn_R_50_C4_SMD2SS########
python tools/train_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS_ship.yaml OUTPUT_DIR ../SWP/logSMDToSSship DATASETS.PSEUDO True DATASETS.SOURCE_TRAIN ('ship_train_SMD_cocostyle',) DATASETS.PSEUDO_TRAIN ship_train_SeaShips_cocostyle DATASETS.TARGET_TRAIN ('ship_train_SeaShips_cocostyle',) DATASETS.TEST ('ship_test_SeaShips_cocostyle',) DATASETS.PSEUDO_PATH ../SW/logSMDToSSship/inference20000/

rem da_faster_rcnn_R_50_C4_SS2SMD########
python tools/train_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml OUTPUT_DIR ../SWP/logSSToSMDship DATASETS.PSEUDO True DATASETS.SOURCE_TRAIN ('ship_train_SeaShips_cocostyle',) DATASETS.PSEUDO_TRAIN ship_train_SMD_cocostyle DATASETS.TARGET_TRAIN ('ship_train_SMD_cocostyle',) DATASETS.TEST ('ship_test_SMD_cocostyle',) DATASETS.PSEUDO_PATH ../SW/logSSToSMDship/inference20000/

rem python tools/train_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml OUTPUT_DIR ../SWP1/logSSToSMDship DATASETS.PSEUDO True DATASETS.SOURCE_TRAIN ('ship_train_SeaShips_cocostyle',) DATASETS.PSEUDO_TRAIN ship_train_SMD_cocostyle DATASETS.TARGET_TRAIN ('ship_train_SMD_cocostyle',) DATASETS.TEST ('ship_test_SMD_cocostyle',) DATASETS.PSEUDO_PATH ../SWP/logSSToSMDship/inference20000/


rem #########################################################
rem LG=True, GC=True, little useful
rem da_faster_rcnn_R_50_C4_SMD2SS#######
rem python tools/train_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS_ship.yaml OUTPUT_DIR ../SW_LCGC/logSMDToSSship DATASETS.TRAIN ('ship_train_SMD_cocostyle',) DATASETS.SOURCE_TRAIN ('ship_train_SMD_cocostyle',) DATASETS.TARGET_TRAIN ('ship_train_SeaShips_cocostyle',) DATASETS.TEST ('ship_test_SeaShips_cocostyle',) SOLVER.MAX_ITER 20000

rem da_faster_rcnn_R_50_C4_SS2SMD########
rem python tools/train_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml OUTPUT_DIR ../SW_LCGC/logSSToSMDship DATASETS.TRAIN ('ship_train_SeaShips_cocostyle',) DATASETS.SOURCE_TRAIN ('ship_train_SeaShips_cocostyle',) DATASETS.TARGET_TRAIN ('ship_train_SMD_cocostyle',) DATASETS.TEST ('ship_test_SMD_cocostyle',) SOLVER.MAX_ITER 20000


rem #########################################################


rem unified class names: useless#######
rem python tools/train_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS.yaml OUTPUT_DIR ../SW4W/logSMDToSS SOLVER.MAX_ITER 20000

rem da_faster_rcnn_R_50_C4_SS2SMD########
rem python tools/train_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD.yaml OUTPUT_DIR ../SW4W/logSSToSMD SOLVER.MAX_ITER 20000

rem #########################################################
rem edge images useless
rem da_faster_rcnn_R_50_C4_SMD2SS#######
rem python tools/train_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS_ship.yaml OUTPUT_DIR ../SWEdge/logSMDToSSship DATASETS.TRAIN ('ship_train_SMD_cocostyle',) DATASETS.SOURCE_TRAIN ('ship_train_SMD_cocostyle',) DATASETS.TARGET_TRAIN ('ship_train_SeaShips_cocostyle',) DATASETS.TEST ('ship_test_SeaShips_cocostyle',) SOLVER.MAX_ITER 20000

rem da_faster_rcnn_R_50_C4_SS2SMD########
rem python tools/train_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml OUTPUT_DIR ../SWEdge/logSSToSMDship DATASETS.TRAIN ('ship_train_SeaShips_cocostyle',) DATASETS.SOURCE_TRAIN ('ship_train_SeaShips_cocostyle',) DATASETS.TARGET_TRAIN ('ship_train_SMD_cocostyle',) DATASETS.TEST ('ship_test_SMD_cocostyle',) SOLVER.MAX_ITER 20000