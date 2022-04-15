rem da_faster_rcnn_R_50_C4_SMD2SS########
python tools/train_net.py --config-file configs/da_ship/sw_faster_rcnn_R_50_C4_SMD2SS_ship.yaml OUTPUT_DIR ../swlogSMDToSSship DATASETS.TRAIN ('ship_train_SMD_cocostyle',) DATASETS.SOURCE_TRAIN ('ship_train_SMD_cocostyle',) DATASETS.TARGET_TRAIN ('ship_train_SeaShips_cocostyle',) DATASETS.TEST ('ship_test_SeaShips_cocostyle',) SOLVER.MAX_ITER 20000


rem da_faster_rcnn_R_50_C4_SS2SMD########
python tools/train_net.py --config-file configs/da_ship/sw_faster_rcnn_R_50_C4_SS2SMD_ship.yaml OUTPUT_DIR ../swlogSSToSMDship DATASETS.TRAIN ('ship_train_SeaShips_cocostyle',) DATASETS.SOURCE_TRAIN ('ship_train_SeaShips_cocostyle',) DATASETS.TARGET_TRAIN ('ship_train_SMD_cocostyle',) DATASETS.TEST ('ship_test_SMD_cocostyle',) SOLVER.MAX_ITER 20000
