python tools/train_net.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS_ship.yaml OUTPUT_DIR ../logSSship DATASETS.TRAIN ('ship_train_SeaShips_cocostyle',) MODEL.DOMAIN_ADAPTATION_ON False

python tools/train_net.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SMD_ship.yaml OUTPUT_DIR ../logSMDship DATASETS.TRAIN ('ship_train_SMD_cocostyle',) MODEL.DOMAIN_ADAPTATION_ON False
python tools/train_net.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SMD_ship.yaml OUTPUT_DIR ../logSMD0ship DATASETS.TRAIN ('ship_train_SMD0_cocostyle',) MODEL.DOMAIN_ADAPTATION_ON False
python tools/train_net.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SMD_ship.yaml OUTPUT_DIR ../logSMD1ship DATASETS.TRAIN ('ship_train_SMD1_cocostyle',) MODEL.DOMAIN_ADAPTATION_ON False
python tools/train_net.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SMD_ship.yaml OUTPUT_DIR ../logSMD2ship DATASETS.TRAIN ('ship_train_SMD2_cocostyle',) MODEL.DOMAIN_ADAPTATION_ON False
python tools/train_net.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SMD_ship.yaml OUTPUT_DIR ../logSMD3ship DATASETS.TRAIN ('ship_train_SMD3_cocostyle',) MODEL.DOMAIN_ADAPTATION_ON False
python tools/train_net.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SMD_ship.yaml OUTPUT_DIR ../logSMD4ship DATASETS.TRAIN ('ship_train_SMD4_cocostyle',) MODEL.DOMAIN_ADAPTATION_ON False


python tools/train_net.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS_ship.yaml OUTPUT_DIR ../logSS0ship DATASETS.TRAIN ('ship_train_SeaShips0_cocostyle',) MODEL.DOMAIN_ADAPTATION_ON False
python tools/train_net.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS_ship.yaml OUTPUT_DIR ../logSS1ship DATASETS.TRAIN ('ship_train_SeaShips1_cocostyle',) MODEL.DOMAIN_ADAPTATION_ON False
python tools/train_net.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS_ship.yaml OUTPUT_DIR ../logSS2ship DATASETS.TRAIN ('ship_train_SeaShips2_cocostyle',) MODEL.DOMAIN_ADAPTATION_ON False
python tools/train_net.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS_ship.yaml OUTPUT_DIR ../logSS3ship DATASETS.TRAIN ('ship_train_SeaShips3_cocostyle',) MODEL.DOMAIN_ADAPTATION_ON False
python tools/train_net.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS_ship.yaml OUTPUT_DIR ../logSS4ship DATASETS.TRAIN ('ship_train_SeaShips4_cocostyle',) MODEL.DOMAIN_ADAPTATION_ON False

rem python tools/train_net.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS_SMD_ship.yaml OUTPUT_DIR ../logSS_SMDship
