python tools/test_net.py --config-file configs/da_ship/ship_e2e_da_faster_rcnn_R_50_C4_SMD2SeaShips.yaml MODEL.WEIGHT ./ MODEL.DOMAIN_ADAPTATION_ON False ^
DATASETS.TEST ('ship_test_SeaShips_cocostyle',) OUTPUT_DIR logSMD2SS

python tools/test_net.py --config-file configs/da_ship/ship_e2e_da_faster_rcnn_R_50_C4_SMD2SeaShips.yaml MODEL.WEIGHT ./ MODEL.DOMAIN_ADAPTATION_ON False ^
DATASETS.TEST ('ship_test_SMD_cocostyle',) OUTPUT_DIR logSMD2SS

python tools/test_net.py --config-file configs/da_ship/ship_e2e_da_faster_rcnn_R_50_C4_SMD2SeaShips.yaml MODEL.WEIGHT ./ MODEL.DOMAIN_ADAPTATION_ON False ^
DATASETS.TEST ('ship_test1283_cocostyle',) OUTPUT_DIR logSMD2SS

python tools/test_net.py --config-file configs/da_ship/ship_e2e_da_faster_rcnn_R_50_C4_SeaShips2SMD.yaml MODEL.WEIGHT ./ MODEL.DOMAIN_ADAPTATION_ON False ^
DATASETS.TEST ('ship_test_SeaShips_cocostyle',) OUTPUT_DIR logSS2SMD

python tools/test_net.py --config-file configs/da_ship/ship_e2e_da_faster_rcnn_R_50_C4_SeaShips2SMD.yaml MODEL.WEIGHT ./ MODEL.DOMAIN_ADAPTATION_ON False ^
DATASETS.TEST ('ship_test_SMD_cocostyle',) OUTPUT_DIR logSS2SMD

python tools/test_net.py --config-file configs/da_ship/ship_e2e_da_faster_rcnn_R_50_C4_SeaShips2SMD.yaml MODEL.WEIGHT ./ MODEL.DOMAIN_ADAPTATION_ON False ^
DATASETS.TEST ('ship_test1283_cocostyle',) OUTPUT_DIR logSS2SMD