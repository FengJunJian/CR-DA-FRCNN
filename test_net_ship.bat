rem python tools/test_net.py --config-file configs/da_ship/ship_e2e_da_faster_rcnn_R_50_C4_SMD2SeaShips.yaml MODEL.WEIGHT ./ MODEL.DOMAIN_ADAPTATION_ON False ^
rem DATASETS.TEST ('ship_test_SeaShips_cocostyle',) OUTPUT_DIR logSMD2SS

rem python tools/test_net.py --config-file configs/da_ship/ship_e2e_da_faster_rcnn_R_50_C4_SMD2SeaShips.yaml MODEL.WEIGHT ./ MODEL.DOMAIN_ADAPTATION_ON False ^
rem DATASETS.TEST ('ship_test_SMD_cocostyle',) OUTPUT_DIR logSMD2SS

rem python tools/test_net.py --config-file configs/da_ship/ship_e2e_da_faster_rcnn_R_50_C4_SMD2SeaShips.yaml MODEL.WEIGHT ./ MODEL.DOMAIN_ADAPTATION_ON False ^
rem DATASETS.TEST ('ship_test1283_cocostyle',) OUTPUT_DIR logSMD2SS

rem python tools/test_net.py --config-file configs/da_ship/ship_e2e_da_faster_rcnn_R_50_C4_SeaShips2SMD.yaml MODEL.WEIGHT ./ MODEL.DOMAIN_ADAPTATION_ON False ^
rem DATASETS.TEST ('ship_test_SeaShips_cocostyle',) OUTPUT_DIR logSS2SMD

rem python tools/test_net.py --config-file configs/da_ship/ship_e2e_da_faster_rcnn_R_50_C4_SeaShips2SMD.yaml MODEL.WEIGHT ./ MODEL.DOMAIN_ADAPTATION_ON False ^
rem DATASETS.TEST ('ship_test_SMD_cocostyle',) OUTPUT_DIR logSS2SMD

rem python tools/test_net.py --config-file configs/da_ship/ship_e2e_da_faster_rcnn_R_50_C4_SeaShips2SMD.yaml MODEL.WEIGHT ./ MODEL.DOMAIN_ADAPTATION_ON False ^
rem DATASETS.TEST ('ship_test1283_cocostyle',) OUTPUT_DIR logSS2SMD

python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../SW/logSSToSMDship DATASETS.TEST ('ship_train_SMD0_cocostyle','ship_train_SMD_cocostyle')
python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../SW/logSMDToSSship DATASETS.TEST ('ship_train_SeaShips0_cocostyle','ship_train_SeaShips_cocostyle')
