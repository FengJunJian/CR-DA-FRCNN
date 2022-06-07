python tools/featureExtractor.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS_ship.yaml OUTPUT_DIR ../SWA1/logSMDToSSship DATASETS.TEST ('ship_test_SMD_cocostyle','ship_test_SeaShips_cocostyle') MODEL.WEIGHT model_0020000.pth
python tools/featureExtractor.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SMD2SS_ship.yaml OUTPUT_DIR ../SWA1/logSSToSMDship DATASETS.TEST ('ship_test_SMD_cocostyle','ship_test_SeaShips_cocostyle') MODEL.WEIGHT model_0020000.pth

rem python tools/featureExtractor.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS.yaml OUTPUT_DIR ../logSSToSMDshipC DATASETS.TEST ('ship_test_SMD_cocostyle','ship_test_SeaShips_cocostyle') MODEL.WEIGHT model_0020000.pth
rem python tools/featureExtractor.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS.yaml OUTPUT_DIR ../logSMDToSSshipC DATASETS.TEST ('ship_test_SMD_cocostyle','ship_test_SeaShips_cocostyle') MODEL.WEIGHT model_0020000.pth


rem python tools/featureExtractor.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS.yaml OUTPUT_DIR ../logSSToSMDshipC DATASETS.TEST ('ship_test_SMD_cocostyle','ship_test_SeaShips_cocostyle') MODEL.WEIGHT model_0020000.pth
rem python tools/featureExtractor.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS.yaml OUTPUT_DIR ../logSMDToSSshipC DATASETS.TEST ('ship_test_SMD_cocostyle','ship_test_SeaShips_cocostyle') MODEL.WEIGHT model_0020000.pth


rem python tools/featureExtractor.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS.yaml OUTPUT_DIR ../logSSToSMDship DATASETS.TEST ('ship_test_SMD_cocostyle','ship_test_SeaShips_cocostyle') MODEL.WEIGHT model_0020000.pth
rem python tools/featureExtractor.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS.yaml OUTPUT_DIR ../logSMDToSSship DATASETS.TEST ('ship_test_SMD_cocostyle','ship_test_SeaShips_cocostyle') MODEL.WEIGHT model_0020000.pth

