rem ship_test_SeaShips_cocostyle########
python tools/featureExtractor.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml OUTPUT_DIR ../logSMDToSSship MODEL.WEIGHT ../logSMDToSSship/model_0020000.pth DATASETS.TEST ('ship_test_SeaShips_cocostyle',)
python tools/featureExtractor.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml OUTPUT_DIR ../logSSToSMDship MODEL.WEIGHT ../logSSToSMDship/model_0020000.pth DATASETS.TEST ('ship_test_SeaShips_cocostyle',)

rem ship_test_SMD_cocostyle########
python tools/featureExtractor.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml OUTPUT_DIR ../logSMDToSSship MODEL.WEIGHT ../logSMDToSSship/model_0020000.pth DATASETS.TEST ('ship_test_SMD_cocostyle',)
python tools/featureExtractor.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml OUTPUT_DIR ../logSSToSMDship MODEL.WEIGHT ../logSSToSMDship/model_0020000.pth DATASETS.TEST ('ship_test_SMD_cocostyle',)

rem #####################Visualization##################
python tools/featureVisualization.py --mode S3
