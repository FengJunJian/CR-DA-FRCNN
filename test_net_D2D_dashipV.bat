rem ship_test_SeaShips_cocostyle########
python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml --flagVisual True --checkpoint model_0020000.pth OUTPUT_DIR ../logSSToSMDship DATASETS.TEST ('ship_test_SeaShips_cocostyle',) 
python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml --flagVisual True --checkpoint model_0020000.pth OUTPUT_DIR ../logSMDToSSship DATASETS.TEST ('ship_test_SeaShips_cocostyle',) 

rem ship_test_SMD_cocostyle########
python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml --flagVisual True --checkpoint model_0020000.pth OUTPUT_DIR ../logSSToSMDship DATASETS.TEST ('ship_test_SMD_cocostyle',) 
python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml --flagVisual True --checkpoint model_0020000.pth OUTPUT_DIR ../logSMDToSSship DATASETS.TEST ('ship_test_SMD_cocostyle',) 

