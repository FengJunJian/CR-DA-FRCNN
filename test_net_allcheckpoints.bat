rem ship_test_SeaShips_cocostyle########
python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml --checkpoint ('model_0002500.pth','model_0005000.pth','model_0007500.pth','model_0010000.pth','model_0012500.pth','model_0015000.pth','model_0017500.pth','model_0020000.pth',) OUTPUT_DIR ../logSSToSMDship DATASETS.TEST ('ship_test_SeaShips_cocostyle',) 
python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml --checkpoint ('model_0002500.pth','model_0005000.pth','model_0007500.pth','model_0010000.pth','model_0012500.pth','model_0015000.pth','model_0017500.pth','model_0020000.pth',) OUTPUT_DIR ../logSMDToSSship DATASETS.TEST ('ship_test_SeaShips_cocostyle',) 

rem ship_test_SMD_cocostyle########
python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml --checkpoint ('model_0002500.pth','model_0005000.pth','model_0007500.pth','model_0010000.pth','model_0012500.pth','model_0015000.pth','model_0017500.pth','model_0020000.pth',) OUTPUT_DIR ../logSSToSMDship DATASETS.TEST ('ship_test_SMD_cocostyle',) 
python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml --checkpoint ('model_0002500.pth','model_0005000.pth','model_0007500.pth','model_0010000.pth','model_0012500.pth','model_0015000.pth','model_0017500.pth','model_0020000.pth',) OUTPUT_DIR ../logSMDToSSship DATASETS.TEST ('ship_test_SMD_cocostyle',) 

