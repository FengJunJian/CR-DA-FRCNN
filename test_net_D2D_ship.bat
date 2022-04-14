rem ship_test_SeaShips_cocostyle########
python tools/test_net.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS_ship.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../logSSship DATASETS.TEST ('ship_test_SMD_cocostyle',)

python tools/test_net.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS_ship.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../logSMDship DATASETS.TEST ('ship_test_SeaShips_cocostyle',) 
python tools/test_net.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS_ship.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../logSMD0ship DATASETS.TEST ('ship_test_SeaShips_cocostyle',) 
python tools/test_net.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS_ship.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../logSMD1ship DATASETS.TEST ('ship_test_SeaShips_cocostyle',) 
python tools/test_net.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS_ship.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../logSMD2ship DATASETS.TEST ('ship_test_SeaShips_cocostyle',) 
python tools/test_net.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS_ship.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../logSMD3ship DATASETS.TEST ('ship_test_SeaShips_cocostyle',) 
python tools/test_net.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS_ship.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../logSMD4ship DATASETS.TEST ('ship_test_SeaShips_cocostyle',) 
python tools/test_net.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS_ship.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../logSSship DATASETS.TEST ('ship_test_SeaShips_cocostyle',) 
python tools/test_net.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS_ship.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../logSS0ship DATASETS.TEST ('ship_test_SeaShips_cocostyle',) 
python tools/test_net.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS_ship.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../logSS1ship DATASETS.TEST ('ship_test_SeaShips_cocostyle',) 
python tools/test_net.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS_ship.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../logSS2ship DATASETS.TEST ('ship_test_SeaShips_cocostyle',) 
python tools/test_net.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS_ship.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../logSS3ship DATASETS.TEST ('ship_test_SeaShips_cocostyle',) 
python tools/test_net.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS_ship.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../logSS4ship DATASETS.TEST ('ship_test_SeaShips_cocostyle',) 

rem ship_test_SMD_cocostyle########
python tools/test_net.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS_ship.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../logSMDship DATASETS.TEST ('ship_test_SMD_cocostyle',) 
python tools/test_net.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS_ship.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../logSMD0ship DATASETS.TEST ('ship_test_SMD_cocostyle',) 
python tools/test_net.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS_ship.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../logSMD1ship DATASETS.TEST ('ship_test_SMD_cocostyle',) 
python tools/test_net.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS_ship.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../logSMD2ship DATASETS.TEST ('ship_test_SMD_cocostyle',) 
python tools/test_net.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS_ship.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../logSMD3ship DATASETS.TEST ('ship_test_SMD_cocostyle',) 
python tools/test_net.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS_ship.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../logSMD4ship DATASETS.TEST ('ship_test_SMD_cocostyle',) 

python tools/test_net.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS_ship.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../logSS0ship DATASETS.TEST ('ship_test_SMD_cocostyle',) 
python tools/test_net.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS_ship.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../logSS1ship DATASETS.TEST ('ship_test_SMD_cocostyle',) 
python tools/test_net.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS_ship.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../logSS2ship DATASETS.TEST ('ship_test_SMD_cocostyle',) 
python tools/test_net.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS_ship.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../logSS3ship DATASETS.TEST ('ship_test_SMD_cocostyle',) 
python tools/test_net.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS_ship.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../logSS4ship DATASETS.TEST ('ship_test_SMD_cocostyle',) 

rem python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SMD2SS.yaml --flagVisual True --checkpoint model_0020000.pth OUTPUT_DIR ../logSMDToSSship DATASETS.TEST ('ship_test_SeaShips_cocostyle',)

rem --config-file ../configs/da_ship/faster_rcnn_R_50_C4_SS_ship.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../../logSMDship DATASETS.TEST ('ship_test_SMD_cocostyle',)
