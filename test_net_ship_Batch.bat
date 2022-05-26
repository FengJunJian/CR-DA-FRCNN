

python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml --flagVisual True --checkpoint model_0020000.pth OUTPUT_DIR ../SW/logSSToSMDship DATASETS.TEST ('ship_test_SeaShips_cocostyle','ship_test_SMD_cocostyle')
python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml --flagVisual True --checkpoint model_0020000.pth OUTPUT_DIR ../SW/logSMDToSSship DATASETS.TEST ('ship_test_SeaShips_cocostyle','ship_test_SMD_cocostyle')
python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../SWA1/logSSToSMDship DATASETS.TEST ('ship_test_SeaShips_cocostyle','ship_test_SMD_cocostyle')
python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../SWA2/logSSToSMDship DATASETS.TEST ('ship_test_SeaShips_cocostyle','ship_test_SMD_cocostyle')
python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../SWA3/logSSToSMDship DATASETS.TEST ('ship_test_SeaShips_cocostyle','ship_test_SMD_cocostyle')


python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml --flagVisual True --checkpoint model_0020000.pth OUTPUT_DIR ../SW_LCGC/logSSToSMDship DATASETS.TEST ('ship_test_SeaShips_cocostyle','ship_test_SMD_cocostyle')
python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml --flagVisual True --checkpoint model_0020000.pth OUTPUT_DIR ../SW_LCGC/logSMDToSSship DATASETS.TEST ('ship_test_SeaShips_cocostyle','ship_test_SMD_cocostyle')
python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml --checkpoint model_0050000.pth OUTPUT_DIR ../SW50000/logSSToSMDship DATASETS.TEST ('ship_test_SeaShips_cocostyle','ship_test_SMD_cocostyle')

python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../SWP/logSSToSMDship DATASETS.TEST ('ship_test_SeaShips_cocostyle','ship_test_SMD_cocostyle')
python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../SWPA/logSSToSMDship DATASETS.TEST ('ship_test_SeaShips_cocostyle','ship_test_SMD_cocostyle')
python tools/test_net.py --config-file configs/da_ship/sw_da_faster_rcnn_R_50_C4_SS2SMD_ship.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../SWPA1/logSSToSMDship DATASETS.TEST ('ship_test_SeaShips_cocostyle','ship_test_SMD_cocostyle')

rem DA
