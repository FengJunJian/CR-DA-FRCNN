rem train:SS test:SS
python tools/featureExtractor.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS.yaml OUTPUT_DIR ../logSS MODEL.WEIGHT ..\logSS\model_0020000.pth DATASETS.TRAIN ('train_SeaShips0_cocostyle',) DATASETS.TEST ('test_SeaShips_cocostyle',) 
python tools/featureExtractor.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS.yaml OUTPUT_DIR ../logSS0 MODEL.WEIGHT ..\logSS0\model_0020000.pth DATASETS.TRAIN ('train_SeaShips0_cocostyle',) DATASETS.TEST ('test_SeaShips_cocostyle',)  
python tools/featureExtractor.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS.yaml OUTPUT_DIR ../logSS1 MODEL.WEIGHT ..\logSS1\model_0020000.pth DATASETS.TRAIN ('train_SeaShips0_cocostyle',) DATASETS.TEST ('test_SeaShips_cocostyle',)  
python tools/featureExtractor.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS.yaml OUTPUT_DIR ../logSS2 MODEL.WEIGHT ..\logSS2\model_0020000.pth DATASETS.TRAIN ('train_SeaShips0_cocostyle',) DATASETS.TEST ('test_SeaShips_cocostyle',)  
python tools/featureExtractor.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS.yaml OUTPUT_DIR ../logSS3 MODEL.WEIGHT ..\logSS3\model_0020000.pth DATASETS.TRAIN ('train_SeaShips0_cocostyle',) DATASETS.TEST ('test_SeaShips_cocostyle',)  
python tools/featureExtractor.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS.yaml OUTPUT_DIR ../logSS4 MODEL.WEIGHT ..\logSS4\model_0020000.pth DATASETS.TRAIN ('train_SeaShips0_cocostyle',) DATASETS.TEST ('test_SeaShips_cocostyle',)  
rem train:SS test:SMD
python tools/featureExtractor.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS.yaml OUTPUT_DIR ../logSS MODEL.WEIGHT ..\logSS\model_0020000.pth DATASETS.TRAIN ('train_SeaShips0_cocostyle',) DATASETS.TEST ('test_SMD_cocostyle',)  
python tools/featureExtractor.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS.yaml OUTPUT_DIR ../logSS0 MODEL.WEIGHT ..\logSS0\model_0020000.pth DATASETS.TRAIN ('train_SeaShips0_cocostyle',) DATASETS.TEST ('test_SMD_cocostyle',)  
python tools/featureExtractor.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS.yaml OUTPUT_DIR ../logSS1 MODEL.WEIGHT ..\logSS1\model_0020000.pth DATASETS.TRAIN ('train_SeaShips0_cocostyle',) DATASETS.TEST ('test_SMD_cocostyle',)  
python tools/featureExtractor.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS.yaml OUTPUT_DIR ../logSS2 MODEL.WEIGHT ..\logSS2\model_0020000.pth DATASETS.TRAIN ('train_SeaShips0_cocostyle',) DATASETS.TEST ('test_SMD_cocostyle',)  
python tools/featureExtractor.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS.yaml OUTPUT_DIR ../logSS3 MODEL.WEIGHT ..\logSS3\model_0020000.pth DATASETS.TRAIN ('train_SeaShips0_cocostyle',) DATASETS.TEST ('test_SMD_cocostyle',)  
python tools/featureExtractor.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS.yaml OUTPUT_DIR ../logSS4 MODEL.WEIGHT ..\logSS4\model_0020000.pth DATASETS.TRAIN ('train_SeaShips0_cocostyle',) DATASETS.TEST ('test_SMD_cocostyle',)  


rem train:SMD test:SS
python tools/featureExtractor.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS.yaml OUTPUT_DIR ../logSMD MODEL.WEIGHT ..\logSMD\model_0020000.pth DATASETS.TRAIN ('train_SeaShips0_cocostyle',) DATASETS.TEST ('test_SeaShips_cocostyle',)  
python tools/featureExtractor.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS.yaml OUTPUT_DIR ../logSMD0 MODEL.WEIGHT ..\logSMD0\model_0020000.pth DATASETS.TRAIN ('train_SeaShips0_cocostyle',) DATASETS.TEST ('test_SeaShips_cocostyle',)  
python tools/featureExtractor.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS.yaml OUTPUT_DIR ../logSMD1 MODEL.WEIGHT ..\logSMD1\model_0020000.pth DATASETS.TRAIN ('train_SeaShips0_cocostyle',) DATASETS.TEST ('test_SeaShips_cocostyle',)  
python tools/featureExtractor.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS.yaml OUTPUT_DIR ../logSMD2 MODEL.WEIGHT ..\logSMD2\model_0020000.pth DATASETS.TRAIN ('train_SeaShips0_cocostyle',) DATASETS.TEST ('test_SeaShips_cocostyle',)  
python tools/featureExtractor.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS.yaml OUTPUT_DIR ../logSMD3 MODEL.WEIGHT ..\logSMD3\model_0020000.pth DATASETS.TRAIN ('train_SeaShips0_cocostyle',) DATASETS.TEST ('test_SeaShips_cocostyle',)  
python tools/featureExtractor.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS.yaml OUTPUT_DIR ../logSMD4 MODEL.WEIGHT ..\logSMD4\model_0020000.pth DATASETS.TRAIN ('train_SeaShips0_cocostyle',) DATASETS.TEST ('test_SeaShips_cocostyle',)  
rem train:SMD test:SMD
python tools/featureExtractor.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS.yaml OUTPUT_DIR ../logSMD MODEL.WEIGHT ..\logSMD\model_0020000.pth DATASETS.TRAIN ('train_SeaShips0_cocostyle',) DATASETS.TEST ('test_SMD_cocostyle',)  
python tools/featureExtractor.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS.yaml OUTPUT_DIR ../logSMD0 MODEL.WEIGHT ..\logSMD0\model_0020000.pth DATASETS.TRAIN ('train_SeaShips0_cocostyle',) DATASETS.TEST ('test_SMD_cocostyle',)  
python tools/featureExtractor.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS.yaml OUTPUT_DIR ../logSMD1 MODEL.WEIGHT ..\logSMD1\model_0020000.pth DATASETS.TRAIN ('train_SeaShips0_cocostyle',) DATASETS.TEST ('test_SMD_cocostyle',)  
python tools/featureExtractor.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS.yaml OUTPUT_DIR ../logSMD2 MODEL.WEIGHT ..\logSMD2\model_0020000.pth DATASETS.TRAIN ('train_SeaShips0_cocostyle',) DATASETS.TEST ('test_SMD_cocostyle',)  
python tools/featureExtractor.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS.yaml OUTPUT_DIR ../logSMD3 MODEL.WEIGHT ..\logSMD3\model_0020000.pth DATASETS.TRAIN ('train_SeaShips0_cocostyle',) DATASETS.TEST ('test_SMD_cocostyle',)  
python tools/featureExtractor.py --config-file configs/da_ship/faster_rcnn_R_50_C4_SS.yaml OUTPUT_DIR ../logSMD4 MODEL.WEIGHT ..\logSMD4\model_0020000.pth DATASETS.TRAIN ('train_SeaShips0_cocostyle',) DATASETS.TEST ('test_SMD_cocostyle',)  

rem #####################Visualization##################
python tools/featureVisualization.py --mode S1
