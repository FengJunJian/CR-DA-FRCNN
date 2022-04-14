rem python tools/train_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD00.yaml OUTPUT_DIR semilog00 MODEL.DOMAIN_ADAPTATION_ON True
rem python tools/train_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD01.yaml OUTPUT_DIR semilog01 MODEL.DOMAIN_ADAPTATION_ON True
rem python tools/train_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD02.yaml OUTPUT_DIR semilog02 MODEL.DOMAIN_ADAPTATION_ON True
rem python tools/train_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD03.yaml OUTPUT_DIR semilog03 MODEL.DOMAIN_ADAPTATION_ON True

rem python tools/train_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD10.yaml OUTPUT_DIR semilog10 MODEL.DOMAIN_ADAPTATION_ON True
rem python tools/train_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD11.yaml OUTPUT_DIR semilog11 MODEL.DOMAIN_ADAPTATION_ON True
python tools/train_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD12.yaml OUTPUT_DIR semilog12 MODEL.DOMAIN_ADAPTATION_ON True MODEL.WEIGHT ./semilog12/model_0007500.pth 
rem python tools/train_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD13.yaml OUTPUT_DIR semilog13 MODEL.DOMAIN_ADAPTATION_ON True

python tools/train_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD20.yaml OUTPUT_DIR semilog20 MODEL.DOMAIN_ADAPTATION_ON True
rem python tools/train_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD21.yaml OUTPUT_DIR semilog21 MODEL.DOMAIN_ADAPTATION_ON True
python tools/train_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD22.yaml OUTPUT_DIR semilog22 MODEL.DOMAIN_ADAPTATION_ON True
rem python tools/train_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD23.yaml OUTPUT_DIR semilog23 MODEL.DOMAIN_ADAPTATION_ON True

python tools/train_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD30.yaml OUTPUT_DIR semilog30 MODEL.DOMAIN_ADAPTATION_ON True
rem python tools/train_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD31.yaml OUTPUT_DIR semilog31 MODEL.DOMAIN_ADAPTATION_ON True
python tools/train_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD32.yaml OUTPUT_DIR semilog32 MODEL.DOMAIN_ADAPTATION_ON True
rem python tools/train_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD33.yaml OUTPUT_DIR semilog33 MODEL.DOMAIN_ADAPTATION_ON True




