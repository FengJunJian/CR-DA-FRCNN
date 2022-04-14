python tools/test_net.py --config-file configs/da_ship/ship_e2e_da_faster_rcnn_R_50_C4_SMD2SeaShips.yaml MODEL.WEIGHT ./ MODEL.DOMAIN_ADAPTATION_ON False ^
DATASETS.TEST ('ship_light_cocostyle',) OUTPUT_DIR logSMD2SS

python tools/test_net.py --config-file configs/da_ship/ship_e2e_da_faster_rcnn_R_50_C4_SMD2SeaShips.yaml MODEL.WEIGHT ./ MODEL.DOMAIN_ADAPTATION_ON False ^
DATASETS.TEST ('ship_occlusion_cocostyle',) OUTPUT_DIR logSMD2SS

python tools/test_net.py --config-file configs/da_ship/ship_e2e_da_faster_rcnn_R_50_C4_SMD2SeaShips.yaml MODEL.WEIGHT ./ MODEL.DOMAIN_ADAPTATION_ON False ^
DATASETS.TEST ('ship_scale_cocostyle',) OUTPUT_DIR logSMD2SS

python tools/test_net.py --config-file configs/da_ship/ship_e2e_da_faster_rcnn_R_50_C4_SeaShips2SMD.yaml MODEL.WEIGHT ./ MODEL.DOMAIN_ADAPTATION_ON False ^
DATASETS.TEST ('ship_light_cocostyle',) OUTPUT_DIR logSS2SMD

python tools/test_net.py --config-file configs/da_ship/ship_e2e_da_faster_rcnn_R_50_C4_SeaShips2SMD.yaml MODEL.WEIGHT ./ MODEL.DOMAIN_ADAPTATION_ON False ^
DATASETS.TEST ('ship_occlusion_cocostyle',) OUTPUT_DIR logSS2SMD

python tools/test_net.py --config-file configs/da_ship/ship_e2e_da_faster_rcnn_R_50_C4_SeaShips2SMD.yaml MODEL.WEIGHT ./ MODEL.DOMAIN_ADAPTATION_ON False ^
DATASETS.TEST ('ship_scale_cocostyle',) OUTPUT_DIR logSS2SMD


rem light
python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD00.yaml OUTPUT_DIR log00 MODEL.DOMAIN_ADAPTATION_ON False DATASETS.TEST ('light_cocostyle',)
python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD01.yaml OUTPUT_DIR log01 MODEL.DOMAIN_ADAPTATION_ON False DATASETS.TEST ('light_cocostyle',)
python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD02.yaml OUTPUT_DIR log02 MODEL.DOMAIN_ADAPTATION_ON False DATASETS.TEST ('light_cocostyle',)
python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD03.yaml OUTPUT_DIR log03 MODEL.DOMAIN_ADAPTATION_ON False DATASETS.TEST ('light_cocostyle',)

python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD10.yaml OUTPUT_DIR log10 MODEL.DOMAIN_ADAPTATION_ON False DATASETS.TEST ('light_cocostyle',)
python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD12.yaml OUTPUT_DIR log12 MODEL.DOMAIN_ADAPTATION_ON False DATASETS.TEST ('light_cocostyle',)

python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD20.yaml OUTPUT_DIR log20 MODEL.DOMAIN_ADAPTATION_ON False DATASETS.TEST ('light_cocostyle',)

python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD30.yaml OUTPUT_DIR log30 MODEL.DOMAIN_ADAPTATION_ON False DATASETS.TEST ('light_cocostyle',)
python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD32.yaml OUTPUT_DIR log32 MODEL.DOMAIN_ADAPTATION_ON False DATASETS.TEST ('light_cocostyle',)
python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD33.yaml OUTPUT_DIR log33 MODEL.DOMAIN_ADAPTATION_ON False DATASETS.TEST ('light_cocostyle',)


rem occlusion
python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD00.yaml OUTPUT_DIR log00 MODEL.DOMAIN_ADAPTATION_ON False DATASETS.TEST ('occlusion_cocostyle',)
python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD01.yaml OUTPUT_DIR log01 MODEL.DOMAIN_ADAPTATION_ON False DATASETS.TEST ('occlusion_cocostyle',)
python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD02.yaml OUTPUT_DIR log02 MODEL.DOMAIN_ADAPTATION_ON False DATASETS.TEST ('occlusion_cocostyle',)
python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD03.yaml OUTPUT_DIR log03 MODEL.DOMAIN_ADAPTATION_ON False DATASETS.TEST ('occlusion_cocostyle',)

python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD10.yaml OUTPUT_DIR log10 MODEL.DOMAIN_ADAPTATION_ON False DATASETS.TEST ('occlusion_cocostyle',)
python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD12.yaml OUTPUT_DIR log12 MODEL.DOMAIN_ADAPTATION_ON False DATASETS.TEST ('occlusion_cocostyle',)

python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD20.yaml OUTPUT_DIR log20 MODEL.DOMAIN_ADAPTATION_ON False DATASETS.TEST ('occlusion_cocostyle',)

python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD30.yaml OUTPUT_DIR log30 MODEL.DOMAIN_ADAPTATION_ON False DATASETS.TEST ('occlusion_cocostyle',)
python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD32.yaml OUTPUT_DIR log32 MODEL.DOMAIN_ADAPTATION_ON False DATASETS.TEST ('occlusion_cocostyle',)
python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD33.yaml OUTPUT_DIR log33 MODEL.DOMAIN_ADAPTATION_ON False DATASETS.TEST ('occlusion_cocostyle',)



rem scale
python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD00.yaml OUTPUT_DIR log00 MODEL.DOMAIN_ADAPTATION_ON False DATASETS.TEST ('scale_cocostyle',)
python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD01.yaml OUTPUT_DIR log01 MODEL.DOMAIN_ADAPTATION_ON False DATASETS.TEST ('scale_cocostyle',)
python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD02.yaml OUTPUT_DIR log02 MODEL.DOMAIN_ADAPTATION_ON False DATASETS.TEST ('scale_cocostyle',)
python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD03.yaml OUTPUT_DIR log03 MODEL.DOMAIN_ADAPTATION_ON False DATASETS.TEST ('scale_cocostyle',)

python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD10.yaml OUTPUT_DIR log10 MODEL.DOMAIN_ADAPTATION_ON False DATASETS.TEST ('scale_cocostyle',)
python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD12.yaml OUTPUT_DIR log12 MODEL.DOMAIN_ADAPTATION_ON False DATASETS.TEST ('scale_cocostyle',)

python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD20.yaml OUTPUT_DIR log20 MODEL.DOMAIN_ADAPTATION_ON False DATASETS.TEST ('scale_cocostyle',)

python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD30.yaml OUTPUT_DIR log30 MODEL.DOMAIN_ADAPTATION_ON False DATASETS.TEST ('scale_cocostyle',)
python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD32.yaml OUTPUT_DIR log32 MODEL.DOMAIN_ADAPTATION_ON False DATASETS.TEST ('scale_cocostyle',)
python tools/test_net.py --config-file configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD33.yaml OUTPUT_DIR log33 MODEL.DOMAIN_ADAPTATION_ON False DATASETS.TEST ('scale_cocostyle',)








