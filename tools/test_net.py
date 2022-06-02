# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)


import argparse
import os
import time
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
#from maskrcnn_benchmark.structures.bounding_box import BoxList


from torch.utils.tensorboard import SummaryWriter
from ForTest import testbbox
import re
from ast import literal_eval


def main_testbbox():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="configs/da_ship/da_faster_rcnn_R_50_C4_SeaShips_SMD12.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--flagVisual",type=bool,default=False)
    parser.add_argument("--flagEven", type=bool, default=False)
    #parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--checkpoint",type=str,default=None)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=['MODEL.DOMAIN_ADAPTATION_ON',False,
                 'OUTPUT_DIR','log12'],

        nargs=argparse.REMAINDER,type=str
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1
    print(args.opts)
    time.sleep(1)
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # cfg.freeze()

    model_dir = cfg.OUTPUT_DIR
    logger = setup_logger("maskrcnn_benchmark", model_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())
    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=model_dir)

    if args.checkpoint:
        try:
            cps = literal_eval(args.checkpoint)
        except:
            cps = [args.checkpoint]

        cfg.MODEL.WEIGHT=[os.path.join(model_dir,cp) for cp in cps]
    else:
        last_checkpoint=os.path.join(model_dir,'last_checkpoint')
        if os.path.exists(last_checkpoint):
            with open(last_checkpoint,'r') as f:
                lines=f.readlines()
                cfg.MODEL.WEIGHT=os.path.join(model_dir,os.path.basename(lines[-1]))
        else:
            cfg.MODEL.WEIGHT=os.path.join(model_dir,'model_final.pth')
    #assert len(cfg.DATASETS.TEST)==1
    if args.flagEven:
        writerT = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'even_'+cfg.DATASETS.TEST[0]))#+time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()).replace(':','-')
    if not isinstance(cfg.MODEL.WEIGHT,list):
        cfg.MODEL.WEIGHT=[cfg.MODEL.WEIGHT]
    for cp in cfg.MODEL.WEIGHT:
        _ = checkpointer.load(cp)
        regxint=re.findall(r'\d+', cp)
        numstr=''
        if len(regxint)>0:
            numstr=str(int(regxint[-1]))
        save_dir = os.path.join(model_dir, 'inference' + numstr)
        # if not os.path.exists(save_dir):
        #     os.mkdir(save_dir)

        logger.info("results will be saved in %s"%(save_dir))

        testResult=testbbox(cfg,model,numstr,flagVisual=args.flagVisual)# will call the model.eval()
        if args.flagEven:
            try:
                for k, v in testResult[0][0].results['bbox'].items():
                    writerT.add_scalar(tag=k, scalar_value=v, global_step=int(numstr))
                    writerT.flush()
            except:
                print('Error:testResult is empty!')
        #model.train()  # see testbbox function


if __name__ == "__main__":
    main_testbbox()
    # main()
