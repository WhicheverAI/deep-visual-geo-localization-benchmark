import torch
from torch import nn
from torchvision import models

# TODO 可以从requirements.txt中读取。
# 目前我们先不管dependencies, 不影响跑起来
dependencies = ['torch', 'torchvision']  

import sys

# 移除模块
if 'parser' in sys.modules:
    del sys.modules['parser']

# 重新导入模块
import parser
import importlib
importlib.reload(parser)

from pathlib import Path
this_file = Path(__file__).resolve()
this_directory = this_file.parent
sys.path.append(this_directory.as_posix())
import os
original_directory = Path.cwd()
os.chdir(this_directory.as_posix())
# import importlib
# spec = importlib.util.spec_from_file_location("parser", this_directory)

# # 如果spec不为空，且对应的文件存在，则可以安全地导入该模块
# if spec and spec.loader:
#     parser = importlib.util.module_from_spec(spec)
#     # 现在可以执行模块
#     spec.loader.exec_module(parser)
#     VPRModel = parser.VPRModel
import parser
from parser import VPRModel
# 任意的python函数

# def full_ft_dinov2_on_pitts30k():
from model import network


import os
import sys
import torch
import logging
import sklearn
from os.path import join
from datetime import datetime
from torch.utils.model_zoo import load_url
from google_drive_downloader import GoogleDriveDownloader as gdd

import test
import util
import commons
import datasets_ws
from model import network
OFF_THE_SHELF_RADENOVIC = {
    'resnet50conv5_sfm'    : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet50-gem-w-97bf910.pth',
    'resnet101conv5_sfm'   : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet101-gem-w-a155e54.pth',
    'resnet50conv5_gldv1'  : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet50-gem-w-83fdc30.pth',
    'resnet101conv5_gldv1' : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet101-gem-w-a4d43db.pth',
}

OFF_THE_SHELF_NAVER = {
    "resnet50conv5"  : "1oPtE_go9tnsiDLkWjN4NMpKjh-_md1G5",
    'resnet101conv5' : "1UWJGDuHtzaQdFhSMojoYVQjmCXhIwVvy"
}

def deep_vpr_model(**kwargs:VPRModel):
    """load the vpr model with the default arguments
    defined in the parser.py file.
    Or overrided by kwargs.
    Returns:
        _type_: _description_
    """
    # args:VPRModel = parser.parse_arguments()
    args = VPRModel()
    # torch.hub.load 传入的比上面的优先。
    args.__dict__.update(kwargs) # 如果没有传参就是空字典。
    model = network.GeoLocalizationNet(args)

    model = model.to(args.device)

    if args.aggregation in ["netvlad", "crn"]:
        args.features_dim *= args.netvlad_clusters

    if args.off_the_shelf.startswith("radenovic") or args.off_the_shelf.startswith("naver"):
        if args.off_the_shelf.startswith("radenovic"):
            pretrain_dataset_name = args.off_the_shelf.split("_")[1]  # sfm or gldv1 datasets
            url = OFF_THE_SHELF_RADENOVIC[f"{args.backbone}_{pretrain_dataset_name}"]
            state_dict = load_url(url, model_dir=join("data", "off_the_shelf_nets"))
        else:
            # This is a hacky workaround to maintain compatibility
            sys.modules['sklearn.decomposition.pca'] = sklearn.decomposition._pca
            zip_file_path = join("data", "off_the_shelf_nets", args.backbone + "_naver.zip")
            if not os.path.exists(zip_file_path):
                gdd.download_file_from_google_drive(file_id=OFF_THE_SHELF_NAVER[args.backbone],
                                                    dest_path=zip_file_path, unzip=True)
            if args.backbone == "resnet50conv5":
                state_dict_filename = "Resnet50-AP-GeM.pt"
            elif args.backbone == "resnet101conv5":
                state_dict_filename = "Resnet-101-AP-GeM.pt"
            state_dict = torch.load(join("data", "off_the_shelf_nets", state_dict_filename))
        state_dict = state_dict["state_dict"]
        model_keys = model.state_dict().keys()
        renamed_state_dict = {k: v for k, v in zip(model_keys, state_dict.values())}
        model.load_state_dict(renamed_state_dict)
    elif args.resume is not None:
        logging.info(f"Resuming model from {args.resume}")
        model = util.resume_model(args, model)
    os.chdir(original_directory.as_posix())
    return model, args.model_dump(mode='json')