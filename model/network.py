
import os
from pathlib import Path
import torch
import logging
import torchvision
from torch import nn
from os.path import join
from transformers import ViTModel
from transformers import AutoImageProcessor, AutoModel
from google_drive_downloader import GoogleDriveDownloader as gdd

from model.cct import cct_14_7x2_384
from model.aggregation import Flatten
from model.normalization import L2Norm
import model.aggregation as aggregation

import peft
# print(f"PEFT version: {peft.__version__}")  # debug用
from peft import LoraConfig, get_peft_model
import opendelta
from opendelta import AutoDeltaConfig, AutoDeltaModel
from opendelta import auto_delta

import wandb
from parser import VPRModel

# Pretrained models on Google Landmarks v2 and Places 365
PRETRAINED_MODELS = {
    'resnet18_places'  : '1DnEQXhmPxtBUrRc81nAvT8z17bk-GBj5',
    'resnet50_places'  : '1zsY4mN4jJ-AsmV3h4hjbT72CBfJsgSGC',
    'resnet101_places' : '1E1ibXQcg7qkmmmyYgmwMTh7Xf1cDNQXa',
    'vgg16_places'     : '1UWl1uz6rZ6Nqmp1K5z3GHAIZJmDh4bDu',
    'resnet18_gldv2'   : '1wkUeUXFXuPHuEvGTXVpuP5BMB-JJ1xke',
    'resnet50_gldv2'   : '1UDUv6mszlXNC1lv6McLdeBNMq9-kaA70',
    'resnet101_gldv2'  : '1apiRxMJpDlV0XmKlC5Na_Drg2jtGL-uE',
    'vgg16_gldv2'      : '10Ov9JdO7gbyz6mB5x0v_VSAUMj91Ta4o'
}


class GeoLocalizationNet(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer.
    """
    def __init__(self, args:VPRModel):
        super().__init__()
        self.backbone = get_backbone(args)
        if args.work_with_tokens:
            self.backbone = nn.Sequential(self.backbone, VitPermuteAsCNN())
        self.arch_name = args.backbone
        self.aggregation = get_aggregation(args)

        if args.aggregation in ["gem", "spoc", "mac", "rmac"]:
            if args.l2 == "before_pool":
                self.aggregation = nn.Sequential(L2Norm(), self.aggregation, Flatten())
            elif args.l2 == "after_pool":
                self.aggregation = nn.Sequential(self.aggregation, L2Norm(), Flatten())
            elif args.l2 == "none":
                self.aggregation = nn.Sequential(self.aggregation, Flatten())
        
        if args.fc_output_dim != None:
            # Concatenate fully connected layer to the aggregation layer
            self.aggregation = nn.Sequential(self.aggregation,
                                             nn.Linear(args.features_dim, args.fc_output_dim),
                                             L2Norm())
            args.features_dim = args.fc_output_dim

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregation(x)
        return x


def get_aggregation(args:VPRModel):
    if args.aggregation == "gem":
        return aggregation.GeM(work_with_tokens=args.work_with_tokens)
    elif args.aggregation == "spoc":
        return aggregation.SPoC()
    elif args.aggregation == "mac":
        return aggregation.MAC()
    elif args.aggregation == "rmac":
        return aggregation.RMAC()
    elif args.aggregation == "netvlad":
        return aggregation.NetVLAD(clusters_num=args.netvlad_clusters, dim=args.features_dim,
                                   work_with_tokens=args.work_with_tokens)
    elif args.aggregation == 'crn':
        return aggregation.CRN(clusters_num=args.netvlad_clusters, dim=args.features_dim)
    elif args.aggregation == "rrm":
        return aggregation.RRM(args.features_dim)
    elif args.aggregation in ['cls', 'seqpool']:
        return nn.Identity()


def get_pretrained_model(args:VPRModel):
    if args.pretrain == 'places':  num_classes = 365
    elif args.pretrain == 'gldv2':  num_classes = 512
    
    if args.backbone.startswith("resnet18"):
        model = torchvision.models.resnet18(num_classes=num_classes)
    elif args.backbone.startswith("resnet50"):
        model = torchvision.models.resnet50(num_classes=num_classes)
    elif args.backbone.startswith("resnet101"):
        model = torchvision.models.resnet101(num_classes=num_classes)
    elif args.backbone.startswith("vgg16"):
        model = torchvision.models.vgg16(num_classes=num_classes)
    
    if args.backbone.startswith('resnet'):
        model_name = args.backbone.split('conv')[0] + "_" + args.pretrain
    else:
        model_name = args.backbone + "_" + args.pretrain
    file_path = join("data", "pretrained_nets", model_name +".pth")
    
    if not os.path.exists(file_path):
        gdd.download_file_from_google_drive(file_id=PRETRAINED_MODELS[model_name],
                                            dest_path=file_path)
    state_dict = torch.load(file_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model


def get_backbone(args:VPRModel):
    # The aggregation layer works differently based on the type of architecture
    # TODO 如果你的模型是vit，那么这里应该设置
    args.work_with_tokens = (args.backbone.startswith('cct') or 
                             args.backbone.startswith('vit') or 
                             args.backbone.startswith('dino'))
    # args.work_with_tokens = False
    if args.backbone.startswith("resnet"):
        if args.pretrain in ['places', 'gldv2']:
            backbone = get_pretrained_model(args)
        elif args.backbone.startswith("resnet18"):
            backbone = torchvision.models.resnet18(pretrained=True)
        elif args.backbone.startswith("resnet50"):
            backbone = torchvision.models.resnet50(pretrained=True)
        elif args.backbone.startswith("resnet101"):
            backbone = torchvision.models.resnet101(pretrained=True)
        for name, child in backbone.named_children():
            # Freeze layers before conv_3
            if name == "layer3":
                break
            for params in child.parameters():
                params.requires_grad = False
        if args.backbone.endswith("conv4"):
            logging.debug(f"Train only conv4_x of the resnet{args.backbone.split('conv')[0]} (remove conv5_x), freeze the previous ones")
            layers = list(backbone.children())[:-3]
        elif args.backbone.endswith("conv5"):
            logging.debug(f"Train only conv4_x and conv5_x of the resnet{args.backbone.split('conv')[0]}, freeze the previous ones")
            layers = list(backbone.children())[:-2]
    elif args.backbone == "vgg16":
        if args.pretrain in ['places', 'gldv2']:
            backbone = get_pretrained_model(args)
        else:
            backbone = torchvision.models.vgg16(pretrained=True)
        layers = list(backbone.features.children())[:-2]
        for l in layers[:-5]:
            for p in l.parameters(): p.requires_grad = False
        logging.debug("Train last layers of the vgg16, freeze the previous ones")
    elif args.backbone == "alexnet":
        backbone = torchvision.models.alexnet(pretrained=True)
        layers = list(backbone.features.children())[:-2]
        for l in layers[:5]:
            for p in l.parameters(): p.requires_grad = False
        logging.debug("Train last layers of the alexnet, freeze the previous ones")
    elif args.backbone.startswith("cct"):
        if args.backbone.startswith("cct384"):
            backbone = cct_14_7x2_384(pretrained=True, progress=True, aggregation=args.aggregation)
        if args.trunc_te:
            logging.debug(f"Truncate CCT at transformers encoder {args.trunc_te}")
            backbone.classifier.blocks = torch.nn.ModuleList(backbone.classifier.blocks[:args.trunc_te].children())
        if args.freeze_te:
            logging.debug(f"Freeze all the layers up to tranformer encoder {args.freeze_te}")
            for p in backbone.parameters():
                p.requires_grad = False
            for name, child in backbone.classifier.blocks.named_children():
                if int(name) > args.freeze_te:
                    for params in child.parameters():
                        params.requires_grad = True
        args.features_dim = 384
        return backbone
    elif args.backbone.startswith("vit"):
        assert args.resize[0] in [224, 384], f'Image size for ViT must be either 224 or 384, but it\'s {args.resize[0]}'
        
        
            
        
        if args.pretrain == 'dinov2':
            # 特殊判断 adapter，使用 selavpr 
            if args.peft == 'sela_vpr_adapter':
                # cricavpr
                if os.path.exists('backbone'):
                    os.remove('backbone')
                # os.symlink('../CricaVPR/backbone', 'backbone')
                os.symlink('../SelaVPR/backbone', 'backbone')
                # 不用sys.path.append, 那样的话其他文件名字比较像，会乱。
                from backbone.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
                backbone = vit_base(patch_size=14,img_size=518,init_values=1,block_chunks=0)  
                foundation_model_path = "backbone/dinov2_vitb14_pretrain.pth"
                assert foundation_model_path is not None, "Please specify foundation model path."
                model_dict = backbone.state_dict()
                state_dict = torch.load(foundation_model_path)
                model_dict.update(state_dict.items())
                backbone.load_state_dict(model_dict)
                ## Freeze parameters except adapter
                for name, param in backbone.named_parameters():
                    if "adapter" not in name:
                        param.requires_grad = False

                ## initialize Adapter
                for n, m in backbone.named_modules():
                    if 'adapter' in n:
                        for n2, m2 in m.named_modules():
                            if 'D_fc2' in n2:
                                if isinstance(m2, nn.Linear):
                                    nn.init.constant_(m2.weight, 0.)
                                    nn.init.constant_(m2.bias, 0.)
                        for n2, m2 in m.named_modules():
                            if 'conv' in n2:
                                if isinstance(m2, nn.Conv2d):
                                    nn.init.constant_(m2.weight, 0.00001)
                                    nn.init.constant_(m2.bias, 0.00001)
                args.features_dim = 768
                return FacebookVitWrapper(backbone, args.aggregation).to(args.device)
            else:
                # backbone = ViTModel.from_pretrained('nateraw/dino_vits8')
                # backbone = ViTModel.from_pretrained('timm/vit_base_patch14_dinov2.lvd142m')
                backbone = AutoModel.from_pretrained('facebook/dinov2-base') # 似乎不对？
                # path = Path('../../../pretrains/facebook/dinov2-base').resolve()
                # backbone = AutoModel.from_pretrained(path.as_posix(), local_files_only=True)
        else:
            if args.resize[0] == 224:
                # backbone = ViTModel.from_pretrained('../../../pretrains/google/vit-base-patch16-224-in21k')
                backbone = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
            elif args.resize[0] == 384:
                backbone = ViTModel.from_pretrained('google/vit-base-patch16-384')

        if args.trunc_te:
            logging.debug(f"Truncate ViT at transformers encoder {args.trunc_te}")
            backbone.encoder.layer = backbone.encoder.layer[:args.trunc_te]
        if args.freeze_te:
            # logging.debug(f"Freeze all the layers up to tranformer encoder {args.freeze_te+1}")
            logging.debug(f"Freeze all the layers up to tranformer encoder {args.freeze_te}")
            for p in backbone.parameters():
                p.requires_grad = False
            for name, child in backbone.encoder.layer.named_children():
                assert name.isdigit(), f"Unexpected name '{name}' in ViT encoder, should be an integer."
                if int(name) >= args.freeze_te: # >=逻辑更加合理，就是冻结多少层。比如8，则冻结0-7层共8层。
                    for params in child.parameters():
                        params.requires_grad = True
        
        # backbone = torch.nn.DataParallel(backbone)
        backbone = VitWrapper(backbone, args.aggregation) # 这里有个问题哦，是不是不小心把NetVLAD给冻结了
        backbone = backbone.to(args.device) # 后面opendelta要参考这个模型的device去初始化参数
        if args.peft:
            logging.info(f"Using PEFT method {args.peft} for fine-tuning. ")
            if args.peft in peft.PeftType.__members__:
                logging.debug("Using Huggingface Approach to implement peft.")
                peft_type = peft.PeftType(args.peft)
                # peft_config_cls = peft.PEFT_TYPE_TO_CONFIG_MAPPING[peft_type]
                # peft_config = peft_config_cls()
                config_dict = dict()
                if peft_type is peft.PeftType.LORA:
                # or peft_type is peft.PeftType.ADALORA:
                    config_dict = peft.LoraConfig(
                            r=16,  # Lora矩阵的中间维度。=r 越小，可训练的参数越少，压缩程度越高
                            # r=32,  # 经过实验还是16好
                            lora_alpha=16,  #  LoRA 矩阵的稀疏性=非零元素的比例。lora_alpha 越小，可训练的参数越少，稀疏程度越高.
                            # lora_dropout=0.5, # 防止过拟合，提高泛化能力
                            lora_dropout=0.1, # 防止过拟合，提高泛化能力
                            bias="none",  # bias是否冻结
                        )
                elif peft_type is peft.PeftType.ADALORA:
                    config_dict = peft.AdaLoraConfig(
                        target_r=16,
                        init_r=24
                    )
                elif peft_type is peft.PeftType.OFT:
                    config_dict = dict()
                elif peft_type is peft.PeftType.GLORA:
                    config_dict = peft.GLoraConfig(r=16)   
                else:
                    logging.warning(f"Unsupported PEFT type {peft_type}, may not work well. ")
                config_dict = config_dict.__dict__ # 上面是为了约束参数的范围，看到文档，这里是为了灵活性
                config_dict['peft_type'] = peft_type
                            # target_modules=['qkv'],  # 这里指定想要被 Lora 微调的模块
                config_dict['target_modules'] = ["query", "value"] # https://github.com/huggingface/peft/blob/main/examples/image_classification/image_classification_peft_lora.ipynb
                peft_config = peft.get_peft_config(config_dict)
                print(f"peft_config: {peft_config}")
                # wandb.config.update({"peft": args.peft})
                if not args.no_wandb:
                    wandb.config['peft_config'] = peft_config # 更新wandb配置
                # lora 微调
                backbone = get_peft_model(backbone , 
                                peft_config,          
                                )
                backbone.print_trainable_parameters()
            elif args.peft in auto_delta.LAZY_CONFIG_MAPPING:
                if args.peft == "adapter":
                    # from opendelta import AdapterModel
                    # delta_model = AdapterModel(backbone, 
                    #     bottleneck_dim=24, 
                    #     non_linearity='gelu_new',
                    #     #   common_structure=False,
                    #     #   common_structure=True,
                    #     modified_modules=[
                    #         # "attention.output.dense",
                    #                     #   "mlp.fc2"
                    #         # "dense", "fc2"
                    #         '[r][\d+]\.attention', "mlp"
                    #                       ], 
                    #     # interactive_modify=True
                    #       )
                    # delta_model.freeze_module(exclude=["deltas", "aggregation"])
                    # delta_model = delta_model.to(args.device)
                    # delta_model.log() # 这里还没初始化

                    # import adapters
                    # adapters.init(backbone)
                    # backbone.add_adapter("my_adapter", 
                    #                      # bottleneck adapter
                    #                      config=adapters.BnConfig(
                    #                          mh_adapter=True, 
                    #                             output_adapter=True,
                    #                      )
                    #                      )
                    # backbone.train_adapter("my_adapter")
                    pass
                    
        
        args.features_dim = 768
        return backbone

    backbone = torch.nn.Sequential(*layers)
    args.features_dim = get_output_channels_dim(backbone)  # Dinamically obtain number of channels in output
    return backbone


# 这是huggingface版本的
class VitWrapper(nn.Module):
    def __init__(self, vit_model, aggregation):
        super().__init__()
        self.vit_model = vit_model
        self.aggregation = aggregation
        if 'config' in dir(vit_model):
            self.config = vit_model.config # hf风格
        if 'dummy_inputs' in dir(vit_model):
            self.dummy_inputs = vit_model.dummy_inputs
        # .to('cuda')
        # b, c, h, w = 1, 3, 224, 224
        # self.dummy_inputs = torch.randn(b, c, h, w)
    def forward(self, x):
        if self.aggregation in ["netvlad", "gem"]:
            res = self.vit_model(x).last_hidden_state[:, 1:, :]
        else:
            res = self.vit_model(x).last_hidden_state[:, 0, :]
        return res
    
class FacebookVitWrapper(nn.Module):
    def __init__(self, vit_model, aggregation):
        super().__init__()
        self.vit_model = vit_model
        self.aggregation = aggregation
    def forward(self, x):
        if self.aggregation in ["netvlad", "gem"]:
            res = self.vit_model(x)['x_norm_patchtokens']
        else:
            res = self.vit_model(x)['x_norm_clstoken']
        return res
    

class VitPermuteAsCNN(nn.Module):
    """Some Information about VitPermuteAsCNN"""
    def __init__(self):
        super(VitPermuteAsCNN, self).__init__()

    def forward(self, x):
        # logging.info(f"x.shape: {x.shape}") # batch, patch数量, embed dim
        batch, patches, embed_dim = x.shape
        patch_side = int(patches ** 0.5)
        assert patch_side * patch_side == patches, f"Patch数量{patches}不是平方数"
        x = x.view(batch, patch_side, patch_side, embed_dim)
        x = x.permute(0, 3, 1, 2)
        return x

def get_output_channels_dim(model):
    """Return the number of channels in the output of a model."""
    return model(torch.ones([1, 3, 224, 224])).shape[1]

