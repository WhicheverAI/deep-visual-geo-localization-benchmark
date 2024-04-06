import os
import torch
import argparse
from pydantic import BaseModel, Field, BaseSettings
import argparse
from typing import List

from pathlib import Path
this_file = Path(__file__).resolve()
this_directory = this_file.parent
#%%
import peft
# T = peft.PeftType
# L = T.LORA
# L.name, L.value
# T(L.name) is L
# T.__members__.items()
# list(T)
from utils import get_env_or_default
#%%
# class ArgumentModel(BaseModel):
class ArgumentModel(BaseSettings):
    """
    1. 代码中写了默认值
    2. 读取环境变量，试图覆盖默认值。 环境变量名字和写的一样。值复杂的话用json。
    3. 实例化，parser读取命令行参数，然后我们重新实例化覆盖参数。
    """
    def create_parser(self):
        """
        Create an ArgumentParser based on the ArgumentModel.
        """
        schema = self.schema()  # 应该得到的是子类的 schema 吧
        schema_properties = schema["properties"]
        parser = argparse.ArgumentParser(description="Process some data.")
        print(self.__annotations__.items())
        for name, pydantic_type in self.__annotations__.items():
            # parser.add_argument(f'--{name.replace("_", "-")}',
            field_info = schema_properties[name]
            filtered_field_info = {
                key: value
                for key, value in field_info.items()
                if key in ["default", "help", "choices"]
            }
            parser.add_argument(
                f"--{name}",
                type=pydantic_type,
                # default=getattr(self, name),
                # default=field_info['default'],
                # help=field_info['help']
                **filtered_field_info,  # 自动填充需要的东西
            )
        return parser
    
    # 保证动态性
    my_extra_fields: dict = Field(default=dict())
    def __setattr__(self, name, value):
        if name not in self.__fields__:
            self.my_extra_fields[name] = value
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        if name not in self.__fields__:
            return self.my_extra_fields[name]
        else:
            super().__getattr__(name)
    class Config:
        case_sensitive = False # 环境变量可以大写也可以小写

class VPRModel(ArgumentModel):
    # no_wandb: bool = Field(False, help="Disable wandb logging")
    no_wandb: bool = Field(True, help="Disable wandb logging")
    train_batch_size: int = Field(
        4,
        # 16,
        help="Number of triplets (query, pos, negs) in a batch. Each triplet consists of 12 images",
    )
    infer_batch_size: int = Field(
        16, help="Batch size for inference (caching and testing)"
    )
    criterion: str = Field(
        "triplet", help="Loss to be used", choices=["triplet", "sare_ind", "sare_joint"]
    )
    margin: float = Field(0.1, help="Margin for the triplet loss")
    epochs_num: int = Field(1000, help="Number of epochs to train for")
    patience: int = Field(3)
    lr: float = Field(
        0.00001,
        # 0.00001 * 4,
        help="_")
    lr_crn_layer: float = Field(5e-3, help="Learning rate for the CRN layer")
    lr_crn_net: float = Field(
        5e-4, help="Learning rate to finetune pretrained network when using CRN"
    )
    optim: str = Field("adam", help="_", choices=["adam", "sgd"])
    cache_refresh_rate: int = Field(
        1000, help="How often to refresh cache, in number of queries"
    )
    queries_per_epoch: int = Field(
        5000,
        help="How many queries to consider for one epoch. Must be multiple of cache_refresh_rate",
    )
    negs_num_per_query: int = Field(
        10, help="How many negatives to consider per each query in the loss"
    )
    neg_samples_num: int = Field(
        1000, help="How many negatives to use to compute the hardest ones"
    )
    mining: str = Field(
        "partial", choices=["partial", "full", "random", "msls_weighted"]
    )
    backbone: str = Field(
        # "resnet18conv4",
        "vit",
        choices=[
            "alexnet",
            "vgg16",
            "resnet18conv4",
            "resnet18conv5",
            "resnet50conv4",
            "resnet50conv5",
            "resnet101conv4",
            "resnet101conv5",
            "cct384",
            "vit",
        ],
        help="_",
    )
    l2: str = Field(
        "before_pool",
        choices=["before_pool", "after_pool", "none"],
        help="When (and if) to apply the l2 norm with shallow aggregation layers",
    )
    aggregation: str = Field(
        "netvlad",
        choices=[
            "netvlad",
            "gem",
            "spoc",
            "mac",
            "rmac",
            "crn",
            "rrm",
            "cls",
            "seqpool",
        ],
    )
    netvlad_clusters: int = Field(64, help="Number of clusters for NetVLAD layer.")
    pca_dim: int = Field(
        None,
        help="PCA dimension (number of principal components). If None, PCA is not used.",
    )
    fc_output_dim: int = Field(
        None,
        help="Output dimension of fully connected layer. If None, don't use a fully connected layer.",
    )
    pretrain: str = Field(
        # "imagenet",
        "dinov2",
        choices=["imagenet", "gldv2", "places", "dinov2"],
        help="Select the pretrained weights for the starting network",
    )
    off_the_shelf: str = Field(
        "imagenet",
        choices=["imagenet", "radenovic_sfm", "radenovic_gldv1", "naver"],
        help="Off-the-shelf networks from popular GitHub repos. Only with ResNet-50/101 + GeM + FC 2048",
    )
    trunc_te: int = Field(None, choices=list(range(0, 14)))
    freeze_te: int = Field(None, choices=list(range(-1, 14))) #TODO 我改了这个参数的意义，需要改下这个范围
    # freeze_te: int = Field(8, choices=list(range(-1, 14)))
    # peft: str = Field(None, choices=['lora'])
    peft: str = Field(
        # None,
        peft.PeftType.LORA.name,
        # peft.PeftType.GLORA.name,
        # peft.PeftType.OFT.name,
        # peft.PeftType.ADALORA.name,
        # peft.PeftType.IA3.name,
        # peft.PeftType.PREFIX_TUNING.name,
        choices=list(peft.PeftType.__members__.keys()))
    seed: int = Field(0)
    resume: str = Field(
        None, help="Path to load checkpoint from, for resuming training or testing."
    )
    device: str = Field("cuda", choices=["cuda", "cpu"])
    num_workers: int = Field(8, help="num_workers for all dataloaders")
    resize: List[int] = Field(
        # [480, 640], nargs=2, help="Resizing shape for images (HxW)."
        # [384, 384], nargs=2, help="Resizing shape for images (HxW)."
        [224, 224], nargs=2, help="Resizing shape for images (HxW)."
    )
    test_method: str = Field(
        "hard_resize",
        choices=[
            "hard_resize",
            "single_query",
            "central_crop",
            "five_crops",
            "nearest_crop",
            "maj_voting",
        ],
        help="This includes pre/post-processing methods and prediction refinement",
    )
    majority_weight: float = Field(
        0.01,
        help="Only for majority voting, scale factor, the higher it is the more importance is given to agreement",
    )
    efficient_ram_testing: bool = Field(False, help="_")
    val_positive_dist_threshold: int = Field(25, help="_")
    train_positives_dist_threshold: int = Field(10, help="_")
    recall_values: List[int] = Field(
        [1, 5, 10, 20], help="Recalls to be computed, such as R@5."
    )
    brightness: float = Field(0, help="_")
    contrast: float = Field(0, help="_")
    saturation: float = Field(0, help="_")
    hue: float = Field(0, help="_")
    rand_perspective: float = Field(0, help="_")
    horizontal_flip: bool = Field(False, help="_")
    random_resized_crop: float = Field(0, help="_")
    random_rotation: float = Field(0, help="_")

    datasets_folder: str = Field(
        # "../VPR-datasets-downloader/datasets", 
        (this_directory.parent/"VPR-datasets-downloader/datasets").as_posix(),
        help="Path with all datasets"
    )
    dataset_name: str = Field(
        "pitts30k", 
        # "pitts250k", 
        help="Relative path of the dataset")
    pca_dataset_folder: str = Field(
        None,
        help="Path with images to be used to compute PCA (ie: pitts30k/images/train",
    )
    save_dir: str = Field(
        "default", 
        help="Folder name of the current run (saved in ./logs/)"
    )
    # addition_experiment_notes:str = Field("big lora, rank 32.")
    addition_experiment_notes:str = Field("")


def parse_arguments()->VPRModel:
    arg_parser = VPRModel().create_parser()
    args_parsed = arg_parser.parse_args()
    args = VPRModel(**vars(args_parsed))
    return post_args_handle(args)

def post_args_handle(args: VPRModel)->VPRModel:

    if args.datasets_folder is None:
        try:
            args.datasets_folder = os.environ["DATASETS_FOLDER"]
        except KeyError:
            raise Exception(
                "You should set the parameter --datasets_folder or export "
                + "the DATASETS_FOLDER environment variable as such \n"
                + "export DATASETS_FOLDER=../datasets_vg/datasets"
            )

    if args.aggregation == "crn" and args.resume is None:
        raise ValueError(
            "CRN must be resumed from a trained NetVLAD checkpoint, but you set resume=None."
        )

    if args.queries_per_epoch % args.cache_refresh_rate != 0:
        raise ValueError(
            "Ensure that queries_per_epoch is divisible by cache_refresh_rate, "
            + f"because {args.queries_per_epoch} is not divisible by {args.cache_refresh_rate}"
        )

    if torch.cuda.device_count() >= 2 and args.criterion in ["sare_joint", "sare_ind"]:
        raise NotImplementedError(
            "SARE losses are not implemented for multiple GPUs, "
            + f"but you're using {torch.cuda.device_count()} GPUs and {args.criterion} loss."
        )

    if args.mining == "msls_weighted" and args.dataset_name != "msls":
        raise ValueError(
            "msls_weighted mining can only be applied to msls dataset, but you're using it on {args.dataset_name}"
        )

    if args.off_the_shelf in ["radenovic_sfm", "radenovic_gldv1", "naver"]:
        if (
            args.backbone not in ["resnet50conv5", "resnet101conv5"]
            or args.aggregation != "gem"
            or args.fc_output_dim != 2048
        ):
            raise ValueError(
                "Off-the-shelf models are trained only with ResNet-50/101 + GeM + FC 2048"
            )

    if args.pca_dim is not None and args.pca_dataset_folder is None:
        raise ValueError("Please specify --pca_dataset_folder when using pca")

    if args.backbone == "vit":
        if args.resize != [224, 224] and args.resize != [384, 384]:
            raise ValueError(
                f"Image size for ViT must be either 224 or 384 {args.resize}"
            )
    if args.backbone == "cct384":
        if args.resize != [384, 384]:
            raise ValueError(
                f"Image size for CCT384 must be 384, but it is {args.resize}"
            )

    if args.backbone in [
        "alexnet",
        "vgg16",
        "resnet18conv4",
        "resnet18conv5",
        "resnet50conv4",
        "resnet50conv5",
        "resnet101conv4",
        "resnet101conv5",
    ]:
        if args.aggregation in ["cls", "seqpool"]:
            raise ValueError(
                f"CNNs like {args.backbone} can't work with aggregation {args.aggregation}"
            )
    if args.backbone in ["cct384"]:
        if args.aggregation in ["spoc", "mac", "rmac", "crn", "rrm"]:
            raise ValueError(
                f"CCT can't work with aggregation {args.aggregation}. Please use one among [netvlad, gem, cls, seqpool]"
            )
    if args.backbone == "vit":
        if args.aggregation not in ["cls", "gem", "netvlad"]:
            raise ValueError(
                f"ViT can't work with aggregation {args.aggregation}. Please use one among [netvlad, gem, cls]"
            )

    return args

#%%
# 设计逻辑：上面我们自己写了 "distribution"字段，表示我们想要让wandb帮我们自动遍历这个参数，注释掉就表示固定下来这个参数。
# if __name__ == "__main__":
    # 生成 wandb sweep yaml
schema = VPRModel.schema()
#%%
# https://docs.wandb.ai/guides/sweeps/define-sweep-configuration
def handle_description_dict(name, description_dict)->dict:
    if 'distribution' not in description_dict: # 这里是如果跑不了，就用常数表示。
        # if 'choices' in description_dict:
        #     return dict(
        #         distribution="categorical",
        #         values=description_dict["choices"]
        #     )
        if 'default' in description_dict:
            return dict(
                distribution="constant", 
                value=description_dict["default"]
                )
        return dict(distribution="constant", 
                value=None)
        # raise ValueError(f"no distribution for {name}")
            
    new_dict = {}
    for k, v in description_dict.items():
        if k=="choices":
            new_dict['values'] = v
        if k in ["distribution", "min", "max"]:
            new_dict[k] = v
    return new_dict
sweep_dict = dict(
    program="sweep.py", 
    name=f"sweep_{schema['title']}", 
    # method="bayes",  # random
    method="grid", 
    metric=dict(
            name="test_R@1",
            # name="R@1",
            goal="maximize",
            target=100,
        ),
    parameters={
        name: handle_description_dict(name, description_dict)
        for name, description_dict in schema["properties"].items() 
        if name!="my_extra_fields" 
        # and 'distribution' in description_dict
    }
)
sweep_dict
#%%
# from yaml import safe_dump
# with open("sweep.yaml", "w") as f:
#     safe_dump(sweep_dict, f, sort_keys=False)
#%%
# import subprocess
# subprocess.check_output("wandb sweep --update handicraft-computing/vpr-benchmark/iyxdn0f6 sweep.yaml", shell=True)
# %%
# wandb agent handicraft-computing/vpr-benchmark/iyxdn0f6