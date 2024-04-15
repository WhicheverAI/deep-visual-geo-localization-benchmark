#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#%%
from bigmodelvis import Visualization
visualize = lambda x:Visualization(x).structure_graph()
from transformers import AutoImageProcessor, AutoModel
backbone = AutoModel.from_pretrained('facebook/dinov2-small')
visualize(backbone) # 默认的print
#%%
from transformers import AutoImageProcessor, AutoModel
backbone = AutoModel.from_pretrained('facebook/dinov2-base')
backbone # 默认的print
visualize(backbone)
#%%
# type(backbone)
# import transformers
# isinstance(backbone, transformers.models.dinov2.modeling_dinov2.Dinov2Model)
# list(filter(lambda x:not x.startswith("_"), dir(backbone)))
# # backbone.__dict__
# # %%
# # backbone.summary() # 没有summary

# backbone.config # hf config
# # %%
# for name, child in backbone.encoder.layer.named_children():
#     print("name:", name)
#     print("child:", child)

# %%

import sys
from pathlib import Path
this_file = Path(__file__).resolve()
this_directory = this_file.parent
print(this_directory)
sys.path.append((this_directory.parent).as_posix())
from backbone.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
backbone = vit_base(patch_size=14,
                    img_size=518,
                    init_values=1,block_chunks=0
                    )  
foundation_model_path = (this_directory.parent/
                         "backbone/dinov2_vitb14_pretrain.pth" # 官网的
                        #  "backbone/dinov2_vitb14_pretrain1.pth" # cricavpr用的
                         ).as_posix()
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
# %%
backbone = backbone.cuda()
# 啊，为什么输入518也行，224也行，都能正确embed到吗？
# data = torch.randn(1, 3, 224, 224).cuda()
data = torch.randn(1, 3, 518, 518).cuda()
res = backbone(data)
print(list(res.keys()))
# print(res['last_hidden_state'][:, 1:, :].shape)
# print(res['last_hidden_state'].shape)
print(res['x_norm_patchtokens'].shape)
print(res['x_norm_clstoken'].shape)

# for k, v in res.items():
#     if isinstance(k, torch.Tensor):
#         print(k, v.shape)
# %%
from bigmodelvis import Visualization
Visualization(backbone).structure_graph()