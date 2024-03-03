#%%
from transformers import AutoImageProcessor, AutoModel
backbone = AutoModel.from_pretrained('facebook/dinov2-base')
backbone # 默认的print
#%%
type(backbone)
import transformers
isinstance(backbone, transformers.models.dinov2.modeling_dinov2.Dinov2Model)
list(filter(lambda x:not x.startswith("_"), dir(backbone)))
# backbone.__dict__
# %%
# backbone.summary() # 没有summary

backbone.config # hf config
# %%
for name, child in backbone.encoder.layer.named_children():
    print("name:", name)
    print("child:", child)

# %%
