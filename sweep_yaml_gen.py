#%%
from parser import *
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