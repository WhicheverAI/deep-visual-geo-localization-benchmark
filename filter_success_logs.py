#%%
from pathlib import Path
from typing import List
import warnings
this_file = Path(__file__).resolve()
this_directory = this_file.parent
#%%
logs = this_directory / "logs"
# success_logs = logs / "success"
# default_logs = logs / "default"
success_logs = logs / "success"
default_logs = logs / "success"
default_logs.mkdir(exist_ok=True, parents=True)
success_logs.mkdir(exist_ok=True, parents=True)
#%%
new_success_logs = []
for subfolder in default_logs.iterdir():
    # print(item)
    if not subfolder.is_dir():
        continue
    if (subfolder/"best_model.pth"
        ).exists() and (subfolder/"last_model.pth"
        ).exists():
        new_success_logs.append(subfolder)
    elif len(list(subfolder.glob("*.pth")))>=1:
        print("need to check ", subfolder)
new_success_logs
#%%
for subfolder in new_success_logs:
    subfolder.rename(success_logs/subfolder.name)
            
#%%
from pydantic import BaseModel, Field
from parser import VPRModel
import datetime 
class ExperimentModel(VPRModel):
    # VPRModel.recall_values 
    experiment_time: datetime.datetime = Field(default=datetime.datetime.now(),
                                               help="The time at which the experiment was launched. "
                                               )
    val_recalls: List[float] = Field([0, 0, 0, 0], 
                                     title="Validation recalls",
                                     description="R@", 
                                     ge=0, le=100
                                     )
    test_recalls: List[float] = Field([0, 0, 0, 0], 
                                      title="Test recalls",
                                      description="test_R@", 
                                      ge=0, le=100)
import tomli
import re
def text_to_experiment(text:str):
    lines = text.splitlines(keepends=False)
    met_args = False
    met_test = False
    best_r5 = 0
    best_val_r_list = []
    def get_r_list_from_line(line:str):
        pattern = r"R@\d+:\s*([\d.]+)"
        matches = re.findall(pattern, line)
        return [float(match) for match in matches]
    # test_r_list = []
    for line in lines:
        if "Arguments: " in line:
            if met_args: raise TypeError()
            met_args = True
            splits = line.split("Arguments:")
            date_time_str = splits[0].strip()
            date_time_object = datetime.datetime.strptime(date_time_str, "%Y-%m-%d %H:%M:%S")
            
            arg_str = splits[1].strip()
            # arguments = arguments.split(" ")
            # arguments = [arg for arg in arguments if "None" not in arg]

            pattern = r"(\w+)\s*=\s*(\[[^\]]*\]|'[^']*'|[^\s,]+)"
            matches = re.findall(pattern, arg_str)
            # arguments = [f"{key}={value}" for key, value in matches 
            #              if value!="None"]
            vpr_dict = {}
            for key, value in matches:
                vpr_dict[key.strip()] = eval(value.strip())  # 使用 eval() 解析字符串值
            
            # toml 不支持 None，这是个严重的缺陷
            # toml_str = "\n".join(arguments
            #                      ).replace("True", "true").replace("False", "false")
            # # print(arguments[49])
            # toml_dict = tomli.loads(toml_str)
            # print(toml_dict)
        elif "Recalls on val set" in line:
            val_r_list = get_r_list_from_line(line)
            if val_r_list[1] > best_r5: # 与训练代码一致，不是>=
                best_val_r_list = val_r_list
                best_r5 = val_r_list[1]
        elif "Recalls on" in line: # else，所以排除上面的情况了
            if met_test: raise TypeError()
            met_test = True
            test_r_list = get_r_list_from_line(line)
            
    if not met_test:
        test_r_list = [0, 0, 0, 0] 
        warnings.warn("Training Not Completed!")
    # return ExperimentModel(experiment_time=date_time_object, 
    return dict(experiment_time=date_time_object, 
                           val_recalls=best_val_r_list, 
                           test_recalls=test_r_list,
                           **vpr_dict)
    
not_interested = ["datasets_folder", "my_extra_fields",
                 "no_wandb", "infer_batch_size", "save_dir", 
                 "pca_dataset_folder", "addition_experiment_notes", 
                 "device", "num_workers", "seed", "recall_values"
                 
                 ]+["val_recalls", "test_recalls"]
experiments = []
for subfolder in success_logs.iterdir():
    if not subfolder.is_dir(): continue
    with open(subfolder/"debug.log", "r") as f:
        contents = f.read()
    experiment = text_to_experiment(contents)
    # exp_dict = experiment.__dict__
    exp_dict = experiment
    for i, r in enumerate(experiment['recall_values']):
        exp_dict[f'val_R@{r}'] = experiment['val_recalls'][i]
    for i, r in enumerate(experiment['recall_values']):
        exp_dict[f'R@{r}'] = experiment['test_recalls'][i]
    for ni in not_interested:
        if ni in exp_dict:
            del exp_dict[ni]
    experiments.append(exp_dict)
    # print(experiment)
    # break
#%%
from pandas import DataFrame
df = DataFrame(experiments).set_index("experiment_time")
df
#%%
# interested_columns = ["dataset_name", 'peft', "R@"]
interested_columns = ["dataset_name", 'peft', "val_R@5", "R@5", "freeze_te", "backbone"]
from delta_residual import matching_strategy
other_columns = [col for col in df.columns if 
                 col not in interested_columns
                # not matching_strategy.is_contains(col, col, interested_columns)
                 ]
interested_columns = [col for col in df.columns if 
                 col not in other_columns]
df = df[interested_columns + other_columns]
df

#%%
df.to_csv("logs/exp_summary.csv")
#%%
# 注意msls，验证集和测试集应该是一样的，0表示没跑完
# 注意pitts30k，验证集应该和测试集不一样，0也是没跑完。
#%%
# 2/24 pit数据集处理好了
# 2024-03-01 17:45:05 的效果不用算进来
# 3.2之前的R@1的效果低，VIT和CNN聚合方法不一样
# 师兄 3.2 +adapter 微调跑1个epoch val pit30
# R@1: 93.3, R@5: 98.6, R@10: 99.4, R@20: 99.7

# cricavpr pit250 97%


#%%
# from autogluon 
# 用机器学习处理数据发现规律，很合理