import train
from parser import VPRModel, post_args_handle, sweep_dict
def sweep_agent(current_run_config):
    # Get hyp dict from sweep agent. Copy because train() modifies parameters which confused wandb.
    # params = vars(wandb.config).get("_items").copy()
    args = VPRModel(current_run_config) # 校验参数合法性； wandb.config不用全部些，只用写自己想要调节的参数就行
    args = post_args_handle(args)
    train.main(args)
    
def sweep():
    pass
if __name__ == "__main__":
    sweep()

