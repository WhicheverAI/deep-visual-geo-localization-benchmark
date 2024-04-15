import wandb
import train
from parser import VPRModel, post_args_handle
def sweep():
    wandb.init()
    # Get hyp dict from sweep agent. Copy because train() modifies parameters which confused wandb.
    # params = vars(wandb.config).get("_items").copy()
    args = VPRModel(**wandb.config) # 校验参数合法性； wandb.config不用全部些，只用写自己想要调节的参数就行
    args = post_args_handle(args)
    train.main(args)

if __name__ == "__main__":
    sweep()

