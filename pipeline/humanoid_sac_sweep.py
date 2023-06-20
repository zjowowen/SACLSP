from SACLSP.config.humanoid_sac import cfg
from SACLSP.algorithmns.sac import SAC
from SACLSP.environment.env import Env
from SACLSP.utils.log import log
from SACLSP.utils.config_utils import merge_dicts
import wandb

def main():
    wandb.init()

    new_cfg=merge_dicts(
        {
        "algo":{
            "train":{
                "gamma":wandb.config.gamma,
                "target_entropy":wandb.config.target_entropy,
                },
            },
        }, cfg
    )

    log.info("config: \n{}".format(new_cfg))

    env = Env(new_cfg.env)
    sac = SAC(new_cfg.algo, env)
    eval_value=sac.train()

    wandb.log({"eval_value": eval_value,})
    

if __name__ == '__main__':

    # Define the search space
    sweep_configuration = {
        'method': 'grid',
        'metric': 
        {
            'goal': 'maximize', 
            'name': 'eval_value'
            },
        'parameters': 
        {
            'gamma': {'values': [0.98, 0.99, 0.993, 0.995]},
            'target_entropy': {'values': [-10, -15, -17, -20, -24, -30, -50]},
        }
    }

    # Start the sweep
    sweep_id = wandb.sweep(
        sweep=sweep_configuration, 
        project='new-humanoid-sac-sweep'
        )
    
    wandb.agent(sweep_id, function=main)

