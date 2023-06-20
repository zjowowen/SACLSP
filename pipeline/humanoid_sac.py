from SACLSP.config.humanoid_sac import cfg
from SACLSP.algorithmns.sac import SAC
from SACLSP.environment.env import Env
from SACLSP.utils.log import log
import wandb

def main():
    env = Env(cfg.env)
    sac = SAC(cfg.algo, env)
    sac.train()

if __name__ == '__main__':
    log.info("config: \n{}".format(cfg))
    wandb.init(project='new-humanoid-sac', config=cfg)
    main()
    