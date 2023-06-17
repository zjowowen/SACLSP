import gym
import torch

class Env:
    def __init__(self, cfg):
        self.cfg = cfg
        self.env = gym.make(cfg.name)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def collect(self, policy, num_episodes, device='cpu'):
        data_list = []
        with torch.no_grad():
            for i in range(num_episodes):
                obs = self.env.reset()
                done = False
                while not done:
                    action, log_prob = policy(torch.tensor(obs).unsqueeze(0).to(torch.float32).to(device))
                    action = action.squeeze(0).cpu().detach().numpy()
                    next_obs, reward, done, _ = self.env.step(action)
                    data_list.append((obs, action, reward, done, next_obs))
                    obs = next_obs
        return data_list
        
    def evaluate(self, policy, device='cpu'):
        obs = self.env.reset()
        done = False
        return_ = 0
        with torch.no_grad():
            while not done:
                action, log_prob= policy(torch.tensor(obs).unsqueeze(0).to(torch.float32).to(device))
                action = action.squeeze(0).cpu().detach().numpy()
                next_obs, reward, done, _ = self.env.step(action)
                return_ += reward
                obs = next_obs
        return return_


if __name__ == "__main__":
    cfg=dict(
        name='Hopper-v3',
    )
    from easydict import EasyDict
    cfg = EasyDict(cfg)
    env = Env(cfg)
    print(env.observation_space)
    print(env.action_space)
    print(env.collect(lambda x: env.action_space.sample(), 1))
    print(env.evaluate(lambda x: env.action_space.sample()))
