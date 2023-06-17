import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from SACLSP.value_networks.q_model import QModel
from SACLSP.policy.sac import SACPolicy
from SACLSP.environment.env import Env
from SACLSP.replay_buffer import ReplayBuffer
from SACLSP.utils.log import log
import easydict

import wandb

class SAC:

    def __init__(self, cfg:easydict, env:Env):
        self.cfg = cfg
        # if cuda is available, use cuda
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.policy = SACPolicy(cfg.policy)
        # TODO: 2 q networks
        # if cfg.twin_critic=True
        self.q = QModel(cfg.q_model)
        self.q_target = QModel(cfg.q_model)

        self.policy.to(self.device)
        self.q.to(self.device)
        self.q_target.to(self.device)

        self.env = env
        self.buffer = ReplayBuffer(cfg.replay_buffer)
        self.q_loss_fn = nn.MSELoss()

        
    def train(self):

        def preprocess(data):
            obs, action, reward, done, next_obs = data
            obs=obs.to(torch.float32).to(self.device)
            action=action.to(torch.float32).to(self.device)
            reward=reward.to(torch.float32).to(self.device)
            done=done.to(torch.float32).to(self.device)
            next_obs=next_obs.to(torch.float32).to(self.device)

            return obs, action, reward, done, next_obs
        
        def compute_q_loss(data):
            obs, action, reward, done, next_obs = data
            with torch.no_grad():
                next_action, next_logp = self.policy(next_obs)
                q_target = reward + self.cfg.train.gamma * (1.0 - done) * (self.q_target(next_obs, next_action).squeeze(1) - self.cfg.train.entropy_coeffi * next_logp)
                
            q_value=self.q(obs, action).squeeze(1)
            q_loss = self.q_loss_fn(q_value, q_target)

            return q_loss, q_value, q_target

        def compute_policy_loss(data):
            obs, action, reward, done, next_obs = data
            # TODO
            # sample more than 1 action 
            action, logp = self.policy(obs)
            q_value = self.q(obs, action).squeeze(1)
            policy_loss = (self.cfg.train.entropy_coeffi * logp - q_value).mean()
            return policy_loss
        
        optimizer_q = torch.optim.Adam(self.q.parameters(), lr=self.cfg.train.q_lr,weight_decay=self.cfg.train.weight_decay)
        optimizer_policy = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.train.policy_lr,weight_decay=self.cfg.train.weight_decay)
        self.q_target.load_state_dict(self.q.state_dict())
        env_step=0

        wandb.watch(models=self.policy, log="all", log_freq=100, idx=0, log_graph=True)
        wandb.watch(models=self.q, log="all", log_freq=100, idx=1, log_graph=True)

        while len(self.buffer.buffer)<self.cfg.train.random_collect_size:
            collected_data=self.env.collect(self.policy, self.cfg.train.num_episodes, device=self.device)
            self.buffer.add_experiences(collected_data)
            env_step+=len(collected_data)

        for train_iter in range(int(self.cfg.train.num_iters)):
            wandb_log = {}

            self.policy.eval()
            self.q.eval()
            self.q_target.eval()
            collected_data=self.env.collect(self.policy, self.cfg.train.num_episodes, device=self.device)
            self.buffer.add_experiences(collected_data)
            env_step+=len(collected_data)
            
            train_data=list(self.buffer.buffer)
            training_data_num=min(len(collected_data), len(train_data))
            ids=np.random.choice(np.array([i for i in range(len(self.buffer.buffer))]), size=training_data_num, replace=False)
            train_data=[train_data[i] for i in ids]
            train_dataloader=DataLoader(train_data, batch_size=self.cfg.train.batch_size, shuffle=True)

            self.policy.train()
            self.q.train()
            self.q_target.train()
            q_loss_sum=0
            q_value_sum=0
            q_value_abs_sum=0
            q_target_sum=0
            q_target_abs_sum=0
            q_grad_norm_sum=0
            policy_loss_sum=0
            policy_grad_norm_sum=0

            q_param_norm_sum=0
            policy_mu_model_param_norm_sum=0
            policy_cov_model_param_norm_sum=0

            for epoch in range(self.cfg.train.num_epochs):

                for batch_data in train_dataloader:
                    # calculate the parameter weight norm of q and policy
                    with torch.no_grad():
                        q_param_norm=0
                        for param in self.q.parameters():
                            q_param_norm+=torch.norm(param)
                        
                        policy_mu_model_param_norm=0
                        for param in self.policy.model.mu_model.parameters():
                            policy_mu_model_param_norm+=torch.norm(param)
                        
                        policy_cov_model_param_norm=0
                        for param in self.policy.model.cov.parameters():
                            policy_cov_model_param_norm+=torch.norm(param)
                        

                    batch_data=preprocess(batch_data)
                    q_loss, q_value, q_target=compute_q_loss(batch_data)
                    q_loss=q_loss*batch_data[0].shape[0]/self.cfg.train.batch_size
                    q_loss.backward()

                    q_grad_norm=torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.train.grad_clip)
                    # log.info(f"q_grad_norm: {q_grad_norm}")

                    optimizer_q.step()
                    optimizer_q.zero_grad()
                    optimizer_policy.zero_grad()
                    policy_loss=compute_policy_loss(batch_data)
                    policy_loss=policy_loss*batch_data[0].shape[0]/self.cfg.train.batch_size
                    policy_loss.backward()

                    policy_grad_norm=torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.train.grad_clip)
                    # log.info(f"policy_grad_norm: {policy_grad_norm}")

                    optimizer_policy.step()
                    optimizer_q.zero_grad()
                    optimizer_policy.zero_grad()

                    with torch.no_grad():
                        for q_target_param, q_param in zip(self.q_target.parameters(), self.q.parameters()):
                            q_target_param.data.mul_(self.cfg.train.q_target_parameter_decay)
                            q_target_param.data.add_((1-self.cfg.train.q_target_parameter_decay)*q_param.data)
                        
                        q_value_sum+=q_value.detach().sum().item()
                        q_value_abs_sum+=torch.abs(q_value.detach()).sum().item()
                        q_target_sum+=q_target.detach().sum().item()
                        q_target_abs_sum+=torch.abs(q_target.detach()).sum().item()
                        q_loss_sum+=q_loss.detach().item()*batch_data[0].shape[0]
                        q_grad_norm_sum+=q_grad_norm.detach().item()*batch_data[0].shape[0]
                        policy_loss_sum+=policy_loss.detach().item()*batch_data[0].shape[0]
                        policy_grad_norm_sum+=policy_grad_norm.detach().item()*batch_data[0].shape[0]
                        q_param_norm_sum+=q_param_norm.detach().item()*batch_data[0].shape[0]
                        policy_mu_model_param_norm_sum+=policy_mu_model_param_norm.detach().item()*batch_data[0].shape[0]
                        policy_cov_model_param_norm_sum+=policy_cov_model_param_norm.detach().item()*batch_data[0].shape[0]

            q_loss_mean=q_loss_sum/len(train_data)
            q_value_mean=q_value_sum/len(train_data)
            q_value_abs_mean=q_value_abs_sum/len(train_data)
            q_target_mean=q_target_sum/len(train_data)
            q_target_abs_mean=q_target_abs_sum/len(train_data)
            q_grad_norm_mean=q_grad_norm_sum/len(train_data)
            policy_loss_mean=policy_loss_sum/len(train_data)
            policy_grad_norm_mean=policy_grad_norm_sum/len(train_data)
            q_param_norm_mean=q_param_norm_sum/len(train_data)
            policy_mu_model_param_norm_mean=policy_mu_model_param_norm_sum/len(train_data)
            policy_cov_model_param_norm_mean=policy_cov_model_param_norm_sum/len(train_data)
            wandb_log.update({
                'q_loss':q_loss_mean, 
                'q_value':q_value_mean,
                'q_value_abs':q_value_abs_mean,
                'q_target':q_target_mean,
                'q_target_abs':q_target_abs_mean, 
                'q_grad_norm':q_grad_norm_mean,
                'policy_loss':policy_loss_mean,
                'policy_grad_norm':policy_grad_norm_mean,
                'env_step':env_step,
                'q_param_norm':q_param_norm_mean,
                'policy_mu_model_param_norm':policy_mu_model_param_norm_mean,
                'policy_cov_model_param_norm':policy_cov_model_param_norm_mean,
                })
            
            if train_iter % self.cfg.train.eval_freq == 0:
                return_ = self.eval()
                wandb_log.update({'return':return_})
                log.info("train_iter: [{}], env_step: [{}], policy_loss: {}, q_loss: {}, return: {}".format(train_iter, env_step, policy_loss_mean, q_loss_mean, return_))
            else:
                log.info("train_iter: [{}], env_step: [{}], policy_loss: {}, q_loss: {}".format(train_iter, env_step, policy_loss_mean, q_loss_mean))

            wandb.log(wandb_log, step=train_iter)
            
    def eval(self):
        self.policy.eval()
        return self.env.evaluate(self.policy, device=self.device)
        
