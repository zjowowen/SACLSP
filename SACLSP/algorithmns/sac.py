import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from SACLSP.value_networks.q_model import QModel
from SACLSP.policy.sac import SACPolicy
from SACLSP.environment.env import Env
from SACLSP.replay_buffer import ReplayBuffer
from SACLSP.utils.log import log
from SACLSP.utils.gym_utils import heuristic_target_entropy
from SACLSP.models.common.parameter import NonegativeParameter
import easydict

import wandb

class SAC:

    def __init__(self, cfg:easydict, env:Env):
        self.cfg = cfg
        # if cuda is available, use cuda
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.policy = SACPolicy(cfg.policy)
        self.q = QModel(cfg.q_model)
        self.q_target = QModel(cfg.q_model)
        if cfg.train.train_entropy_coeffi:
            self.entropy_coeffi = NonegativeParameter(torch.tensor(cfg.train.entropy_coeffi))
            self.target_entropy = cfg.train.target_entropy if hasattr(cfg.train,'target_entropy') \
                else heuristic_target_entropy(env.action_space) * cfg.train.relative_target_entropy_scale if hasattr(cfg.train,'relative_target_entropy_scale') \
                    else heuristic_target_entropy(env.action_space)
        else:
            self.entropy_coeffi = NonegativeParameter(torch.tensor(cfg.train.entropy_coeffi), requires_grad=False)

        self.policy.to(self.device)
        self.q.to(self.device)
        self.q_target.to(self.device)
        self.entropy_coeffi.to(self.device)

        self.env = env
        self.buffer = ReplayBuffer(cfg.replay_buffer)
        self.q_loss_fn = nn.MSELoss()

        
    def train(self):

        def collect_data(num_steps=None, num_episodes=None):
            if num_steps is not None:
                collected_data=self.env.collect(self.policy, num_steps=num_steps, device=self.device)
            elif num_episodes is not None:
                collected_data=self.env.collect(self.policy, num_episodes=num_episodes, device=self.device)
            else:
                if hasattr(self.cfg.train, 'num_episodes_collected') and self.cfg.train.num_episodes_collected>0:
                    collected_data=self.env.collect(self.policy, num_episodes=self.cfg.train.num_episodes_collected, device=self.device)
                elif hasattr(self.cfg.train, 'num_steps_collected') and self.cfg.train.num_steps_collected>0:
                    collected_data=self.env.collect(self.policy, num_steps=self.cfg.train.num_steps_collected, device=self.device)
            return collected_data

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
                q_target = reward + self.cfg.train.gamma * (1.0 - done) * (self.q_target.min_q(next_obs, next_action) - self.entropy_coeffi.data * next_logp)
                q_target_repeat = q_target.unsqueeze(-1).repeat(1, self.q.model_num)
            
            q_value=self.q(obs, action)
            q_loss = self.q_loss_fn(q_value, q_target_repeat)

            return q_loss, q_value, q_target

        def compute_policy_loss(data):
            obs, action, reward, done, next_obs = data
            # TODO
            # sample more than 1 action 
            action, logp = self.policy(obs)
            q_value = self.q.min_q(obs, action)
            policy_loss = (self.entropy_coeffi.data * logp - q_value).mean()
            return policy_loss, logp
        
        def compute_entropy_coeffi_loss(data):
            obs, action, reward, done, next_obs = data
            with torch.no_grad():
                action, logp = self.policy(obs)
                average_action_entropy = -torch.mean(logp)
            entropy_coeffi_loss = self.entropy_coeffi.data * (average_action_entropy - self.target_entropy)

            return entropy_coeffi_loss, average_action_entropy

        optimizer_q = torch.optim.Adam(self.q.parameters(), lr=self.cfg.train.q_lr,weight_decay=self.cfg.train.weight_decay)
        optimizer_policy = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.train.policy_lr,weight_decay=self.cfg.train.weight_decay)
        if self.cfg.train.train_entropy_coeffi:
            optimizer_entropy_coeffi = torch.optim.Adam(self.entropy_coeffi.parameters(), lr=self.cfg.train.entropy_coeffi_lr,weight_decay=self.cfg.train.weight_decay)

        self.q_target.load_state_dict(self.q.state_dict())
        env_step=0

        wandb.watch(models=self.policy, log="all", log_freq=100, idx=0, log_graph=True)
        wandb.watch(models=self.q, log="all", log_freq=100, idx=1, log_graph=True)
        wandb.watch(models=self.entropy_coeffi, log="all", log_freq=100, idx=2, log_graph=True)

        collected_data=collect_data(num_steps=self.cfg.train.random_collect_size)
        self.buffer.add_experiences(collected_data)
        env_step+=len(collected_data)

        for train_iter in range(int(self.cfg.train.num_iters)):
            wandb_log = {}

            self.policy.eval()
            self.q.eval()
            self.q_target.eval()
            collected_data=collect_data()
            self.buffer.add_experiences(collected_data)
            env_step+=len(collected_data)

            train_data=list(self.buffer.buffer)
            training_data_num=min(self.cfg.train.num_steps_training, len(train_data))

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

            if self.cfg.train.train_entropy_coeffi:
                entropy_coeffi_loss_sum=0
            average_action_entropy_sum=0

            for epoch in range(self.cfg.train.num_epochs):
                ids=np.random.choice(np.array([i for i in range(len(self.buffer.buffer))]), size=training_data_num, replace=False)
                train_data=[train_data[i] for i in ids]
                train_dataloader=DataLoader(train_data, batch_size=self.cfg.train.batch_size, shuffle=True)

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
                    optimizer_q.zero_grad()
                    q_loss.backward()
                    q_grad_norm=torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.train.q_grad_clip*self.q.model_num)
                    optimizer_q.step()
                    optimizer_q.zero_grad()
                    

                    policy_loss, logp=compute_policy_loss(batch_data)
                    policy_loss=policy_loss*batch_data[0].shape[0]/self.cfg.train.batch_size
                    optimizer_policy.zero_grad()
                    policy_loss.backward()
                    policy_grad_norm=torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.train.policy_grad_clip)
                    optimizer_policy.step()
                    optimizer_policy.zero_grad()

                    if self.cfg.train.train_entropy_coeffi:
                        entropy_coeffi_loss, average_action_entropy=compute_entropy_coeffi_loss(batch_data)
                        optimizer_entropy_coeffi.zero_grad()
                        entropy_coeffi_loss.backward()
                        optimizer_entropy_coeffi.step()
                        optimizer_entropy_coeffi.zero_grad()
                    else:
                        with torch.no_grad():
                            average_action_entropy = -torch.mean(logp)
                    
                        
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
                        if self.cfg.train.train_entropy_coeffi:
                            entropy_coeffi_loss_sum+=entropy_coeffi_loss.detach().item()*batch_data[0].shape[0]
                        average_action_entropy_sum+=average_action_entropy.detach().item()*batch_data[0].shape[0]

            q_loss_mean=q_loss_sum/len(train_data)/self.q.model_num
            q_value_mean=q_value_sum/len(train_data)/self.q.model_num
            q_value_abs_mean=q_value_abs_sum/len(train_data)/self.q.model_num
            q_target_mean=q_target_sum/len(train_data)
            q_target_abs_mean=q_target_abs_sum/len(train_data)
            q_grad_norm_mean=q_grad_norm_sum/len(train_data)/self.q.model_num
            policy_loss_mean=policy_loss_sum/len(train_data)
            policy_grad_norm_mean=policy_grad_norm_sum/len(train_data)
            q_param_norm_mean=q_param_norm_sum/len(train_data)
            policy_mu_model_param_norm_mean=policy_mu_model_param_norm_sum/len(train_data)
            policy_cov_model_param_norm_mean=policy_cov_model_param_norm_sum/len(train_data)
            if self.cfg.train.train_entropy_coeffi:
                entropy_coeffi_loss_mean=entropy_coeffi_loss_sum/len(train_data)
            average_action_entropy_mean=average_action_entropy_sum/len(train_data)
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
            if self.cfg.train.train_entropy_coeffi:
                wandb_log.update({'entropy_coeffi_loss':entropy_coeffi_loss_mean})
            wandb_log.update({
                'average_action_entropy':average_action_entropy_mean,
                'entropy_coeffi':self.entropy_coeffi.data.detach().item(),
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
        
