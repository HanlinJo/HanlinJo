import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from collections import deque
from torch.distributions import Normal
from env_v2x_attacker import V2XRLEnvironment  # 使用新的环境文件
from torch.distributions import Categorical
import logging
import pandas as pd
import argparse

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - V2X-RL-Attacker - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("attacker_train.log", mode='a')
    ]
)
logger = logging.getLogger('V2X-RL-Attacker')

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.states_ = []
        self.actions_probs = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.states_[:]
        del self.actions_probs[:]

class CombinedActor(nn.Module):
    """组合动作网络 - 输出单一动作（时隙+子信道组合）"""
    def __init__(self, state_dim, action_dim):
        super(CombinedActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc_action = nn.Linear(128, action_dim)  # 组合动作输出
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        action_probs = F.softmax(self.fc_action(x), dim=-1)
        return action_probs

class SeparatedActor(nn.Module):
    """分离动作网络 - 分别输出时隙和子信道选择"""
    def __init__(self, state_dim, num_slots, num_subchannels):
        super(SeparatedActor, self).__init__()
        # 共享特征提取层
        self.shared_fc1 = nn.Linear(state_dim, 512)
        self.shared_fc2 = nn.Linear(512, 256)
        
        # 时隙选择头
        self.slot_fc1 = nn.Linear(256, 128)
        self.slot_fc2 = nn.Linear(128, num_slots)
        
        # 子信道选择头
        self.subchannel_fc1 = nn.Linear(256, 128)
        self.subchannel_fc2 = nn.Linear(128, num_subchannels)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # 共享特征
        shared = F.relu(self.shared_fc1(x))
        shared = self.dropout(shared)
        shared = F.relu(self.shared_fc2(x))
        shared = self.dropout(shared)
        
        # 时隙概率
        slot_features = F.relu(self.slot_fc1(shared))
        slot_probs = F.softmax(self.slot_fc2(slot_features), dim=-1)
        
        # 子信道概率
        subchannel_features = F.relu(self.subchannel_fc1(shared))
        subchannel_probs = F.softmax(self.subchannel_fc2(subchannel_features), dim=-1)
        
        return slot_probs, subchannel_probs

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class AttackerActorCritic(nn.Module):
    def __init__(self, state_dim, num_slots, num_subchannels, action_mode='combined'):
        super(AttackerActorCritic, self).__init__()
        self.action_mode = action_mode
        self.num_slots = num_slots
        self.num_subchannels = num_subchannels
        
        if action_mode == 'combined':
            action_dim = num_slots * num_subchannels
            self.actor = CombinedActor(state_dim, action_dim).to(device)
        else:  # separated
            self.actor = SeparatedActor(state_dim, num_slots, num_subchannels).to(device)
        
        self.critic = Critic(state_dim).to(device)
    
    def act(self, state):
        if self.action_mode == 'combined':
            return self._act_combined(state)
        else:
            return self._act_separated(state)
    
    def _act_combined(self, state):
        """组合动作模式"""
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)
        
        return action.detach(), action_logprob.detach(), state_val.detach(), action_probs
    
    def _act_separated(self, state):
        """分离动作模式"""
        slot_probs, subchannel_probs = self.actor(state)
        
        # 创建分布
        slot_dist = Categorical(slot_probs)
        subchannel_dist = Categorical(subchannel_probs)
        
        # 采样动作
        slot_action = slot_dist.sample()
        subchannel_action = subchannel_dist.sample()
        
        # 计算联合对数概率
        slot_logprob = slot_dist.log_prob(slot_action)
        subchannel_logprob = subchannel_dist.log_prob(subchannel_action)
        joint_logprob = slot_logprob + subchannel_logprob
        
        # 组合动作
        combined_action = slot_action * self.num_subchannels + subchannel_action
        
        # 获取状态值
        state_val = self.critic(state)
        
        return (combined_action.detach(), joint_logprob.detach(), state_val.detach(), 
                (slot_probs, subchannel_probs))
    
    def evaluate(self, state, action):
        if self.action_mode == 'combined':
            return self._evaluate_combined(state, action)
        else:
            return self._evaluate_separated(state, action)
    
    def _evaluate_combined(self, state, action):
        """组合动作评估"""
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action.squeeze())
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy
    
    def _evaluate_separated(self, state, action):
        """分离动作评估"""
        slot_probs, subchannel_probs = self.actor(state)
        
        # 将组合动作分解为时隙和子信道
        slot_actions = action.squeeze() // self.num_subchannels
        subchannel_actions = action.squeeze() % self.num_subchannels
        
        # 创建分布
        slot_dist = Categorical(slot_probs)
        subchannel_dist = Categorical(subchannel_probs)
        
        # 计算对数概率
        slot_logprobs = slot_dist.log_prob(slot_actions)
        subchannel_logprobs = subchannel_dist.log_prob(subchannel_actions)
        joint_logprobs = slot_logprobs + subchannel_logprobs
        
        # 计算熵
        slot_entropy = slot_dist.entropy()
        subchannel_entropy = subchannel_dist.entropy()
        joint_entropy = slot_entropy + subchannel_entropy
        
        # 获取状态值
        state_values = self.critic(state)
        
        return joint_logprobs, state_values, joint_entropy

class AttackerPPO:
    def __init__(self, state_dim, num_slots, num_subchannels, action_mode, 
                 actor_lr, critic_lr, gamma, betas, K_epochs, eps_clip):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = RolloutBuffer()
        self.action_mode = action_mode
        
        # 策略网络
        self.policy = AttackerActorCritic(state_dim, num_slots, num_subchannels, action_mode).to(device)
        self.policy_old = AttackerActorCritic(state_dim, num_slots, num_subchannels, action_mode).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 优化器
        self.optimizer_actor = torch.optim.Adam(self.policy.actor.parameters(), lr=actor_lr, betas=betas)
        self.optimizer_critic = torch.optim.Adam(self.policy.critic.parameters(), lr=critic_lr, betas=betas)
        
        # 损失函数
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val, action_probs = self.policy_old.act(state_tensor)
            
            # 保存到缓冲区
            self.buffer.states.append(state_tensor)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)
            self.buffer.actions_probs.append(action_probs)
            
            return action.cpu().numpy()
    
    def update(self):
        # 转换列表为张量
        old_states = torch.stack(self.buffer.states).to(device).detach()
        old_actions = torch.stack(self.buffer.actions).to(device).detach()
        old_logprobs = torch.stack(self.buffer.logprobs).to(device).detach()
        old_state_values = torch.stack(self.buffer.state_values).to(device).detach()
        
        # 计算折扣奖励
        rewards = []
        discounted_reward = 0
        
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # 归一化奖励
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # 计算优势函数
        advantages = rewards.detach() - old_state_values.squeeze(1).detach()
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        
        # 优化策略K个epochs
        total_actor_loss = 0
        total_critic_loss = 0
        
        for epoch in range(self.K_epochs):
            # 评估旧状态和动作
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            
            # 计算比率
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # 代理损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # 演员损失（包括熵正则化）
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * dist_entropy.mean()
            
            # 评论家损失
            critic_loss = self.MseLoss(state_values, rewards.detach())
            
            # 梯度清零
            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            
            # 反向传播
            actor_loss.backward()
            critic_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), 0.5)
            
            # 优化步骤
            self.optimizer_actor.step()
            self.optimizer_critic.step()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
        
        # 复制新权重到旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 清空缓冲区
        self.buffer.clear()
        
        return total_actor_loss / self.K_epochs, total_critic_loss / self.K_epochs
    
    def save(self, checkpoint_path):
        torch.save({
            'policy_state_dict': self.policy_old.state_dict(),
            'action_mode': self.action_mode
        }, checkpoint_path)
    
    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.policy_old.load_state_dict(checkpoint['policy_state_dict'])
        self.policy.load_state_dict(checkpoint['policy_state_dict'])

def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='V2X RL Attacker Training')
    parser.add_argument('--action-mode', type=str, default='combined', choices=['combined', 'separated'],
                        help='Action selection mode: combined or separated')
    parser.add_argument('--num-vehicles', type=int, default=10,
                        help='Number of vehicles in the environment')
    parser.add_argument('--num-attackers', type=int, default=1,
                        help='Number of attackers in the environment')
    parser.add_argument('--episode-duration', type=int, default=20000,
                        help='Duration of each episode in ms')
    parser.add_argument('--max-episodes', type=int, default=5000,
                        help='Maximum number of training episodes')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Interval for logging training progress')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='Interval for saving models')
    parser.add_argument('--communication-range', type=int, default=320,
                        help='Communication range in meters')
    parser.add_argument('--render', action='store_true',
                        help='Enable environment rendering')
    args = parser.parse_args()
    
    # 初始化环境
    env = V2XRLEnvironment(
        num_vehicles=args.num_vehicles,
        num_attackers=args.num_attackers,
        episode_duration=args.episode_duration,
        communication_range=args.communication_range,
        action_mode=args.action_mode,
        render_mode='human' if args.render else None
    )
    
    # 获取状态和动作空间维度
    state_dim = env.observation_space.shape[0]
    num_slots = env.num_slots
    num_subchannels = env.num_subchannels
    
    # 训练参数
    update_timestep = 2048    # 更新策略的时间步间隔
    actor_lr = 0.0003         # 演员学习率
    critic_lr = 0.001         # 评论家学习率
    gamma = 0.99              # 折扣因子
    betas = (0.9, 0.999)      # Adam优化器参数
    K_epochs = 10             # 更新策略的epoch数
    eps_clip = 0.2            # PPO裁剪参数
    random_seed = 42          # 随机种子
    
    # 设置随机种子
    if random_seed:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    # 初始化PPO攻击者
    ppo_attacker = AttackerPPO(
        state_dim=state_dim,
        num_slots=num_slots,
        num_subchannels=num_subchannels,
        action_mode=args.action_mode,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        gamma=gamma,
        betas=betas,
        K_epochs=K_epochs,
        eps_clip=eps_clip
    )
    
    logger.info(f"Using RL Attacker with {args.action_mode} action mode")
    logger.info(f"State dimension: {state_dim}, Slots: {num_slots}, Subchannels: {num_subchannels}")
    
    # 训练统计
    time_step = 0
    all_rewards = []
    avg_rewards = []
    transmission_failures = []
    attack_success_rates = []
    prr_values = []
    actor_losses = []
    critic_losses = []
    best_avg_reward = -float('inf')
    
    # 模型保存目录
    save_dir = f"attacker_models_{args.action_mode}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 训练循环
    start_time = time.time()
    
    for episode in range(args.max_episodes):
        state = env.reset()
        episode_reward = 0
        episode_failures = 0
        done = False
        current_prr = 0
        current_attack_success = 0
        
        while not done:
            time_step += 1
            
            # 选择攻击动作
            action = ppo_attacker.select_action(state)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 更新统计
            episode_reward += reward
            episode_failures += info.get('transmission_failures', 0)
            current_prr = info.get('prr', current_prr)
            current_attack_success = info.get('attack_success_rate', current_attack_success)
            
            # 保存到缓冲区
            ppo_attacker.buffer.rewards.append(reward)
            ppo_attacker.buffer.is_terminals.append(done)
            
            # 更新状态
            state = next_state
            
            # 更新策略
            if time_step % update_timestep == 0:
                actor_loss, critic_loss = ppo_attacker.update()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                time_step = 0
        
        # 获取回合统计
        stats = env.get_episode_stats()
        vehicle_prrs = stats['vehicle_prrs']
        avg_vehicle_prr = np.mean(list(vehicle_prrs.values())) if vehicle_prrs else 0
        
        # 记录统计数据
        all_rewards.append(episode_reward)
        transmission_failures.append(episode_failures)
        attack_success_rates.append(current_attack_success)
        prr_values.append(current_prr)
        
        # 打印详细信息
        if (episode + 1) % args.log_interval == 0:
            logger.info(f"Episode {episode+1}/{args.max_episodes}, "
                        f"Action Mode: {args.action_mode}, "
                        f"Reward: {episode_reward:.2f}, "
                        f"Failures: {episode_failures}, "
                        f"Attack Success: {current_attack_success:.4f}, "
                        f"Total PRR: {current_prr:.4f}, "
                        f"Avg Vehicle PRR: {avg_vehicle_prr:.4f}")
            
            # 打印每个车辆的PRR
            for vehicle_id, prr in vehicle_prrs.items():
                logger.info(f"  Vehicle {vehicle_id}: PRR = {prr:.4f}")
        
        # 每10回合计算一次平均奖励
        if (episode + 1) % 10 == 0:
            start_idx = max(0, episode - 9)
            recent_rewards = all_rewards[start_idx:episode+1]
            avg_reward = np.mean(recent_rewards)
            avg_rewards.append(avg_reward)
            
            # 保存最佳模型
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                ppo_attacker.save(os.path.join(save_dir, "best_model.pth"))
        
        # 定期保存模型
        if (episode+1) % args.save_interval == 0:
            ppo_attacker.save(os.path.join(save_dir, f"model_ep_{episode+1}.pth"))
    
    # 训练结束
    total_time = time.time() - start_time
    logger.info(f"Training completed! Total time: {total_time/60:.2f} minutes")
    
    # 保存最终模型
    ppo_attacker.save(os.path.join(save_dir, "final_model.pth"))
    
    # 绘制训练曲线
    plt.figure(figsize=(20, 12))
    
    plt.subplot(2, 3, 1)
    plt.plot(all_rewards)
    if len(all_rewards) > 100:
        plt.plot(pd.Series(all_rewards).rolling(window=100, min_periods=1).mean(), 'r-', linewidth=2)
    plt.title(f"Reward Curve ({args.action_mode} mode)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend(["Episode Reward", "Moving Average"])
    
    plt.subplot(2, 3, 2)
    plt.plot(transmission_failures)
    plt.title("Transmission Failures")
    plt.xlabel("Episode")
    plt.ylabel("Failures")
    
    plt.subplot(2, 3, 3)
    plt.plot(attack_success_rates)
    plt.title("Attack Success Rate")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    
    plt.subplot(2, 3, 4)
    plt.plot(prr_values)
    plt.title("Packet Reception Rate (PRR)")
    plt.xlabel("Episode")
    plt.ylabel("PRR")
    
    plt.subplot(2, 3, 5)
    if actor_losses:
        plt.plot(actor_losses)
        plt.title("Actor Loss")
        plt.xlabel("Update")
        plt.ylabel("Loss")
    
    plt.subplot(2, 3, 6)
    if critic_losses:
        plt.plot(critic_losses)
        plt.title("Critic Loss")
        plt.xlabel("Update")
        plt.ylabel("Loss")
    
    plt.tight_layout()
    plt.savefig(f"attacker_training_results_{args.action_mode}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 保存训练数据
    training_data = {
        'rewards': all_rewards,
        'transmission_failures': transmission_failures,
        'attack_success_rates': attack_success_rates,
        'prr_values': prr_values,
        'actor_losses': actor_losses,
        'critic_losses': critic_losses
    }
    
    np.save(os.path.join(save_dir, 'training_data.npy'), training_data)
    logger.info(f"Training data saved to {save_dir}/training_data.npy")

if __name__ == "__main__":
    main()