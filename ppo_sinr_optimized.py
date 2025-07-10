import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from collections import deque
from torch.distributions import Categorical
import logging
import pandas as pd
import argparse
from v2x_environment_sinr import V2XRLEnvironment  # 使用新的SINR环境

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - V2X-RL-Training - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("train_sinr.log", mode='a')
    ]
)
logger = logging.getLogger('V2X-RL-Training')

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

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        演员网络 - 修改为单输出头
        :param state_dim: 状态维度
        :param action_dim: 动作维度（时隙选择数量）
        """
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_action = nn.Linear(256, action_dim)  # 动作输出
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc_action(x), dim=-1)
        return action_probs
    
class Critic(nn.Module):
    def __init__(self, state_dim):
        """
        评论家网络
        :param state_dim: 状态维度
        """
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
    
    def act(self, state):
        # 获取动作概率
        action_probs = self.actor(state)
        
        # 创建分布
        dist = Categorical(action_probs)
        
        # 采样动作
        action = dist.sample()
        
        # 计算对数概率
        action_logprob = dist.log_prob(action)
        
        # 获取状态值
        state_val = self.critic(state)
        
        return action.detach(), action_logprob.detach(), state_val.detach(), action_probs
    
    def evaluate(self, state, action):
        # 获取动作概率
        action_probs = self.actor(state)
        
        # 创建分布
        dist = Categorical(action_probs)
        
        # 计算对数概率
        action_logprobs = dist.log_prob(action.squeeze())
        
        # 计算熵（用于正则化）
        dist_entropy = dist.entropy()
        
        # 获取状态值
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, actor_lr, critic_lr, gamma, betas, K_epochs, eps_clip):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = RolloutBuffer()
        
        # 策略网络
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 优化器
        self.optimizer_actor = torch.optim.Adam(self.policy.actor.parameters(), lr=actor_lr)
        self.optimizer_critic = torch.optim.Adam(self.policy.critic.parameters(), lr=critic_lr)
        
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
        old_states_ = torch.stack(self.buffer.states_).to(device).detach()
        
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
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # 计算优势函数
        advantages = rewards.detach() - old_state_values.squeeze(1).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        
        # 计算TD目标
        next_state_values = self.policy_old.critic(old_states_).squeeze(1)
        mask = (~torch.tensor(self.buffer.is_terminals).to(device).detach()).type(torch.long)
        td_target = rewards + self.gamma * next_state_values * mask
        td_target = (td_target - td_target.mean()) / (td_target.std() + 1e-7)
        
        # 优化策略K个epochs
        total_actor_loss = 0
        total_critic_loss = 0
        
        for _ in range(self.K_epochs):
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
            critic_loss = self.MseLoss(state_values, td_target.detach())
            
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
        
        return {
            'actor_loss': total_actor_loss / self.K_epochs,
            'critic_loss': total_critic_loss / self.K_epochs
        }
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
    
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=device))

def analyze_sinr_records(sinr_records, step_count, episode, detailed_output=False):
    """分析SINR记录并输出关键信息"""
    if not sinr_records:
        return
    
    print(f"\n=== Episode {episode+1}, Step {step_count}: SINR Analysis ===")
    
    for record in sinr_records[-3:]:  # 只显示最近3条记录
        print(f"\n时间: {record['time']}ms, 资源块: (时隙{record['resource'][0]}, 子信道{record['resource'][1]})")
        
        # 发送者信息
        print("发送者:")
        for sender in record['senders']:
            pos = sender['sender_position']
            sender_type = '攻击者' if sender['is_attacker'] else '正常车辆'
            print(f"  ID: {sender['sender_id']}, 位置: ({pos[0]:.1f}, {pos[1]:.1f}), 类型: {sender_type}")
        
        # 接收者信息（简化显示）
        if detailed_output:
            print("接收者详情:")
            for receiver in record['receivers']:
                pos = receiver['receiver_position']
                print(f"  接收者 {receiver['receiver_id']}, 位置: ({pos[0]:.1f}, {pos[1]:.1f})")
                
                for i in range(len(receiver['sender_ids'])):
                    sender_id = receiver['sender_ids'][i]
                    sinr = receiver['sinr_values'][i]
                    distance = receiver['distances'][i]
                    success = "成功" if sinr >= 10.0 else "失败"  # 假设阈值为10dB
                    print(f"    从发送者 {sender_id}: SINR = {sinr:.2f} dB, 距离 = {distance:.2f} m, {success}")
        else:
            # 简化显示：只显示统计信息
            total_receivers = len(record['receivers'])
            successful_receptions = 0
            total_sinr_values = []
            
            for receiver in record['receivers']:
                for sinr in receiver['sinr_values']:
                    total_sinr_values.append(sinr)
                    if sinr >= 10.0:  # 假设阈值为10dB
                        successful_receptions += 1
            
            if total_sinr_values:
                avg_sinr = np.mean(total_sinr_values)
                min_sinr = np.min(total_sinr_values)
                max_sinr = np.max(total_sinr_values)
                success_rate = successful_receptions / len(total_sinr_values)
                
                print(f"  接收统计: {total_receivers}个接收者, 平均SINR: {avg_sinr:.2f}dB, "
                      f"范围: [{min_sinr:.2f}, {max_sinr:.2f}]dB, 成功率: {success_rate:.2%}")

def print_step_info(step_count, action, reward, info, detailed=True):
    """打印每步的详细信息"""
    if detailed:
        print(f"\n--- Step {step_count} ---")
        print(f"Action: {action}")
        print(f"Reward: {reward:.4f}")
        print(f"Collisions Caused: {info.get('collisions_caused', 0)}")
        print(f"Total Collisions: {info.get('total_collisions', 0)}")
        print(f"Attack Success Rate: {info.get('attack_success_rate', 0):.4f}")
        print(f"PRR: {info.get('prr', 0):.4f}")
        print(f"Message Failures: {info.get('message_failures', 0)}")
        print(f"Resource Block Attacks: {info.get('resource_block_attacks', 0)}")
        print(f"Resource Block Collisions: {info.get('resource_block_collisions', 0)}")
    else:
        # 简化输出
        print(f"Step {step_count}: Action={action}, Reward={reward:.3f}, "
              f"Collisions={info.get('collisions_caused', 0)}, "
              f"PRR={info.get('prr', 0):.3f}")

def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='V2X RL Training with SINR-based Collision Detection')
    parser.add_argument('--attacker-type', type=str, default='RL', choices=['RL', 'Fix'],
                        help='Type of attacker: RL (RLAttacker) or Fix (FixAttacker)')
    parser.add_argument('--fix-cycle', type=int, default=20, choices=[20, 30, 50, 100],
                        help='FixAttacker transmission cycle (ms)')
    parser.add_argument('--fix-num-subchannels', type=int, default=2, choices=[1, 2, 3, 4, 5],
                        help='Number of subchannels to occupy for FixAttacker')
    parser.add_argument('--num-vehicles', type=int, default=10,
                        help='Number of vehicles in the environment')
    parser.add_argument('--episode-duration', type=int, default=20000,
                        help='Duration of each episode in ms')
    parser.add_argument('--max-episodes', type=int, default=3000,
                        help='Maximum number of training episodes')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Interval for logging training progress')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='Interval for saving models')
    parser.add_argument('--num-attackers', type=int, default=1,
                        help='Number of attackers')
    parser.add_argument('--communication-range', type=int, default=320,
                        help='Communication range in meters')
    parser.add_argument('--use-sinr', action='store_true', default=True,
                        help='Use SINR-based collision detection')
    parser.add_argument('--sinr-threshold', type=float, default=10.0,
                        help='SINR threshold in dB')
    parser.add_argument('--detailed-output', action='store_true',
                        help='Enable detailed step-by-step output')
    parser.add_argument('--step-output-interval', type=int, default=100,
                        help='Interval for step output (every N steps)')
    args = parser.parse_args()
    
    # 初始化环境
    env = V2XRLEnvironment(
        num_vehicles=args.num_vehicles,
        num_attackers=args.num_attackers,
        episode_duration=args.episode_duration,
        communication_range=args.communication_range,
        vehicle_resource_mode='Combine',
        attacker_type=args.attacker_type,
        fix_attacker_params={
            'cycle': args.fix_cycle,
            'num_subchannels': args.fix_num_subchannels
        },
        render_mode='human',
        use_sinr=args.use_sinr,
        sinr_threshold=args.sinr_threshold,
        tx_power=23.0,
        noise_power=-95.0,
        path_loss_exponent=3.8
    )
    
    # 获取状态和动作空间维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"Environment initialized with SINR detection: {args.use_sinr}")
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
    # 训练参数
    max_episodes = args.max_episodes
    update_timestep = 1024
    save_interval = args.save_interval
    actor_lr = 0.0003
    critic_lr = 0.001
    gamma = 0.99
    betas = (0.9, 0.999)
    K_epochs = 8
    eps_clip = 0.2
    random_seed = 42
    
    # 设置随机种子
    if random_seed:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    # 只有当使用RLAttacker时才初始化PPO
    if args.attacker_type == 'RL':
        ppo = PPO(
            state_dim=state_dim,
            action_dim=action_dim,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            betas=betas,
            K_epochs=K_epochs,
            eps_clip=eps_clip
        )
        logger.info("Using RLAttacker with PPO training")
    else:
        ppo = None
        logger.info(f"Using FixAttacker with cycle={args.fix_cycle}ms and {args.fix_num_subchannels} subchannels")
    
    # 训练统计
    time_step = 0
    all_rewards = []
    avg_rewards = []
    collision_rates = []
    attack_success_rates = []
    prr_values = []
    best_avg_reward = -float('inf')
    
    # 模型保存目录
    save_dir = "ppo_models_sinr"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 训练循环
    start_time = time.time()
    
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        episode_collisions = 0
        done = False
        current_prr = 0
        current_attack_success = 0
        step_count = 0
        
        print(f"\n{'='*60}")
        print(f"Episode {episode+1}/{max_episodes} Started")
        print(f"Attacker Type: {args.attacker_type}")
        print(f"SINR Detection: {'Enabled' if args.use_sinr else 'Disabled'}")
        print(f"{'='*60}")
        
        while not done:
            time_step += 1
            step_count += 1
            
            # 选择动作
            if args.attacker_type == 'RL':
                action = ppo.select_action(state)
            else:
                action = 0  # 对于FixAttacker，传递虚拟动作
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 更新统计
            episode_reward += reward
            episode_collisions += info.get('collisions_caused', 0)
            current_prr = info.get('prr', current_prr)
            current_attack_success = info.get('attack_success_rate', current_attack_success)
            
            # 每N步输出详细信息
            if step_count % args.step_output_interval == 0 or args.detailed_output:
                print_step_info(step_count, action, reward, info, args.detailed_output)
                
                # 获取并分析SINR记录
                sinr_records = env.get_resource_block_sinr_records()
                if sinr_records:
                    analyze_sinr_records(sinr_records, step_count, episode, args.detailed_output)
            
            # 只有当使用RLAttacker时才需要保存到缓冲区
            if args.attacker_type == 'RL':
                ppo.buffer.rewards.append(reward)
                ppo.buffer.is_terminals.append(done)
                ppo.buffer.states_.append(torch.FloatTensor(next_state).to(device))
            
            # 更新状态
            state = next_state
            
            # 只有当使用RLAttacker时才更新策略
            if args.attacker_type == 'RL' and time_step % update_timestep == 0:
                update_info = ppo.update()
                print(f"\nPPO Update at step {time_step}:")
                print(f"  Actor Loss: {update_info['actor_loss']:.6f}")
                print(f"  Critic Loss: {update_info['critic_loss']:.6f}")
                time_step = 0
        
        # Episode结束后的统计
        stats = env.get_episode_stats()
        vehicle_prrs = stats['vehicle_prrs']
        avg_vehicle_prr = np.mean(list(vehicle_prrs.values())) if vehicle_prrs else 0
        
        # 记录统计数据
        all_rewards.append(episode_reward)
        collision_rates.append(episode_collisions)
        attack_success_rates.append(current_attack_success)
        prr_values.append(current_prr)
        
        # 打印Episode总结
        print(f"\n{'='*60}")
        print(f"Episode {episode+1} Summary:")
        print(f"  Total Steps: {step_count}")
        print(f"  Total Reward: {episode_reward:.4f}")
        print(f"  Total Collisions: {episode_collisions}")
        print(f"  Attack Success Rate: {current_attack_success:.4f}")
        print(f"  Network PRR: {current_prr:.4f}")
        print(f"  Average Vehicle PRR: {avg_vehicle_prr:.4f}")
        
        # 显示个人PRR分布
        if vehicle_prrs:
            prr_values_list = list(vehicle_prrs.values())
            print(f"  Vehicle PRR Range: [{min(prr_values_list):.4f}, {max(prr_values_list):.4f}]")
            print(f"  Vehicle PRR Std: {np.std(prr_values_list):.4f}")
        
        # 获取最终SINR统计
        sinr_records = env.get_resource_block_sinr_records()
        if sinr_records:
            all_sinr_values = []
            attack_sinr_values = []
            normal_sinr_values = []
            
            for record in sinr_records:
                has_attacker = any(sender['is_attacker'] for sender in record['senders'])
                
                for receiver in record['receivers']:
                    for sinr in receiver['sinr_values']:
                        all_sinr_values.append(sinr)
                        if has_attacker:
                            attack_sinr_values.append(sinr)
                        else:
                            normal_sinr_values.append(sinr)
            
            if all_sinr_values:
                print(f"  SINR Statistics:")
                print(f"    All transmissions - Mean: {np.mean(all_sinr_values):.2f}dB, "
                      f"Std: {np.std(all_sinr_values):.2f}dB")
                if attack_sinr_values:
                    print(f"    During attacks - Mean: {np.mean(attack_sinr_values):.2f}dB, "
                          f"Std: {np.std(attack_sinr_values):.2f}dB")
                if normal_sinr_values:
                    print(f"    Normal periods - Mean: {np.mean(normal_sinr_values):.2f}dB, "
                          f"Std: {np.std(normal_sinr_values):.2f}dB")
        
        print(f"{'='*60}")
        
        # 每10回合计算一次平均奖励
        if (episode + 1) % 10 == 0:
            start_idx = max(0, episode - 9)
            recent_rewards = all_rewards[start_idx:episode+1]
            avg_reward = np.mean(recent_rewards)
            avg_rewards.append(avg_reward)
        
            # 只有当使用RLAttacker时才保存最佳模型
            if args.attacker_type == 'RL' and avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                ppo.save(os.path.join(save_dir, "best_model.pth"))
                print(f"New best model saved with average reward: {best_avg_reward:.4f}")

        # 定期保存模型
        if args.attacker_type == 'RL' and (episode+1) % save_interval == 0:
            ppo.save(os.path.join(save_dir, f"model_ep_{episode+1}.pth"))
            print(f"Model saved at episode {episode+1}")
        
        # 打印进度日志
        if (episode+1) % args.log_interval == 0:
            avg_collision = np.mean(collision_rates[-args.log_interval:])
            avg_attack_success = np.mean(attack_success_rates[-args.log_interval:])
            avg_prr = np.mean(prr_values[-args.log_interval:])
            
            logger.info(f"Episode: {episode+1}/{max_episodes}, "
                        f"Attacker: {args.attacker_type}, "
                        f"Avg Reward: {np.mean(all_rewards[-args.log_interval:]):.4f}, "
                        f"Avg Collisions: {avg_collision:.2f}, "
                        f"Avg Attack Success: {avg_attack_success:.4f}, "
                        f"Avg PRR: {avg_prr:.4f}")
    
    # 训练结束
    total_time = time.time() - start_time
    logger.info(f"Training completed! Total time: {total_time/60:.2f} minutes")
    
    # 保存最终模型
    if args.attacker_type == 'RL':
        ppo.save(os.path.join(save_dir, "final_model.pth"))
    
    # 绘制训练曲线
    plt.figure(figsize=(15, 12))
    
    plt.subplot(2, 3, 1)
    plt.plot(all_rewards)
    plt.plot(pd.Series(all_rewards).rolling(window=100, min_periods=1).mean(), 'r-', linewidth=2)
    plt.title(f"Reward Curve ({args.attacker_type} Attacker)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend(["Episode Reward", "Moving Average"])
    
    plt.subplot(2, 3, 2)
    plt.plot(collision_rates)
    plt.title("Collision Counts")
    plt.xlabel("Episode")
    plt.ylabel("Collisions")
    
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
    
    # 新增：SINR分布图
    plt.subplot(2, 3, 5)
    sinr_records = env.get_resource_block_sinr_records()
    if sinr_records:
        all_sinr_values = []
        for record in sinr_records:
            for receiver in record['receivers']:
                all_sinr_values.extend(receiver['sinr_values'])
        
        if all_sinr_values:
            plt.hist(all_sinr_values, bins=50, alpha=0.7)
            plt.axvline(x=args.sinr_threshold, color='r', linestyle='--', 
                       label=f'Threshold ({args.sinr_threshold}dB)')
            plt.title("SINR Distribution")
            plt.xlabel("SINR (dB)")
            plt.ylabel("Frequency")
            plt.legend()
    
    # 新增：个人PRR分布
    plt.subplot(2, 3, 6)
    if vehicle_prrs:
        plt.bar(range(len(vehicle_prrs)), list(vehicle_prrs.values()))
        plt.title("Final Vehicle PRR Distribution")
        plt.xlabel("Vehicle ID")
        plt.ylabel("PRR")
    
    plt.tight_layout()
    plt.savefig(f"training_results_{args.attacker_type}_attacker_sinr.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()