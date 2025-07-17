import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict, deque
import random
import math

class Vehicle:
    def __init__(self, vehicle_id, x, y, velocity=50):
        self.id = vehicle_id
        self.x = x
        self.y = y
        self.velocity = velocity  # km/h
        self.velocity_ms = velocity / 3.6  # m/s
        self.transmission_cycle = 100  # ms
        self.last_transmission_time = 0
        self.packet_count = 0
        self.successful_transmissions = 0
        self.failed_transmissions = 0
        self.resource_pool = []  # 可用资源池
        self.selected_resource = None
        self.sinr_records = []  # SINR记录
        
    def update_position(self, dt):
        """更新车辆位置"""
        self.x += self.velocity_ms * dt / 1000  # dt是毫秒
        
    def should_transmit(self, current_time):
        """判断是否应该传输"""
        return current_time - self.last_transmission_time >= self.transmission_cycle
    
    def get_prr(self):
        """计算包接收率"""
        total = self.successful_transmissions + self.failed_transmissions
        return self.successful_transmissions / total if total > 0 else 0

class RLAttacker:
    def __init__(self, attacker_id, x, y, num_slots, num_subchannels, action_mode='combined'):
        self.id = attacker_id
        self.x = x
        self.y = y
        self.num_slots = num_slots
        self.num_subchannels = num_subchannels
        self.action_mode = action_mode
        self.transmission_cycle = 20  # 攻击者传输周期更短
        self.last_transmission_time = 0
        self.attack_count = 0
        self.successful_attacks = 0
        self.monitoring_data = {}  # 监听到的资源使用情况
        self.target_resources = []  # 目标攻击资源
        
    def update_monitoring_data(self, resource_usage):
        """更新监听到的资源使用情况"""
        self.monitoring_data = resource_usage
        
    def should_attack(self, current_time):
        """判断是否应该发起攻击"""
        return current_time - self.last_transmission_time >= self.transmission_cycle
    
    def get_attack_success_rate(self):
        """计算攻击成功率"""
        return self.successful_attacks / self.attack_count if self.attack_count > 0 else 0

class V2XRLEnvironment(gym.Env):
    def __init__(self, num_vehicles=10, num_attackers=1, episode_duration=20000, 
                 communication_range=320, action_mode='combined', render_mode=None):
        super(V2XRLEnvironment, self).__init__()
        
        # 环境参数
        self.num_vehicles = num_vehicles
        self.num_attackers = num_attackers
        self.episode_duration = episode_duration
        self.communication_range = communication_range
        self.action_mode = action_mode
        self.render_mode = render_mode
        
        # V2X参数
        self.num_slots = 20
        self.num_subchannels = 5
        self.slot_duration = 1  # ms
        self.subchannel_bandwidth = 180  # kHz
        
        # 物理层参数
        self.tx_power = 23  # dBm
        self.noise_power = -110  # dBm
        self.path_loss_exponent = 2.0
        self.sinr_threshold = 10  # dB
        
        # 动作空间和观测空间
        if action_mode == 'combined':
            self.action_space = spaces.Discrete(self.num_slots * self.num_subchannels)
        else:  # separated
            self.action_space = spaces.Discrete(self.num_slots * self.num_subchannels)  # 仍然返回组合动作
        
        # 状态空间：资源使用情况 + 车辆位置信息 + 历史攻击效果
        state_dim = (self.num_slots * self.num_subchannels +  # 资源占用状态
                    self.num_vehicles * 3 +  # 车辆位置和速度
                    self.num_slots * self.num_subchannels +  # 历史攻击效果
                    10)  # 其他统计信息
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        
        # 初始化
        self.reset()
        
    def reset(self):
        """重置环境"""
        self.current_time = 0
        self.vehicles = []
        self.attackers = []
        
        # 创建车辆
        for i in range(self.num_vehicles):
            x = random.uniform(0, 2000)  # 2km道路
            y = random.uniform(-10, 10)  # 车道宽度
            velocity = random.uniform(40, 80)  # km/h
            vehicle = Vehicle(i, x, y, velocity)
            self.vehicles.append(vehicle)
        
        # 创建攻击者
        for i in range(self.num_attackers):
            x = random.uniform(0, 2000)
            y = random.uniform(-10, 10)
            attacker = RLAttacker(i, x, y, self.num_slots, self.num_subchannels, self.action_mode)
            self.attackers.append(attacker)
        
        # 重置统计信息
        self.resource_usage_history = deque(maxlen=100)
        self.attack_history = deque(maxlen=100)
        self.transmission_failures = 0
        self.total_transmissions = 0
        self.episode_rewards = []
        
        # 资源使用跟踪
        self.current_resource_usage = np.zeros((self.num_slots, self.num_subchannels))
        self.resource_interference = defaultdict(list)
        
        return self._get_state()
    
    def step(self, action):
        """执行一步"""
        reward = 0
        done = False
        info = {}
        
        # 时间步进
        dt = 1  # 1ms时间步
        self.current_time += dt
        
        # 更新车辆位置
        for vehicle in self.vehicles:
            vehicle.update_position(dt)
        
        # 处理正常车辆传输
        self._process_vehicle_transmissions()
        
        # 处理攻击者行为
        if self.attackers[0].should_attack(self.current_time):
            reward = self._process_attacker_action(action)
            self.attackers[0].last_transmission_time = self.current_time
            self.attackers[0].attack_count += 1
        
        # 更新资源使用历史
        self.resource_usage_history.append(self.current_resource_usage.copy())
        
        # 检查回合结束
        if self.current_time >= self.episode_duration:
            done = True
        
        # 计算信息
        info = self._get_info()
        
        return self._get_state(), reward, done, info
    
    def _process_vehicle_transmissions(self):
        """处理正常车辆的传输"""
        self.current_resource_usage.fill(0)
        
        for vehicle in self.vehicles:
            if vehicle.should_transmit(self.current_time):
                # 选择资源
                resource = self._select_resource_for_vehicle(vehicle)
                if resource:
                    slot, subchannel = resource
                    self.current_resource_usage[slot, subchannel] += 1
                    vehicle.selected_resource = resource
                    vehicle.last_transmission_time = self.current_time
                    vehicle.packet_count += 1
                    self.total_transmissions += 1
    
    def _select_resource_for_vehicle(self, vehicle):
        """为车辆选择传输资源"""
        # 简化的资源选择：随机选择可用资源
        available_resources = []
        for slot in range(self.num_slots):
            for subchannel in range(self.num_subchannels):
                if self.current_resource_usage[slot, subchannel] == 0:
                    available_resources.append((slot, subchannel))
        
        if available_resources:
            return random.choice(available_resources)
        else:
            # 如果没有可用资源，随机选择一个（会造成冲突）
            return (random.randint(0, self.num_slots-1), 
                   random.randint(0, self.num_subchannels-1))
    
    def _process_attacker_action(self, action):
        """处理攻击者动作并计算奖励"""
        if self.action_mode == 'combined':
            slot = action // self.num_subchannels
            subchannel = action % self.num_subchannels
        else:  # separated - action仍然是组合后的值
            slot = action // self.num_subchannels
            subchannel = action % self.num_subchannels
        
        # 确保动作在有效范围内
        slot = max(0, min(slot, self.num_slots - 1))
        subchannel = max(0, min(subchannel, self.num_subchannels - 1))
        
        # 计算攻击效果
        reward = self._calculate_attack_reward(slot, subchannel)
        
        # 更新攻击者状态
        self.attackers[0].target_resources.append((slot, subchannel))
        
        # 记录攻击历史
        self.attack_history.append({
            'time': self.current_time,
            'resource': (slot, subchannel),
            'reward': reward
        })
        
        return reward
    
    def _calculate_attack_reward(self, attack_slot, attack_subchannel):
        """计算攻击奖励"""
        reward = 0
        failures_caused = 0
        
        # 检查是否有车辆在使用相同资源
        vehicles_using_resource = []
        for vehicle in self.vehicles:
            if (vehicle.selected_resource and 
                vehicle.selected_resource[0] == attack_slot and 
                vehicle.selected_resource[1] == attack_subchannel):
                vehicles_using_resource.append(vehicle)
        
        if vehicles_using_resource:
            # 计算干扰效果
            for vehicle in vehicles_using_resource:
                # 计算距离
                distance = self._calculate_distance(self.attackers[0], vehicle)
                
                if distance <= self.communication_range:
                    # 计算SINR
                    sinr = self._calculate_sinr(vehicle, self.attackers[0], distance)
                    
                    if sinr < self.sinr_threshold:
                        # 攻击成功，造成传输失败
                        failures_caused += 1
                        vehicle.failed_transmissions += 1
                        self.transmission_failures += 1
                        
                        # 主要奖励：造成传输失败
                        reward += 10.0
                        
                        # 额外奖励：基于SINR降低程度
                        sinr_degradation = max(0, self.sinr_threshold - sinr)
                        reward += sinr_degradation * 0.5
                    else:
                        # 攻击未成功
                        vehicle.successful_transmissions += 1
                        reward -= 2.0  # 惩罚
                else:
                    # 距离太远，攻击无效
                    vehicle.successful_transmissions += 1
                    reward -= 1.0
            
            if failures_caused > 0:
                self.attackers[0].successful_attacks += 1
                # 额外奖励：同时攻击多个目标
                if failures_caused > 1:
                    reward += failures_caused * 2.0
        else:
            # 没有车辆使用该资源，给予小惩罚
            reward -= 5.0
        
        # 资源利用效率奖励
        resource_utilization = self.current_resource_usage[attack_slot, attack_subchannel]
        if resource_utilization > 1:
            # 攻击高利用率资源给予额外奖励
            reward += resource_utilization * 1.0
        
        return reward
    
    def _calculate_distance(self, obj1, obj2):
        """计算两个对象之间的距离"""
        return math.sqrt((obj1.x - obj2.x)**2 + (obj1.y - obj2.y)**2)
    
    def _calculate_sinr(self, receiver, interferer, distance):
        """计算SINR"""
        # 路径损失模型
        path_loss = 20 * math.log10(distance) + 20 * math.log10(5.9) + 32.44  # dB
        
        # 接收功率
        rx_power = self.tx_power - path_loss  # dBm
        
        # 干扰功率（攻击者）
        interference_power = self.tx_power - path_loss  # dBm
        
        # 转换为线性值
        signal_linear = 10**(rx_power / 10)
        interference_linear = 10**(interference_power / 10)
        noise_linear = 10**(self.noise_power / 10)
        
        # 计算SINR
        sinr_linear = signal_linear / (interference_linear + noise_linear)
        sinr_db = 10 * math.log10(sinr_linear)
        
        return sinr_db
    
    def _get_state(self):
        """获取当前状态"""
        state = []
        
        # 1. 资源使用情况 (num_slots * num_subchannels)
        resource_state = self.current_resource_usage.flatten()
        state.extend(resource_state)
        
        # 2. 车辆位置和速度信息 (num_vehicles * 3)
        for vehicle in self.vehicles:
            state.extend([
                vehicle.x / 2000.0,  # 归一化位置
                vehicle.y / 20.0,    # 归一化位置
                vehicle.velocity / 100.0  # 归一化速度
            ])
        
        # 3. 历史攻击效果 (num_slots * num_subchannels)
        attack_effect = np.zeros(self.num_slots * self.num_subchannels)
        if len(self.attack_history) > 0:
            recent_attacks = list(self.attack_history)[-10:]  # 最近10次攻击
            for attack in recent_attacks:
                slot, subchannel = attack['resource']
                idx = slot * self.num_subchannels + subchannel
                attack_effect[idx] += attack['reward'] / 10.0  # 归一化
        state.extend(attack_effect)
        
        # 4. 其他统计信息 (10维)
        stats = [
            self.current_time / self.episode_duration,  # 时间进度
            self.transmission_failures / max(1, self.total_transmissions),  # 失败率
            len([v for v in self.vehicles if self._calculate_distance(self.attackers[0], v) <= self.communication_range]) / self.num_vehicles,  # 通信范围内车辆比例
            self.attackers[0].get_attack_success_rate(),  # 攻击成功率
            np.mean([v.get_prr() for v in self.vehicles]),  # 平均PRR
            np.std([v.get_prr() for v in self.vehicles]),   # PRR标准差
            self.attackers[0].x / 2000.0,  # 攻击者位置
            self.attackers[0].y / 20.0,
            np.sum(self.current_resource_usage) / (self.num_slots * self.num_subchannels),  # 资源利用率
            len(self.attack_history) / 100.0  # 攻击频率
        ]
        state.extend(stats)
        
        return np.array(state, dtype=np.float32)
    
    def _get_info(self):
        """获取额外信息"""
        vehicle_prrs = {v.id: v.get_prr() for v in self.vehicles}
        avg_prr = np.mean(list(vehicle_prrs.values())) if vehicle_prrs else 0
        
        info = {
            'transmission_failures': self.transmission_failures,
            'total_transmissions': self.total_transmissions,
            'attack_success_rate': self.attackers[0].get_attack_success_rate(),
            'prr': avg_prr,
            'vehicle_prrs': vehicle_prrs,
            'current_time': self.current_time,
            'resource_utilization': np.sum(self.current_resource_usage) / (self.num_slots * self.num_subchannels)
        }
        
        return info
    
    def get_episode_stats(self):
        """获取回合统计信息"""
        vehicle_prrs = {v.id: v.get_prr() for v in self.vehicles}
        
        stats = {
            'episode_duration': self.current_time,
            'total_transmissions': self.total_transmissions,
            'transmission_failures': self.transmission_failures,
            'failure_rate': self.transmission_failures / max(1, self.total_transmissions),
            'vehicle_prrs': vehicle_prrs,
            'avg_prr': np.mean(list(vehicle_prrs.values())) if vehicle_prrs else 0,
            'attack_count': self.attackers[0].attack_count,
            'successful_attacks': self.attackers[0].successful_attacks,
            'attack_success_rate': self.attackers[0].get_attack_success_rate()
        }
        
        return stats
    
    def render(self, mode='human'):
        """渲染环境"""
        if mode != 'human':
            return
        
        plt.clf()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 左图：车辆和攻击者位置
        ax1.set_xlim(0, 2000)
        ax1.set_ylim(-20, 20)
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title(f'V2X Network at t={self.current_time}ms')
        
        # 绘制车辆
        for vehicle in self.vehicles:
            color = 'green' if vehicle.get_prr() > 0.8 else 'orange' if vehicle.get_prr() > 0.5 else 'red'
            ax1.scatter(vehicle.x, vehicle.y, c=color, s=100, marker='o', alpha=0.7)
            ax1.text(vehicle.x, vehicle.y+2, f'V{vehicle.id}', ha='center', fontsize=8)
        
        # 绘制攻击者
        for attacker in self.attackers:
            ax1.scatter(attacker.x, attacker.y, c='black', s=150, marker='^', alpha=0.8)
            ax1.text(attacker.x, attacker.y+2, f'A{attacker.id}', ha='center', fontsize=8, color='red')
            
            # 绘制通信范围
            circle = plt.Circle((attacker.x, attacker.y), self.communication_range, 
                              fill=False, color='red', linestyle='--', alpha=0.5)
            ax1.add_patch(circle)
        
        # 右图：资源使用情况
        im = ax2.imshow(self.current_resource_usage.T, cmap='YlOrRd', aspect='auto')
        ax2.set_xlabel('Time Slot')
        ax2.set_ylabel('Subchannel')
        ax2.set_title('Resource Usage')
        plt.colorbar(im, ax=ax2)
        
        # 标记攻击者目标资源
        if len(self.attackers[0].target_resources) > 0:
            recent_target = self.attackers[0].target_resources[-1]
            ax2.scatter(recent_target[0], recent_target[1], c='red', s=200, marker='x', linewidth=3)
        
        plt.tight_layout()
        plt.pause(0.01)
    
    def close(self):
        """关闭环境"""
        plt.close('all')

# 测试代码
if __name__ == "__main__":
    env = V2XRLEnvironment(num_vehicles=10, num_attackers=1, action_mode='combined')
    
    state = env.reset()
    print(f"State shape: {state.shape}")
    print(f"Action space: {env.action_space}")
    
    for step in range(100):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        
        if step % 20 == 0:
            print(f"Step {step}: Reward={reward:.2f}, Failures={info['transmission_failures']}, PRR={info['prr']:.3f}")
        
        if done:
            break
    
    stats = env.get_episode_stats()
    print("\nEpisode Statistics:")
    for key, value in stats.items():
        if key != 'vehicle_prrs':
            print(f"{key}: {value}")
    
    env.close()