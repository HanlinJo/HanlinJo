import numpy as np
import random
import time
from collections import defaultdict, deque
import logging
import gym
from gym import spaces
import matplotlib
import matplotlib.pyplot as plt
import math

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('V2X-RL-Environment-Enhanced-SINR')

# 全局攻击模式切换
TARGETED_ATTACK_MODE = True

class Packet:
    """表示用于传输的V2X数据包"""
    
    def __init__(self, sender_id, timestamp, position, packet_id, size=190, is_attack=False):
        self.sender_id = sender_id
        self.timestamp = timestamp
        self.position = position
        self.size = size
        self.is_attack = is_attack
        self.packet_id = packet_id
        self.expected_receivers = 0

class SensingData:
    """表示感知数据"""
    def __init__(self, slot_id, subchannel, pRsvp, sender_id, timestamp):
        self.slot_id = slot_id
        self.subchannel = subchannel
        self.pRsvp = pRsvp
        self.sender_id = sender_id
        self.timestamp = timestamp

class ResourceInfo:
    """表示资源块 (时隙+子信道)"""
    def __init__(self, slot_id, subchannel):
        self.slot_id = slot_id
        self.subchannel = subchannel
    
    def __eq__(self, other):
        if not isinstance(other, ResourceInfo):
            return False
        return (self.slot_id == other.slot_id and 
                self.subchannel == other.subchannel)
    
    def __repr__(self):
        return f"(slot:{self.slot_id}, subchannel:{self.subchannel})"

class SINRCalculator:
    """SINR计算器 - 封装SINR计算逻辑"""
    
    def __init__(self, tx_power=23.0, noise_power=-95.0, use_simple_interference=False):
        self.tx_power = tx_power  # 发射功率 (dBm)
        self.noise_power = noise_power  # 噪声功率 (dBm)
        self.use_simple_interference = use_simple_interference  # 是否使用简化干扰计算
        
    def calculate_path_loss(self, distance):
        """
        优化的路径损耗模型 - 确保远距离单发送者也能接收成功
        """
        if distance < 1:
            distance = 1
        
        # 修正的路径损耗模型，减少远距离损耗
        if distance <= 50:
            # 近距离使用标准模型
            return 32.45 + 20 * np.log10(distance) + 20 * np.log10(5.9)
        elif distance <= 200:
            # 中距离优化
            return 32.45 + 20 * np.log10(50) + 20 * np.log10(5.9) + 15 * np.log10(distance / 50)
        else:
            # 远距离进一步优化，确保300m+也能接收
            return 32.45 + 20 * np.log10(50) + 20 * np.log10(5.9) + 15 * np.log10(200 / 50) + 10 * np.log10(distance / 200)
    
    def calculate_sinr_optimized(self, receiver_pos, sender_pos, interferers_pos):
        """
        优化的SINR计算：
        1. 单发送者时确保远距离也能接收
        2. 多发送者时距离起作用
        """
        # 计算目标信号接收功率
        distance = np.linalg.norm(receiver_pos - sender_pos)
        path_loss = self.calculate_path_loss(distance)
        rx_power = self.tx_power - path_loss  # dBm
        
        # 计算干扰功率
        if len(interferers_pos) == 0:
            # 没有干扰者 - 确保远距离也能成功接收
            interference_power_mw = 0
        else:
            # 有干扰者 - 距离开始起作用
            interference_power_mw = 0
            for intf_pos in interferers_pos:
                intf_distance = np.linalg.norm(receiver_pos - intf_pos)
                intf_path_loss = self.calculate_path_loss(intf_distance)
                intf_rx_power = self.tx_power - intf_path_loss  # dBm
                # 转换为毫瓦并累加
                interference_power_mw += 10 ** (intf_rx_power / 10)
        
        # 添加噪声功率
        noise_mw = 10 ** (self.noise_power / 10)
        total_interference_mw = interference_power_mw + noise_mw
        
        # 计算SINR
        signal_mw = 10 ** (rx_power / 10)
        sinr_linear = signal_mw / total_interference_mw
        sinr_db = 10 * np.log10(sinr_linear) if sinr_linear > 0 else -100
        
        return sinr_db
    
    def calculate_sinr_simple(self, receiver_pos, sender_pos, num_interferers):
        """
        简化的SINR计算：仅基于发送者数量和路径损耗
        """
        # 计算目标信号接收功率
        distance = np.linalg.norm(receiver_pos - sender_pos)
        path_loss = self.calculate_path_loss(distance)
        rx_power = self.tx_power - path_loss  # dBm
        
        # 简化干扰计算：基于干扰者数量
        if num_interferers == 0:
            # 没有干扰 - 只有噪声
            interference_power_db = self.noise_power
        else:
            # 有干扰 - 假设平均干扰功率
            avg_interference_power = self.tx_power - 60  # 假设平均60dB路径损耗
            total_interference_mw = num_interferers * (10 ** (avg_interference_power / 10))
            noise_mw = 10 ** (self.noise_power / 10)
            interference_power_db = 10 * np.log10(total_interference_mw + noise_mw)
        
        # 计算SINR
        signal_mw = 10 ** (rx_power / 10)
        interference_mw = 10 ** (interference_power_db / 10)
        sinr_linear = signal_mw / interference_mw
        sinr_db = 10 * np.log10(sinr_linear) if sinr_linear > 0 else -100
        
        return sinr_db
    
class Message:
    """表示从发送者到接收者的V2X消息"""
    def __init__(self, sender_id, receiver_id, packet_id):
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.packet_id = packet_id
        self.resources_received = 0  # 已接收的资源块数量
        self.success = True  # 初始假设成功，一旦失败就变为False
        self.completed = False  # 是否已完成（两个资源块都处理）
        
    def record_reception(self, success):
        """记录资源块接收情况"""
        self.resources_received += 1
        if not success:
            self.success = False  # 一旦失败就不可逆
            
    def is_completed(self):
        """检查消息是否完成（两个资源块都处理）"""
        return self.resources_received >= 2
    
class Vehicle:
    """表示具有V2X功能的车辆"""
    
    def __init__(self, vehicle_id, initial_position, initial_velocity, sim, resource_selection_mode='Separate'):
        self.id = vehicle_id
        self.position = initial_position
        self.velocity = initial_velocity
        self.sim = sim
        self.resource_selection_mode = resource_selection_mode
        
        # 获取资源池参数
        self.num_slots = sim.resource_pool.num_slots
        self.num_subchannels = sim.resource_pool.num_subchannels
        
        # 资源选择参数
        self.resel_counter = 0
        self.prob_resource_keep = random.uniform(0.2, 0.8)
        self.current_resources = None
        self.sensing_data = []
        self.next_transmission_time = 0
        self.sent_resources_count = 0
        self.current_packet_id = 0
        # 感知窗口参数
        self.sensing_window_duration = 1000
        self.has_transmitted = False
        self.total_faild = 0
        # 初始化统计
        self.packets_sent = 0
        self.packets_received = 0
        self.packets_received_succeed = 0
        self.collisions = 0
        self.successful_transmissions = 0
        self.prr = 0.0
        self.expected_receptions = 0
        self.successful_receptions = 0
    
    def reset(self):
        """重置车辆状态"""
        self.resel_counter = 0
        self.packets_received_succeed = 0
        self.current_resources = None
        self.sensing_data = []
        self.next_transmission_time = 0
        self.sent_resources_count = 0
        self.current_packet_id = 0
        self.packets_sent = 0
        self.packets_received = 0
        self.collisions = 0
        self.successful_transmissions = 0
        self.prr = 0.0
        self.expected_receptions = 0
        self.successful_receptions = 0
        self.total_faild = 0
        
    def move(self, delta_time):
        """基于速度和时间增量更新车辆位置"""
        self.position = self.position + self.velocity * delta_time
        
        # 处理边界条件（反射）
        if self.position[0] >= 1000:
            self.position[0] = 1000 - (self.position[0] - 1000)
            self.velocity = -self.velocity
            self.position[1] = 10.0
        if self.position[0] <= 0:
            self.position[0] = -self.position[0]
            self.velocity = -self.velocity
            self.position[1] = 5.0
            
    def get_sensed_resource_occupancy(self):
        """获取监听窗中的资源占用状态矩阵"""
        occupancy = np.zeros((self.num_slots, self.num_subchannels), dtype=int)
        
        for data in self.sensing_data:
            slot = data.slot_id % self.num_slots
            if 0 <= slot < self.num_slots and 0 <= data.subchannel < self.num_subchannels:
                occupancy[slot, data.subchannel] = 1
                
        return occupancy  
    
    def select_future_resource(self, current_time):
        """选择未来资源 - 根据模式选择资源块"""
        self._update_sensing_window(current_time)
        selection_window = self._create_selection_window(current_time)
        
        # 创建已占用资源集合
        occupied_resources = set()
        for data in self.sensing_data:
            resource_key = (data.slot_id, data.subchannel)
            occupied_resources.add(resource_key)
        
        # 根据模式选择资源
        if self.resource_selection_mode == 'Combine':
            selected_resources = self._select_combined_resources(selection_window, occupied_resources)
        else:  # Separate模式
            selected_resources = self._select_separate_resources(selection_window, occupied_resources)
        
        self.resel_counter = random.randint(5, 15)
        return selected_resources

    def _select_separate_resources(self, selection_window, occupied_resources):
        """Separate模式：选择两个独立的资源块"""
        candidate_resources = []
        for resource in selection_window:
            resource_key = (resource.slot_id, resource.subchannel)
            if resource_key not in occupied_resources:
                candidate_resources.append(resource)
        
        min_candidates = max(1, int(0.2 * len(selection_window)))
        if len(candidate_resources) < min_candidates:
            candidate_resources = selection_window[:]
        
        selected_resources = []
        if len(candidate_resources) >= 2:
            selected = random.sample(candidate_resources, 2)
            selected_resources = selected
        elif len(candidate_resources) == 1:
            selected_resources = [candidate_resources[0], random.choice(selection_window)]
        else:
            slot1 = random.randint(0, self.num_slots-1)
            subchannel1 = random.randint(0, self.num_subchannels-1)
            slot2 = random.randint(0, self.num_slots-1)
            subchannel2 = random.randint(0, self.num_subchannels-1)
            selected_resources = [ResourceInfo(slot1, subchannel1), ResourceInfo(slot2, subchannel2)]
        
        return selected_resources

    def _select_combined_resources(self, selection_window, occupied_resources):
        """Combine模式：选择同一时隙的两个相邻子信道"""
        slot_resources = defaultdict(list)
        for resource in selection_window:
            slot_resources[resource.slot_id].append(resource)
        
        valid_slots = []
        for slot_id, resources in slot_resources.items():
            free_subchannels = [r.subchannel for r in resources 
                               if (slot_id, r.subchannel) not in occupied_resources]
            
            adjacent_pairs = []
            for i in range(self.num_subchannels - 1):
                if i in free_subchannels and (i+1) in free_subchannels:
                    adjacent_pairs.append((i, i+1))
            
            if adjacent_pairs:
                valid_slots.append((slot_id, adjacent_pairs))
        
        if valid_slots:
            slot_id, adjacent_pairs = random.choice(valid_slots)
            sc1, sc2 = random.choice(adjacent_pairs)
            return [ResourceInfo(slot_id, sc1), ResourceInfo(slot_id, sc2)]
        
        if slot_resources:
            slot_id = random.choice(list(slot_resources.keys()))
            sc_pair = random.choice([(i, i+1) for i in range(self.num_subchannels - 1)])
            sc1, sc2 = sc_pair
            return [ResourceInfo(slot_id, sc1), ResourceInfo(slot_id, sc2)]
        
        return []
    
    def _create_selection_window(self, current_time):
        """创建选择窗口 (T1=4到T2=100)"""
        selection_window = []
        current_slot = current_time % self.num_slots
        start_slot = (current_slot + 4) % self.num_slots
        end_slot = (current_slot + 100) % self.num_slots
        
        if start_slot < end_slot:
            slots = range(start_slot, end_slot)
        else:
            slots = list(range(start_slot, self.num_slots)) + list(range(0, end_slot))
        
        for slot in slots:
            for subchannel in range(self.num_subchannels):
                selection_window.append(ResourceInfo(slot, subchannel))
        
        return selection_window

    def _update_sensing_window(self, current_time):
        """通过移除旧条目更新感知窗口"""
        sensing_window_start = current_time - self.sensing_window_duration
        self.sensing_data = [data for data in self.sensing_data 
                            if data.timestamp >= sensing_window_start]
    
    def add_sensing_data(self, slot_id, subchannel, pRsvp, sender_id, timestamp):
        """添加感知数据"""
        sensing_data = SensingData(
            slot_id=slot_id,
            subchannel=subchannel,
            pRsvp=pRsvp,
            sender_id=sender_id,
            timestamp=timestamp
        )
        self.sensing_data.append(sensing_data)
    
    def handle_periodic_resource_reselection(self, current_time):
        """在周期开始时处理资源重选"""
        self.prob_resource_keep = random.uniform(0.2, 0.8)
        if current_time % self.num_slots == 0:
            if self.resel_counter <= 0:
                if random.random() < self.prob_resource_keep:
                    self.resel_counter = random.randint(5, 15)
                else:
                    self.current_resources = None
                    self.resel_counter = random.randint(5, 15)
            
            if self.current_resources is None:
                self.current_resources = self.select_future_resource(current_time)
                self.sent_resources_count = 0
                self.current_packet_id += 1
    
    def send_packet(self, current_time):
        """使用选定的资源发送数据包（现在使用两个资源块）"""
        if self.current_resources is None:
            return None
        
        current_slot = current_time % self.num_slots
        
        resources_to_send = []
        for resource in self.current_resources:
            if resource.slot_id == current_slot:
                resources_to_send.append(resource)
        
        if not resources_to_send:
            return None
        
        packet = Packet(self.id, current_time, self.position, self.current_packet_id)
        packet.expected_receivers = self._calculate_expected_receivers()
        
        transmissions = []
        for resource in resources_to_send:
            transmissions.append((packet, resource))
        
        self.sent_resources_count += len(resources_to_send)
        self.packets_sent += len(resources_to_send)
        
        if self.sent_resources_count >= 2:
            self.has_transmitted = True
            self.resel_counter -= 1
            self.sent_resources_count = 0
        
        return transmissions
    
    def _calculate_expected_receivers(self):
        """计算当前时刻能接收到该包的车辆数量（通信范围内）"""
        count = 0
        for vehicle in self.sim.vehicles:
            if vehicle.id != self.id and vehicle.should_receive_packet(self.position):
                count += 1
        return count
    
    # def record_reception(self, success):
    #     """记录数据包接收情况（车辆级别）"""
    #     self.expected_receptions += expected_receivers
    #     if success:
    #         self.successful_receptions += expected_receivers
            
    def record_reception(self, success):
        """记录接收状态（作为接收者）"""
        self.expected_receptions += 1
        if success:
            self.successful_receptions += 1
    
    def calculate_prr(self):
        """计算个人PRR"""
        if self.expected_receptions > 0:
            return self.successful_receptions / self.expected_receptions
        return 0.0
    
    def receive_packet(self, packet, resource, success):
        """处理接收到的数据包 - 修改为始终添加感知数据"""
        if hasattr(packet, 'is_attack') and packet.is_attack:
            pRsvp = 100
        else:
            pRsvp = 100
        
        self.add_sensing_data(
            resource.slot_id,
            resource.subchannel,
            pRsvp,
            packet.sender_id,
            packet.timestamp
        )
        if not packet.is_attack:
            self.packets_received += 1
        if not packet.is_attack and success:
            self.packets_received_succeed += 1
            return True
        return False
    
    def should_receive_packet(self, sender_position):
        """确定该车辆是否应接收来自发送者的数据包"""
        distance = np.linalg.norm(self.position - sender_position)
        return distance <= self.sim.communication_range

class RLAttacker:
    """基于RL的攻击者"""
    
    def __init__(self, attacker_id, initial_position, initial_velocity, sim):
        self.id = attacker_id
        self.position = initial_position
        self.velocity = initial_velocity
        self.sim = sim
        self.last_collison = 0
        
        self.num_slots = sim.resource_pool.num_slots
        self.num_subchannels = sim.resource_pool.num_subchannels
        
        self.next_transmission_time = 0
        self.transmission_cycle = 20
        self.current_resource = None
        self.attack_packets_sent = 0
        self.attack_success_count = 0
        self.collisions_caused = 0
        self.target_slot = -1
        self.sensing_data = []
        self.sensing_window_duration = 200
        self.last_action = None
        self.last_reward = 0
        
        self.action_history = deque(maxlen=100)
        
        self.target_vehicle_id = 0
        self.target_vehicle_resources = []
        self.target_vehicle_tracking_time = 0
        self.targeted_resources = []
    
    def reset(self):
        """重置攻击者状态"""
        self.last_collison = 0
        self.next_transmission_time = 0
        self.current_resource = None
        self.attack_packets_sent = 0
        self.attack_success_count = 0
        self.collisions_caused = 0
        self.target_slot = -1
        self.sensing_data = []
        self.last_action = None
        self.last_reward = 0
        self.action_history.clear()
        
        self.target_vehicle_id = -1
        self.target_vehicle_resources = []
        self.target_vehicle_tracking_time = 0
        self.targeted_resources = []
         
    def move(self, delta_time):
        """更新攻击者位置"""
        self.position = self.position + self.velocity * delta_time
    
    def _action_to_tuple(self, a):
        if isinstance(a, np.ndarray):
            if a.ndim == 0:
                return (a.item(),)
            else:
                return tuple(a.tolist())
        if isinstance(a, (list, tuple)):
            return tuple(a)
        return (a,)
    
    
    def select_attack_resources(self, action):
        """基于RL动作选择两个攻击资源"""
        # 解析动作：每个动作值对应一个资源块
        resource1_idx = action[0]
        resource2_idx = action[1]
        
        # 计算目标时隙和子信道
        slot1 = resource1_idx // self.num_subchannels
        sc1 = resource1_idx % self.num_subchannels
        
        slot2 = resource2_idx // self.num_subchannels
        sc2 = resource2_idx % self.num_subchannels
        
        # 创建两个资源对象
        resource1 = ResourceInfo(slot1, sc1)
        resource2 = ResourceInfo(slot2, sc2)
        
        return [resource1, resource2]

    def send_attack_packet_with_action(self, current_time, action):
        """发送攻击数据包 - 使用选定的资源块"""
        # 基于动作选择资源
        self.current_resources = self.select_attack_resources(action)
        
        current_slot = current_time % self.num_slots
        attack_packets = []
        
        # 检查每个资源是否在当前时隙
        for resource in self.current_resources:
            if resource.slot_id == current_slot:
                attack_packet = Packet(self.id, current_time, self.position, packet_id=0, is_attack=True)
                attack_packets.append((attack_packet, resource))
                self.attack_packets_sent += 1
        
        return attack_packets
    # def send_attack_packet_with_action(self, current_time, action):
    #     """使用RL动作发送攻击数据包"""
    #     if TARGETED_ATTACK_MODE:
    #         return self._send_targeted_attack(current_time)
    #     else:
    #         return self._send_exhaustion_attack(current_time, action)
    
    
    def _send_exhaustion_attack(self, current_time, action):
        """侧链资源耗尽攻击模式"""
        slot_offset = action
        
        current_slot = current_time % self.num_slots
        self.target_slot = (current_slot + slot_offset) % self.num_slots
        
        if current_slot != self.target_slot:
            return []
        
        attack_packets = []
        for subchannel in range(self.num_subchannels):
            resource = ResourceInfo(self.target_slot, subchannel)
            attack_packet = Packet(self.id, current_time, self.position, packet_id=0, is_attack=True)
            attack_packets.append((attack_packet, resource))
        self.attack_packets_sent += 1
        self.action_history.append(action)
        
        return attack_packets
    
    def _send_targeted_attack(self, current_time):
        """目标侧链攻击模式"""
        if not self.targeted_resources:
            return []
        
        current_slot = current_time % self.num_slots
        attack_packets = []
        
        for resource in self.targeted_resources:
            if resource.slot_id == current_slot:
                attack_packet = Packet(self.id, current_time, self.position, packet_id=0, is_attack=True)
                attack_packets.append((attack_packet, resource))
                self.attack_packets_sent += 1
        
        return attack_packets
    
    def get_state(self, current_time):
        """获取RL代理的当前状态"""
        self._update_sensing_window(current_time)
        
        resource_state = np.zeros((self.num_slots, self.num_subchannels))
        for sensing_data in self.sensing_data:
            slot_id = sensing_data.slot_id % self.num_slots
            if 0 <= slot_id < self.num_slots and 0 <= sensing_data.subchannel < self.num_subchannels:
                resource_state[slot_id, sensing_data.subchannel] = 1
        
        occupied_resources = np.sum(resource_state)
        total_resources = self.num_slots * self.num_subchannels
        free_ratio = 1.0 - (occupied_resources / total_resources)
        
        collision_stats = [
            free_ratio,
            self.sim.recent_collision_rate
        ]
        
        full_state = np.concatenate([
            resource_state.flatten(),
            collision_stats
        ])
        
        return full_state.astype(np.float32)
    
    def calculate_reward(self, collision_occurred, num_collisions_caused, current_time, collision_count):
        """计算RL代理的奖励"""
        reward = 0.0
        
        if collision_occurred:
            reward += 1.0 * num_collisions_caused
            self.attack_success_count += 1
        else:
            reward -= 2.0
        
        if len(self.action_history) > 10:
            unique_actions = len(set(self._action_to_tuple(a) for a in self.action_history))
            diversity_ratio = unique_actions / len(self.action_history)
            reward += 0.02 * diversity_ratio
        
        if current_time > 0:
            avg_collisions = collision_count - self.last_collison
            reward += avg_collisions * 0.01
            reward = np.clip(reward, 0.0, 0.2)
        self.last_collison = collision_count
            
        prr = self.sim._calculate_current_prr()
        reward += (1.0 - prr) * 0.5
        reward = np.clip(reward, -1.0, 5.0)
        self.last_reward = reward

        return reward
    
    def record_attack_success(self, collision_occurred):
        """记录攻击成功"""
        if collision_occurred:
            self.collisions_caused += 1
    
    def should_receive_packet(self, sender_position):
        """攻击者可以接收通信范围内的数据包"""
        distance = np.linalg.norm(self.position - sender_position)
        return distance <= self.sim.communication_range

    def add_sensing_data(self, slot_id, subchannel, pRsvp, sender_id, timestamp):
        """添加来自接收传输的感知数据"""
        sensing_data = SensingData(
            slot_id=slot_id,
            subchannel=subchannel,
            pRsvp=pRsvp,
            sender_id=sender_id,
            timestamp=timestamp
        )
        self.sensing_data.append(sensing_data)
        
        if TARGETED_ATTACK_MODE and sender_id == self.target_vehicle_id:
            resource = ResourceInfo(slot_id, subchannel)
            self.target_vehicle_resources.append(resource)
    
    def _update_sensing_window(self, current_time):
        """更新监听窗，移除过期数据"""
        sensing_window_start = current_time - self.sensing_window_duration
        
        self.sensing_data = [data for data in self.sensing_data 
                            if data.timestamp >= sensing_window_start]

class FixAttacker(Vehicle):
    """固定策略攻击者"""
    
    def __init__(self, attacker_id, initial_position, initial_velocity, sim, 
                 attack_cycle=20, num_subchannels=2, resource_selection_mode='Combine'):
        super().__init__(attacker_id, initial_position, initial_velocity, sim, resource_selection_mode)
        self.is_attack = True
        self.attack_cycle = attack_cycle
        self.num_subchannels = num_subchannels
        self.next_attack_time = 0
        self.attack_packets_sent = 0
        self.collisions_caused = 0
        self.has_transmitted = False
        
        self.num_slots = sim.resource_pool.num_slots
        self.sensing_window_duration = 200
        self.prob_resource_keep = 0.2
        
        self.cycle_groups = self._calculate_cycle_groups() if not TARGETED_ATTACK_MODE else []
        
        self.target_vehicle_id = 0  # 默认攻击车辆0
        self.target_vehicle_resources = []
        self.target_vehicle_tracking_time = 0
        self.targeted_resources = []
        
        # 目标攻击相关
        self.target_tracking_enabled = True
        self.last_target_update_time = 0
    
    def reset(self):
        """重置攻击者状态"""
        super().reset()
        self.next_attack_time = 0
        self.attack_packets_sent = 0
        self.collisions_caused = 0
        self.sensing_data = []
        self.cycle_groups = self._calculate_cycle_groups() if not TARGETED_ATTACK_MODE else []
        
        self.target_vehicle_id = 0  # 重置后仍然攻击车辆0
        self.target_vehicle_resources = []
        self.target_vehicle_tracking_time = 0
        self.targeted_resources = []
        self.last_target_update_time = 0
        
    def _calculate_cycle_groups(self):
        """根据攻击周期计算时隙组"""
        num_groups = self.num_slots // self.attack_cycle
        groups = []
        start = 0
        
        for _ in range(num_groups):
            end = start + self.attack_cycle
            groups.append((start, end))
            start = end
        
        if start < self.num_slots:
            groups.append((start, self.num_slots))
        
        return groups
    
    def _get_current_cycle_group(self, current_time):
        """获取当前时间所属的周期组"""
        if not self.cycle_groups:
            return (0, self.num_slots)
            
        current_slot = current_time % self.num_slots
        
        for start, end in self.cycle_groups:
            if start <= current_slot < end:
                return (start, end)
        
        return self.cycle_groups[-1]
    
    def send_packet(self, current_time):
        """重写发送方法实现攻击逻辑"""
        if TARGETED_ATTACK_MODE:
            # 每次发送前更新目标资源
            self._update_target_resources()
            return self._send_targeted_attack(current_time)
        else:
            return self._send_cycle_group_attack(current_time)
    
    def _send_cycle_group_attack(self, current_time):
        """执行周期组攻击模式"""
        if self.current_resources is None:
            return None
        
        if self.has_transmitted:
            return None
        
        current_slot = current_time % self.num_slots
        
        resources_to_send = []
        for resource in self.current_resources:
            if resource.slot_id == current_slot:
                resources_to_send.append(resource)
        
        if not resources_to_send:
            return None
        
        packet = Packet(self.id, current_time, self.position, self.current_packet_id, is_attack=True)
        
        transmissions = []
        for resource in resources_to_send:
            transmissions.append((packet, resource))
        
        self.sent_resources_count += len(transmissions)
        self.attack_packets_sent += len(transmissions)
        
        if self.sent_resources_count >= 2:
            self.resel_counter -= 1
            if self.resel_counter <= 0:
                self.current_resources = None
            self.sent_resources_count = 0
        
        self.has_transmitted = True
        
        return transmissions
    # def _get_targeted_resources(self):
    #     """获取目标车辆的资源（从感知数据中提取）"""
    #     # 从感知数据中提取目标车辆最近的资源
    #     target_resources = []
    #     for data in self.sensing_data:
    #         if data.sender_id == self.target_vehicle_id:
    #             resource = ResourceInfo(data.slot_id, data.subchannel)
    #             target_resources.append(resource)
        
    #     # 返回最新的资源（最多2个）
    #     return target_resources[:2]
    
    def _send_targeted_attack(self, current_time):
        """执行目标侧链攻击模式 - 改进版"""
        # # 定期更新目标资源（每100ms更新一次）
        # if current_time % 100 == 0:
        #     self._update_target_resources()
        #     self.last_target_update_time = current_time
            
            # # 调试输出
            # if self.targeted_resources:
            #     logger.info(f"攻击者 {self.id} 更新目标资源，针对车辆 {self.target_vehicle_id}:")
            #     for res in self.targeted_resources:
            #         logger.info(f"  目标资源: 时隙 {res.slot_id}, 子信道 {res.subchannel}")
        
        # if not self.targeted_resources:
        #     logger.debug(f"攻击者 {self.id} 没有找到目标车辆 {self.target_vehicle_id} 的资源")
        #     return None
        
        current_slot = current_time % self.num_slots
        transmissions = []
        
        for resource in self.targeted_resources:
            if resource.slot_id == current_slot:
                packet = Packet(self.id, current_time, self.position, self.current_packet_id, is_attack=True)
                transmissions.append((packet, resource))
                self.attack_packets_sent += 1
                # logger.info(f"攻击者 {self.id} 在时隙 {current_slot} 攻击资源 (时隙:{resource.slot_id}, 子信道:{resource.subchannel})")
        # print(transmissions)
        return transmissions
    
    def _update_target_resources(self):
        """更新目标车辆的资源选择信息 - 改进版"""
        if self.target_vehicle_id < 0:
            return
        
        self.targeted_resources = []
        
        # 收集目标车辆最近的所有资源
        target_resources_with_time = []
        
        for data in self.sensing_data:
            if data.sender_id == self.target_vehicle_id:
                resource = ResourceInfo(data.slot_id, data.subchannel)
                target_resources_with_time.append((resource, data.timestamp))
        # print(len(self.sensing_data))
        if target_resources_with_time:
            # 按时间排序，获取最新的资源
            target_resources_with_time.sort(key=lambda x: x[1], reverse=True)
            
            # 获取最新时间戳的所有资源
            latest_timestamp = target_resources_with_time[0][1]
            latest_resources = [res for res, ts in target_resources_with_time if ts == latest_timestamp]
            
            self.targeted_resources = latest_resources
            
            logger.debug(f"攻击者 {self.id} 找到目标车辆 {self.target_vehicle_id} 的 {len(latest_resources)} 个资源")
        else:
            logger.debug(f"攻击者 {self.id} 未找到目标车辆 {self.target_vehicle_id} 的感知数据")
    
    def add_sensing_data(self, slot_id, subchannel, pRsvp, sender_id, timestamp):
        """添加感知数据 - 扩展以支持目标攻击"""
        super().add_sensing_data(slot_id, subchannel, pRsvp, sender_id, timestamp)
        
        # 如果是目标攻击模式且来自目标车辆，记录资源
        if TARGETED_ATTACK_MODE and sender_id == self.target_vehicle_id:
            logger.debug(f"攻击者 {self.id} 感知到目标车辆 {self.target_vehicle_id} 使用资源 (时隙:{slot_id}, 子信道:{subchannel})")
    
    def should_receive_packet(self, sender_position):
        """攻击者可以接收通信范围内的数据包"""
        distance = np.linalg.norm(self.position - sender_position)
        return distance <= self.sim.communication_range

class ResourcePool:
    """管理V2X通信的侧链路资源池"""
    
    def __init__(self, num_slots=100, num_subchannels=5, subchannel_size=10):
        self.num_slots = num_slots
        self.num_subchannels = num_subchannels
        self.subchannel_size = subchannel_size
        self.total_rbs = num_subchannels * num_slots

class V2XRLEnvironment(gym.Env):
    """V2X攻击优化的RL环境 - 支持优化的SINR碰撞检测"""
    
    def __init__(self, num_vehicles=20, num_attackers=1, episode_duration=20000, 
                 communication_range=320.0, vehicle_resource_mode='Separate',
                 attacker_type='RL', fix_attacker_params=None, render_mode='human',
                 num_slots=100, num_subchannels=5, use_sinr=True, sinr_threshold=5.0,
                 tx_power=23.0, noise_power=-95.0, use_simple_interference=False):
        super(V2XRLEnvironment, self).__init__()
        
        self.num_vehicles = num_vehicles
        self.num_attackers = num_attackers
        self.episode_duration = episode_duration
        self.communication_range = communication_range
        self.vehicle_resource_mode = vehicle_resource_mode
        self.attacker_type = attacker_type
        self.fix_attacker_params = fix_attacker_params or {'cycle': 20, 'num_subchannels': 2}
        self.num_slots = num_slots
        self.num_subchannels = num_subchannels
        self.tx_power = 23.0
        self.noise_power = -95.0  # dBm
        # SINR相关参数
        self.use_sinr = use_sinr
        self.sinr_threshold = sinr_threshold  # 优化后的SINR阈值
        self.use_simple_interference = use_simple_interference  # 是否使用简化干扰计算
        
        # 初始化SINR计算器
        self.sinr_calculator = SINRCalculator(
            tx_power=tx_power,
            noise_power=noise_power,
            use_simple_interference=use_simple_interference
        )
        
        # 初始化组件
        self.resource_pool = ResourcePool(num_slots=num_slots, num_subchannels=num_subchannels, subchannel_size=10)
        self.initial_vehicle_states = None
        self.initial_attacker_states = None
        
        # 碰撞统计
        self.recent_collision_queue = deque(maxlen=100)
        self.recent_collision_rate = 0.0
        
        # RL空间
        state_size = (num_slots * num_subchannels) + 2
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(state_size,), dtype=np.float32
        )
        
        self.action_space = spaces.MultiDiscrete([500, 500])
        
        # 初始化模拟组件
        self.vehicles = []
        self.attackers = []
        self.current_time = 0
        self.message_status_dict = {}
        self.message_pool = {}  # 消息池字典，key: (sender_id, packet_id, receiver_id)
        # SINR记录存储
        self.sinr_records = []
        
        # 统计
        self.reset_stats()
        
    def _update_resource_grid(self):
        """更新资源网格状态"""
        # 重置资源网格
        self.resource_grid = [[set() for _ in range(self.num_subchannels)] for _ in range(self.num_slots)]
        
        # 添加车辆选择的资源
        for vehicle in self.vehicles:
            if vehicle.current_resources:
                for resource in vehicle.current_resources:
                    slot = resource.slot_id
                    sc = resource.subchannel
                    if 0 <= slot < self.num_slots and 0 <= sc < self.num_subchannels:
                        self.resource_grid[slot][sc].add(vehicle.id)
        
        # 添加攻击者选择的资源
        for attacker in self.attackers:
            if isinstance(attacker, RLAttacker) and attacker.target_slot >= 0:
                for sc in range(self.num_subchannels):
                    if 0 <= attacker.target_slot < self.num_slots:
                        self.resource_grid[attacker.target_slot][sc].add(attacker.id)
            elif isinstance(attacker, FixAttacker) and attacker.current_resources:
                for resource in attacker.current_resources:
                    slot = resource.slot_id
                    sc = resource.subchannel
                    if 0 <= slot < self.num_slots and 0 <= sc < self.num_subchannels:
                        self.resource_grid[slot][sc].add(attacker.id)    
                        
    def render_sensing_view(self, vehicle_id=0):
        """渲染指定车辆的监听窗视图 - 增强版：红色标注攻击者资源"""
        if not hasattr(self, 'sensing_fig') or not hasattr(self, 'sensing_ax'):
            plt.ion()
            self.sensing_fig, self.sensing_ax = plt.subplots(figsize=(15, 8))
            self.sensing_cbar = None
        
        if vehicle_id >= len(self.vehicles+self.attackers):
            return
            
        if vehicle_id < self.num_vehicles:
            vehicle = self.vehicles[vehicle_id]
        else:
            vehicle = self.attackers[vehicle_id - self.num_vehicles]
        occupancy = vehicle.get_sensed_resource_occupancy()
        
        # 创建攻击者资源矩阵（标记攻击者使用的资源）
        attacker_occupancy = np.zeros((self.num_slots, self.num_subchannels), dtype=int)
        for data in vehicle.sensing_data:
            # 检查是否是攻击者发送的数据
            if data.sender_id >= self.num_vehicles:  # 攻击者ID >= 车辆数
                slot = data.slot_id % self.num_slots
                if 0 <= slot < self.num_slots and 0 <= data.subchannel < self.num_subchannels:
                    attacker_occupancy[slot, data.subchannel] = 1
        
        self.sensing_ax.clear()
        
        # 创建组合视图
        combined_view = np.zeros((self.num_slots, self.num_subchannels))
        # 普通占用标记为1
        combined_view[occupancy == 1] = 1
        # 攻击者占用标记为2
        combined_view[attacker_occupancy == 1] = 2
        
        # 绘制热力图
        cmap = matplotlib.colors.ListedColormap(['white', 'blue', 'red'])
        bounds = [0, 1, 2, 3]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        
        im = self.sensing_ax.imshow(combined_view.T, cmap=cmap, norm=norm, aspect='auto', 
                                   origin='lower', extent=[0, self.num_slots, 0, self.num_subchannels])
        
        # 标记当前时隙
        current_slot = self.current_time % self.num_slots
        self.sensing_ax.axvline(x=current_slot, color='r', linestyle='-', linewidth=1)
        
        # 添加网格
        self.sensing_ax.set_xticks(np.arange(0, self.num_slots+1, max(1, self.num_slots//10)))
        self.sensing_ax.set_yticks(np.arange(0, self.num_subchannels+1, 1))
        self.sensing_ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
        
        # 设置坐标轴标签
        self.sensing_ax.set_xlabel(f'Slot (0-{self.num_slots-1})')
        self.sensing_ax.set_ylabel(f'Subchannel (0-{self.num_subchannels-1})')
        self.sensing_ax.set_title(f'Perceived Resource Occupancy by Vehicle {vehicle_id} at Time: {self.current_time} ms')
        
        # 创建图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='white', edgecolor='black', label='Free'),
            Patch(facecolor='blue', edgecolor='black', label='Occupied (Normal)'),
            Patch(facecolor='red', edgecolor='black', label='Occupied (Attacker)')
        ]
        self.sensing_ax.legend(handles=legend_elements, loc='upper right')
        
        # 刷新显示
        plt.draw()
        plt.pause(0.001)
    
    def render(self, mode='human'):
        """渲染资源选择图"""
        if mode != 'human' or self.current_time % 1 != 0:  # 每1ms渲染一次
            return

        # 确保只创建一个图形对象
        if not hasattr(self, 'fig') or not hasattr(self, 'ax'):
            plt.ion()  # 开启交互模式
            self.fig, self.ax = plt.subplots(figsize=(15, 8))
            self.cbar = None

        self._update_resource_grid()
        self.ax.clear()

        # 创建可视化矩阵
        grid_data = np.zeros((self.num_subchannels, self.num_slots))
        # 创建攻击资源矩阵（标记攻击者选择的资源）
        attack_data = np.zeros((self.num_subchannels, self.num_slots))

        for slot in range(self.num_slots):
            for sc in range(self.num_subchannels):
                users = self.resource_grid[slot][sc]
                if users:
                    grid_data[sc, slot] = len(users)
                    # 检查是否有攻击者使用该资源
                    if any(uid >= self.num_vehicles for uid in users):  # 攻击者ID >= 车辆数
                        attack_data[sc, slot] = 1  # 标记为攻击资源

        # 绘制热力图
        im = self.ax.imshow(grid_data, cmap='viridis', aspect='auto', origin='lower',
                           vmin=0, vmax=3, extent=[0, self.num_slots, 0, self.num_subchannels])

        # 标记当前时隙
        current_slot = self.current_time % self.num_slots
        self.ax.axvline(x=current_slot, color='r', linestyle='-', linewidth=1)

        # 添加文本标签
        for slot in range(self.num_slots):
            for sc in range(self.num_subchannels):
                users = self.resource_grid[slot][sc]
                if users:
                    user_text = ','.join(str(uid) for uid in users)
                    # 如果是攻击者，使用红色文本
                    text_color = 'red' if any(uid >= self.num_vehicles for uid in users) else 'white'
                    self.ax.text(slot + 0.5, sc + 0.5, user_text,
                                 ha='center', va='center', fontsize=8, color=text_color)

        # 在攻击者使用的资源块上绘制红色矩形
        for slot in range(self.num_slots):
            for sc in range(self.num_subchannels):
                if attack_data[sc, slot] == 1:
                    # 绘制红色边框矩形
                    rect = plt.Rectangle((slot, sc), 1, 1, 
                                        fill=False, edgecolor='red', linewidth=2)
                    self.ax.add_patch(rect)

        # 添加网格
        self.ax.set_xticks(np.arange(0, self.num_slots+1, max(1, self.num_slots//10)))
        self.ax.set_yticks(np.arange(0, self.num_subchannels+1, 1))
        self.ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)

        # 设置坐标轴标签
        self.ax.set_xlabel(f'Slot (0-{self.num_slots-1})')
        self.ax.set_ylabel(f'Subchannel (0-{self.num_subchannels-1})')
        title = f'Resource Allocation at Time: {self.current_time} ms (Current Slot: {current_slot})'
        if self.attackers:
            # 添加攻击者信息
            attacker = self.attackers[0]
            if isinstance(attacker, RLAttacker):
                title += f'\nAttacker Target Slot: {attacker.target_slot}'
                if TARGETED_ATTACK_MODE:
                    title += f' | Target Vehicle: {attacker.target_vehicle_id}'
            elif isinstance(attacker, FixAttacker):
                if TARGETED_ATTACK_MODE:
                    title += f'\nFixAttacker Target Vehicle: {attacker.target_vehicle_id}'
                else:
                    if attacker.current_resources:
                        slots = set(r.slot_id for r in attacker.current_resources)
                        title += f'\nFixAttacker Slot(s): {", ".join(map(str, slots))}'
        self.ax.set_title(title)

        # 添加/更新颜色条
        if self.cbar is None:
            self.cbar = self.fig.colorbar(im, ax=self.ax)
            self.cbar.set_label('Number of Users')
        else:
            self.cbar.update_normal(im)

        # 刷新显示
        plt.draw()
        plt.pause(0.001)
        self.fig.canvas.flush_events()  # 确保GUI事件被处理
        
    def analyze_sinr_threshold(self, test_episodes=100):
        """
        分析最佳SINR阈值
        返回在不同阈值下的PRR性能
        """
        logger.info("开始分析最佳SINR阈值...")
        
        # 测试不同的SINR阈值
        thresholds = [-10, -5, 0, 5, 10, 15, 20]
        results = {}
        
        original_threshold = self.sinr_threshold
        
        for threshold in thresholds:
            logger.info(f"测试SINR阈值: {threshold} dB")
            self.sinr_threshold = threshold
            
            prr_values = []
            collision_rates = []
            
            for episode in range(test_episodes):
                self.reset()
                
                # 运行一个短期测试
                for step in range(100):  # 2秒测试
                    action = self.action_space.sample()
                    _, _, done, info = self.step(action)
                    if done:
                        break
                
                prr_values.append(info['prr'])
                collision_rates.append(info['collision_rate'])
            
            results[threshold] = {
                'avg_prr': np.mean(prr_values),
                'std_prr': np.std(prr_values),
                'avg_collision_rate': np.mean(collision_rates),
                'std_collision_rate': np.std(collision_rates)
            }
            
            logger.info(f"阈值 {threshold} dB: 平均PRR = {results[threshold]['avg_prr']:.4f}")
        
        # 恢复原始阈值
        self.sinr_threshold = original_threshold
        
        # 找到最佳阈值（最高PRR）
        best_threshold = max(results.keys(), key=lambda k: results[k]['avg_prr'])
        logger.info(f"推荐的最佳SINR阈值: {best_threshold} dB (PRR: {results[best_threshold]['avg_prr']:.4f})")
        
        return results, best_threshold
    
    def get_resource_block_sinr_records(self):
        """获取资源块SINR记录"""
        return self.sinr_records
    
    def reset(self):
        """重置环境开始新的一轮"""
        self.current_time = 0
        self.recent_collision_queue.clear()
        self.recent_collision_rate = 0.0
        self.message_status_dict = {}
        self.sinr_records = []
        self.message_pool = {}
        self.reset_stats()
        for vehicle in self.vehicles:
            vehicle.reset()
            
        for attacker in self.attackers:
            attacker.reset()
        
        if not hasattr(self, 'initial_vehicle_states') or self.initial_vehicle_states is None:
            self.vehicles = []
            self.attackers = []
            self._initialize_vehicles()
            self._initialize_attackers()
            self.initial_vehicle_states = [(v.position.copy(), v.velocity.copy()) for v in self.vehicles]
            self.initial_attacker_states = [(a.position.copy(), a.velocity.copy()) for a in self.attackers]
        else:
            self.vehicles = []
            for i, (position, velocity) in enumerate(self.initial_vehicle_states):
                vehicle = Vehicle(i, position.copy(), velocity.copy(), self, self.vehicle_resource_mode)
                self.vehicles.append(vehicle)
            self.attackers = []
            for i, (position, velocity) in enumerate(self.initial_attacker_states):
                attacker_id = self.num_vehicles + i
                if self.attacker_type == 'RL':
                    attacker = RLAttacker(attacker_id, position.copy(), velocity.copy(), self)
                else:
                    attacker = FixAttacker(
                        attacker_id, 
                        position.copy(), 
                        velocity.copy(), 
                        self,
                        attack_cycle=self.fix_attacker_params['cycle'],
                        num_subchannels=self.fix_attacker_params['num_subchannels']
                    )
                self.attackers.append(attacker)

        if self.attacker_type == 'RL' and self.attackers:
            initial_state = self.attackers[0].get_state(self.current_time)
            return initial_state.astype(np.float32)
        else:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
    
    def step(self, action):
        """在环境中执行一步"""
        episode_reward = 0
        collision_count_before = self.collision_count
        message_failures_before = self.message_failures
        
        for attacker in self.attackers:
            if isinstance(attacker, RLAttacker):
                attacker.target_slot = (self.current_time + action) % self.num_slots
                attacker.current_resource = True
            elif isinstance(attacker, FixAttacker):
                if not TARGETED_ATTACK_MODE:
                    attacker.current_resources = attacker.select_future_resource(self.current_time)
                else:
                    # 目标攻击模式：直接设置目标资源
                    attacker._update_target_resources()
                attacker.sent_resources_count = 0
                attacker.current_packet_id += 1
                attacker.has_transmitted = False
        
        # 模拟20ms (20个1ms的时间步)
        for step in range(100):
            if self.current_time % self.num_slots == 0:
                for vehicle in self.vehicles:
                    vehicle.handle_periodic_resource_reselection(self.current_time)
                    vehicle._update_sensing_window(self.current_time)
            
            for vehicle in self.vehicles:
                vehicle.move(0.001)
            
            for attacker in self.attackers:
                attacker.move(0.001)
                
            step_reward, step_collisions = self._process_transmissions_with_rl(action)
            episode_reward += step_reward
            
            self.recent_collision_queue.append(step_collisions)
            if len(self.recent_collision_queue) > 0:
                self.recent_collision_rate = sum(self.recent_collision_queue) / len(self.recent_collision_queue)
            # if self.current_time%5==0:
            #     self.render()
            #     self.render_sensing_view(vehicle_id=0)
            self._process_message_pool()
            self.current_time += 1
            
        for attacker in self.attackers:
            if isinstance(attacker, RLAttacker):
                attacker.current_resource = None
            if isinstance(attacker, FixAttacker):
                attacker.current_resource = None
        
        collision_count_after = self.collision_count
        collisions_caused = collision_count_after - collision_count_before
        
        if self.attacker_type == 'RL' and self.attackers:
            next_state = self.attackers[0].get_state(self.current_time)
        else:
            next_state = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        done = self.current_time >= self.episode_duration
        
        info = {
            'collisions_caused': collisions_caused,
            'total_collisions': self.collision_count,
            'attack_success_rate': self.total_attack_success / max(1, self.attackers[0].attack_packets_sent) if self.attackers else 0,
            'prr': self._calculate_current_prr(),
            'step_reward': episode_reward,
            'message_failures': self.message_failures - message_failures_before,
            'resource_block_attacks': self.resource_block_attacks,
            'resource_block_collisions': self.resource_block_collisions,
            'collision_rate': self.collision_count / max(1, self.transmission_count)
        }
        
        return next_state.astype(np.float32), episode_reward, done, info
    
    def _process_message_pool(self):
        """处理消息池中的消息，更新统计信息"""
        for msg_key, message in list(self.message_pool.items()):
            if message.is_completed():
                # 更新接收者状态
                receiver = next((v for v in self.vehicles if v.id == message.receiver_id), None)
                if receiver:
                    receiver.record_reception(message.success)
                
                # 更新发送者状态
                sender = next((v for v in self.vehicles if v.id == message.sender_id), None)
                if sender and not isinstance(sender, (RLAttacker, FixAttacker)):
                    if not message.success:
                        sender.collisions += 1
                
                # 更新全局统计
                self.total_expected_packets += 1
                if message.success:
                    self.total_received_packets += 1
                else:
                    self.message_failures += 1
                    self.collision_count += 1
                
                # 从消息池中移除已完成的消息
                del self.message_pool[msg_key]
                
    def _process_transmissions_with_rl(self, action):
        """处理传输，包括RL引导的攻击"""
        transmissions = []
        attack_transmissions = []
        current_slot = self.current_time % self.num_slots
        attacker_sent = False
        
        for vehicle in self.vehicles:
            tx_result = vehicle.send_packet(self.current_time)
            if tx_result:
                for packet, resource in tx_result:
                    transmissions.append((vehicle, packet, resource))
                    self.transmission_count += 1
        
        for attacker in self.attackers:
            if isinstance(attacker, RLAttacker):
                attack_result = attacker.send_attack_packet_with_action(self.current_time, action)
            else:
                attack_result = attacker.send_packet(self.current_time)
            if attack_result:
                attacker_sent = True
                # print(attacker_sent)
                for attack_packet, resource in attack_result:
                    attack_transmissions.append((attacker, attack_packet, resource))
            
                self.attack_transmission_count += 1    
        
        all_transmissions = transmissions + attack_transmissions
        reward = 0.0
        step_collisions = 0
        
        if all_transmissions:
            collision_info = self._handle_transmissions_with_enhanced_sinr(all_transmissions)
            step_collisions = collision_info.get('collisions_caused', 0)
            
            if attacker_sent and self.attacker_type == 'RL':
                collision_occurred = step_collisions > 0
                
                for attacker in self.attackers:
                    if isinstance(attacker, RLAttacker) and attacker_sent:
                        reward = attacker.calculate_reward(collision_occurred, step_collisions, self.current_time, self.collision_count)
        
        return reward, step_collisions
    
    def _distance(self, pos1, pos2):
        """计算两点之间的欧几里得距离"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _handle_transmissions_with_enhanced_sinr(self, transmissions):
        """处理传输并使用SINR进行碰撞检测"""
        # 按时隙分组
        tx_by_slot = defaultdict(list)
        for sender, packet, resource in transmissions:
            tx_by_slot[resource.slot_id].append((sender, packet, resource))
            # print(sender.id)
        collision_info = {'collisions_caused': 0}
        # 处理每个时隙的传输
        for slot_id, slot_transmissions in tx_by_slot.items():
            # 检测碰撞：记录每个子信道的使用情况
            subchannel_usage = defaultdict(list)
            for sender, packet, resource in slot_transmissions:
                subchannel_usage[resource.subchannel].append((sender, packet, resource))

            # 消息状态跟踪 - 重构为接收者粒度
            # 格式: {(sender_id, packet_id, receiver_id): status}
            receiver_message_status = defaultdict(lambda: {
                'success': False,
                'resources_used': 0,
                'failed_resources': set()
            })
            slot_sinr_records = []
            # 处理每个子信道
            for subchannel, users in subchannel_usage.items():
                # 检查是否有攻击者参与
                has_attacker = any(isinstance(sender, (RLAttacker, FixAttacker)) for sender, _, _ in users)
                normal_users = [user for user in users if not isinstance(user[0], (RLAttacker, FixAttacker))]
                if has_attacker:
                    for attacker in self.attackers:
                        attacker.attack_packets_sent += 1
                        # self.attack_transmission_count += 1
                # 传统碰撞检测：多个发送者使用同一资源块
                collision_occurred = len(normal_users) > 1
                
                # SINR-based接收检测
                if self.use_sinr:
                    # 记录该资源块的信息
                    resource_record = {
                        'time': self.current_time,
                        'resource': (slot_id, subchannel),
                        'senders': [],
                        'receivers': []
                    }
                    
                    # 记录发送者信息
                    for sender, packet, resource in users:
                        resource_record['senders'].append({
                            'sender_id': sender.id,
                            'sender_position': sender.position.tolist(),
                            'is_attacker': isinstance(sender, (RLAttacker, FixAttacker))
                        })
                    # 对每个接收者计算SINR
                    for receiver in self.vehicles:
                        # 只考虑不是发送者的接收者
                        receiver_sinr_info = {
                            'receiver_id': receiver.id,
                            'receiver_position': receiver.position.tolist(),
                            'sinr_values': [],
                            'sender_ids': [],
                            'distances': []
                        }
                        if not any(sender.id == receiver.id for sender, _, _ in users):
                            for sender, packet, resource in users:
                                if receiver.should_receive_packet(sender.position):
                                    
                                    # if(sender.id==20):
                                        # print("接受20的信息")
                                    # 收集干扰源
                                    # interferers = []
                                    # for other_sender, _, _ in users:
                                    #     if other_sender.id != sender.id:
                                    #         interferers.append(other_sender.position)
                                    # if sender.id == 20:
                                        # print(receiver.id,"获取",sender.id,"信息")
                                    interferers_pos = []
                                    for other_sender, _, _ in users:
                                        if other_sender.id != sender.id:
                                            interferers_pos.append(other_sender.position)
                                    # 计算SINR

                                    if self.use_simple_interference:
                                        sinr = self.sinr_calculator.calculate_sinr_simple(
                                            receiver.position, sender.position, len(interferers_pos)
                                        )
                                    else:
                                        sinr = self.sinr_calculator.calculate_sinr_optimized(
                                            receiver.position, sender.position, interferers_pos
                                        )
                                    distance = self._distance(receiver.position, sender.position)
                                    # 如果SINR高于阈值，标记为成功接收
                                    success = sinr >= self.sinr_threshold
                                    # if has_attacker and normal_users:
                                    #     print("攻击者干扰正常用户")
                                    #     print(sender.id,success)
                                    # if sender.id == 20:
                                    #     print(success)
                                    if success:
                                        msg_key = (sender.id, packet.packet_id, receiver.id)
                                        receiver_message_status[msg_key]['success'] = True
                                    else:
                                        # 记录失败原因
                                        receiver.total_faild += 1
                                        msg_key = (sender.id, packet.packet_id, receiver.id)
                                        receiver_message_status[msg_key]['failed_resources'].add(receiver.id)
                                        if sender.id in self.attackers:
                                            collision_info['collisions_caused'] += 1
                                            # 记录攻击者碰撞
                                            # if isinstance(sender, RLAttacker):
                                            #     sender.record_attack_success(False)
                                            # elif isinstance(sender, FixAttacker):
                                            #     sender.collisions_caused += 1
                                            #     self.total_attack_success += 1

                                    if isinstance(receiver, Vehicle):
                                            receiver.receive_packet(packet, resource, success)
                                    
                                    receiver_sinr_info['sinr_values'].append(sinr)
                                    receiver_sinr_info['sender_ids'].append(sender.id)
                                    receiver_sinr_info['distances'].append(distance)
                        # 只有当接收者有SINR记录时才添加
                        if receiver_sinr_info['sinr_values']:
                            resource_record['receivers'].append(receiver_sinr_info)
                    # 保存资源块记录
                    if resource_record['receivers']:
                        slot_sinr_records.append(resource_record)
                        
                    for receiver in self.attackers:
                        if not any(sender.id == receiver.id for sender, _, _ in users):
                            for sender, packet, resource in normal_users:
                                if receiver.should_receive_packet(sender.position):
                                    if isinstance(sender, (RLAttacker, FixAttacker)):
                                        pRsvp = 20
                                    else:
                                        pRsvp = 100

                                    receiver.add_sensing_data(
                                        resource.slot_id,
                                        resource.subchannel,
                                        pRsvp,
                                        sender.id,
                                        packet.timestamp
                                    )

                # 更新消息状态
                for sender, packet, resource in normal_users:
                    # 为每个预期的接收者初始化状态
                    # print(packet.expected_receivers)
                    expected_receiver_ids = []
                    for vehicle in self.vehicles + self.attackers:
                        if vehicle.id != sender.id and vehicle.should_receive_packet(sender.position):
                            expected_receiver_ids.append(vehicle.id)
                            
                    for receiver_id in expected_receiver_ids:
                        msg_key = (sender.id, packet.packet_id, receiver_id)
                        if msg_key not in receiver_message_status:
                            receiver_message_status[msg_key] = {
                                'success': False,
                                'resources_used': 0,
                                'failed_resources': set()
                            }
                        # 更新使用的资源计数
                        receiver_message_status[msg_key]['resources_used'] += 1

                        # 如果没有使用SINR，使用传统碰撞检测
                        if not self.use_sinr:
                            if has_attacker or collision_occurred:
                                receiver_message_status[msg_key]['failed_resources'].add(receiver_id)

                # 更新资源块级失效原因统计
                if not self.use_sinr:
                    if has_attacker:
                        self.resource_block_attacks += 1
                    elif collision_occurred:
                        self.resource_block_collisions += 1
            self.sinr_records.extend(slot_sinr_records)
            # 检查完成的消息
            # finished_messages = set()
            # for msg_key, status in receiver_message_status.items():
            #     sender_id, packet_id, receiver_id = msg_key
            #     if status['resources_used'] == 2:  # 两个资源块都传输完毕
            #         # 如果任何资源块失败，整个消息失败
            #         if status['failed_resources']:
            #             status['success'] = False

            #         # 更新接收者状态
            #         receiver = next((v for v in self.vehicles if v.id == receiver_id), None)
            #         if receiver:
            #             receiver.record_reception(status['success'])

            #         # 更新发送者状态
            #         sender = next((v for v in self.vehicles if v.id == sender_id), None)
            #         if sender and not isinstance(sender, (RLAttacker, FixAttacker)):
            #             if not status['success']:
            #                 sender.collisions += 1

            #         # 更新攻击者状态
            #         if not status['success']:
            #             for _, _, resource in slot_transmissions:
            #                 for attacker in [a for a in self.attackers if a.id in [s.id for s,_,_ in subchannel_usage[resource.subchannel]]]:
            #                     attacker.collisions_caused += 1
            #                     # attacker.record_attack_success(True)

            #         # 更新全局统计
            #         self.total_expected_packets += 1
            #         if status['success']:
            #             self.total_received_packets += 1
            #         else:
            #             self.message_failures += 1
            #             self.collision_count += 1

            #         finished_messages.add(msg_key)

            # # 移除已完成的消息
            # for msg_key in finished_messages:
            #     del receiver_message_status[msg_key]
            finished_messages = set()
            for msg_key, status in receiver_message_status.items():
                sender_id, packet_id, receiver_id = msg_key
                if status['resources_used'] == 2:  # 两个资源块都传输完毕
                    # 如果任何资源块失败，整个消息失败
                    if status['failed_resources']:
                        status['success'] = False

                    # 更新发送者状态 (关键修改点)
                    sender = next((v for v in self.vehicles if v.id == sender_id), None)
                    if sender:
                        # 无论成功与否都增加发送计数
                        sender.expected_receptions += 1  # 新增发送计数

                        # 仅当成功时增加发送者的成功接收计数
                        if status['success']:
                            sender.successful_receptions += 1  # 新增成功接收计数

                        # 保留原有的碰撞计数（仅限普通车辆）
                        if not isinstance(sender, (RLAttacker, FixAttacker)) and not status['success']:
                            sender.collisions += 1

                    # 更新接收者状态 (关键修改点：仅在失败时记录)
                    # if not status['success']:
                    receiver = next((v for v in self.vehicles if v.id == receiver_id), None)
                    if receiver:
                        receiver.record_reception(status['success'])  # 只记录失败

                    # 更新攻击者状态（保留原逻辑）
                    if not status['success']:
                        for _, _, resource in slot_transmissions:
                            for attacker in [a for a in self.attackers if a.id in [s.id for s,_,_ in subchannel_usage[resource.subchannel]]]:
                                attacker.collisions_caused += 1
                                self.total_attack_success += 1

                    # 更新全局统计（保留原逻辑）
                    self.total_expected_packets += 1
                    if status['success']:
                        self.total_received_packets += 1
                    else:
                        self.message_failures += 1
                        self.collision_count += 1

                    finished_messages.add(msg_key)

            # 移除已完成的消息
            for msg_key in finished_messages:
                del receiver_message_status[msg_key]
        return collision_info
    
    
    def _calculate_sinr(self, receiver_pos, sender_pos, interferers_pos):
        """
        计算SINR
        :param receiver_pos: 接收者位置
        :param sender_pos: 目标发送者位置
        :param interferers_pos: 干扰源位置列表
        :return: SINR (dB)
        """
        # 计算目标信号接收功率
        distance = np.linalg.norm(receiver_pos - sender_pos)
        path_loss = self._calculate_path_loss(distance)
        rx_power = self.tx_power - path_loss  # dBm
        
        # 计算干扰信号总功率
        interference_power = 0.0  # 毫瓦
        for intf_pos in interferers_pos:
            intf_distance = np.linalg.norm(receiver_pos - intf_pos)
            intf_path_loss = self._calculate_path_loss(intf_distance)
            intf_rx_power = self.tx_power - intf_path_loss  # dBm
            # 转换为毫瓦并累加
            interference_power += 10 ** (intf_rx_power / 10)
        
        # 添加噪声功率 (转换为毫瓦)
        noise_mw = 10 ** (self.noise_power / 10)
        total_interference = interference_power + noise_mw
        
        # 计算SINR (dB)
        signal_mw = 10 ** (rx_power / 10)
        sinr_linear = signal_mw / total_interference
        return 10 * np.log10(sinr_linear) if sinr_linear > 0 else -100
    
    def _calculate_path_loss(self, distance):
        """更符合V2X场景的路径损耗模型 - 修正版本"""
        # 使用更准确的3GPP V2V路径损耗模型
        # 参考: 3GPP TR 36.885
        if distance < 3:  # 最小距离保护
            distance = 3
            
        # LOS（视距）路径损耗
        return 38.77 + 16.7 * np.log10(distance) + 18.2 * np.log10(max(distance, 1))
   
    
    def _calculate_current_prr(self):
        """计算当前分组接收率(PRR)"""
        if self.total_expected_packets > 0:
            return self.total_received_packets / self.total_expected_packets
        return 0.0
    
    def get_vehicle_prrs(self):
        """获取所有车辆的个人PRR"""
        vehicle_prrs = {}
        for vehicle in self.vehicles:
            # print(vehicle.id,vehicle.packets_received,vehicle.packets_received_succeed,vehicle.total_faild)
            vehicle_prrs[vehicle.id] = vehicle.calculate_prr()
        return vehicle_prrs
    
    def get_episode_stats(self):
        """获取当前轮的统计信息"""
        vehicle_prrs = self.get_vehicle_prrs()
        
        return {
            'total_collisions': self.collision_count,
            'total_transmissions': self.transmission_count,
            'prr': self._calculate_current_prr(),
            'attack_success_rate': self.total_attack_success / max(1, self.attack_transmission_count),
            'collision_rate': self.collision_count / max(1, self.transmission_count),
            'message_failures': self.message_failures,
            'resource_block_attacks': self.resource_block_attacks,
            'resource_block_collisions': self.resource_block_collisions,
            'vehicle_prrs': vehicle_prrs
        }
    
    def reset_stats(self):
        """重置所有统计"""
        self.collision_count = 0
        self.transmission_count = 0
        self.total_expected_packets = 0
        self.total_received_packets = 0
        self.attack_transmission_count = 0
        self.total_attack_success = 0
        self.message_failures = 0
        self.resource_block_attacks = 0
        self.resource_block_collisions = 0
    
    def _initialize_vehicles(self):
        """初始化车辆位置和速度，并保存初始状态"""
        lane1_y = 5.0
        lane2_y = 10.0
        highway_length = 1000.0
        self.vehicles = []

        vehicle_states = []

        for i in range(self.num_vehicles):
            lane_y = lane1_y if i % 2 == 0 else lane2_y
            pos_x = random.uniform(0, highway_length)
            position = np.array([pos_x, lane_y])
            velocity = np.array([10.0 if lane_y == lane1_y else -10.0, 0.0])
            vehicle = Vehicle(i, position.copy(), velocity.copy(), self, self.vehicle_resource_mode)
            self.vehicles.append(vehicle)
            vehicle_states.append((position.copy(), velocity.copy()))

        if self.initial_vehicle_states is None:
            self.initial_vehicle_states = vehicle_states

    def _initialize_attackers(self):
        """初始化攻击者并保存初始状态"""
        highway_length = 1000.0
        self.attackers = []

        attacker_states = []

        for i in range(self.num_attackers):
            attacker_id = self.num_vehicles + i
            position = np.array([highway_length/2, 0])
            velocity = np.array([0.0, 0.0])
            
            if self.attacker_type == 'RL':
                attacker = RLAttacker(attacker_id, position.copy(), velocity.copy(), self)
            else:
                attacker = FixAttacker(
                    attacker_id, 
                    position.copy(), 
                    velocity.copy(), 
                    self,
                    attack_cycle=self.fix_attacker_params['cycle'],
                    num_subchannels=self.fix_attacker_params['num_subchannels']
                )
                
            self.attackers.append(attacker)
            attacker_states.append((position.copy(), velocity.copy()))

        if self.initial_attacker_states is None:
            self.initial_attacker_states = attacker_states
    
    def close(self):
        """关闭环境，清理资源"""
        if hasattr(self, 'fig'):
            plt.ioff()
            plt.close()

# 使用示例和测试函数
def test_sinr_optimization():
    """测试SINR优化效果"""
    print("=== SINR优化测试 ===")
    
    # 创建环境，使用优化的SINR计算
    env_optimized = V2XRLEnvironment(
        num_vehicles=10,
        num_attackers=1,
        episode_duration=5000,
        use_sinr=True,
        sinr_threshold=5.0,  # 优化后的阈值
        use_simple_interference=False,  # 使用完整SINR计算
        attacker_type='Fix'
    )
    
    # 创建对比环境，使用简化干扰计算
    env_simple = V2XRLEnvironment(
        num_vehicles=10,
        num_attackers=1,
        episode_duration=5000,
        use_sinr=True,
        sinr_threshold=5.0,
        use_simple_interference=True,  # 使用简化干扰计算
        attacker_type='Fix'
    )
    
    # 测试优化版本
    print("\n测试优化的SINR计算...")
    state = env_optimized.reset()
    for step in range(100):
        action = env_optimized.action_space.sample()
        state, reward, done, info = env_optimized.step(action)
        if done:
            break
    
    optimized_prr = info['prr']
    print(f"优化版本PRR: {optimized_prr:.4f}")
    
    # 测试简化版本
    print("\n测试简化的SINR计算...")
    state = env_simple.reset()
    for step in range(100):
        action = env_simple.action_space.sample()
        state, reward, done, info = env_simple.step(action)
        if done:
            break
    
    simple_prr = info['prr']
    print(f"简化版本PRR: {simple_prr:.4f}")
    
    print(f"\n性能提升: {((optimized_prr - simple_prr) / simple_prr * 100):.2f}%")
    
    # 分析最佳SINR阈值
    print("\n=== 分析最佳SINR阈值 ===")
    results, best_threshold = env_optimized.analyze_sinr_threshold(test_episodes=20)
    
    print("\n不同阈值下的性能:")
    for threshold, result in results.items():
        print(f"阈值 {threshold:2d} dB: PRR = {result['avg_prr']:.4f} ± {result['std_prr']:.4f}")
    
    print(f"\n推荐阈值: {best_threshold} dB")
    
    env_optimized.close()
    env_simple.close()

# if __name__ == "__main__":
#     test_sinr_optimization()