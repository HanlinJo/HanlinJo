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
logger = logging.getLogger('V2X-RL-Environment-SINR')

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

class ResourceBlockReception:
    """资源块接收状态跟踪"""
    def __init__(self, sender_id, packet_id, resource_block_id):
        self.sender_id = sender_id
        self.packet_id = packet_id
        self.resource_block_id = resource_block_id  # 0或1（第一个或第二个资源块）
        self.received_count = 0  # 成功接收的接收者数量
        self.total_receivers = 0  # 总接收者数量
        self.success = False  # 该资源块是否成功接收

class SINRRecord:
    """SINR记录结构"""
    def __init__(self, time, resource_block, sender_info, receiver_info):
        self.time = time
        self.resource_block = resource_block  # (slot_id, subchannel)
        self.sender_info = sender_info  # {'id': sender_id, 'position': position, 'type': 'vehicle'/'attacker'}
        self.receiver_info = receiver_info  # [{'id': receiver_id, 'position': position, 'distance': distance, 'sinr': sinr}]

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
        
        # 初始化统计
        self.packets_sent = 0
        self.packets_received = 0
        self.collisions = 0
        self.successful_transmissions = 0
        self.prr = 0.0
        self.expected_receptions = 0
        self.successful_receptions = 0
    
    def reset(self):
        """重置车辆状态"""
        self.resel_counter = 0
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
    
    def record_reception(self, success, expected_receivers):
        """记录数据包接收情况（车辆级别）"""
        self.expected_receptions += expected_receivers
        if success:
            self.successful_receptions += expected_receivers
    
    def calculate_prr(self):
        """计算个人PRR"""
        if self.expected_receptions > 0:
            return self.successful_receptions / self.expected_receptions
        return 0.0
    
    def receive_packet(self, packet, resource, collision_occurred):
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
        
        if not packet.is_attack and not collision_occurred:
            self.packets_received += 1
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
        self.sensing_window_duration = 100
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
    
    def send_attack_packet_with_action(self, current_time, action):
        """使用RL动作发送攻击数据包"""
        if TARGETED_ATTACK_MODE:
            return self._send_targeted_attack(current_time)
        else:
            return self._send_exhaustion_attack(current_time, action)
    
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
    
    def _update_target_resources(self):
        """更新目标车辆的资源选择信息"""
        if self.target_vehicle_id == -1:
            return
        
        self.targeted_resources = []
        
        for data in self.sensing_data:
            if data.sender_id == self.target_vehicle_id:
                resource = ResourceInfo(data.slot_id, data.subchannel)
                self.targeted_resources.append(resource)
        
        unique_resources = {}
        for resource in self.targeted_resources:
            unique_resources[(resource.slot_id, resource.subchannel)] = resource
        self.targeted_resources = list(unique_resources.values())
        
        if len(self.targeted_resources) > 5:
            self.targeted_resources = self.targeted_resources[-5:]
    
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
        
        if TARGETED_ATTACK_MODE:
            self._update_target_resources()

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
        self.sensing_window_duration = 100
        self.prob_resource_keep = 0.2
        
        self.cycle_groups = self._calculate_cycle_groups() if not TARGETED_ATTACK_MODE else []
        
        self.target_vehicle_id = 0
        self.target_vehicle_resources = []
        self.target_vehicle_tracking_time = 0
        self.targeted_resources = []
    
    def reset(self):
        """重置攻击者状态"""
        super().reset()
        self.next_attack_time = 0
        self.attack_packets_sent = 0
        self.collisions_caused = 0
        self.sensing_data = []
        self.cycle_groups = self._calculate_cycle_groups() if not TARGETED_ATTACK_MODE else []
        
        self.target_vehicle_id = -1
        self.target_vehicle_resources = []
        self.target_vehicle_tracking_time = 0
        self.targeted_resources = []
        
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
    
    def _send_targeted_attack(self, current_time):
        """执行目标侧链攻击模式"""
        self._update_target_resources()
        
        if not self.targeted_resources:
            return None
        
        current_slot = current_time % self.num_slots
        transmissions = []
        
        for resource in self.targeted_resources:
            if resource.slot_id == current_slot:
                packet = Packet(self.id, current_time, self.position, self.current_packet_id, is_attack=True)
                transmissions.append((packet, resource))
                self.attack_packets_sent += 1
        
        return transmissions
    
    def _update_target_resources(self):
        """更新目标车辆的资源选择信息"""
        if self.target_vehicle_id == -1:
            return
        
        self.targeted_resources = []
        
        latest_timestamp = 0
        latest_resources = []
        
        for data in self.sensing_data:
            if data.sender_id == self.target_vehicle_id:
                resource = ResourceInfo(data.slot_id, data.subchannel)
                
                if data.timestamp > latest_timestamp:
                    latest_timestamp = data.timestamp
                    latest_resources = [resource]
                elif data.timestamp == latest_timestamp:
                    latest_resources.append(resource)
        
        self.targeted_resources = latest_resources
    
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
    """V2X攻击优化的RL环境 - 支持SINR碰撞检测"""
    
    def __init__(self, num_vehicles=20, num_attackers=1, episode_duration=20000, 
                 communication_range=320.0, vehicle_resource_mode='Separate',
                 attacker_type='RL', fix_attacker_params=None, render_mode='human',
                 num_slots=100, num_subchannels=5, use_sinr=True, sinr_threshold=10.0,
                 tx_power=23.0, noise_power=-95.0, path_loss_exponent=3.8):
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
        
        # SINR相关参数
        self.use_sinr = use_sinr
        self.sinr_threshold = sinr_threshold  # SINR阈值 (dB)
        self.tx_power = tx_power  # 发射功率 (dBm)
        self.noise_power = noise_power  # 噪声功率 (dBm)
        self.path_loss_exponent = path_loss_exponent  # 路径损耗指数
        self.d0 = 1.0  # 参考距离 (m)
        self.pl0 = 32.45 + 20 * np.log10(5900)  # 参考路径损耗 (dB)
        
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
        
        self.action_space = spaces.Discrete(20)
        
        # 初始化模拟组件
        self.vehicles = []
        self.attackers = []
        self.current_time = 0
        self.message_status_dict = {}
        
        # SINR记录存储
        self.sinr_records = []
        
        # 资源选择可视化
        if render_mode == 'human':
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(15, 8))
            self.resource_grid = [[set() for _ in range(num_subchannels)] for _ in range(num_slots)]
        
        # 统计
        self.reset_stats()
    
    def _calculate_path_loss(self, distance):
        """计算路径损耗 (dB)"""
        if distance < self.d0:
            distance = self.d0
        return self.pl0 + 10 * self.path_loss_exponent * np.log10(distance / self.d0)
    
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
    
    def get_resource_block_sinr_records(self):
        """获取资源块SINR记录"""
        return self.sinr_records
    
    def reset(self):
        """重置环境开始新的一轮"""
        self.current_time = 0
        self.recent_collision_queue.clear()
        self.recent_collision_rate = 0.0
        self.message_status_dict = {}
        self.sinr_records = []  # 清空SINR记录
        
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
                attacker.sent_resources_count = 0
                attacker.current_packet_id += 1
                attacker.has_transmitted = False
        
        # 模拟20ms (20个1ms的时间步)
        for step in range(20):
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
            'resource_block_collisions': self.resource_block_collisions
        }
        
        return next_state.astype(np.float32), episode_reward, done, info
    
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
                for attack_packet, resource in attack_result:
                    attack_transmissions.append((attacker, attack_packet, resource))
                self.attack_transmission_count += 1    
            
        all_transmissions = transmissions + attack_transmissions
        reward = 0.0
        step_collisions = 0
        
        if all_transmissions:
            collision_info = self._handle_transmissions_with_sinr(all_transmissions)
            step_collisions = collision_info.get('collisions_caused', 0)
            
            if attacker_sent and self.attacker_type == 'RL':
                collision_occurred = step_collisions > 0
                
                for attacker in self.attackers:
                    if isinstance(attacker, RLAttacker) and attacker_sent:
                        reward = attacker.calculate_reward(collision_occurred, step_collisions, self.current_time, self.collision_count)
        
        return reward, step_collisions
    
    def _handle_transmissions_with_sinr(self, transmissions):
        """处理传输并使用SINR进行碰撞检测"""
        # 按时隙分组
        tx_by_slot = defaultdict(list)
        for sender, packet, resource in transmissions:
            tx_by_slot[resource.slot_id].append((sender, packet, resource))

        collision_info = {'collisions_caused': 0}
        
        # 处理每个时隙的传输
        for slot_id, slot_transmissions in tx_by_slot.items():
            # 检测碰撞：记录每个子信道的使用情况
            subchannel_usage = defaultdict(list)
            for sender, packet, resource in slot_transmissions:
                subchannel_usage[resource.subchannel].append((sender, packet, resource))

            # 消息状态跟踪
            resource_block_status = defaultdict(lambda: {'attack': False, 'collision': False, 'failed_receivers': set()})
            
            # 处理每个子信道
            for subchannel, users in subchannel_usage.items():
                # 检查是否有攻击者参与
                has_attacker = any(isinstance(sender, (RLAttacker, FixAttacker)) for sender, _, _ in users)
                normal_users = [user for user in users if not isinstance(user[0], (RLAttacker, FixAttacker))]
                
                # 传统碰撞检测：多个发送者使用同一资源块
                collision_occurred = len(normal_users) > 1
                
                # 更新资源块状态
                if has_attacker:
                    resource_block_status[subchannel]['attack'] = True
                if collision_occurred:
                    resource_block_status[subchannel]['collision'] = True
                
                # SINR-based接收检测
                if self.use_sinr and len(users) > 1:
                    # 对每个接收者计算SINR
                    for receiver in self.vehicles:
                        if not any(sender.id == receiver.id for sender, _, _ in users):  # 不是发送者
                            for sender, packet, resource in normal_users:
                                if receiver.should_receive_packet(sender.position):
                                    # 收集干扰源
                                    interferers = []
                                    for other_sender, _, _ in users:
                                        if other_sender.id != sender.id:
                                            interferers.append(other_sender.position)
                                    
                                    # 计算SINR
                                    sinr = self._calculate_sinr(receiver.position, sender.position, interferers)
                                    
                                    # 记录SINR信息
                                    sender_info = {
                                        'id': sender.id,
                                        'position': sender.position.copy(),
                                        'type': 'attacker' if isinstance(sender, (RLAttacker, FixAttacker)) else 'vehicle'
                                    }
                                    
                                    receiver_info = [{
                                        'id': receiver.id,
                                        'position': receiver.position.copy(),
                                        'distance': np.linalg.norm(receiver.position - sender.position),
                                        'sinr': sinr
                                    }]
                                    
                                    sinr_record = SINRRecord(
                                        time=self.current_time,
                                        resource_block=(slot_id, subchannel),
                                        sender_info=sender_info,
                                        receiver_info=receiver_info
                                    )
                                    self.sinr_records.append(sinr_record)
                                    
                                    # 如果SINR低于阈值，标记为失败
                                    if sinr < self.sinr_threshold:
                                        resource_block_status[subchannel]['failed_receivers'].add(receiver.id)
                
                # 处理消息状态
                for sender, packet, resource in normal_users:
                    msg_key = (sender.id, packet.packet_id)
                    if msg_key not in self.message_status_dict:
                        self.message_status_dict[msg_key] = {
                            'resources': 0,
                            'success': True,
                            'expected_receivers': packet.expected_receivers,
                            'failed_resource_blocks': set()
                        }
                    
                    # 更新消息的资源块计数
                    self.message_status_dict[msg_key]['resources'] += 1
                    
                    # 检查该资源块是否失败
                    resource_block_failed = False
                    if self.use_sinr:
                        # 基于SINR判断
                        if resource_block_status[subchannel]['failed_receivers']:
                            resource_block_failed = True
                            self.message_status_dict[msg_key]['failed_resource_blocks'].add(len(self.message_status_dict[msg_key]['failed_resource_blocks']))
                    else:
                        # 传统碰撞检测
                        if has_attacker or collision_occurred:
                            resource_block_failed = True
                            self.message_status_dict[msg_key]['failed_resource_blocks'].add(len(self.message_status_dict[msg_key]['failed_resource_blocks']))
                    
                    # 如果任何资源块失败，整个消息失败
                    if resource_block_failed:
                        self.message_status_dict[msg_key]['success'] = False
                
                # 更新资源块级失效原因统计
                if has_attacker:
                    self.resource_block_attacks += 1
                elif collision_occurred:
                    self.resource_block_collisions += 1
            
            # 检查完成的消息
            finished_msgs = []
            for msg_key, status in self.message_status_dict.items():
                if status['resources'] == 2:  # 两个资源块都传输完毕
                    self.total_expected_packets += 1
                    sender_id, packet_id = msg_key

                    # 更新车辆级别PRR统计
                    sender_vehicle = next((v for v in self.vehicles if v.id == sender_id), None)
                    if sender_vehicle:
                        sender_vehicle.record_reception(status['success'], status['expected_receivers'])

                    if status['success']:
                        self.total_received_packets += 1
                    else:
                        self.message_failures += 1
                    finished_msgs.append(msg_key)
                    
            for msg_key in finished_msgs:
                del self.message_status_dict[msg_key]
            
            # 处理接收
            for receiver in self.vehicles + self.attackers:
                for sender, packet, resource in slot_transmissions:
                    if sender.id == receiver.id:
                        continue
                    
                    if not receiver.should_receive_packet(sender.position):
                        continue
                    
                    # 判断是否发生碰撞
                    collision_occurred = resource_block_status[resource.subchannel]['attack'] or \
                                        resource_block_status[resource.subchannel]['collision']
                    
                    # 如果使用SINR，还要检查接收者是否在失败列表中
                    if self.use_sinr and receiver.id in resource_block_status[resource.subchannel]['failed_receivers']:
                        collision_occurred = True
                    
                    # 处理接收
                    if isinstance(receiver, Vehicle):
                        receiver.receive_packet(packet, resource, collision_occurred)
                    else:
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
            
            # 更新碰撞统计
            normal_senders = set()
            attack_success = set()
            for sender, packet, resource in slot_transmissions:
                if not isinstance(sender, (RLAttacker, FixAttacker)):
                    normal_senders.add(sender.id)
            
            # 碰撞检测
            for subchannel, users in subchannel_usage.items():
                if len(users) > 1:  # 发生碰撞
                    has_attacker = any(isinstance(sender, (RLAttacker, FixAttacker)) for sender, _, _ in users)
                    if has_attacker:
                        self.total_attack_success += len(normal_senders)
                        collision_info['collisions_caused'] += len(normal_senders)
                        for sender_id in normal_senders:
                            attack_success.add(sender_id)
                    self.collision_count += len(normal_senders)
            
            # 更新发送者碰撞计数
            for sender_id in normal_senders:
                for vehicle in self.vehicles:
                    if vehicle.id == sender_id:
                        vehicle.collisions += 1
                        break
                if sender_id in attack_success:
                    for attacker in self.attackers:
                        if isinstance(attacker, FixAttacker):
                            attacker.collisions_caused += 1
                        elif isinstance(attacker, RLAttacker):
                            attacker.record_attack_success(True)
        
        return collision_info
    
    def _calculate_current_prr(self):
        """计算当前分组接收率(PRR)"""
        if self.total_expected_packets > 0:
            return self.total_received_packets / self.total_expected_packets
        return 0.0
    
    def get_vehicle_prrs(self):
        """获取所有车辆的个人PRR"""
        vehicle_prrs = {}
        for vehicle in self.vehicles:
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
    
    def set_attacker_type(self, attacker_type, fix_params=None):
        """设置攻击者类型和参数"""
        self.attacker_type = attacker_type
        if fix_params:
            self.fix_attacker_params = fix_params
    
    def toggle_attack_mode(self):
        """切换攻击模式"""
        global TARGETED_ATTACK_MODE
        TARGETED_ATTACK_MODE = not TARGETED_ATTACK_MODE
        mode_name = "目标侧链攻击" if TARGETED_ATTACK_MODE else "侧链资源耗尽"
        logger.info(f"攻击模式已切换为: {mode_name}")
        return TARGETED_ATTACK_MODE
    
    def close(self):
        """关闭环境，清理资源"""
        if hasattr(self, 'fig'):
            plt.ioff()
            plt.close()

# 使用示例
if __name__ == "__main__":
    # 创建环境，启用SINR检测
    env = V2XRLEnvironment(
        num_vehicles=10,
        num_attackers=1,
        episode_duration=5000,
        use_sinr=True,
        sinr_threshold=10.0,
        attacker_type='Fix'
    )
    
    # 运行一个简单的测试
    state = env.reset()
    
    for step in range(100):
        action = env.action_space.sample()  # 随机动作
        state, reward, done, info = env.step(action)
        
        if step % 20 == 0:
            print(f"Step {step}: Reward={reward:.3f}, PRR={info['prr']:.3f}")
        
        if done:
            break
    
    # 获取SINR记录
    sinr_records = env.get_resource_block_sinr_records()
    print(f"\n总共记录了 {len(sinr_records)} 条SINR记录")
    
    if sinr_records:
        # 显示前几条记录
        for i, record in enumerate(sinr_records[:5]):
            print(f"记录 {i+1}:")
            print(f"  时间: {record.time}")
            print(f"  资源块: {record.resource_block}")
            print(f"  发送者: ID={record.sender_info['id']}, 类型={record.sender_info['type']}")
            for j, receiver in enumerate(record.receiver_info):
                print(f"  接收者{j+1}: ID={receiver['id']}, 距离={receiver['distance']:.1f}m, SINR={receiver['sinr']:.1f}dB")
            print()
    
    env.close()