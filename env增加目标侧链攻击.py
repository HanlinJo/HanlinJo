# import numpy as np
# # import matplotlib.pyplot as plt
# import random
# import time
# from collections import defaultdict, deque
# import logging
# import gym
# from gym import spaces
# # import matplotlib
# # import os

# import matplotlib
# # matplotlib.use('TkAgg')  

# import matplotlib.pyplot as plt
# # plt.switch_backend('TkAgg')
# # plt.interactive(False)
# # 设置日志
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger('V2X-RL-Environment')

# # 全局攻击模式切换
# TARGETED_ATTACK_MODE = False  # 默认是侧链资源耗尽模式

# class Packet:
#     """表示用于传输的V2X数据包"""
    
#     def __init__(self, sender_id, timestamp, position, packet_id, size=190, is_attack=False):
#         self.sender_id = sender_id
#         self.timestamp = timestamp
#         self.position = position
#         self.size = size
#         self.is_attack = is_attack
#         self.packet_id = packet_id  # 唯一标识一个完整的数据包

# class SensingData:
#     """表示感知数据"""
#     def __init__(self, slot_id, subchannel, pRsvp, sender_id, timestamp):
#         self.slot_id = slot_id          # 0-99
#         self.subchannel = subchannel    # 0-4
#         self.pRsvp = pRsvp              # 预留周期
#         self.sender_id = sender_id      # 发送者ID
#         self.timestamp = timestamp      # 时间戳

# class ResourceInfo:
#     """表示资源块 (时隙+子信道)"""
#     def __init__(self, slot_id, subchannel):
#         self.slot_id = slot_id        # 0-99
#         self.subchannel = subchannel  # 0-4
    
#     def __eq__(self, other):
#         if not isinstance(other, ResourceInfo):
#             return False
#         return (self.slot_id == other.slot_id and 
#                 self.subchannel == other.subchannel)
    
#     def __repr__(self):
#         return f"(slot:{self.slot_id}, subchannel:{self.subchannel})"

# class Vehicle:
#     """表示具有V2X功能的车辆"""
    
#     def __init__(self, vehicle_id, initial_position, initial_velocity, sim, resource_selection_mode='Separate'):
#         self.id = vehicle_id
#         self.position = initial_position
#         self.velocity = initial_velocity
#         self.sim = sim
#         self.resource_selection_mode = resource_selection_mode  # 资源选择模式: 'Separate' 或 'Combine'
        
#         # 获取资源池参数
#         self.num_slots = sim.resource_pool.num_slots
#         self.num_subchannels = sim.resource_pool.num_subchannels
        
#         # 资源选择参数
#         self.resel_counter = 0
#         self.prob_resource_keep = random.uniform(0.2, 0.8)
#         self.current_resources = None  # 存储两个资源块
#         self.sensing_data = []
#         self.next_transmission_time = 0
#         self.sent_resources_count = 0  # 已发送的资源块计数
#         self.current_packet_id = 0     # 当前数据包ID
        
#         # 感知窗口参数
#         self.sensing_window_duration = 1000  # 100ms感知窗口
#         self.has_transmitted = False  # 标记是否已发送数据包
#         # 初始化统计
#         self.packets_sent = 0
#         self.packets_received = 0
#         self.collisions = 0
#         self.successful_transmissions = 0
    
#     def reset(self):
#         """重置车辆状态"""
#         self.resel_counter = 0
#         self.current_resources = None
#         self.sensing_data = []  # 清空感知数据
#         self.next_transmission_time = 0
#         self.sent_resources_count = 0
#         self.current_packet_id = 0
#         self.packets_sent = 0
#         self.packets_received = 0
#         self.collisions = 0
#         self.successful_transmissions = 0
        
#     def move(self, delta_time):
#         """基于速度和时间增量更新车辆位置"""
#         self.position = self.position + self.velocity * delta_time
        
#         # 处理边界条件（反射）
#         if self.position[0] >= 1000:
#             self.position[0] = 1000 - (self.position[0] - 1000)
#             self.velocity = -self.velocity
#             self.position[1] = 10.0
#         if self.position[0] <= 0:
#             self.position[0] = -self.position[0]
#             self.velocity = -self.velocity
#             self.position[1] = 5.0
            
#     def get_sensed_resource_occupancy(self):
#         """获取监听窗中的资源占用状态矩阵"""
#         occupancy = np.zeros((self.num_slots, self.num_subchannels), dtype=int)
        
#         for data in self.sensing_data:
#             slot = data.slot_id % self.num_slots
#             if 0 <= slot < self.num_slots and 0 <= data.subchannel < self.num_subchannels:
#                 occupancy[slot, data.subchannel] = 1
                
#         return occupancy  
    
#     def select_future_resource(self, current_time):
#         """选择未来资源 - 根据模式选择资源块"""
#         self._update_sensing_window(current_time)
#         selection_window = self._create_selection_window(current_time)
        
#         # 创建已占用资源集合
#         occupied_resources = set()
#         for data in self.sensing_data:
#             resource_key = (data.slot_id, data.subchannel)
#             occupied_resources.add(resource_key)
        
#         # 根据模式选择资源
#         if self.resource_selection_mode == 'Combine':
#             selected_resources = self._select_combined_resources(selection_window, occupied_resources)
#         else:  # Separate模式
#             selected_resources = self._select_separate_resources(selection_window, occupied_resources)
        
#         self.resel_counter = random.randint(5, 15)
#         return selected_resources

#     def _select_separate_resources(self, selection_window, occupied_resources):
#         """Separate模式：选择两个独立的资源块"""
#         # 创建候选资源列表（排除已占用资源）
#         candidate_resources = []
#         for resource in selection_window:
#             resource_key = (resource.slot_id, resource.subchannel)
#             if resource_key not in occupied_resources:
#                 candidate_resources.append(resource)
        
#         # 如果候选资源不足选择窗的20%，则从整个选择窗中随机选择
#         min_candidates = max(1, int(0.2 * len(selection_window)))
#         if len(candidate_resources) < min_candidates:
#             candidate_resources = selection_window[:]  # 使用整个选择窗
        
#         # 从候选资源中随机选择两个不同的资源块
#         selected_resources = []
#         if len(candidate_resources) >= 2:
#             selected = random.sample(candidate_resources, 2)
#             selected_resources = selected
#         elif len(candidate_resources) == 1:
#             selected_resources = [candidate_resources[0], random.choice(selection_window)]
#         else:
#             # 如果没有候选资源，随机创建两个资源
#             slot1 = random.randint(0, self.num_slots-1)
#             subchannel1 = random.randint(0, self.num_subchannels-1)
#             slot2 = random.randint(0, self.num_slots-1)
#             subchannel2 = random.randint(0, self.num_subchannels-1)
#             selected_resources = [ResourceInfo(slot1, subchannel1), ResourceInfo(slot2, subchannel2)]
        
#         return selected_resources

#     def _select_combined_resources(self, selection_window, occupied_resources):
#         """Combine模式：选择同一时隙的两个相邻子信道"""
#         # 按时隙分组资源
#         slot_resources = defaultdict(list)
#         for resource in selection_window:
#             slot_resources[resource.slot_id].append(resource)
        
#         # 收集所有有空闲相邻子信道的时隙
#         valid_slots = []
#         for slot_id, resources in slot_resources.items():
#             # 获取该时隙所有空闲子信道
#             free_subchannels = [r.subchannel for r in resources 
#                                if (slot_id, r.subchannel) not in occupied_resources]
            
#             # 检查是否有相邻的子信道对可用
#             adjacent_pairs = []
#             for i in range(self.num_subchannels - 1):  # 确保i+1不越界
#                 if i in free_subchannels and (i+1) in free_subchannels:
#                     adjacent_pairs.append((i, i+1))
            
#             if adjacent_pairs:
#                 valid_slots.append((slot_id, adjacent_pairs))
        
#         # 如果有可用的时隙和相邻子信道对
#         if valid_slots:
#             # 随机选择一个时隙
#             slot_id, adjacent_pairs = random.choice(valid_slots)
#             # 随机选择一对相邻子信道
#             sc1, sc2 = random.choice(adjacent_pairs)
#             return [ResourceInfo(slot_id, sc1), ResourceInfo(slot_id, sc2)]
        
#         if slot_resources:
#             # 随机选择一个时隙
#             slot_id = random.choice(list(slot_resources.keys()))
#             # 在该时隙中随机选择一对相邻子信道
#             sc_pair = random.choice([(i, i+1) for i in range(self.num_subchannels - 1)])
#             sc1, sc2 = sc_pair

#             # 创建资源对象（即使子信道可能被占用）
#             return [ResourceInfo(slot_id, sc1), ResourceInfo(slot_id, sc2)]
        
#         return []
    
#     def _create_selection_window(self, current_time):
#         """创建选择窗口 (T1=4到T2=100)"""
#         selection_window = []
#         current_slot = current_time % self.num_slots  # 当前时隙
#         start_slot = (current_slot + 4) % self.num_slots
#         end_slot = (current_slot + 100) % self.num_slots
        
#         # 考虑周期性，处理跨周期的情况
#         if start_slot < end_slot:
#             slots = range(start_slot, end_slot)
#         else:
#             slots = list(range(start_slot, self.num_slots)) + list(range(0, end_slot))
        
#         # 为每个时隙创建所有可能的子信道组合
#         for slot in slots:
#             for subchannel in range(self.num_subchannels):
#                 selection_window.append(ResourceInfo(slot, subchannel))
        
#         return selection_window

#     def _update_sensing_window(self, current_time):
#         """通过移除旧条目更新感知窗口"""
#         sensing_window_start = current_time - self.sensing_window_duration
        
#         # 移除感知窗口外的数据
#         self.sensing_data = [data for data in self.sensing_data 
#                             if data.timestamp >= sensing_window_start]
    
#     def add_sensing_data(self, slot_id, subchannel, pRsvp, sender_id, timestamp):
#         """添加感知数据"""
#         sensing_data = SensingData(
#             slot_id=slot_id,
#             subchannel=subchannel,
#             pRsvp=pRsvp,
#             sender_id=sender_id,
#             timestamp=timestamp
#         )
#         self.sensing_data.append(sensing_data)
    
#     def handle_periodic_resource_reselection(self, current_time):
#         """在周期开始时处理资源重选"""
#         self.prob_resource_keep = random.uniform(0.2, 0.8)  # 每次重选时更新概率
#         if current_time % self.num_slots == 0:  # 每周期开始时
#             # 检查资源重选计数器
#             if self.resel_counter <= 0:
#                 # 依概率决定是否保留资源
#                 if random.random() < self.prob_resource_keep:
#                     # 保留当前资源，重置计数器
#                     self.resel_counter = random.randint(5, 15)
#                 else:
#                     # 不保留资源，重置资源
#                     self.current_resources = None
#                     self.resel_counter = random.randint(5, 15)
            
#             # 如果当前没有资源，选择新资源
#             if self.current_resources is None:
#                 self.current_resources = self.select_future_resource(current_time)
#                 self.sent_resources_count = 0
#                 self.current_packet_id += 1
    
#     def send_packet(self, current_time):
#         """使用选定的资源发送数据包（现在使用两个资源块）"""
#         # 如果当前没有资源，直接返回
#         if self.current_resources is None:
#             return None
        
#         current_slot = current_time % self.num_slots
        
#         # 检查是否有资源块需要在当前时隙发送
#         resources_to_send = []
#         for resource in self.current_resources:
#             if resource.slot_id == current_slot:
#                 resources_to_send.append(resource)
        
#         if not resources_to_send:
#             return None
        
#         # 创建数据包
#         packet = Packet(self.id, current_time, self.position, self.current_packet_id)
        
#         # 为每个要发送的资源块创建传输记录
#         transmissions = []
#         for resource in resources_to_send:
#             transmissions.append((packet, resource))
        
#         # 更新已发送资源块计数
#         self.sent_resources_count += len(resources_to_send)
        
#         # 更新统计
#         self.packets_sent += len(resources_to_send)
        
#         # 如果两个资源块都已发送，重置状态
#         if self.sent_resources_count >= 2:
#             self.has_transmitted = True
#             self.resel_counter -= 1
#             self.sent_resources_count = 0
        
#         return transmissions
    
#     def receive_packet(self, packet, resource, collision_occurred):
#         """处理接收到的数据包 - 修改为始终添加感知数据"""
#         # 无论数据包是否来自攻击者，都添加到感知数据
#         if hasattr(packet, 'is_attack') and packet.is_attack:
#             pRsvp = 100  # 攻击者的发送周期
#         else:
#             pRsvp = 100  # 普通车辆的发送周期
        
#         # 添加感知数据
#         self.add_sensing_data(
#             resource.slot_id,
#             resource.subchannel,
#             pRsvp,
#             packet.sender_id,
#             packet.timestamp
#         )
        
#         # 处理接收（仅对非攻击者数据包进行统计）
#         if not packet.is_attack and not collision_occurred:
#             self.packets_received += 1
#             return True
#         return False
    
#     def should_receive_packet(self, sender_position):
#         """确定该车辆是否应接收来自发送者的数据包"""
#         distance = np.linalg.norm(self.position - sender_position)
#         return distance <= self.sim.communication_range

# class RLAttacker:
#     """基于RL的攻击者，支持两种攻击模式：侧链资源耗尽和目标侧链攻击"""
    
#     def __init__(self, attacker_id, initial_position, initial_velocity, sim):
#         self.id = attacker_id
#         self.position = initial_position
#         self.velocity = initial_velocity
#         self.sim = sim
#         self.last_collison = 0
        
#         # 获取资源池参数
#         self.num_slots = sim.resource_pool.num_slots
#         self.num_subchannels = sim.resource_pool.num_subchannels
        
#         # 攻击者特定参数
#         self.next_transmission_time = 0
#         self.transmission_cycle = 20  # 20ms传输周期
#         self.current_resource = None
#         # 攻击统计
#         self.attack_packets_sent = 0
#         self.attack_success_count = 0
#         self.collisions_caused = 0
#         self.target_slot = -1  # 跟踪最后攻击时隙
#         # 用于RL状态的感知数据
#         self.sensing_data = []
#         self.sensing_window_duration = 100  # 100ms感知窗口
#         # RL特定属性
#         self.last_action = None
#         self.last_reward = 0
        
#         # 用于多样性奖励
#         self.action_history = deque(maxlen=100)  # 跟踪最后100个动作
        
#         # 目标攻击模式相关属性
#         self.target_vehicle_id = -1  # 目标车辆ID
#         self.target_vehicle_resources = []  # 目标车辆选择的资源
#         self.target_vehicle_tracking_time = 0  # 跟踪目标车辆的时间
        
#         # 目标资源信息
#         self.targeted_resources = []  # 目标车辆最近使用的资源
    
#     def reset(self):
#         """重置攻击者状态"""
#         self.last_collison = 0
#         self.next_transmission_time = 0
#         self.current_resource = None
#         self.attack_packets_sent = 0
#         self.attack_success_count = 0
#         self.collisions_caused = 0
#         self.target_slot = -1
#         self.sensing_data = []  # 清空感知数据
#         self.last_action = None
#         self.last_reward = 0
#         self.action_history.clear()
        
#         # 重置目标攻击相关状态
#         self.target_vehicle_id = -1
#         self.target_vehicle_resources = []
#         self.target_vehicle_tracking_time = 0
#         self.targeted_resources = []
         
#     def move(self, delta_time):
#         """更新攻击者位置"""
#         self.position = self.position + self.velocity * delta_time
    
#     def _action_to_tuple(self,a):
#         # 如果是0维numpy数组，转为标量tuple
#         if isinstance(a, np.ndarray):
#             if a.ndim == 0:
#                 return (a.item(),)
#             else:
#                 return tuple(a.tolist())
#         # 如果是list或tuple，直接转tuple
#         if isinstance(a, (list, tuple)):
#             return tuple(a)
#         # 其他情况（如int/float），包装成tuple
#         return (a,)
    
#     def send_attack_packet_with_action(self, current_time, action):
#         """使用RL动作发送攻击数据包 - 支持两种攻击模式"""
#         # 根据全局模式选择攻击策略
#         if TARGETED_ATTACK_MODE:
#             # 目标侧链攻击模式
#             return self._send_targeted_attack(current_time)
#         else:
#             # 侧链资源耗尽模式
#             return self._send_exhaustion_attack(current_time, action)
    
#     def _send_exhaustion_attack(self, current_time, action):
#         """侧链资源耗尽攻击模式"""
#         slot_offset = action
        
#         # 计算目标时隙
#         current_slot = current_time % self.num_slots
#         self.target_slot = (current_slot + slot_offset) % self.num_slots
        
#         # 如果当前时隙不是目标时隙，不发送
#         if current_slot != self.target_slot:
#             return []
        
#         # 在该时隙的所有子信道上发送攻击
#         attack_packets = []
#         for subchannel in range(self.num_subchannels):  # 所有子信道
#             resource = ResourceInfo(self.target_slot, subchannel)
#             attack_packet = Packet(self.id, current_time, self.position, packet_id=0, is_attack=True)
#             attack_packets.append((attack_packet, resource))
#         self.attack_packets_sent += 1
#         # 记录动作用于多样性奖励
#         self.action_history.append(action)
        
#         return attack_packets
    
#     def _send_targeted_attack(self, current_time):
#         """目标侧链攻击模式 - 攻击特定车辆使用的资源"""
#         # 如果没有目标车辆，随机选择一个
#         if self.target_vehicle_id == -1 or current_time - self.target_vehicle_tracking_time > 1000:
#             self._select_new_target()
        
#         # 如果没有有效的目标资源，返回空
#         if not self.targeted_resources:
#             return []
        
#         current_slot = current_time % self.num_slots
#         attack_packets = []
        
#         # 检查目标资源是否在当前时隙
#         for resource in self.targeted_resources:
#             if resource.slot_id == current_slot:
#                 # 在该资源块上发送攻击
#                 attack_packet = Packet(self.id, current_time, self.position, packet_id=0, is_attack=True)
#                 attack_packets.append((attack_packet, resource))
#                 self.attack_packets_sent += 1
        
#         return attack_packets
    
#     def _select_new_target(self):
#         """选择一个新的目标车辆"""
#         # 选择通信范围内的车辆作为目标
#         possible_targets = []
#         for vehicle in self.sim.vehicles:
#             distance = np.linalg.norm(self.position - vehicle.position)
#             if distance <= self.sim.communication_range:
#                 possible_targets.append(vehicle.id)
        
#         if possible_targets:
#             self.target_vehicle_id = random.choice(possible_targets)
#             self.target_vehicle_tracking_time = self.sim.current_time
#             logger.info(f"攻击者 {self.id} 选择了新的目标车辆: {self.target_vehicle_id}")
#         else:
#             self.target_vehicle_id = -1
    
#     def _update_target_resources(self):
#         """更新目标车辆的资源选择信息"""
#         if self.target_vehicle_id == -1:
#             return
        
#         # 清空旧资源
#         self.targeted_resources = []
        
#         # 从感知数据中提取目标车辆的资源选择
#         for data in self.sensing_data:
#             if data.sender_id == self.target_vehicle_id:
#                 # 记录目标车辆使用的资源
#                 resource = ResourceInfo(data.slot_id, data.subchannel)
#                 self.targeted_resources.append(resource)
        
#         # 去重 - 保留最近出现的资源
#         unique_resources = {}
#         for resource in self.targeted_resources:
#             unique_resources[(resource.slot_id, resource.subchannel)] = resource
#         self.targeted_resources = list(unique_resources.values())
        
#         # 如果资源过多，只保留最近的5个
#         if len(self.targeted_resources) > 5:
#             self.targeted_resources = self.targeted_resources[-5:]
    
#     def get_state(self, current_time):
#         """获取RL代理的当前状态 - 更新后的状态空间"""
#         self._update_sensing_window(current_time)
        
#         # 1. 资源池占用矩阵 (num_slots时隙 x num_subchannels子信道)
#         resource_state = np.zeros((self.num_slots, self.num_subchannels))
#         for sensing_data in self.sensing_data:
#             slot_id = sensing_data.slot_id % self.num_slots
#             if 0 <= slot_id < self.num_slots and 0 <= sensing_data.subchannel < self.num_subchannels:
#                 resource_state[slot_id, sensing_data.subchannel] = 1
        
#         # 计算空闲资源比例
#         occupied_resources = np.sum(resource_state)
#         total_resources = self.num_slots * self.num_subchannels
#         free_ratio = 1.0 - (occupied_resources / total_resources)
        
#         # 5. 碰撞统计
#         collision_stats = [
#             free_ratio,
#             self.sim.recent_collision_rate  # 最近碰撞率
#         ]
        
#         # 组合所有状态组件
#         full_state = np.concatenate([
#             resource_state.flatten(),  # num_slots * num_subchannels维
#             collision_stats            # 2维
#         ])
        
#         return full_state.astype(np.float32)
    
#     def _calculate_vehicle_density(self):
#         """计算高速公路5个路段的车辆密度"""
#         segments = np.zeros(5)
#         segment_length = 1000.0 / 5
        
#         for vehicle in self.sim.vehicles:
#             segment_idx = min(int(vehicle.position[0] / segment_length), 4)
#             segments[segment_idx] += 1
        
#         # 通过最大可能车辆数归一化
#         max_vehicles = max(1, len(self.sim.vehicles) / 2.0)
#         return segments / max_vehicles
    
#     def _get_nearest_vehicle_state(self):
#         """获取最近车辆的状态信息"""
#         min_distance = float('inf')
#         rel_velocity = 0.0
        
#         for vehicle in self.sim.vehicles:
#             distance = np.linalg.norm(self.position - vehicle.position)
#             if distance < min_distance:
#                 min_distance = distance
#                 # 计算朝向攻击者的相对速度分量
#                 direction_vector = (self.position - vehicle.position) / max(distance, 1e-5)
#                 rel_velocity = np.dot(vehicle.velocity, direction_vector)
        
#         # 归一化值
#         min_distance = min(min_distance, 500.0) / 500.0  # 归一化到[0,1]
#         rel_velocity = (rel_velocity + 30.0) / 60.0  # 归一化到[0,1]，假设速度在[-30,30]范围内
        
#         return [min_distance, rel_velocity]
    
#     def calculate_reward(self, collision_occurred, num_collisions_caused,current_time,collision_count):
#         """计算RL代理的奖励 - 改进的奖励函数"""
#         reward = 0.0
        
#         # 1. 主要奖励：造成的碰撞次数（与干扰成功的发送者数量成正比）
#         if collision_occurred:
#             # 根据造成的碰撞次数给予奖励
#             reward += 1.0 * num_collisions_caused
#             self.attack_success_count += 1
#         else:
#             # 攻击失败的惩罚
#             reward -= 2.0
        
#         # 2. 动作多样性奖励：鼓励探索
#         if len(self.action_history) > 10:
#             unique_actions = len(set(self._action_to_tuple(a) for a in self.action_history))
#             diversity_ratio = unique_actions / len(self.action_history)
#             reward += 0.02 * diversity_ratio
        
#         # 3. 鼓励本轮平均碰撞数
#         if current_time > 0:
#             avg_collisions = collision_count - self.last_collison
#             reward += avg_collisions * 0.01  # 10为权重，可调
#             reward = np.clip(reward, 0.0, 0.2)
#         self.last_collison = collision_count
            
#         # 4. 鼓励降低全局PRR
#         prr = self.sim._calculate_current_prr()
#         reward += (1.0 - prr) * 0.5  # 权重2.0可调，prr越低奖励越高   
#         reward = np.clip(reward, -1.0, 5.0)
#         self.last_reward = reward

#         return reward
        
#     def _calculate_resource_utilization(self):
#         """计算感知窗口中的资源利用率"""
#         if not self.sensing_data:
#             return 0.0
        
#         # 计算使用的唯一资源
#         unique_resources = set()
#         for data in self.sensing_data:
#             resource_key = (data.slot_id % self.num_slots, data.subchannel)
#             unique_resources.add(resource_key)
        
#         # 感知窗口中总可能资源
#         total_resources = self.num_slots * self.num_subchannels
        
#         return len(unique_resources) / total_resources
    
#     def record_attack_success(self, collision_occurred):
#         """记录攻击成功"""
#         if collision_occurred:
#             self.collisions_caused += 1
    
#     def should_receive_packet(self, sender_position):
#         """攻击者可以接收通信范围内的数据包"""
#         distance = np.linalg.norm(self.position - sender_position)
#         return distance <= self.sim.communication_range

#     def add_sensing_data(self, slot_id, subchannel, pRsvp, sender_id, timestamp):
#         """添加来自接收传输的感知数据"""
#         sensing_data = SensingData(
#             slot_id=slot_id,
#             subchannel=subchannel,
#             pRsvp=pRsvp,
#             sender_id=sender_id,
#             timestamp=timestamp
#         )
#         self.sensing_data.append(sensing_data)
        
#         # 如果是目标攻击模式且来自目标车辆，记录资源
#         if TARGETED_ATTACK_MODE and sender_id == self.target_vehicle_id:
#             resource = ResourceInfo(slot_id, subchannel)
#             self.target_vehicle_resources.append(resource)
    
#     def _update_sensing_window(self, current_time):
#         """更新监听窗，移除过期数据"""
#         sensing_window_start = current_time - self.sensing_window_duration
        
#         # 移除监听窗口外的数据
#         self.sensing_data = [data for data in self.sensing_data 
#                             if data.timestamp >= sensing_window_start]
        
#         # 在目标攻击模式下，更新目标资源
#         if TARGETED_ATTACK_MODE:
#             self._update_target_resources()
        
# class FixAttacker(Vehicle):
#     """固定策略攻击者，使用周期组选择资源，但使用Combine模式选择空闲子信道对"""
    
#     def __init__(self, attacker_id, initial_position, initial_velocity, sim, 
#                  attack_cycle=20, num_subchannels=2, resource_selection_mode='Combine'):
#         super().__init__(attacker_id, initial_position, initial_velocity, sim, resource_selection_mode)
#         self.is_attack = True
#         self.attack_cycle = attack_cycle  # 攻击周期: 20,30,50,100ms
#         self.num_subchannels = num_subchannels  # 占用相邻子信道数: 1-5
#         self.next_attack_time = 0  # 下一次攻击时间
#         self.attack_packets_sent = 0
#         self.collisions_caused = 0
#         self.has_transmitted = False  # 是否正在发送攻击包
        
#         # 获取资源池参数
#         self.num_slots = sim.resource_pool.num_slots
        
#         # 使用短监听窗口增强攻击效果
#         self.sensing_window_duration = 200  # 200ms监听窗口（正常为1000ms）
        
#         # 高重选概率
#         self.prob_resource_keep = 0.2  # 20%概率保留资源（正常为20-80%）
        
#         # 计算周期组
#         self.cycle_groups = self._calculate_cycle_groups()
    
#     def reset(self):
#         """重置攻击者状态"""
#         super().reset()  # 调用父类重置方法
#         self.next_attack_time = 0
#         self.attack_packets_sent = 0
#         self.collisions_caused = 0
#         self.sensing_data = []  # 清空感知数据
#         self.cycle_groups = self._calculate_cycle_groups()
        
#     def _calculate_cycle_groups(self):
#         """根据攻击周期计算时隙组"""
#         num_groups = self.num_slots // self.attack_cycle
#         groups = []
#         start = 0
        
#         # 创建完整的周期组
#         for _ in range(num_groups):
#             end = start + self.attack_cycle
#             groups.append((start, end))
#             start = end
        
#         # 处理剩余的时隙
#         if start < self.num_slots:
#             groups.append((start, self.num_slots))
        
#         return groups
    
#     def _get_current_cycle_group(self, current_time):
#         """获取当前时间所属的周期组"""
#         current_slot = current_time % self.num_slots
        
#         # 查找当前时隙所属的组
#         for start, end in self.cycle_groups:
#             if start <= current_slot < end:
#                 return (start, end)
        
#         # 处理跨周期的情况（最后一个组）
#         return self.cycle_groups[-1]
    
#     def _create_selection_window(self, current_time):
#         """创建基于攻击周期的选择窗 - 重写父类方法"""
#         # 获取当前周期组
#         group_start, group_end = self._get_current_cycle_group(current_time)
#         selection_window = []
        
#         # 创建选择窗口（当前周期组内的所有时隙和子信道）
#         for slot in range(group_start, group_end):
#             for subchannel in range(self.num_subchannels):
#                 selection_window.append(ResourceInfo(slot % self.num_slots, subchannel))
        
#         return selection_window
    
#     def select_future_resource(self, current_time):
#         """重写资源选择方法，使用Combine模式在周期组内选择空闲子信道对"""
#         self._update_sensing_window(current_time)
#         selection_window = self._create_selection_window(current_time)
        
#         # 创建已占用资源集合
#         occupied_resources = set()
#         for data in self.sensing_data:
#             resource_key = (data.slot_id, data.subchannel)
#             occupied_resources.add(resource_key)
#         self.resel_counter = 1
#         # 使用Combine模式选择资源
#         return self._select_combined_resources(selection_window, occupied_resources)
    
#     def _select_combined_resources(self, selection_window, occupied_resources):
#         """Combine模式：在周期组内选择同一时隙的两个相邻子信道"""
#         # 按时隙分组资源
#         slot_resources = defaultdict(list)
#         for resource in selection_window:
#             slot_resources[resource.slot_id].append(resource)
        
#         # 收集所有有空闲相邻子信道的时隙
#         valid_slots = []
#         for slot_id, resources in slot_resources.items():
#             # 获取该时隙所有空闲子信道
#             free_subchannels = [r.subchannel for r in resources 
#                                if (slot_id, r.subchannel) not in occupied_resources]
            
#             # 检查是否有相邻的子信道对可用
#             adjacent_pairs = []
#             for i in range(self.num_subchannels - 1):  # 确保i+1不越界
#                 if i in free_subchannels and (i+1) in free_subchannels:
#                     adjacent_pairs.append((i, i+1))
            
#             if adjacent_pairs:
#                 valid_slots.append((slot_id, adjacent_pairs))
        
#         # 如果有可用的时隙和相邻子信道对
#         if valid_slots:
#             # 随机选择一个时隙
#             slot_id, adjacent_pairs = random.choice(valid_slots)
#             # 随机选择一对相邻子信道
#             sc1, sc2 = random.choice(adjacent_pairs)
#             return [ResourceInfo(slot_id, sc1), ResourceInfo(slot_id, sc2)]
        
#         # 如果没有完全空闲的相邻子信道对，选择部分空闲的
#         if slot_resources:
#             # 随机选择一个时隙
#             slot_id = random.choice(list(slot_resources.keys()))
#             # 在该时隙中随机选择一对相邻子信道
#             sc_pair = random.choice([(i, i+1) for i in range(self.num_subchannels - 1)])
#             sc1, sc2 = sc_pair

#             # 创建资源对象（即使子信道可能被占用）
#             return [ResourceInfo(slot_id, sc1), ResourceInfo(slot_id, sc2)]
        
#         # 如果没有任何资源可用，随机创建两个资源
#         group_start, group_end = self._get_current_cycle_group(self.sim.current_time)
#         slot_id = random.randint(group_start, group_end - 1) % self.num_slots
#         sc_pair = random.choice([(i, i+1) for i in range(self.num_subchannels - 1)])
#         sc1, sc2 = sc_pair
#         return [ResourceInfo(slot_id, sc1), ResourceInfo(slot_id, sc2)]
    
#     def send_packet(self, current_time):
#         """重写发送方法实现攻击逻辑"""
#         # 检查是否到达攻击时间
#         if self.current_resources is None:
#             return None
        
#         if self.has_transmitted:
#             return None
        
#         current_slot = current_time % self.num_slots
        
#         # 检查是否有资源块需要在当前时隙发送
#         resources_to_send = []
#         for resource in self.current_resources:
#             if resource.slot_id == current_slot:
#                 resources_to_send.append(resource)
        
#         if not resources_to_send:
#             return None
#         # print(f"攻击者 {self.id} 当前时隙: {current_slot}, 目标时隙: {resources_to_send[0].slot_id}, 是否发送: {current_slot == resources_to_send[0].slot_id}")
#         # 创建攻击包
#         packet = Packet(self.id, current_time, self.position, self.current_packet_id, is_attack=True)
        
#         # 记录日志 - 调试用
#         logger.debug(f"Attacker {self.id} sends packet at time {current_time}, slot {current_slot}, "
#                      f"resources: {[(r.slot_id, r.subchannel) for r in resources_to_send]}")
        
#         # 为每个要发送的资源块创建传输记录
#         transmissions = []
#         for resource in resources_to_send:
#             transmissions.append((packet, resource))
        
#         # 更新已发送资源块计数
#         self.sent_resources_count += len(transmissions)
#         self.attack_packets_sent += len(transmissions)
        
#         # 如果两个资源块都已发送，重置状态
#         if self.sent_resources_count >= 2:
#             self.resel_counter -= 1
#             if self.resel_counter <= 0:
#                 self.current_resources = None
#             self.sent_resources_count = 0
        
#         # 更新下一次攻击时间
#         # self.next_attack_time = current_time + self.attack_cycle
#         self.has_transmitted = True
        
#         return transmissions
    
#     def should_receive_packet(self, sender_position):
#         """攻击者可以接收通信范围内的数据包"""
#         distance = np.linalg.norm(self.position - sender_position)
#         return distance <= self.sim.communication_range
    
# class ResourcePool:
#     """管理V2X通信的侧链路资源池"""
    
#     def __init__(self, num_slots=100, num_subchannels=5, subchannel_size=10):
#         self.num_slots = num_slots
#         self.num_subchannels = num_subchannels
#         self.subchannel_size = subchannel_size
#         self.total_rbs = num_subchannels * num_slots

# class V2XRLEnvironment(gym.Env):
#     """V2X攻击优化的RL环境 - 简化版本"""
    
#     def __init__(self, num_vehicles=20, num_attackers=1, episode_duration=20000, 
#                  communication_range=320.0, vehicle_resource_mode='Separate',
#                  attacker_type='RL', fix_attacker_params=None, render_mode='human',
#                  num_slots=100, num_subchannels=5):
#         super(V2XRLEnvironment, self).__init__()
        
#         self.num_vehicles = num_vehicles
#         self.num_attackers = num_attackers
#         self.episode_duration = episode_duration  # 每轮20秒
#         self.communication_range = communication_range
#         self.vehicle_resource_mode = vehicle_resource_mode  # 车辆资源选择模式
#         self.attacker_type = attacker_type  # 'RL' 或 'Fix'
#         self.fix_attacker_params = fix_attacker_params or {'cycle': 20, 'num_subchannels': 2}
#         self.num_slots = num_slots
#         self.num_subchannels = num_subchannels
        
#         # 初始化组件
#         self.resource_pool = ResourcePool(num_slots=num_slots, num_subchannels=num_subchannels, subchannel_size=10)
#         self.initial_vehicle_states = None
#         self.initial_attacker_states = None
        
#         # 碰撞统计
#         self.recent_collision_queue = deque(maxlen=100)  # 记录最近100ms的碰撞次数
#         self.recent_collision_rate = 0.0
        
#         # RL空间 - 状态大小动态变化
#         state_size = (num_slots * num_subchannels) + 2
#         self.observation_space = spaces.Box(
#             low=0, high=1, shape=(state_size,), dtype=np.float32
#         )
        
#         # 动作：只选择时隙 (20个选择)
#         self.action_space = spaces.Discrete(20)  # 时隙偏移: 0-19
        
#         # 初始化模拟组件
#         self.vehicles = []
#         self.attackers = []
#         self.current_time = 0
#         self.message_status_dict = {}
        
#         # 资源选择可视化
#         if self.render_mode == 'human':
#             plt.ion()
#             self.fig, self.ax = plt.subplots(figsize=(15, 8))
#             self.resource_grid = [[set() for _ in range(num_subchannels)] for _ in range(num_slots)]
#         # 统计
#         self.reset_stats()
    
#     def _update_resource_grid(self):
#         """更新资源网格状态"""
#         # 重置资源网格
#         self.resource_grid = [[set() for _ in range(self.num_subchannels)] for _ in range(self.num_slots)]
        
#         # 添加车辆选择的资源
#         for vehicle in self.vehicles:
#             if vehicle.current_resources:
#                 for resource in vehicle.current_resources:
#                     slot = resource.slot_id
#                     sc = resource.subchannel
#                     if 0 <= slot < self.num_slots and 0 <= sc < self.num_subchannels:
#                         self.resource_grid[slot][sc].add(vehicle.id)
        
#         # 添加攻击者选择的资源
#         for attacker in self.attackers:
#             if isinstance(attacker, RLAttacker) and attacker.target_slot >= 0:
#                 for sc in range(self.num_subchannels):
#                     if 0 <= attacker.target_slot < self.num_slots:
#                         self.resource_grid[attacker.target_slot][sc].add(attacker.id)
#             elif isinstance(attacker, FixAttacker) and attacker.current_resources:
#                 for resource in attacker.current_resources:
#                     slot = resource.slot_id
#                     sc = resource.subchannel
#                     if 0 <= slot < self.num_slots and 0 <= sc < self.num_subchannels:
#                         self.resource_grid[slot][sc].add(attacker.id)
    
#     def render_sensing_view(self, vehicle_id=0):
#         """渲染指定车辆的监听窗视图 - 增强版：红色标注攻击者资源"""
#         if not hasattr(self, 'sensing_fig') or not hasattr(self, 'sensing_ax'):
#             plt.ion()
#             self.sensing_fig, self.sensing_ax = plt.subplots(figsize=(15, 8))
#             self.sensing_cbar = None
        
#         if vehicle_id >= len(self.vehicles+self.attackers):
#             return
            
#         if vehicle_id < self.num_vehicles:
#             vehicle = self.vehicles[vehicle_id]
#         else:
#             vehicle = self.attackers[vehicle_id - self.num_vehicles]
#         occupancy = vehicle.get_sensed_resource_occupancy()
        
#         # 创建攻击者资源矩阵（标记攻击者使用的资源）
#         attacker_occupancy = np.zeros((self.num_slots, self.num_subchannels), dtype=int)
#         for data in vehicle.sensing_data:
#             # 检查是否是攻击者发送的数据
#             if data.sender_id >= self.num_vehicles:  # 攻击者ID >= 车辆数
#                 slot = data.slot_id % self.num_slots
#                 if 0 <= slot < self.num_slots and 0 <= data.subchannel < self.num_subchannels:
#                     attacker_occupancy[slot, data.subchannel] = 1
        
#         self.sensing_ax.clear()
        
#         # 创建组合视图
#         combined_view = np.zeros((self.num_slots, self.num_subchannels))
#         # 普通占用标记为1
#         combined_view[occupancy == 1] = 1
#         # 攻击者占用标记为2
#         combined_view[attacker_occupancy == 1] = 2
        
#         # 绘制热力图
#         cmap = matplotlib.colors.ListedColormap(['white', 'blue', 'red'])
#         bounds = [0, 1, 2, 3]
#         norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        
#         im = self.sensing_ax.imshow(combined_view.T, cmap=cmap, norm=norm, aspect='auto', 
#                                    origin='lower', extent=[0, self.num_slots, 0, self.num_subchannels])
        
#         # 标记当前时隙
#         current_slot = self.current_time % self.num_slots
#         self.sensing_ax.axvline(x=current_slot, color='r', linestyle='-', linewidth=1)
        
#         # 添加网格
#         self.sensing_ax.set_xticks(np.arange(0, self.num_slots+1, max(1, self.num_slots//10)))
#         self.sensing_ax.set_yticks(np.arange(0, self.num_subchannels+1, 1))
#         self.sensing_ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
        
#         # 设置坐标轴标签
#         self.sensing_ax.set_xlabel(f'Slot (0-{self.num_slots-1})')
#         self.sensing_ax.set_ylabel(f'Subchannel (0-{self.num_subchannels-1})')
#         self.sensing_ax.set_title(f'Perceived Resource Occupancy by Vehicle {vehicle_id} at Time: {self.current_time} ms')
        
#         # 创建图例
#         from matplotlib.patches import Patch
#         legend_elements = [
#             Patch(facecolor='white', edgecolor='black', label='Free'),
#             Patch(facecolor='blue', edgecolor='black', label='Occupied (Normal)'),
#             Patch(facecolor='red', edgecolor='black', label='Occupied (Attacker)')
#         ]
#         self.sensing_ax.legend(handles=legend_elements, loc='upper right')
        
#         # 刷新显示
#         plt.draw()
#         plt.pause(0.001)
    
#     def render(self, mode='human'):
#         """渲染资源选择图"""
#         if mode != 'human' or self.current_time % 1 != 0:  # 每1ms渲染一次
#             return

#         # 确保只创建一个图形对象
#         if not hasattr(self, 'fig') or not hasattr(self, 'ax'):
#             plt.ion()  # 开启交互模式
#             self.fig, self.ax = plt.subplots(figsize=(15, 8))
#             self.cbar = None

#         self._update_resource_grid()
#         self.ax.clear()

#         # 创建可视化矩阵
#         grid_data = np.zeros((self.num_subchannels, self.num_slots))
#         # 创建攻击资源矩阵（标记攻击者选择的资源）
#         attack_data = np.zeros((self.num_subchannels, self.num_slots))

#         for slot in range(self.num_slots):
#             for sc in range(self.num_subchannels):
#                 users = self.resource_grid[slot][sc]
#                 if users:
#                     grid_data[sc, slot] = len(users)
#                     # 检查是否有攻击者使用该资源
#                     if any(uid >= self.num_vehicles for uid in users):  # 攻击者ID >= 车辆数
#                         attack_data[sc, slot] = 1  # 标记为攻击资源

#         # 绘制热力图
#         im = self.ax.imshow(grid_data, cmap='viridis', aspect='auto', origin='lower',
#                            vmin=0, vmax=3, extent=[0, self.num_slots, 0, self.num_subchannels])

#         # 标记当前时隙
#         current_slot = self.current_time % self.num_slots
#         self.ax.axvline(x=current_slot, color='r', linestyle='-', linewidth=1)

#         # 添加文本标签
#         for slot in range(self.num_slots):
#             for sc in range(self.num_subchannels):
#                 users = self.resource_grid[slot][sc]
#                 if users:
#                     user_text = ','.join(str(uid) for uid in users)
#                     # 如果是攻击者，使用红色文本
#                     text_color = 'red' if any(uid >= self.num_vehicles for uid in users) else 'white'
#                     self.ax.text(slot + 0.5, sc + 0.5, user_text,
#                                  ha='center', va='center', fontsize=8, color=text_color)

#         # 在攻击者使用的资源块上绘制红色矩形
#         for slot in range(self.num_slots):
#             for sc in range(self.num_subchannels):
#                 if attack_data[sc, slot] == 1:
#                     # 绘制红色边框矩形
#                     rect = plt.Rectangle((slot, sc), 1, 1, 
#                                         fill=False, edgecolor='red', linewidth=2)
#                     self.ax.add_patch(rect)

#         # 添加网格
#         self.ax.set_xticks(np.arange(0, self.num_slots+1, max(1, self.num_slots//10)))
#         self.ax.set_yticks(np.arange(0, self.num_subchannels+1, 1))
#         self.ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)

#         # 设置坐标轴标签
#         self.ax.set_xlabel(f'Slot (0-{self.num_slots-1})')
#         self.ax.set_ylabel(f'Subchannel (0-{self.num_subchannels-1})')
#         title = f'Resource Allocation at Time: {self.current_time} ms (Current Slot: {current_slot})'
#         if self.attackers:
#             # 添加攻击者信息
#             attacker = self.attackers[0]
#             if isinstance(attacker, RLAttacker):
#                 title += f'\nAttacker Target Slot: {attacker.target_slot}'
#                 if TARGETED_ATTACK_MODE:
#                     title += f' | Target Vehicle: {attacker.target_vehicle_id}'
#             elif isinstance(attacker, FixAttacker) and attacker.current_resources:
#                 slots = set(r.slot_id for r in attacker.current_resources)
#                 title += f'\nAttacker Slot(s): {", ".join(map(str, slots))}'
#         self.ax.set_title(title)

#         # 添加/更新颜色条
#         if self.cbar is None:
#             self.cbar = self.fig.colorbar(im, ax=self.ax)
#             self.cbar.set_label('Number of Users')
#         else:
#             self.cbar.update_normal(im)

#         # 刷新显示
#         plt.draw()
#         plt.pause(0.001)
#         self.fig.canvas.flush_events()  # 确保GUI事件被处理
        
#     def reset(self):
#         """重置环境开始新的一轮"""
#         # 重置时间
#         self.current_time = 0

#         # 重置碰撞统计
#         self.recent_collision_queue.clear()
#         self.recent_collision_rate = 0.0
#         self.message_status_dict = {}
#         # 重置统计
#         self.reset_stats()
#         for vehicle in self.vehicles:
#             vehicle.reset()
            
#         for attacker in self.attackers:
#             attacker.reset()
#         # 只在第一次时进行随机初始化并保存初始状态
#         if not hasattr(self, 'initial_vehicle_states') or self.initial_vehicle_states is None:
#             self.vehicles = []
#             self.attackers = []
#             self._initialize_vehicles()
#             self._initialize_attackers()
#             # 保存初始状态
#             self.initial_vehicle_states = [(v.position.copy(), v.velocity.copy()) for v in self.vehicles]
#             self.initial_attacker_states = [(a.position.copy(), a.velocity.copy()) for a in self.attackers]
#         else:
#             # 之后每次reset直接恢复初始状态
#             self.vehicles = []
#             for i, (position, velocity) in enumerate(self.initial_vehicle_states):
#                 vehicle = Vehicle(i, position.copy(), velocity.copy(), self, self.vehicle_resource_mode)
#                 self.vehicles.append(vehicle)
#             self.attackers = []
#             for i, (position, velocity) in enumerate(self.initial_attacker_states):
#                 attacker_id = self.num_vehicles + i
#                 if self.attacker_type == 'RL':
#                     attacker = RLAttacker(attacker_id, position.copy(), velocity.copy(), self)
#                 else:  # Fix attacker
#                     attacker = FixAttacker(
#                         attacker_id, 
#                         position.copy(), 
#                         velocity.copy(), 
#                         self,
#                         attack_cycle=self.fix_attacker_params['cycle'],
#                         num_subchannels=self.fix_attacker_params['num_subchannels']
#                     )
#                 self.attackers.append(attacker)

#         # 获取初始状态
#         if self.attacker_type == 'RL' and self.attackers:
#             initial_state = self.attackers[0].get_state(self.current_time)
#             return initial_state.astype(np.float32)
#         else:
#             # 对于FixAttacker，返回一个空状态（因为不需要RL状态）
#             return np.zeros(self.observation_space.shape, dtype=np.float32)
    
#     def step(self, action):
#         """在环境中执行一步"""
#         # 运行一个传输周期(20ms)的模拟
#         episode_reward = 0
#         collision_count_before = self.collision_count
#         message_failures_before = self.message_failures
#         for attacker in self.attackers:
#             if isinstance(attacker, RLAttacker):
#                 # 设置目标时隙
#                 attacker.target_slot = (self.current_time + action) % self.num_slots
#                 # 标记已有资源
#                 attacker.current_resource = True
#             elif isinstance(attacker, FixAttacker):
#                 # FixAttacker在每个20ms周期开始时选择新资源
#                 attacker.current_resources = attacker.select_future_resource(self.current_time)
#                 attacker.sent_resources_count = 0
#                 attacker.current_packet_id += 1
#                 attacker.has_transmitted = False
#         # 模拟20ms (20个1ms的时间步)
#         for step in range(20):
#             # 在每周期开始时处理资源重选
#             if self.current_time % self.num_slots == 0:  # 每周期开始时
#                 for vehicle in self.vehicles:
#                     vehicle.handle_periodic_resource_reselection(self.current_time)
#                     vehicle._update_sensing_window(self.current_time)
#                 # for attacker in self.attackers:
#                 #     if isinstance(attacker, FixAttacker):
#                 #         attacker.handle_periodic_resource_reselection(self.current_time)
            
#             # 更新位置
#             for vehicle in self.vehicles:
#                 vehicle.move(0.001)  # 1ms时间步
            
#             for attacker in self.attackers:
#                 attacker.move(0.001)
                
#             # 处理传输
#             step_reward, step_collisions = self._process_transmissions_with_rl(action)
#             episode_reward += step_reward
            
#             # 更新碰撞统计
#             self.recent_collision_queue.append(step_collisions)
#             if len(self.recent_collision_queue) > 0:
#                 self.recent_collision_rate = sum(self.recent_collision_queue) / len(self.recent_collision_queue)
            
#             # if self.current_time % 1 == 0:  # 每10ms渲染一次
#             #     self.render()
#             #     # time.sleep(0.001)  # 稍微延迟以便观察
#             # # self.render()
#             # if self.current_time % 100 == 0:  # 每100ms
#             #     self.render_sensing_view(vehicle_id=0)  # 显示车辆0的监听窗
            
#             self.current_time += 1
            
#         # 每步后重置资源
#         for attacker in self.attackers:
#             if isinstance(attacker, RLAttacker):
#                 attacker.current_resource = None
#             if isinstance(attacker, FixAttacker):
#                 attacker.current_resource = None
        
#         # 基于造成的碰撞计算奖励
#         collision_count_after = self.collision_count
#         collisions_caused = collision_count_after - collision_count_before
        
#         # 获取下一个状态
#         if self.attacker_type == 'RL' and self.attackers:
#             next_state = self.attackers[0].get_state(self.current_time)
#         else:
#             next_state = np.zeros(self.observation_space.shape, dtype=np.float32)
        
#         # 检查轮次是否结束
#         done = self.current_time >= self.episode_duration
        
#         # 额外信息
#         info = {
#             'collisions_caused': collisions_caused,
#             'total_collisions': self.collision_count,
#             'attack_success_rate': self.total_attack_success / max(1, self.attackers[0].attack_packets_sent) if self.attackers else 0,
#             'prr': self._calculate_current_prr(),
#             'step_reward': episode_reward,
#             'message_failures': self.message_failures - message_failures_before,
#             'resource_block_attacks': self.resource_block_attacks,
#             'resource_block_collisions': self.resource_block_collisions
#         }
        
#         return next_state.astype(np.float32), episode_reward, done, info
    
#     def close(self):
#         """关闭环境，清理资源"""
#         if self.render_mode == 'human':
#             plt.ioff()
#             plt.close()
            
#     def _process_transmissions_with_rl(self, action):
#         """处理传输，包括RL引导的攻击"""
#         transmissions = []
#         attack_transmissions = []
#         current_slot = self.current_time % self.num_slots
#         attacker_sent = False
        
#         # 收集车辆传输
#         for vehicle in self.vehicles:
#             # 发送数据包
#             tx_result = vehicle.send_packet(self.current_time)
#             if tx_result:
#                 # 添加传输记录
#                 for packet, resource in tx_result:
#                     transmissions.append((vehicle, packet, resource))
#                     self.transmission_count += 1
        
#         # 收集RL攻击者传输
#         for attacker in self.attackers:
#             if isinstance(attacker, RLAttacker):
#                 attack_result = attacker.send_attack_packet_with_action(self.current_time, action)
#             else:  # FixAttacker
#                 attack_result = attacker.send_packet(self.current_time)
                
#             if attack_result:
#                 attacker_sent = True
#                 for attack_packet, resource in attack_result:
#                     attack_transmissions.append((attacker, attack_packet, resource))
#                 self.attack_transmission_count += 1    
            
#         # 处理所有传输并计算奖励
#         all_transmissions = transmissions + attack_transmissions
#         reward = 0.0
#         step_collisions = 0
        
#         if all_transmissions:
#             collision_info = self._handle_transmissions_with_reward(all_transmissions)
#             step_collisions = collision_info.get('collisions_caused', 0)
            
#             # 如果攻击者传输了数据，则计算攻击者奖励
#             if attacker_sent and self.attacker_type == 'RL':
#                 # 获取碰撞信息
#                 collision_occurred = step_collisions > 0
                
#                 # 使用改进函数计算奖励
#                 for attacker in self.attackers:
#                     if isinstance(attacker, RLAttacker) and attacker_sent:
#                         reward = attacker.calculate_reward(collision_occurred, step_collisions,self.current_time,self.collision_count)
        
#         return reward, step_collisions
    
#     def _handle_transmissions_with_reward(self, transmissions):
#         """处理传输并计算奖励（支持双资源块传输）"""
#         # print(f"处理 {len(transmissions)} 个传输 at time {self.current_time}")
#         # 1. 按时隙分组
#         tx_by_slot = defaultdict(list)
#         for sender, packet, resource in transmissions:
#             tx_by_slot[resource.slot_id].append((sender, packet, resource))

#         collision_info = {'collisions_caused': 0}
        
#         # 2. 处理每个时隙的传输
#         for slot_id, slot_transmissions in tx_by_slot.items():
#             # 2.1 检测碰撞：记录每个子信道的使用情况
#             subchannel_usage = defaultdict(list)
#             for sender, packet, resource in slot_transmissions:
#                 subchannel_usage[resource.subchannel].append((sender, packet, resource))

#             # 2.2 消息状态跟踪
#             resource_block_status = defaultdict(lambda: {'attack': False, 'collision': False})
            
#             # 2.3 处理每个子信道
#             for subchannel, users in subchannel_usage.items():
#                 # 检查是否有攻击者参与
#                 has_attacker = any(isinstance(sender, (RLAttacker, FixAttacker)) for sender, _, _ in users)
#                 normal_users = [user for user in users if not isinstance(user[0], (RLAttacker, FixAttacker))]
                
#                 # 碰撞检测：多个发送者使用同一资源块
#                 collision_occurred = len(normal_users) > 1
                
#                 # 更新资源块状态
#                 if has_attacker:
#                     resource_block_status[subchannel]['attack'] = True
#                 if collision_occurred:
#                     resource_block_status[subchannel]['collision'] = True
#                 for sender, packet, resource in normal_users:
#                     msg_key = (sender.id, packet.packet_id)
#                     if msg_key not in self.message_status_dict:
#                         self.message_status_dict[msg_key] = {'resources': 0, 'success': True}
#                     # 更新消息的资源块计数
#                     self.message_status_dict[msg_key]['resources'] += 1
#                     # 如果发生攻击或碰撞，标记消息失败
#                     if has_attacker or collision_occurred:
#                         self.message_status_dict[msg_key]['success'] = False
#                 # 更新资源块级失效原因统计
#                 if has_attacker:
#                     self.resource_block_attacks += 1
#                 elif collision_occurred:
#                     self.resource_block_collisions += 1
#             finished_msgs = []
#             for msg_key, status in self.message_status_dict.items():
#                 if status['resources'] == 2:
#                     self.total_expected_packets += 1
#                     if status['success']:
#                         self.total_received_packets += 1
#                     else:
#                         self.message_failures += 1
#                     finished_msgs.append(msg_key)
                    
#             for msg_key in finished_msgs:
#                 del self.message_status_dict[msg_key]
#             # 2.4 处理接收 - 关键修改：确保车辆感知到所有传输
#             for receiver in self.vehicles + self.attackers:
#                 for sender, packet, resource in slot_transmissions:
#                     if sender.id == receiver.id:  # 跳过自己发送的包
#                         continue
                    
#                     if not receiver.should_receive_packet(sender.position):  # 超出通信范围
#                         continue
                    
#                     # 判断是否发生碰撞
#                     collision_occurred = resource_block_status[resource.subchannel]['attack'] or \
#                                         resource_block_status[resource.subchannel]['collision']
                    
#                     # 处理接收
#                     if isinstance(receiver, Vehicle):
#                         # 车辆会记录所有传输（包括攻击者的）
#                         receiver.receive_packet(packet, resource, collision_occurred)
#                     else:
#                         # 攻击者接收包用于感知
#                         if isinstance(sender, (RLAttacker, FixAttacker)):
#                             pRsvp = 20  # 攻击者发送周期
#                         else:
#                             pRsvp = 100  # 普通车辆发送周期
                            
#                         receiver.add_sensing_data(
#                             resource.slot_id,
#                             resource.subchannel,
#                             pRsvp,
#                             sender.id,
#                             packet.timestamp
#                         )
#                     # if receiver.id == 0:  # 只监控车辆0
#                     #     print(f"车辆0接收: 发送者 {sender.id}, 时隙 {resource.slot_id}, 子信道 {resource.subchannel}")
#             # 2.5 更新碰撞统计
#             normal_senders = set()
#             attack_success = set()
#             for sender, packet, resource in slot_transmissions:
#                 if not isinstance(sender, (RLAttacker, FixAttacker)):
#                     normal_senders.add(sender.id)
            
#             # 碰撞检测
#             for subchannel, users in subchannel_usage.items():
#                 if len(users) > 1:  # 发生碰撞
#                     # 检查是否有攻击者参与
#                     has_attacker = any(isinstance(sender, (RLAttacker, FixAttacker)) for sender, _, _ in users)
#                     if has_attacker:
#                         # 攻击者成功造成碰撞
                        
#                         self.total_attack_success += len(normal_senders)  # 攻击者成功造成碰撞
#                         collision_info['collisions_caused'] += len(normal_senders)
#                         for sender_id in normal_senders:
#                             attack_success.add(sender_id)
#                     self.collision_count += len(normal_senders)
            
#             # 更新发送者碰撞计数
#             for sender_id in normal_senders:
#                 # 更新车辆碰撞计数
#                 for vehicle in self.vehicles:
#                     if vehicle.id == sender_id:
#                         vehicle.collisions += 1
#                         break
#                 # 更新攻击者成功次数
#                 if sender_id in attack_success:
#                     for attacker in self.attackers:
#                         if isinstance(attacker, FixAttacker):
#                             attacker.collisions_caused += 1
#                         elif isinstance(attacker, RLAttacker):
#                             attacker.record_attack_success(True)
        
#         return collision_info
    
#     def _calculate_current_prr(self):
#         """计算当前分组接收率(PRR)"""
#         if self.total_expected_packets > 0:
#             return self.total_received_packets / self.total_expected_packets
#         return 0.0
    
#     def reset_stats(self):
#         """重置所有统计"""
#         self.collision_count = 0
#         self.transmission_count = 0
#         self.total_expected_packets = 0
#         self.total_received_packets = 0
#         self.attack_transmission_count = 0
#         self.total_attack_success = 0
#         self.message_failures = 0  # 消息层失败计数
#         self.resource_block_attacks = 0  # 资源块攻击失效计数
#         self.resource_block_collisions = 0  # 资源块碰撞失效计数
    
#     def _initialize_vehicles(self):
#         """初始化车辆位置和速度，并保存初始状态"""
#         lane1_y = 5.0
#         lane2_y = 10.0
#         highway_length = 1000.0
#         self.vehicles = []

#         vehicle_states = []

#         for i in range(self.num_vehicles):
#             lane_y = lane1_y if i % 2 == 0 else lane2_y
#             pos_x = random.uniform(0, highway_length)
#             position = np.array([pos_x, lane_y])
#             velocity = np.array([10.0 if lane_y == lane1_y else -10.0, 0.0])
#             vehicle = Vehicle(i, position.copy(), velocity.copy(), self, self.vehicle_resource_mode)
#             self.vehicles.append(vehicle)
#             vehicle_states.append((position.copy(), velocity.copy()))

#         # 只在第一次初始化时保存
#         if self.initial_vehicle_states is None:
#             self.initial_vehicle_states = vehicle_states

#     def _initialize_attackers(self):
#         """初始化攻击者并保存初始状态"""
#         highway_length = 1000.0
#         self.attackers = []

#         attacker_states = []

#         for i in range(self.num_attackers):
#             attacker_id = self.num_vehicles + i
#             position = np.array([highway_length/2, 0])
#             velocity = np.array([0.0, 0.0])
            
#             if self.attacker_type == 'RL':
#                 attacker = RLAttacker(attacker_id, position.copy(), velocity.copy(), self)
#             else:  # Fix attacker
#                 attacker = FixAttacker(
#                     attacker_id, 
#                     position.copy(), 
#                     velocity.copy(), 
#                     self,
#                     attack_cycle=self.fix_attacker_params['cycle'],
#                     num_subchannels=self.fix_attacker_params['num_subchannels']
#                 )
                
#             self.attackers.append(attacker)
#             attacker_states.append((position.copy(), velocity.copy()))

#         # 只在第一次初始化时保存
#         if self.initial_attacker_states is None:
#             self.initial_attacker_states = attacker_states
    
#     def get_episode_stats(self):
#         """获取当前轮的统计信息"""
#         return {
#             'total_collisions': self.collision_count,
#             'total_transmissions': self.transmission_count,
#             'prr': self._calculate_current_prr(),
#             'attack_success_rate': self.total_attack_success / max(1, self.attack_transmission_count),
#             'collision_rate': self.collision_count / max(1, self.transmission_count),
#             'message_failures': self.message_failures,
#             'resource_block_attacks': self.resource_block_attacks,
#             'resource_block_collisions': self.resource_block_collisions
#         }
    
#     def set_attacker_type(self, attacker_type, fix_params=None):
#         """设置攻击者类型和参数"""
#         self.attacker_type = attacker_type
#         if fix_params:
#             self.fix_attacker_params = fix_params
    
#     def toggle_attack_mode(self):
#         """切换攻击模式"""
#         global TARGETED_ATTACK_MODE
#         TARGETED_ATTACK_MODE = not TARGETED_ATTACK_MODE
#         mode_name = "目标侧链攻击" if TARGETED_ATTACK_MODE else "侧链资源耗尽"
#         logger.info(f"攻击模式已切换为: {mode_name}")
#         return TARGETED_ATTACK_MODE

# # 添加一个简单的按钮功能（在实际环境中可能需要GUI集成）
# def create_gui_button(env):
#     """创建一个简单的控制台按钮来切换攻击模式"""
#     print("\n" + "="*50)
#     print("按 't' 切换攻击模式 (当前: " + 
#           ("目标侧链攻击" if TARGETED_ATTACK_MODE else "侧链资源耗尽") + ")")
#     print("按 'q' 退出")
#     print("="*50)
    
#     while True:
#         key = input("输入命令: ").strip().lower()
#         if key == 't':
#             new_mode = env.toggle_attack_mode()
#             print(f"攻击模式已切换为: {'目标侧链攻击' if new_mode else '侧链资源耗尽'}")
#         elif key == 'q':
#             break
#         else:
#             print("无效命令，请重新输入")


import numpy as np
# import matplotlib.pyplot as plt
import random
import time
from collections import defaultdict, deque
import logging
import gym
from gym import spaces
# import matplotlib
# import os

import matplotlib
# matplotlib.use('TkAgg')  

import matplotlib.pyplot as plt
# plt.switch_backend('TkAgg')
# plt.interactive(False)
# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('V2X-RL-Environment')

# 全局攻击模式切换
TARGETED_ATTACK_MODE = True  # 默认是侧链资源耗尽模式

class Packet:
    """表示用于传输的V2X数据包"""
    
    def __init__(self, sender_id, timestamp, position, packet_id, size=190, is_attack=False):
        self.sender_id = sender_id
        self.timestamp = timestamp
        self.position = position
        self.size = size
        self.is_attack = is_attack
        self.packet_id = packet_id  # 唯一标识一个完整的数据包
        self.expected_receivers = 0  # 新增：预期接收者数量

class SensingData:
    """表示感知数据"""
    def __init__(self, slot_id, subchannel, pRsvp, sender_id, timestamp):
        self.slot_id = slot_id          # 0-99
        self.subchannel = subchannel    # 0-4
        self.pRsvp = pRsvp              # 预留周期
        self.sender_id = sender_id      # 发送者ID
        self.timestamp = timestamp      # 时间戳

class ResourceInfo:
    """表示资源块 (时隙+子信道)"""
    def __init__(self, slot_id, subchannel):
        self.slot_id = slot_id        # 0-99
        self.subchannel = subchannel  # 0-4
    
    def __eq__(self, other):
        if not isinstance(other, ResourceInfo):
            return False
        return (self.slot_id == other.slot_id and 
                self.subchannel == other.subchannel)
    
    def __repr__(self):
        return f"(slot:{self.slot_id}, subchannel:{self.subchannel})"

class Vehicle:
    """表示具有V2X功能的车辆"""
    
    def __init__(self, vehicle_id, initial_position, initial_velocity, sim, resource_selection_mode='Separate'):
        self.id = vehicle_id
        self.position = initial_position
        self.velocity = initial_velocity
        self.sim = sim
        self.resource_selection_mode = resource_selection_mode  # 资源选择模式: 'Separate' 或 'Combine'
        
        # 获取资源池参数
        self.num_slots = sim.resource_pool.num_slots
        self.num_subchannels = sim.resource_pool.num_subchannels
        
        # 资源选择参数
        self.resel_counter = 0
        self.prob_resource_keep = random.uniform(0.2, 0.8)
        self.current_resources = None  # 存储两个资源块
        self.sensing_data = []
        self.next_transmission_time = 0
        self.sent_resources_count = 0  # 已发送的资源块计数
        self.current_packet_id = 0     # 当前数据包ID
        
        # 感知窗口参数
        self.sensing_window_duration = 1000  # 100ms感知窗口
        self.has_transmitted = False  # 标记是否已发送数据包
        # 初始化统计
        self.packets_sent = 0
        self.packets_received = 0
        self.collisions = 0
        self.successful_transmissions = 0
        self.prr = 0.0  # 分组接收率
    
        self.expected_receptions = 0  # 预期接收总数（车辆级别）
        self.successful_receptions = 0  # 成功接收总数（车辆级别）
    def reset(self):
        """重置车辆状态"""
        self.resel_counter = 0
        self.current_resources = None
        self.sensing_data = []  # 清空感知数据
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
        # 创建候选资源列表（排除已占用资源）
        candidate_resources = []
        for resource in selection_window:
            resource_key = (resource.slot_id, resource.subchannel)
            if resource_key not in occupied_resources:
                candidate_resources.append(resource)
        
        # 如果候选资源不足选择窗的20%，则从整个选择窗中随机选择
        min_candidates = max(1, int(0.2 * len(selection_window)))
        if len(candidate_resources) < min_candidates:
            candidate_resources = selection_window[:]  # 使用整个选择窗
        
        # 从候选资源中随机选择两个不同的资源块
        selected_resources = []
        if len(candidate_resources) >= 2:
            selected = random.sample(candidate_resources, 2)
            selected_resources = selected
        elif len(candidate_resources) == 1:
            selected_resources = [candidate_resources[0], random.choice(selection_window)]
        else:
            # 如果没有候选资源，随机创建两个资源
            slot1 = random.randint(0, self.num_slots-1)
            subchannel1 = random.randint(0, self.num_subchannels-1)
            slot2 = random.randint(0, self.num_slots-1)
            subchannel2 = random.randint(0, self.num_subchannels-1)
            selected_resources = [ResourceInfo(slot1, subchannel1), ResourceInfo(slot2, subchannel2)]
        
        return selected_resources

    def _select_combined_resources(self, selection_window, occupied_resources):
        """Combine模式：选择同一时隙的两个相邻子信道"""
        # 按时隙分组资源
        slot_resources = defaultdict(list)
        for resource in selection_window:
            slot_resources[resource.slot_id].append(resource)
        
        # 收集所有有空闲相邻子信道的时隙
        valid_slots = []
        for slot_id, resources in slot_resources.items():
            # 获取该时隙所有空闲子信道
            free_subchannels = [r.subchannel for r in resources 
                               if (slot_id, r.subchannel) not in occupied_resources]
            
            # 检查是否有相邻的子信道对可用
            adjacent_pairs = []
            for i in range(self.num_subchannels - 1):  # 确保i+1不越界
                if i in free_subchannels and (i+1) in free_subchannels:
                    adjacent_pairs.append((i, i+1))
            
            if adjacent_pairs:
                valid_slots.append((slot_id, adjacent_pairs))
        
        # 如果有可用的时隙和相邻子信道对
        if valid_slots:
            # 随机选择一个时隙
            slot_id, adjacent_pairs = random.choice(valid_slots)
            # 随机选择一对相邻子信道
            sc1, sc2 = random.choice(adjacent_pairs)
            return [ResourceInfo(slot_id, sc1), ResourceInfo(slot_id, sc2)]
        
        if slot_resources:
            # 随机选择一个时隙
            slot_id = random.choice(list(slot_resources.keys()))
            # 在该时隙中随机选择一对相邻子信道
            sc_pair = random.choice([(i, i+1) for i in range(self.num_subchannels - 1)])
            sc1, sc2 = sc_pair

            # 创建资源对象（即使子信道可能被占用）
            return [ResourceInfo(slot_id, sc1), ResourceInfo(slot_id, sc2)]
        
        return []
    
    def _create_selection_window(self, current_time):
        """创建选择窗口 (T1=4到T2=100)"""
        selection_window = []
        current_slot = current_time % self.num_slots  # 当前时隙
        start_slot = (current_slot + 4) % self.num_slots
        end_slot = (current_slot + 100) % self.num_slots
        
        # 考虑周期性，处理跨周期的情况
        if start_slot < end_slot:
            slots = range(start_slot, end_slot)
        else:
            slots = list(range(start_slot, self.num_slots)) + list(range(0, end_slot))
        
        # 为每个时隙创建所有可能的子信道组合
        for slot in slots:
            for subchannel in range(self.num_subchannels):
                selection_window.append(ResourceInfo(slot, subchannel))
        
        return selection_window

    def _update_sensing_window(self, current_time):
        """通过移除旧条目更新感知窗口"""
        sensing_window_start = current_time - self.sensing_window_duration
        
        # 移除感知窗口外的数据
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
        self.prob_resource_keep = random.uniform(0.2, 0.8)  # 每次重选时更新概率
        if current_time % self.num_slots == 0:  # 每周期开始时
            # 检查资源重选计数器
            if self.resel_counter <= 0:
                # 依概率决定是否保留资源
                if random.random() < self.prob_resource_keep:
                    # 保留当前资源，重置计数器
                    self.resel_counter = random.randint(5, 15)
                else:
                    # 不保留资源，重置资源
                    self.current_resources = None
                    self.resel_counter = random.randint(5, 15)
            
            # 如果当前没有资源，选择新资源
            if self.current_resources is None:
                self.current_resources = self.select_future_resource(current_time)
                self.sent_resources_count = 0
                self.current_packet_id += 1
    
    def send_packet(self, current_time):
        """使用选定的资源发送数据包（现在使用两个资源块）"""
        # 如果当前没有资源，直接返回
        if self.current_resources is None:
            return None
        
        current_slot = current_time % self.num_slots
        
        # 检查是否有资源块需要在当前时隙发送
        resources_to_send = []
        for resource in self.current_resources:
            if resource.slot_id == current_slot:
                resources_to_send.append(resource)
        
        if not resources_to_send:
            return None
        
        # 创建数据包
        packet = Packet(self.id, current_time, self.position, self.current_packet_id)
        
        packet.expected_receivers = self._calculate_expected_receivers()  # 新增：计算预期接收者
        # 为每个要发送的资源块创建传输记录
        transmissions = []
        for resource in resources_to_send:
            transmissions.append((packet, resource))
        
        # 更新已发送资源块计数
        self.sent_resources_count += len(resources_to_send)
        
        # 更新统计
        self.packets_sent += len(resources_to_send)
        
        # 如果两个资源块都已发送，重置状态
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
        # 无论数据包是否来自攻击者，都添加到感知数据
        if hasattr(packet, 'is_attack') and packet.is_attack:
            pRsvp = 100  # 攻击者的发送周期
        else:
            pRsvp = 100  # 普通车辆的发送周期
        
        # 添加感知数据
        self.add_sensing_data(
            resource.slot_id,
            resource.subchannel,
            pRsvp,
            packet.sender_id,
            packet.timestamp
        )
        
        # 处理接收（仅对非攻击者数据包进行统计）
        if not packet.is_attack and not collision_occurred:
            self.packets_received += 1
            return True
        return False
    
    def should_receive_packet(self, sender_position):
        """确定该车辆是否应接收来自发送者的数据包"""
        distance = np.linalg.norm(self.position - sender_position)
        return distance <= self.sim.communication_range

    

class RLAttacker:
    """基于RL的攻击者，支持两种攻击模式：侧链资源耗尽和目标侧链攻击"""
    
    def __init__(self, attacker_id, initial_position, initial_velocity, sim):
        self.id = attacker_id
        self.position = initial_position
        self.velocity = initial_velocity
        self.sim = sim
        self.last_collison = 0
        
        # 获取资源池参数
        self.num_slots = sim.resource_pool.num_slots
        self.num_subchannels = sim.resource_pool.num_subchannels
        
        # 攻击者特定参数
        self.next_transmission_time = 0
        self.transmission_cycle = 20  # 20ms传输周期
        self.current_resource = None
        # 攻击统计
        self.attack_packets_sent = 0
        self.attack_success_count = 0
        self.collisions_caused = 0
        self.target_slot = -1  # 跟踪最后攻击时隙
        # 用于RL状态的感知数据
        self.sensing_data = []
        self.sensing_window_duration = 100  # 100ms感知窗口
        # RL特定属性
        self.last_action = None
        self.last_reward = 0
        
        # 用于多样性奖励
        self.action_history = deque(maxlen=100)  # 跟踪最后100个动作
        
        # 目标攻击模式相关属性
        self.target_vehicle_id = 0  # 目标车辆ID
        self.target_vehicle_resources = []  # 目标车辆选择的资源
        self.target_vehicle_tracking_time = 0  # 跟踪目标车辆的时间
        
        # 目标资源信息
        self.targeted_resources = []  # 目标车辆最近使用的资源
    
    def reset(self):
        """重置攻击者状态"""
        self.last_collison = 0
        self.next_transmission_time = 0
        self.current_resource = None
        self.attack_packets_sent = 0
        self.attack_success_count = 0
        self.collisions_caused = 0
        self.target_slot = -1
        self.sensing_data = []  # 清空感知数据
        self.last_action = None
        self.last_reward = 0
        self.action_history.clear()
        
        # 重置目标攻击相关状态
        self.target_vehicle_id = -1
        self.target_vehicle_resources = []
        self.target_vehicle_tracking_time = 0
        self.targeted_resources = []
         
    def move(self, delta_time):
        """更新攻击者位置"""
        self.position = self.position + self.velocity * delta_time
    
    def _action_to_tuple(self,a):
        # 如果是0维numpy数组，转为标量tuple
        if isinstance(a, np.ndarray):
            if a.ndim == 0:
                return (a.item(),)
            else:
                return tuple(a.tolist())
        # 如果是list或tuple，直接转tuple
        if isinstance(a, (list, tuple)):
            return tuple(a)
        # 其他情况（如int/float），包装成tuple
        return (a,)
    
    def send_attack_packet_with_action(self, current_time, action):
        """使用RL动作发送攻击数据包 - 支持两种攻击模式"""
        # 根据全局模式选择攻击策略
        if TARGETED_ATTACK_MODE:
            # 目标侧链攻击模式
            return self._send_targeted_attack(current_time)
        else:
            # 侧链资源耗尽模式
            return self._send_exhaustion_attack(current_time, action)
    
    def _send_exhaustion_attack(self, current_time, action):
        """侧链资源耗尽攻击模式"""
        slot_offset = action
        
        # 计算目标时隙
        current_slot = current_time % self.num_slots
        self.target_slot = (current_slot + slot_offset) % self.num_slots
        
        # 如果当前时隙不是目标时隙，不发送
        if current_slot != self.target_slot:
            return []
        
        # 在该时隙的所有子信道上发送攻击
        attack_packets = []
        for subchannel in range(self.num_subchannels):  # 所有子信道
            resource = ResourceInfo(self.target_slot, subchannel)
            attack_packet = Packet(self.id, current_time, self.position, packet_id=0, is_attack=True)
            attack_packets.append((attack_packet, resource))
        self.attack_packets_sent += 1
        # 记录动作用于多样性奖励
        self.action_history.append(action)
        
        return attack_packets
    
    def _send_targeted_attack(self, current_time):
        """目标侧链攻击模式 - 攻击特定车辆使用的资源"""
        # 如果没有目标车辆，随机选择一个
        # if self.target_vehicle_id == -1 or current_time - self.target_vehicle_tracking_time > 1000:
        #     self._select_new_target()
        
        # 如果没有有效的目标资源，返回空
        if not self.targeted_resources:
            return []
        
        current_slot = current_time % self.num_slots
        attack_packets = []
        
        # 检查目标资源是否在当前时隙
        for resource in self.targeted_resources:
            if resource.slot_id == current_slot:
                # 在该资源块上发送攻击
                attack_packet = Packet(self.id, current_time, self.position, packet_id=0, is_attack=True)
                attack_packets.append((attack_packet, resource))
                self.attack_packets_sent += 1
        
        return attack_packets
    
    def _select_new_target(self):
        """选择一个新的目标车辆"""
        # 选择通信范围内的车辆作为目标
        possible_targets = []
        for vehicle in self.sim.vehicles:
            distance = np.linalg.norm(self.position - vehicle.position)
            if distance <= self.sim.communication_range:
                possible_targets.append(vehicle.id)
        
        if possible_targets:
            self.target_vehicle_id = random.choice(possible_targets)
            self.target_vehicle_tracking_time = self.sim.current_time
            logger.info(f"攻击者 {self.id} 选择了新的目标车辆: {self.target_vehicle_id}")
        else:
            self.target_vehicle_id = -1
    
    def _update_target_resources(self):
        """更新目标车辆的资源选择信息"""
        if self.target_vehicle_id == -1:
            return
        
        # 清空旧资源
        self.targeted_resources = []
        
        # 从感知数据中提取目标车辆的资源选择
        for data in self.sensing_data:
            if data.sender_id == self.target_vehicle_id:
                # 记录目标车辆使用的资源
                resource = ResourceInfo(data.slot_id, data.subchannel)
                self.targeted_resources.append(resource)
        
        # 去重 - 保留最近出现的资源
        unique_resources = {}
        for resource in self.targeted_resources:
            unique_resources[(resource.slot_id, resource.subchannel)] = resource
        self.targeted_resources = list(unique_resources.values())
        
        # 如果资源过多，只保留最近的5个
        if len(self.targeted_resources) > 5:
            self.targeted_resources = self.targeted_resources[-5:]
    
    def get_state(self, current_time):
        """获取RL代理的当前状态 - 更新后的状态空间"""
        self._update_sensing_window(current_time)
        
        # 1. 资源池占用矩阵 (num_slots时隙 x num_subchannels子信道)
        resource_state = np.zeros((self.num_slots, self.num_subchannels))
        for sensing_data in self.sensing_data:
            slot_id = sensing_data.slot_id % self.num_slots
            if 0 <= slot_id < self.num_slots and 0 <= sensing_data.subchannel < self.num_subchannels:
                resource_state[slot_id, sensing_data.subchannel] = 1
        
        # 计算空闲资源比例
        occupied_resources = np.sum(resource_state)
        total_resources = self.num_slots * self.num_subchannels
        free_ratio = 1.0 - (occupied_resources / total_resources)
        
        # 5. 碰撞统计
        collision_stats = [
            free_ratio,
            self.sim.recent_collision_rate  # 最近碰撞率
        ]
        
        # 组合所有状态组件
        full_state = np.concatenate([
            resource_state.flatten(),  # num_slots * num_subchannels维
            collision_stats            # 2维
        ])
        
        return full_state.astype(np.float32)
    
    def _calculate_vehicle_density(self):
        """计算高速公路5个路段的车辆密度"""
        segments = np.zeros(5)
        segment_length = 1000.0 / 5
        
        for vehicle in self.sim.vehicles:
            segment_idx = min(int(vehicle.position[0] / segment_length), 4)
            segments[segment_idx] += 1
        
        # 通过最大可能车辆数归一化
        max_vehicles = max(1, len(self.sim.vehicles) / 2.0)
        return segments / max_vehicles
    
    def _get_nearest_vehicle_state(self):
        """获取最近车辆的状态信息"""
        min_distance = float('inf')
        rel_velocity = 0.0
        
        for vehicle in self.sim.vehicles:
            distance = np.linalg.norm(self.position - vehicle.position)
            if distance < min_distance:
                min_distance = distance
                # 计算朝向攻击者的相对速度分量
                direction_vector = (self.position - vehicle.position) / max(distance, 1e-5)
                rel_velocity = np.dot(vehicle.velocity, direction_vector)
        
        # 归一化值
        min_distance = min(min_distance, 500.0) / 500.0  # 归一化到[0,1]
        rel_velocity = (rel_velocity + 30.0) / 60.0  # 归一化到[0,1]，假设速度在[-30,30]范围内
        
        return [min_distance, rel_velocity]
    
    def calculate_reward(self, collision_occurred, num_collisions_caused,current_time,collision_count):
        """计算RL代理的奖励 - 改进的奖励函数"""
        reward = 0.0
        
        # 1. 主要奖励：造成的碰撞次数（与干扰成功的发送者数量成正比）
        if collision_occurred:
            # 根据造成的碰撞次数给予奖励
            reward += 1.0 * num_collisions_caused
            self.attack_success_count += 1
        else:
            # 攻击失败的惩罚
            reward -= 2.0
        
        # 2. 动作多样性奖励：鼓励探索
        if len(self.action_history) > 10:
            unique_actions = len(set(self._action_to_tuple(a) for a in self.action_history))
            diversity_ratio = unique_actions / len(self.action_history)
            reward += 0.02 * diversity_ratio
        
        # 3. 鼓励本轮平均碰撞数
        if current_time > 0:
            avg_collisions = collision_count - self.last_collison
            reward += avg_collisions * 0.01  # 10为权重，可调
            reward = np.clip(reward, 0.0, 0.2)
        self.last_collison = collision_count
            
        # 4. 鼓励降低全局PRR
        prr = self.sim._calculate_current_prr()
        reward += (1.0 - prr) * 0.5  # 权重2.0可调，prr越低奖励越高   
        reward = np.clip(reward, -1.0, 5.0)
        self.last_reward = reward

        return reward
        
    def _calculate_resource_utilization(self):
        """计算感知窗口中的资源利用率"""
        if not self.sensing_data:
            return 0.0
        
        # 计算使用的唯一资源
        unique_resources = set()
        for data in self.sensing_data:
            resource_key = (data.slot_id % self.num_slots, data.subchannel)
            unique_resources.add(resource_key)
        
        # 感知窗口中总可能资源
        total_resources = self.num_slots * self.num_subchannels
        
        return len(unique_resources) / total_resources
    
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
        
        # 如果是目标攻击模式且来自目标车辆，记录资源
        if TARGETED_ATTACK_MODE and sender_id == self.target_vehicle_id:
            resource = ResourceInfo(slot_id, subchannel)
            self.target_vehicle_resources.append(resource)
    
    def _update_sensing_window(self, current_time):
        """更新监听窗，移除过期数据"""
        sensing_window_start = current_time - self.sensing_window_duration
        
        # 移除监听窗口外的数据
        self.sensing_data = [data for data in self.sensing_data 
                            if data.timestamp >= sensing_window_start]
        
        # 在目标攻击模式下，更新目标资源
        if TARGETED_ATTACK_MODE:
            self._update_target_resources()
        
# class FixAttacker(Vehicle):
#     """固定策略攻击者，支持两种攻击模式：周期组攻击和目标侧链攻击"""
    
#     def __init__(self, attacker_id, initial_position, initial_velocity, sim, 
#                  attack_cycle=20, num_subchannels=2, resource_selection_mode='Combine'):
#         super().__init__(attacker_id, initial_position, initial_velocity, sim, resource_selection_mode)
#         self.is_attack = True
#         self.attack_cycle = attack_cycle  # 攻击周期: 20,30,50,100ms
#         self.num_subchannels = num_subchannels  # 占用相邻子信道数: 1-5
#         self.next_attack_time = 0  # 下一次攻击时间
#         self.attack_packets_sent = 0
#         self.collisions_caused = 0
#         self.has_transmitted = False  # 是否正在发送攻击包
        
#         # 获取资源池参数
#         self.num_slots = sim.resource_pool.num_slots
        
#         # 使用短监听窗口增强攻击效果
#         self.sensing_window_duration = 200  # 200ms监听窗口（正常为1000ms）
        
#         # 高重选概率
#         self.prob_resource_keep = 0.2  # 20%概率保留资源（正常为20-80%）
        
#         # 计算周期组
#         self.cycle_groups = self._calculate_cycle_groups()
        
#         # 目标攻击模式相关属性
#         self.target_vehicle_id = -1  # 目标车辆ID
#         self.target_vehicle_resources = []  # 目标车辆选择的资源
#         self.target_vehicle_tracking_time = 0  # 跟踪目标车辆的时间
        
#         # 目标资源信息
#         self.targeted_resources = []  # 目标车辆最近使用的资源
    
#     def reset(self):
#         """重置攻击者状态"""
#         super().reset()  # 调用父类重置方法
#         self.next_attack_time = 0
#         self.attack_packets_sent = 0
#         self.collisions_caused = 0
#         self.sensing_data = []  # 清空感知数据
#         self.cycle_groups = self._calculate_cycle_groups()
        
#         # 重置目标攻击相关状态
#         self.target_vehicle_id = -1
#         self.target_vehicle_resources = []
#         self.target_vehicle_tracking_time = 0
#         self.targeted_resources = []
        
#     def _calculate_cycle_groups(self):
#         """根据攻击周期计算时隙组"""
#         num_groups = self.num_slots // self.attack_cycle
#         groups = []
#         start = 0
        
#         # 创建完整的周期组
#         for _ in range(num_groups):
#             end = start + self.attack_cycle
#             groups.append((start, end))
#             start = end
        
#         # 处理剩余的时隙
#         if start < self.num_slots:
#             groups.append((start, self.num_slots))
        
#         return groups
    
#     def _get_current_cycle_group(self, current_time):
#         """获取当前时间所属的周期组"""
#         current_slot = current_time % self.num_slots
        
#         # 查找当前时隙所属的组
#         for start, end in self.cycle_groups:
#             if start <= current_slot < end:
#                 return (start, end)
        
#         # 处理跨周期的情况（最后一个组）
#         return self.cycle_groups[-1]
    
#     def _create_selection_window(self, current_time):
#         """创建基于攻击周期的选择窗 - 重写父类方法"""
#         # 获取当前周期组
#         group_start, group_end = self._get_current_cycle_group(current_time)
#         selection_window = []
        
#         # 创建选择窗口（当前周期组内的所有时隙和子信道）
#         for slot in range(group_start, group_end):
#             for subchannel in range(self.num_subchannels):
#                 selection_window.append(ResourceInfo(slot % self.num_slots, subchannel))
        
#         return selection_window
    
#     def select_future_resource(self, current_time):
#         """重写资源选择方法，使用Combine模式在周期组内选择空闲子信道对"""
#         self._update_sensing_window(current_time)
#         selection_window = self._create_selection_window(current_time)
        
#         # 创建已占用资源集合
#         occupied_resources = set()
#         for data in self.sensing_data:
#             resource_key = (data.slot_id, data.subchannel)
#             occupied_resources.add(resource_key)
#         self.resel_counter = 1
#         # 使用Combine模式选择资源
#         return self._select_combined_resources(selection_window, occupied_resources)
    
#     def _select_combined_resources(self, selection_window, occupied_resources):
#         """Combine模式：在周期组内选择同一时隙的两个相邻子信道"""
#         # 按时隙分组资源
#         slot_resources = defaultdict(list)
#         for resource in selection_window:
#             slot_resources[resource.slot_id].append(resource)
        
#         # 收集所有有空闲相邻子信道的时隙
#         valid_slots = []
#         for slot_id, resources in slot_resources.items():
#             # 获取该时隙所有空闲子信道
#             free_subchannels = [r.subchannel for r in resources 
#                                if (slot_id, r.subchannel) not in occupied_resources]
            
#             # 检查是否有相邻的子信道对可用
#             adjacent_pairs = []
#             for i in range(self.num_subchannels - 1):  # 确保i+1不越界
#                 if i in free_subchannels and (i+1) in free_subchannels:
#                     adjacent_pairs.append((i, i+1))
            
#             if adjacent_pairs:
#                 valid_slots.append((slot_id, adjacent_pairs))
        
#         # 如果有可用的时隙和相邻子信道对
#         if valid_slots:
#             # 随机选择一个时隙
#             slot_id, adjacent_pairs = random.choice(valid_slots)
#             # 随机选择一对相邻子信道
#             sc1, sc2 = random.choice(adjacent_pairs)
#             return [ResourceInfo(slot_id, sc1), ResourceInfo(slot_id, sc2)]
        
#         # 如果没有完全空闲的相邻子信道对，选择部分空闲的
#         if slot_resources:
#             # 随机选择一个时隙
#             slot_id = random.choice(list(slot_resources.keys()))
#             # 在该时隙中随机选择一对相邻子信道
#             sc_pair = random.choice([(i, i+1) for i in range(self.num_subchannels - 1)])
#             sc1, sc2 = sc_pair

#             # 创建资源对象（即使子信道可能被占用）
#             return [ResourceInfo(slot_id, sc1), ResourceInfo(slot_id, sc2)]
        
#         # 如果没有任何资源可用，随机创建两个资源
#         group_start, group_end = self._get_current_cycle_group(self.sim.current_time)
#         slot_id = random.randint(group_start, group_end - 1) % self.num_slots
#         sc_pair = random.choice([(i, i+1) for i in range(self.num_subchannels - 1)])
#         sc1, sc2 = sc_pair
#         return [ResourceInfo(slot_id, sc1), ResourceInfo(slot_id, sc2)]
    
#     def send_packet(self, current_time):
#         """重写发送方法实现攻击逻辑 - 支持目标攻击模式"""
#         # 如果处于目标攻击模式，执行目标攻击
#         if TARGETED_ATTACK_MODE:
#             return self._send_targeted_attack(current_time)
#         else:
#             return self._send_cycle_group_attack(current_time)
    
#     def _send_cycle_group_attack(self, current_time):
#         """执行周期组攻击模式"""
#         # 检查是否到达攻击时间
#         if self.current_resources is None:
#             return None
        
#         if self.has_transmitted:
#             return None
        
#         current_slot = current_time % self.num_slots
        
#         # 检查是否有资源块需要在当前时隙发送
#         resources_to_send = []
#         for resource in self.current_resources:
#             if resource.slot_id == current_slot:
#                 resources_to_send.append(resource)
        
#         if not resources_to_send:
#             return None
        
#         # 创建攻击包
#         packet = Packet(self.id, current_time, self.position, self.current_packet_id, is_attack=True)
        
#         # 记录日志 - 调试用
#         logger.debug(f"Attacker {self.id} sends packet at time {current_time}, slot {current_slot}, "
#                      f"resources: {[(r.slot_id, r.subchannel) for r in resources_to_send]}")
        
#         # 为每个要发送的资源块创建传输记录
#         transmissions = []
#         for resource in resources_to_send:
#             transmissions.append((packet, resource))
        
#         # 更新已发送资源块计数
#         self.sent_resources_count += len(transmissions)
#         self.attack_packets_sent += len(transmissions)
        
#         # 如果两个资源块都已发送，重置状态
#         if self.sent_resources_count >= 2:
#             self.resel_counter -= 1
#             if self.resel_counter <= 0:
#                 self.current_resources = None
#             self.sent_resources_count = 0
        
#         # 更新下一次攻击时间
#         self.has_transmitted = True
        
#         return transmissions
    
#     def _send_targeted_attack(self, current_time):
#         """执行目标侧链攻击模式"""
#         # 如果没有目标车辆，选择一个新的目标
#         # if self.target_vehicle_id == -1 or current_time - self.target_vehicle_tracking_time > 1000:
#             # self._select_new_target()
#         self.target_vehicle_id = 0
#         # 如果没有目标资源，不发送攻击
#         if not self.targeted_resources:
#             return None
        
#         current_slot = current_time % self.num_slots
#         transmissions = []
        
#         # 检查目标资源是否在当前时隙
#         for resource in self.targeted_resources:
#             if resource.slot_id == current_slot:
#                 # 在该资源块上发送攻击
#                 packet = Packet(self.id, current_time, self.position, self.current_packet_id, is_attack=True)
#                 transmissions.append((packet, resource))
#                 self.attack_packets_sent += 1
        
#         return transmissions
    
#     def _select_new_target(self):
#         """选择一个新的目标车辆"""
#         # 选择通信范围内的车辆作为目标
#         possible_targets = []
#         for vehicle in self.sim.vehicles:
#             distance = np.linalg.norm(self.position - vehicle.position)
#             if distance <= self.sim.communication_range:
#                 possible_targets.append(vehicle.id)
        
#         if possible_targets:
#             self.target_vehicle_id = random.choice(possible_targets)
#             self.target_vehicle_tracking_time = self.sim.current_time
#             logger.info(f"FixAttacker {self.id} 选择了新的目标车辆: {self.target_vehicle_id}")
#         else:
#             self.target_vehicle_id = -1
    
#     def _update_target_resources(self):
#         """更新目标车辆的资源选择信息"""
#         if self.target_vehicle_id == -1:
#             return
        
#         # 清空旧资源
#         self.targeted_resources = []
        
#         # 从感知数据中提取目标车辆的资源选择
#         for data in self.sensing_data:
#             if data.sender_id == self.target_vehicle_id:
#                 # 记录目标车辆使用的资源
#                 resource = ResourceInfo(data.slot_id, data.subchannel)
#                 self.targeted_resources.append(resource)
        
#         # 去重 - 保留最近出现的资源
#         unique_resources = {}
#         for resource in self.targeted_resources:
#             unique_resources[(resource.slot_id, resource.subchannel)] = resource
#         self.targeted_resources = list(unique_resources.values())
        
#         # 如果资源过多，只保留最近的5个
#         if len(self.targeted_resources) > 5:
#             self.targeted_resources = self.targeted_resources[-5:]
    
#     def add_sensing_data(self, slot_id, subchannel, pRsvp, sender_id, timestamp):
#         """添加感知数据 - 扩展以支持目标攻击"""
#         super().add_sensing_data(slot_id, subchannel, pRsvp, sender_id, timestamp)
        
#         # 如果是目标攻击模式且来自目标车辆，记录资源
#         if TARGETED_ATTACK_MODE and sender_id == self.target_vehicle_id:
#             resource = ResourceInfo(slot_id, subchannel)
#             self.target_vehicle_resources.append(resource)
    
#     def _update_sensing_window(self, current_time):
#         """更新监听窗 - 扩展以支持目标攻击"""
#         super()._update_sensing_window(current_time)
        
#         # 在目标攻击模式下，更新目标资源
#         if TARGETED_ATTACK_MODE:
#             self._update_target_resources()
    
#     def should_receive_packet(self, sender_position):
#         """攻击者可以接收通信范围内的数据包"""
#         distance = np.linalg.norm(self.position - sender_position)
#         return distance <= self.sim.communication_range

class FixAttacker(Vehicle):
    """固定策略攻击者，支持两种攻击模式：周期组攻击和目标侧链攻击"""
    
    def __init__(self, attacker_id, initial_position, initial_velocity, sim, 
                 attack_cycle=20, num_subchannels=2, resource_selection_mode='Combine'):
        super().__init__(attacker_id, initial_position, initial_velocity, sim, resource_selection_mode)
        self.is_attack = True
        self.attack_cycle = attack_cycle  # 攻击周期: 20,30,50,100ms
        self.num_subchannels = num_subchannels  # 占用相邻子信道数: 1-5
        self.next_attack_time = 0  # 下一次攻击时间
        self.attack_packets_sent = 0
        self.collisions_caused = 0
        self.has_transmitted = False  # 是否正在发送攻击包
        
        # 获取资源池参数
        self.num_slots = sim.resource_pool.num_slots
        
        # 使用短监听窗口增强攻击效果
        self.sensing_window_duration = 100  # 200ms监听窗口（正常为1000ms）
        
        # 高重选概率
        self.prob_resource_keep = 0.2  # 20%概率保留资源（正常为20-80%）
        
        # 计算周期组（仅在资源耗尽模式下使用）
        self.cycle_groups = self._calculate_cycle_groups() if not TARGETED_ATTACK_MODE else []
        
        # 目标攻击模式相关属性
        self.target_vehicle_id = 0  # 目标车辆ID
        self.target_vehicle_resources = []  # 目标车辆选择的资源
        self.target_vehicle_tracking_time = 0  # 跟踪目标车辆的时间
        
        # 目标资源信息
        self.targeted_resources = []  # 目标车辆最近使用的资源
    
    def reset(self):
        """重置攻击者状态"""
        super().reset()  # 调用父类重置方法
        self.next_attack_time = 0
        self.attack_packets_sent = 0
        self.collisions_caused = 0
        self.sensing_data = []  # 清空感知数据
        self.cycle_groups = self._calculate_cycle_groups() if not TARGETED_ATTACK_MODE else []
        
        # 重置目标攻击相关状态
        self.target_vehicle_id = -1
        self.target_vehicle_resources = []
        self.target_vehicle_tracking_time = 0
        self.targeted_resources = []
        
    def _calculate_cycle_groups(self):
        """根据攻击周期计算时隙组（仅在资源耗尽模式下使用）"""
        num_groups = self.num_slots // self.attack_cycle
        groups = []
        start = 0
        
        # 创建完整的周期组
        for _ in range(num_groups):
            end = start + self.attack_cycle
            groups.append((start, end))
            start = end
        
        # 处理剩余的时隙
        if start < self.num_slots:
            groups.append((start, self.num_slots))
        
        return groups
    
    def _get_current_cycle_group(self, current_time):
        """获取当前时间所属的周期组（仅在资源耗尽模式下使用）"""
        if not self.cycle_groups:
            return (0, self.num_slots)
            
        current_slot = current_time % self.num_slots
        
        # 查找当前时隙所属的组
        for start, end in self.cycle_groups:
            if start <= current_slot < end:
                return (start, end)
        
        # 处理跨周期的情况（最后一个组）
        return self.cycle_groups[-1]
    
    def _create_selection_window(self, current_time):
        """创建基于攻击周期的选择窗 - 仅在资源耗尽模式下使用"""
        # 获取当前周期组
        group_start, group_end = self._get_current_cycle_group(current_time)
        selection_window = []
        
        # 创建选择窗口（当前周期组内的所有时隙和子信道）
        for slot in range(group_start, group_end):
            for subchannel in range(self.num_subchannels):
                selection_window.append(ResourceInfo(slot % self.num_slots, subchannel))
        
        return selection_window
    
    def select_future_resource(self, current_time):
        """重写资源选择方法，使用Combine模式在周期组内选择空闲子信道对"""
        # 在目标攻击模式下，不需要选择新资源
        if TARGETED_ATTACK_MODE:
            return []
            
        self._update_sensing_window(current_time)
        selection_window = self._create_selection_window(current_time)
        
        # 创建已占用资源集合
        occupied_resources = set()
        for data in self.sensing_data:
            resource_key = (data.slot_id, data.subchannel)
            occupied_resources.add(resource_key)
        self.resel_counter = 1
        # 使用Combine模式选择资源
        return self._select_combined_resources(selection_window, occupied_resources)
    
    def _select_combined_resources(self, selection_window, occupied_resources):
        """Combine模式：在周期组内选择同一时隙的两个相邻子信道"""
        # 按时隙分组资源
        slot_resources = defaultdict(list)
        for resource in selection_window:
            slot_resources[resource.slot_id].append(resource)
        
        # 收集所有有空闲相邻子信道的时隙
        valid_slots = []
        for slot_id, resources in slot_resources.items():
            # 获取该时隙所有空闲子信道
            free_subchannels = [r.subchannel for r in resources 
                               if (slot_id, r.subchannel) not in occupied_resources]
            
            # 检查是否有相邻的子信道对可用
            adjacent_pairs = []
            for i in range(self.num_subchannels - 1):  # 确保i+1不越界
                if i in free_subchannels and (i+1) in free_subchannels:
                    adjacent_pairs.append((i, i+1))
            
            if adjacent_pairs:
                valid_slots.append((slot_id, adjacent_pairs))
        
        # 如果有可用的时隙和相邻子信道对
        if valid_slots:
            # 随机选择一个时隙
            slot_id, adjacent_pairs = random.choice(valid_slots)
            # 随机选择一对相邻子信道
            sc1, sc2 = random.choice(adjacent_pairs)
            return [ResourceInfo(slot_id, sc1), ResourceInfo(slot_id, sc2)]
        
        # 如果没有完全空闲的相邻子信道对，选择部分空闲的
        if slot_resources:
            # 随机选择一个时隙
            slot_id = random.choice(list(slot_resources.keys()))
            # 在该时隙中随机选择一对相邻子信道
            sc_pair = random.choice([(i, i+1) for i in range(self.num_subchannels - 1)])
            sc1, sc2 = sc_pair

            # 创建资源对象（即使子信道可能被占用）
            return [ResourceInfo(slot_id, sc1), ResourceInfo(slot_id, sc2)]
        
        # 如果没有任何资源可用，随机创建两个资源
        group_start, group_end = self._get_current_cycle_group(self.sim.current_time)
        slot_id = random.randint(group_start, group_end - 1) % self.num_slots
        sc_pair = random.choice([(i, i+1) for i in range(self.num_subchannels - 1)])
        sc1, sc2 = sc_pair
        return [ResourceInfo(slot_id, sc1), ResourceInfo(slot_id, sc2)]
    
    def send_packet(self, current_time):
        """重写发送方法实现攻击逻辑 - 支持目标攻击模式"""
        # 如果处于目标攻击模式，执行目标攻击
        if TARGETED_ATTACK_MODE:
            return self._send_targeted_attack(current_time)
        else:
            return self._send_cycle_group_attack(current_time)
    
    def _send_cycle_group_attack(self, current_time):
        """执行周期组攻击模式"""
        # 检查是否到达攻击时间
        if self.current_resources is None:
            return None
        
        if self.has_transmitted:
            return None
        
        current_slot = current_time % self.num_slots
        
        # 检查是否有资源块需要在当前时隙发送
        resources_to_send = []
        for resource in self.current_resources:
            if resource.slot_id == current_slot:
                resources_to_send.append(resource)
        
        if not resources_to_send:
            return None
        
        # 创建攻击包
        packet = Packet(self.id, current_time, self.position, self.current_packet_id, is_attack=True)
        
        # 记录日志 - 调试用
        logger.debug(f"Attacker {self.id} sends packet at time {current_time}, slot {current_slot}, "
                     f"resources: {[(r.slot_id, r.subchannel) for r in resources_to_send]}")
        
        # 为每个要发送的资源块创建传输记录
        transmissions = []
        for resource in resources_to_send:
            transmissions.append((packet, resource))
        
        # 更新已发送资源块计数
        self.sent_resources_count += len(transmissions)
        self.attack_packets_sent += len(transmissions)
        
        # 如果两个资源块都已发送，重置状态
        if self.sent_resources_count >= 2:
            self.resel_counter -= 1
            if self.resel_counter <= 0:
                self.current_resources = None
            self.sent_resources_count = 0
        
        # 更新下一次攻击时间
        self.has_transmitted = True
        
        return transmissions
    
    def _send_targeted_attack(self, current_time):
        """执行目标侧链攻击模式"""
        # 如果没有目标车辆，选择一个新的目标
        # if self.target_vehicle_id == -1 or current_time - self.target_vehicle_tracking_time > 1000:
        #     self._select_new_target()
        
        # 更新目标资源
        self._update_target_resources()
        
        # 如果没有目标资源，不发送攻击
        if not self.targeted_resources:
            return None
        
        current_slot = current_time % self.num_slots
        transmissions = []
        
        # 检查目标资源是否在当前时隙
        for resource in self.targeted_resources:
            if resource.slot_id == current_slot:
                # 在该资源块上发送攻击
                packet = Packet(self.id, current_time, self.position, self.current_packet_id, is_attack=True)
                transmissions.append((packet, resource))
                self.attack_packets_sent += 1
        
        return transmissions
    
    def _select_new_target(self):
        """选择一个新的目标车辆"""
        # 选择通信范围内的车辆作为目标
        possible_targets = []
        for vehicle in self.sim.vehicles:
            distance = np.linalg.norm(self.position - vehicle.position)
            if distance <= self.sim.communication_range:
                possible_targets.append(vehicle.id)
        
        if possible_targets:
            self.target_vehicle_id = random.choice(possible_targets)
            self.target_vehicle_tracking_time = self.sim.current_time
            logger.info(f"FixAttacker {self.id} 选择了新的目标车辆: {self.target_vehicle_id}")
        else:
            self.target_vehicle_id = -1
    def print_attacker_sensing_window(self, attacker_id=0):
        """打印攻击者的监听窗（感知到的资源占用情况）"""
        if attacker_id >= len(self.attackers):
            print(f"攻击者 {attacker_id} 不存在")
            return
        attacker = self.attackers[attacker_id]
        occupancy = attacker.get_sensed_resource_occupancy()
        print(f"\n攻击者 {attacker_id} 监听窗（占用=1，空闲=0）：")
        print(occupancy)
        
    def print_attacker_selected_resources(self, attacker_id=0):
        """打印攻击者当前选择的资源"""
        if attacker_id >= len(self.attackers):
            print(f"攻击者 {attacker_id} 不存在")
            return
        attacker = self.attackers[attacker_id]
        if hasattr(attacker, 'current_resources') and attacker.current_resources:
            print(f"\n攻击者 {attacker_id} 当前选择的资源：")
            for res in attacker.current_resources:
                print(f"  slot: {res.slot_id}, subchannel: {res.subchannel}")
        elif hasattr(attacker, 'target_slot') and attacker.target_slot >= 0:
            print(f"\n攻击者 {attacker_id} 当前目标时隙: {attacker.target_slot}")
        else:
            print(f"\n攻击者 {attacker_id} 当前没有选择资源")
    def _update_target_resources(self):
        """更新目标车辆的资源选择信息 - 只保留最近一次的资源"""
        if self.target_vehicle_id == -1:
            return
        
        # 清空旧资源
        self.targeted_resources = []
        
        # 从感知数据中提取目标车辆最近使用的资源
        latest_timestamp = 0
        latest_resources = []
        
        for data in self.sensing_data:
            if data.sender_id == self.target_vehicle_id:
                # 记录目标车辆使用的资源
                resource = ResourceInfo(data.slot_id, data.subchannel)
                
                # 只保留最近一次的资源
                if data.timestamp > latest_timestamp:
                    latest_timestamp = data.timestamp
                    latest_resources = [resource]
                elif data.timestamp == latest_timestamp:
                    latest_resources.append(resource)
        
        self.targeted_resources = latest_resources
    
    def add_sensing_data(self, slot_id, subchannel, pRsvp, sender_id, timestamp):
        """添加感知数据 - 扩展以支持目标攻击"""
        super().add_sensing_data(slot_id, subchannel, pRsvp, sender_id, timestamp)
        
        # 如果是目标攻击模式且来自目标车辆，记录资源
        if TARGETED_ATTACK_MODE and sender_id == self.target_vehicle_id:
            resource = ResourceInfo(slot_id, subchannel)
            self.target_vehicle_resources.append(resource)
    
    def _update_sensing_window(self, current_time):
        """更新监听窗 - 扩展以支持目标攻击"""
        super()._update_sensing_window(current_time)
        
        # 在目标攻击模式下，更新目标资源
        if TARGETED_ATTACK_MODE:
            self._update_target_resources()
    
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
    """V2X攻击优化的RL环境 - 简化版本"""
    
    def __init__(self, num_vehicles=20, num_attackers=1, episode_duration=20000, 
                 communication_range=320.0, vehicle_resource_mode='Separate',
                 attacker_type='RL', fix_attacker_params=None, render_mode='human',
                 num_slots=100, num_subchannels=5,useSINR = False):
        super(V2XRLEnvironment, self).__init__()
        
        self.num_vehicles = num_vehicles
        self.num_attackers = num_attackers
        self.episode_duration = episode_duration  # 每轮20秒
        self.communication_range = communication_range
        self.vehicle_resource_mode = vehicle_resource_mode  # 车辆资源选择模式
        self.attacker_type = attacker_type  # 'RL' 或 'Fix'
        self.fix_attacker_params = fix_attacker_params or {'cycle': 20, 'num_subchannels': 2}
        self.num_slots = num_slots
        self.num_subchannels = num_subchannels
        
        # 初始化组件
        self.resource_pool = ResourcePool(num_slots=num_slots, num_subchannels=num_subchannels, subchannel_size=10)
        self.initial_vehicle_states = None
        self.initial_attacker_states = None
        
        # 碰撞统计
        self.recent_collision_queue = deque(maxlen=100)  # 记录最近100ms的碰撞次数
        self.recent_collision_rate = 0.0
        
        # RL空间 - 状态大小动态变化
        state_size = (num_slots * num_subchannels) + 2
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(state_size,), dtype=np.float32
        )
        
        # 动作：只选择时隙 (20个选择)
        self.action_space = spaces.Discrete(20)  # 时隙偏移: 0-19
        
        # 初始化模拟组件
        self.vehicles = []
        self.attackers = []
        self.current_time = 0
        self.message_status_dict = {}
        # self.useSINR = useSINR  # 是否使用SINR判断接收
        
        # 资源选择可视化
        if self.render_mode == 'human':
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(15, 8))
            self.resource_grid = [[set() for _ in range(num_subchannels)] for _ in range(num_slots)]
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
        
    def reset(self):
        """重置环境开始新的一轮"""
        # 重置时间
        self.current_time = 0

        # 重置碰撞统计
        self.recent_collision_queue.clear()
        self.recent_collision_rate = 0.0
        self.message_status_dict = {}
        # 重置统计
        self.reset_stats()
        for vehicle in self.vehicles:
            vehicle.reset()
            
        for attacker in self.attackers:
            attacker.reset()
        # 只在第一次时进行随机初始化并保存初始状态
        if not hasattr(self, 'initial_vehicle_states') or self.initial_vehicle_states is None:
            self.vehicles = []
            self.attackers = []
            self._initialize_vehicles()
            self._initialize_attackers()
            # 保存初始状态
            self.initial_vehicle_states = [(v.position.copy(), v.velocity.copy()) for v in self.vehicles]
            self.initial_attacker_states = [(a.position.copy(), a.velocity.copy()) for a in self.attackers]
        else:
            # 之后每次reset直接恢复初始状态
            self.vehicles = []
            for i, (position, velocity) in enumerate(self.initial_vehicle_states):
                vehicle = Vehicle(i, position.copy(), velocity.copy(), self, self.vehicle_resource_mode)
                self.vehicles.append(vehicle)
            self.attackers = []
            for i, (position, velocity) in enumerate(self.initial_attacker_states):
                attacker_id = self.num_vehicles + i
                if self.attacker_type == 'RL':
                    attacker = RLAttacker(attacker_id, position.copy(), velocity.copy(), self)
                else:  # Fix attacker
                    attacker = FixAttacker(
                        attacker_id, 
                        position.copy(), 
                        velocity.copy(), 
                        self,
                        attack_cycle=self.fix_attacker_params['cycle'],
                        num_subchannels=self.fix_attacker_params['num_subchannels']
                    )
                self.attackers.append(attacker)

        # 获取初始状态
        if self.attacker_type == 'RL' and self.attackers:
            initial_state = self.attackers[0].get_state(self.current_time)
            return initial_state.astype(np.float32)
        else:
            # 对于FixAttacker，返回一个空状态（因为不需要RL状态）
            return np.zeros(self.observation_space.shape, dtype=np.float32)
    
    def step(self, action):
        """在环境中执行一步"""
        # 运行一个传输周期(20ms)的模拟
        episode_reward = 0
        collision_count_before = self.collision_count
        message_failures_before = self.message_failures
        for attacker in self.attackers:
            if isinstance(attacker, RLAttacker):
                # 设置目标时隙
                attacker.target_slot = (self.current_time + action) % self.num_slots
                # 标记已有资源
                attacker.current_resource = True
            elif isinstance(attacker, FixAttacker):
                # FixAttacker在每个20ms周期开始时选择新资源
                if not TARGETED_ATTACK_MODE:
                    attacker.current_resources = attacker.select_future_resource(self.current_time)
                attacker.sent_resources_count = 0
                attacker.current_packet_id += 1
                attacker.has_transmitted = False
        # 模拟20ms (20个1ms的时间步)
        for step in range(20):
            # 在每周期开始时处理资源重选
            if self.current_time % self.num_slots == 0:  # 每周期开始时
                for vehicle in self.vehicles:
                    vehicle.handle_periodic_resource_reselection(self.current_time)
                    vehicle._update_sensing_window(self.current_time)
                # for attacker in self.attackers:
                #     if isinstance(attacker, FixAttacker):
                #         attacker.handle_periodic_resource_reselection(self.current_time)
            
            # 更新位置
            for vehicle in self.vehicles:
                vehicle.move(0.001)  # 1ms时间步
            
            for attacker in self.attackers:
                attacker.move(0.001)
                
            # 处理传输
            step_reward, step_collisions = self._process_transmissions_with_rl(action)
            episode_reward += step_reward
            
            # 更新碰撞统计
            self.recent_collision_queue.append(step_collisions)
            if len(self.recent_collision_queue) > 0:
                self.recent_collision_rate = sum(self.recent_collision_queue) / len(self.recent_collision_queue)
            
            # if self.current_time % 5 == 0:  # 每10ms渲染一次
            #     self.render()
            #     # time.sleep(0.001)  # 稍微延迟以便观察
            # # self.render()
            # if self.current_time % 100 == 0:  # 每100ms
            #     self.render_sensing_view(vehicle_id=0)  # 显示车辆0的监听窗
            
            self.current_time += 1
            
        # 每步后重置资源
        for attacker in self.attackers:
            if isinstance(attacker, RLAttacker):
                attacker.current_resource = None
            if isinstance(attacker, FixAttacker):
                attacker.current_resource = None
        
        # 基于造成的碰撞计算奖励
        collision_count_after = self.collision_count
        collisions_caused = collision_count_after - collision_count_before
        
        # 获取下一个状态
        if self.attacker_type == 'RL' and self.attackers:
            next_state = self.attackers[0].get_state(self.current_time)
        else:
            next_state = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # 检查轮次是否结束
        done = self.current_time >= self.episode_duration
        
        # 额外信息
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
    
    def close(self):
        """关闭环境，清理资源"""
        if self.render_mode == 'human':
            plt.ioff()
            plt.close()
            
    def _process_transmissions_with_rl(self, action):
        """处理传输，包括RL引导的攻击"""
        transmissions = []
        attack_transmissions = []
        current_slot = self.current_time % self.num_slots
        attacker_sent = False
        
        # 收集车辆传输
        for vehicle in self.vehicles:
            # 发送数据包
            tx_result = vehicle.send_packet(self.current_time)
            if tx_result:
                # 添加传输记录
                for packet, resource in tx_result:
                    transmissions.append((vehicle, packet, resource))
                    self.transmission_count += 1
        
        # 收集RL攻击者传输
        for attacker in self.attackers:
            if isinstance(attacker, RLAttacker):
                attack_result = attacker.send_attack_packet_with_action(self.current_time, action)
            else:  # FixAttacker
                attack_result = attacker.send_packet(self.current_time)
                
            if attack_result:
                attacker_sent = True
                for attack_packet, resource in attack_result:
                    attack_transmissions.append((attacker, attack_packet, resource))
                self.attack_transmission_count += 1    
            
        # 处理所有传输并计算奖励
        all_transmissions = transmissions + attack_transmissions
        reward = 0.0
        step_collisions = 0
        
        if all_transmissions:
            collision_info = self._handle_transmissions_with_reward(all_transmissions)
            step_collisions = collision_info.get('collisions_caused', 0)
            
            # 如果攻击者传输了数据，则计算攻击者奖励
            if attacker_sent and self.attacker_type == 'RL':
                # 获取碰撞信息
                collision_occurred = step_collisions > 0
                
                # 使用改进函数计算奖励
                for attacker in self.attackers:
                    if isinstance(attacker, RLAttacker) and attacker_sent:
                        reward = attacker.calculate_reward(collision_occurred, step_collisions,self.current_time,self.collision_count)
        
        return reward, step_collisions
    
    
    # def _calculate_path_loss(self, distance):
    #     """计算路径损耗 (dB)"""
    #     if distance < self.d0:
    #         distance = self.d0
    #     return self.pl0 + 10 * self.n * np.log10(distance / self.d0)
    
    # def _calculate_sinr(self, receiver_pos, sender_pos, interferers_pos):
    #     """
    #     计算SINR
    #     :param receiver_pos: 接收者位置
    #     :param sender_pos: 目标发送者位置
    #     :param interferers_pos: 干扰源位置列表
    #     :return: SINR (dB)
    #     """
    #     # 计算目标信号接收功率
    #     distance = np.linalg.norm(receiver_pos - sender_pos)
    #     path_loss = self._calculate_path_loss(distance)
    #     rx_power = self.tx_power - path_loss  # dBm
        
    #     # 计算干扰信号总功率
    #     interference_power = 0.0  # 毫瓦
    #     for intf_pos in interferers_pos:
    #         intf_distance = np.linalg.norm(receiver_pos - intf_pos)
    #         intf_path_loss = self._calculate_path_loss(intf_distance)
    #         intf_rx_power = self.tx_power - intf_path_loss  # dBm
    #         # 转换为毫瓦并累加
    #         interference_power += 10 ** (intf_rx_power / 10)
        
    #     # 添加噪声功率 (转换为毫瓦)
    #     noise_mw = 10 ** (self.noise_power / 10)
    #     total_interference = interference_power + noise_mw
        
    #     # 计算SINR (dB)
    #     signal_mw = 10 ** (rx_power / 10)
    #     sinr_linear = signal_mw / total_interference
    #     return 10 * np.log10(sinr_linear) if sinr_linear > 0 else -100
    
    def _handle_transmissions_with_reward(self, transmissions):
        """处理传输并计算奖励（支持双资源块传输）"""
        # print(f"处理 {len(transmissions)} 个传输 at time {self.current_time}")
        # 1. 按时隙分组
        tx_by_slot = defaultdict(list)
        for sender, packet, resource in transmissions:
            tx_by_slot[resource.slot_id].append((sender, packet, resource))

        collision_info = {'collisions_caused': 0}
        
        # 2. 处理每个时隙的传输
        for slot_id, slot_transmissions in tx_by_slot.items():
            # 2.1 检测碰撞：记录每个子信道的使用情况
            subchannel_usage = defaultdict(list)
            for sender, packet, resource in slot_transmissions:
                subchannel_usage[resource.subchannel].append((sender, packet, resource))

            # 2.2 消息状态跟踪
            resource_block_status = defaultdict(lambda: {'attack': False, 'collision': False})
            
            # 2.3 处理每个子信道
            for subchannel, users in subchannel_usage.items():
                # 检查是否有攻击者参与
                has_attacker = any(isinstance(sender, (RLAttacker, FixAttacker)) for sender, _, _ in users)
                normal_users = [user for user in users if not isinstance(user[0], (RLAttacker, FixAttacker))]
                
                # 碰撞检测：多个发送者使用同一资源块
                collision_occurred = len(normal_users) > 1
                
                # 更新资源块状态
                if has_attacker:
                    resource_block_status[subchannel]['attack'] = True
                if collision_occurred:
                    resource_block_status[subchannel]['collision'] = True
                for sender, packet, resource in normal_users:
                    msg_key = (sender.id, packet.packet_id)
                    if msg_key not in self.message_status_dict:
                        self.message_status_dict[msg_key] = {'resources': 0,'success': True,'expected_receivers': packet.expected_receivers}
                    # 更新消息的资源块计数
                    self.message_status_dict[msg_key]['resources'] += 1
                    # 如果发生攻击或碰撞，标记消息失败
                    if has_attacker or collision_occurred:
                        self.message_status_dict[msg_key]['success'] = False
                # 更新资源块级失效原因统计
                if has_attacker:
                    self.resource_block_attacks += 1
                elif collision_occurred:
                    self.resource_block_collisions += 1
                finished_msgs = []
                for msg_key, status in self.message_status_dict.items():
                    if status['resources'] == 2:  # 两个资源块都传输完毕
                        self.total_expected_packets += 1
                        sender_id, packet_id = msg_key

                        # 更新车辆级别PRR统计
                        sender_vehicle = next((v for v in self.vehicles if v.id == sender_id), None)
                        if sender_vehicle:
                            # 记录该数据包的接收情况
                            sender_vehicle.record_reception(status['success'], status['expected_receivers'])

                        if status['success']:
                            self.total_received_packets += 1
                        else:
                            self.message_failures += 1
                        finished_msgs.append(msg_key)
                    
            for msg_key in finished_msgs:
                del self.message_status_dict[msg_key]
            # 2.4 处理接收 - 关键修改：确保车辆感知到所有传输
            for receiver in self.vehicles + self.attackers:
                for sender, packet, resource in slot_transmissions:
                    if sender.id == receiver.id:  # 跳过自己发送的包
                        continue
                    
                    if not receiver.should_receive_packet(sender.position):  # 超出通信范围
                        continue
                    
                    # 判断是否发生碰撞
                    collision_occurred = resource_block_status[resource.subchannel]['attack'] or \
                                        resource_block_status[resource.subchannel]['collision']
                    
                    # 处理接收
                    if isinstance(receiver, Vehicle):
                        # 车辆会记录所有传输（包括攻击者的）
                        receiver.receive_packet(packet, resource, collision_occurred)
                    else:
                        # 攻击者接收包用于感知
                        if isinstance(sender, (RLAttacker, FixAttacker)):
                            pRsvp = 20  # 攻击者发送周期
                        else:
                            pRsvp = 100  # 普通车辆发送周期
                            
                        receiver.add_sensing_data(
                            resource.slot_id,
                            resource.subchannel,
                            pRsvp,
                            sender.id,
                            packet.timestamp
                        )
                    # if receiver.id == 0:  # 只监控车辆0
                    #     print(f"车辆0接收: 发送者 {sender.id}, 时隙 {resource.slot_id}, 子信道 {resource.subchannel}")
            # 2.5 更新碰撞统计
            normal_senders = set()
            attack_success = set()
            for sender, packet, resource in slot_transmissions:
                if not isinstance(sender, (RLAttacker, FixAttacker)):
                    normal_senders.add(sender.id)
            
            # 碰撞检测
            for subchannel, users in subchannel_usage.items():
                if len(users) > 1:  # 发生碰撞
                    # 检查是否有攻击者参与
                    has_attacker = any(isinstance(sender, (RLAttacker, FixAttacker)) for sender, _, _ in users)
                    if has_attacker:
                        # 攻击者成功造成碰撞
                        
                        self.total_attack_success += len(normal_senders)  # 攻击者成功造成碰撞
                        collision_info['collisions_caused'] += len(normal_senders)
                        for sender_id in normal_senders:
                            attack_success.add(sender_id)
                    self.collision_count += len(normal_senders)
            
            # 更新发送者碰撞计数
            for sender_id in normal_senders:
                # 更新车辆碰撞计数
                for vehicle in self.vehicles:
                    if vehicle.id == sender_id:
                        vehicle.collisions += 1
                        break
                # 更新攻击者成功次数
                if sender_id in attack_success:
                    for attacker in self.attackers:
                        if isinstance(attacker, FixAttacker):
                            attacker.collisions_caused += 1
                        elif isinstance(attacker, RLAttacker):
                            attacker.record_attack_success(True)
        
        return collision_info
    
    # def _handle_transmissions_with_reward(self, transmissions):
    #     """处理传输并计算奖励（支持双资源块传输）"""
    #     # 按时隙分组
    #     tx_by_slot = defaultdict(list)
    #     for sender, packet, resource in transmissions:
    #         tx_by_slot[resource.slot_id].append((sender, packet, resource))

    #     collision_info = {'collisions_caused': 0}
        
    #     # 处理每个时隙的传输
    #     for slot_id, slot_transmissions in tx_by_slot.items():
    #         # 检测碰撞：记录每个子信道的使用情况
    #         subchannel_usage = defaultdict(list)
    #         for sender, packet, resource in slot_transmissions:
    #             subchannel_usage[resource.subchannel].append((sender, packet, resource))

    #         # 消息状态跟踪
    #         resource_block_status = defaultdict(lambda: {'attack': False, 'collision': False})
            
    #         # 处理每个子信道
    #         for subchannel, users in subchannel_usage.items():
    #             # 检查是否有攻击者参与
    #             has_attacker = any(isinstance(sender, (RLAttacker, FixAttacker)) for sender, _, _ in users)
    #             normal_users = [user for user in users if not isinstance(user[0], (RLAttacker, FixAttacker))]
                
    #             # 碰撞检测：多个发送者使用同一资源块
    #             collision_occurred = len(normal_users) > 1
                
    #             # 更新资源块状态
    #             if has_attacker:
    #                 resource_block_status[subchannel]['attack'] = True
    #             if collision_occurred:
    #                 resource_block_status[subchannel]['collision'] = True
                
    #             # 处理每个传输
    #             for sender, packet, resource in normal_users:
    #                 msg_key = (sender.id, packet.packet_id)
    #                 if msg_key not in self.message_status_dict:
    #                     self.message_status_dict[msg_key] = {
    #                         'resources': 0,
    #                         'success': True,
    #                         'expected_receivers': packet.expected_receivers
    #                     }
    #                 # 更新消息的资源块计数
    #                 self.message_status_dict[msg_key]['resources'] += 1
                    
    #                 # 如果使用SINR判断，先不标记失败，稍后处理
    #                 if not self.useSINR and (has_attacker or collision_occurred):
    #                     # 非SINR模式下，发生攻击或碰撞直接标记失败
    #                     self.message_status_dict[msg_key]['success'] = False
                
    #             # 更新资源块级失效原因统计
    #             if has_attacker:
    #                 self.resource_block_attacks += 1
    #             elif collision_occurred:
    #                 self.resource_block_collisions += 1
            
    #         # 处理接收
    #         for receiver in self.vehicles + self.attackers:
    #             for sender, packet, resource in slot_transmissions:
    #                 if sender.id == receiver.id:  # 跳过自己发送的包
    #                     continue
                    
    #                 if not receiver.should_receive_packet(sender.position):  # 超出通信范围
    #                     continue
                    
    #                 # 判断是否发生碰撞（用于非SINR模式或感知数据）
    #                 collision_occurred = resource_block_status[resource.subchannel]['attack'] or \
    #                                     resource_block_status[resource.subchannel]['collision']
                    
    #                 # 实际接收成功判断（使用SINR或简单冲突检测）
    #                 reception_success = False
                    
    #                 if self.useSINR:
    #                     # SINR模式：收集同资源块上的其他发送者作为干扰源
    #                     interferers = []
    #                     for other_sender, _, _ in users:
    #                         if other_sender.id != sender.id and isinstance(other_sender, (Vehicle, RLAttacker, FixAttacker)):
    #                             interferers.append(other_sender.position)
                        
    #                     # 计算SINR
    #                     sinr = self._calculate_sinr(receiver.position, sender.position, interferers)
    #                     reception_success = sinr >= self.sinr_threshold
    #                 else:
    #                     # 非SINR模式：无冲突即为成功
    #                     reception_success = not collision_occurred
                    
    #                 # 处理接收
    #                 if isinstance(receiver, Vehicle):
    #                     # 车辆会记录所有传输（包括攻击者的）
    #                     receiver.receive_packet(packet, resource, not reception_success)
    #                 else:
    #                     # 攻击者接收包用于感知
    #                     if isinstance(sender, (RLAttacker, FixAttacker)):
    #                         pRsvp = 20  # 攻击者发送周期
    #                     else:
    #                         pRsvp = 100  # 普通车辆发送周期
                            
    #                     receiver.add_sensing_data(
    #                         resource.slot_id,
    #                         resource.subchannel,
    #                         pRsvp,
    #                         sender.id,
    #                         packet.timestamp
    #                     )
                    
    #                 # 更新消息状态（仅对非攻击者数据包）
    #                 if not packet.is_attack and isinstance(receiver, Vehicle):
    #                     msg_key = (sender.id, packet.packet_id)
    #                     if msg_key in self.message_status_dict:
    #                         # 如果此接收者接收失败，标记整个消息失败
    #                         if not reception_success:
    #                             self.message_status_dict[msg_key]['success'] = False
            
    #         # 更新碰撞统计
    #         normal_senders = set()
    #         attack_success = set()
    #         for sender, packet, resource in slot_transmissions:
    #             if not isinstance(sender, (RLAttacker, FixAttacker)):
    #                 normal_senders.add(sender.id)
            
    #         # 碰撞检测
    #         for subchannel, users in subchannel_usage.items():
    #             if len(users) > 1:  # 发生碰撞
    #                 # 检查是否有攻击者参与
    #                 has_attacker = any(isinstance(sender, (RLAttacker, FixAttacker)) for sender, _, _ in users)
    #                 if has_attacker:
    #                     # 攻击者成功造成碰撞
    #                     self.total_attack_success += len(normal_senders)
    #                     collision_info['collisions_caused'] += len(normal_senders)
    #                     for sender_id in normal_senders:
    #                         attack_success.add(sender_id)
    #                 self.collision_count += len(normal_senders)
            
    #         # 更新发送者碰撞计数
    #         for sender_id in normal_senders:
    #             # 更新车辆碰撞计数
    #             for vehicle in self.vehicles:
    #                 if vehicle.id == sender_id:
    #                     vehicle.collisions += 1
    #                     break
    #             # 更新攻击者成功次数
    #             if sender_id in attack_success:
    #                 for attacker in self.attackers:
    #                     if isinstance(attacker, FixAttacker):
    #                         attacker.collisions_caused += 1
    #                     elif isinstance(attacker, RLAttacker):
    #                         attacker.record_attack_success(True)
        
    #     return collision_info
    
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
    
    # def get_episode_stats(self):
    #     """获取当前轮的统计信息"""
    #     # 获取个人PRR
    #     vehicle_prrs = self.get_vehicle_prrs()
        
    #     return {
    #         'total_collisions': self.collision_count,
    #         'total_transmissions': self.transmission_count,
    #         'prr': self._calculate_current_prr(),
    #         'attack_success_rate': self.total_attack_success / max(1, self.attack_transmission_count),
    #         'collision_rate': self.collision_count / max(1, self.transmission_count),
    #         'message_failures': self.message_failures,
    #         'resource_block_attacks': self.resource_block_attacks,
    #         'resource_block_collisions': self.resource_block_collisions,
    #         'vehicle_prrs': vehicle_prrs  # 新增：个人PRR
    #     }
    def get_episode_stats(self):
        """获取当前轮的统计信息"""
        # 获取个人PRR
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
            'vehicle_prrs': vehicle_prrs  # 确保包含这个键
        }
    
    def get_vehicle_prrs(self):
        """获取所有车辆的个人PRR"""
        vehicle_prrs = {}
        for vehicle in self.vehicles:
            vehicle_prrs[vehicle.id] = vehicle.calculate_prr()
        return vehicle_prrs
    def reset_stats(self):
        """重置所有统计"""
        self.collision_count = 0
        self.transmission_count = 0
        self.total_expected_packets = 0
        self.total_received_packets = 0
        self.attack_transmission_count = 0
        self.total_attack_success = 0
        self.message_failures = 0  # 消息层失败计数
        self.resource_block_attacks = 0  # 资源块攻击失效计数
        self.resource_block_collisions = 0  # 资源块碰撞失效计数
    
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

        # 只在第一次初始化时保存
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
            else:  # Fix attacker
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

        # 只在第一次初始化时保存
        if self.initial_attacker_states is None:
            self.initial_attacker_states = attacker_states
    
    # def get_episode_stats(self):
    #     """获取当前轮的统计信息"""
    #     return {
    #         'total_collisions': self.collision_count,
    #         'total_transmissions': self.transmission_count,
    #         'prr': self._calculate_current_prr(),
    #         'attack_success_rate': self.total_attack_success / max(1, self.attack_transmission_count),
    #         'collision_rate': self.collision_count / max(1, self.transmission_count),
    #         'message_failures': self.message_failures,
    #         'resource_block_attacks': self.resource_block_attacks,
    #         'resource_block_collisions': self.resource_block_collisions
    #     }
    
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

# 添加一个简单的按钮功能（在实际环境中可能需要GUI集成）
def create_gui_button(env):
    """创建一个简单的控制台按钮来切换攻击模式"""
    print("\n" + "="*50)
    print("按 't' 切换攻击模式 (当前: " + 
          ("目标侧链攻击" if TARGETED_ATTACK_MODE else "侧链资源耗尽") + ")")
    print("按 'q' 退出")
    print("="*50)
    
    while True:
        key = input("输入命令: ").strip().lower()
        if key == 't':
            new_mode = env.toggle_attack_mode()
            print(f"攻击模式已切换为: {'目标侧链攻击' if new_mode else '侧链资源耗尽'}")
        elif key == 'q':
            break
        else:
            print("无效命令，请重新输入")