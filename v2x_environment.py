import numpy as np
import gym
from gym import spaces
import math
import random
from collections import defaultdict
import logging

logger = logging.getLogger('V2X-Environment')

class Packet:
    """Represents a V2X packet for transmission"""
    
    def __init__(self, sender_id, timestamp, position, size=190, is_attack=False):
        self.sender_id = sender_id
        self.timestamp = timestamp
        self.position = position
        self.size = size
        self.is_attack = is_attack

class SensingData:
    """Represents sensing data from previous transmissions"""
    
    def __init__(self, subframe_info, rb_start, rb_len, rsrp, rssi, pRsvp, sender_id, timestamp):
        self.subframe_info = subframe_info
        self.rb_start = rb_start
        self.rb_len = rb_len
        self.rsrp = rsrp
        self.rssi = rssi
        self.pRsvp = pRsvp
        self.sender_id = sender_id
        self.timestamp = timestamp

class SubframeInfo:
    """Represents a specific subframe in the LTE frame structure"""
    
    def __init__(self, frame_no, subframe_no):
        self.frame_no = frame_no
        self.subframe_no = subframe_no
    
    def __eq__(self, other):
        if not isinstance(other, SubframeInfo):
            return False
        return self.frame_no == other.frame_no and self.subframe_no == other.subframe_no
    
    def __lt__(self, other):
        if self.frame_no < other.frame_no:
            return True
        elif self.frame_no == other.frame_no:
            return self.subframe_no < other.subframe_no
        return False
    
    def __hash__(self):
        return hash((self.frame_no, self.subframe_no))
    
    def __repr__(self):
        return f"({self.frame_no}, {self.subframe_no})"

class ResourceInfo:
    """Represents a resource allocation for transmission"""
    
    def __init__(self, subframe, subchannel, rb_start=None, rb_len=None):
        self.subframe = subframe
        self.subchannel = subchannel
        self.rb_start = rb_start
        self.rb_len = rb_len
    
    def __eq__(self, other):
        if not isinstance(other, ResourceInfo):
            return False
        return (self.subframe == other.subframe and 
                self.subchannel == other.subchannel)
    
    def __hash__(self):
        return hash((self.subframe, self.subchannel))

class Vehicle:
    """Represents a vehicle with V2X capability"""
    
    def __init__(self, vehicle_id, initial_position, initial_velocity, resource_pool):
        self.id = vehicle_id
        self.position = initial_position
        self.velocity = initial_velocity
        self.resource_pool = resource_pool
        
        # Resource selection parameters
        self.resel_counter = 0
        self.prob_resource_keep = random.uniform(0.2, 0.8)
        self.current_resource = None
        self.sensing_data = []
        self.next_transmission_time = 0
        
        # Sensing window parameters
        self.sensing_window_duration = 1000  # 1000ms sensing window
        self.rsrp_threshold = -110  # RSRP threshold in dBm for resource exclusion
        
        # Initialize stats
        self.packets_sent = 0
        self.packets_received = 0
        self.collisions = 0
        self.successful_transmissions = 0
    
    def move(self, delta_time):
        """Update vehicle position based on velocity and time delta"""
        self.position = self.position + self.velocity * delta_time
        
        # Handle boundary conditions with reflection
        if self.position[0] >= 1000:
            self.position[0] = 1000 - (self.position[0] - 1000)
            self.velocity = -self.velocity
            self.position[1] = 10.0
        if self.position[0] <= 0:
            self.position[0] = -self.position[0]
            self.velocity = -self.velocity
            self.position[1] = 5.0
    
    def select_resources(self, current_time):
        """Enhanced resource selection using sensing and selection windows"""
        # Step 1: Update sensing window
        self._update_sensing_window(current_time)
        
        # Step 2: Create selection window (T1=4ms to T2=100ms)
        selection_window = self._create_selection_window(current_time)
        
        # Step 3: Calculate candidate resources using sensing data
        candidate_resources = self._calculate_candidate_resources(selection_window, current_time)
        
        # Step 4: Select best resource from candidates
        if not candidate_resources:
            selected_resource = self._select_random_resource(selection_window)
        else:
            selected_resource = self._select_best_resource(candidate_resources)
        
        # Set resource block parameters
        selected_resource.rb_start = selected_resource.subchannel * self.resource_pool.subchannel_size
        selected_resource.rb_len = self.resource_pool.subchannel_size
        
        # Update reselection counter (5-15 cycles)
        self.resel_counter = random.randint(5, 15)
        self.current_resource = selected_resource
        
        return selected_resource
    
    def _create_selection_window(self, current_time):
        """Create the resource selection window based on T1 and T2 parameters"""
        t1_time = current_time + 4  # T1 = 4ms
        t2_time = current_time + 100  # T2 = 100ms
        
        selection_window = []
        
        # Generate all possible resources in the time window
        for time_offset in range(4, 101):  # From T1 to T2
            future_time = current_time + time_offset
            
            # Convert to subframe information
            frame_no = math.floor(future_time / 10) % 1024
            subframe_no = (math.floor(future_time % 10)) + 1
            if subframe_no > 10:
                subframe_no = 1
                frame_no = (frame_no + 1) % 1024
            
            subframe = SubframeInfo(frame_no, subframe_no)
            
            # For each subframe, consider all subchannels
            for subchannel in range(self.resource_pool.num_subchannels):
                selection_window.append(ResourceInfo(subframe, subchannel))
        
        return selection_window
    
    def _calculate_candidate_resources(self, selection_window, current_time):
        """Calculate candidate resources by excluding those with high RSRP"""
        candidate_resources = []
        
        for resource in selection_window:
            excluded = False
            
            # Check if this resource conflicts with sensed transmissions
            for sensing_data in self.sensing_data:
                if self._would_overlap_in_future(resource, sensing_data, current_time):
                    # Exclude if RSRP is above threshold
                    if sensing_data.rsrp > self.rsrp_threshold:
                        excluded = True
                        break
            
            if not excluded:
                candidate_resources.append(resource)
        
        # Ensure at least 20% of resources remain as candidates
        min_candidates = max(1, int(0.2 * len(selection_window)))
        if len(candidate_resources) < min_candidates:
            candidate_resources = self._get_resources_by_lowest_rsrp(selection_window, min_candidates, current_time)
        
        return candidate_resources
    
    def _would_overlap_in_future(self, resource, sensing_data, current_time):
        """Check if a future resource would overlap with a sensed transmission considering periodicity"""
        sensed_time = sensing_data.timestamp
        resource_time = self._resource_to_time(resource, current_time)
        
        time_diff = resource_time - sensed_time
        
        # Check if the time difference matches the reservation period
        if time_diff > 0 and (time_diff % sensing_data.pRsvp) == 0:
            # Check frequency domain overlap
            resource_rb_start = resource.subchannel * self.resource_pool.subchannel_size
            resource_rb_end = resource_rb_start + self.resource_pool.subchannel_size - 1
            
            sensed_rb_start = sensing_data.rb_start
            sensed_rb_end = sensed_rb_start + sensing_data.rb_len - 1
            
            # Check for overlap in frequency domain
            return not (resource_rb_end < sensed_rb_start or resource_rb_start > sensed_rb_end)
        
        return False
    
    def _resource_to_time(self, resource, current_time):
        """Convert resource subframe to absolute time"""
        frame_time = resource.subframe.frame_no * 10
        subframe_time = resource.subframe.subframe_no - 1
        return frame_time + subframe_time
    
    def _get_resources_by_lowest_rsrp(self, resources, count, current_time):
        """Get resources with lowest average RSRP values"""
        resource_rsrp = []
        
        for resource in resources:
            avg_rsrp = self._calculate_avg_rsrp_for_resource(resource, current_time)
            resource_rsrp.append((resource, avg_rsrp))
        
        # Sort by RSRP (lowest first)
        resource_rsrp.sort(key=lambda x: x[1])
        
        return [r[0] for r in resource_rsrp[:count]]
    
    def _calculate_avg_rsrp_for_resource(self, resource, current_time):
        """Calculate average RSRP for a resource based on sensing data"""
        relevant_rsrp = []
        
        for sensing_data in self.sensing_data:
            if self._would_overlap_in_future(resource, sensing_data, current_time):
                relevant_rsrp.append(sensing_data.rsrp)
        
        if not relevant_rsrp:
            return -140  # Very low RSRP if no data
        
        return sum(relevant_rsrp) / len(relevant_rsrp)
    
    def _select_best_resource(self, candidate_resources):
        """Select the best resource from candidates based on S-RSSI metric"""
        resource_metrics = []
        
        for resource in candidate_resources:
            # Calculate average S-RSSI for this resource
            avg_s_rssi = self._calculate_avg_s_rssi(resource)
            resource_metrics.append((resource, avg_s_rssi))
        
        # Sort by S-RSSI (lowest first - better reception conditions)
        resource_metrics.sort(key=lambda x: x[1])
        
        # Select randomly from the best 20%
        num_best = max(1, int(0.2 * len(resource_metrics)))
        best_resources = resource_metrics[:num_best]
        
        selected = random.choice(best_resources)
        return selected[0]
    
    def _calculate_avg_s_rssi(self, resource):
        """Calculate average S-RSSI for a resource"""
        relevant_rssi = []
        
        for sensing_data in self.sensing_data:
            # Check if this sensing data is relevant to the resource
            if self._sensing_data_relevant_to_resource(resource, sensing_data):
                relevant_rssi.append(sensing_data.rssi)
        
        if not relevant_rssi:
            return -140  # Very low S-RSSI if no data
        
        return sum(relevant_rssi) / len(relevant_rssi)
    
    def _sensing_data_relevant_to_resource(self, resource, sensing_data):
        """Check if sensing data is relevant to a resource for S-RSSI calculation"""
        # Check frequency domain overlap
        resource_rb_start = resource.subchannel * self.resource_pool.subchannel_size
        resource_rb_end = resource_rb_start + self.resource_pool.subchannel_size - 1
        
        sensed_rb_start = sensing_data.rb_start
        sensed_rb_end = sensed_rb_start + sensing_data.rb_len - 1
        
        return not (resource_rb_end < sensed_rb_start or resource_rb_start > sensed_rb_end)
    
    def _select_random_resource(self, selection_window):
        """Select a random resource from the selection window (fallback)"""
        resource = random.choice(selection_window)
        resource.rb_start = resource.subchannel * self.resource_pool.subchannel_size
        resource.rb_len = self.resource_pool.subchannel_size
        return resource
    
    def _update_sensing_window(self, current_time):
        """Update sensing window by removing old entries"""
        sensing_window_start = current_time - self.sensing_window_duration
        
        # Remove data outside the sensing window
        self.sensing_data = [data for data in self.sensing_data 
                            if data.timestamp >= sensing_window_start]
    
    def add_sensing_data(self, subframe_info, rb_start, rb_len, rsrp, rssi, pRsvp, sender_id, timestamp):
        """Add sensing data from a received transmission"""
        sensing_data = SensingData(subframe_info, rb_start, rb_len, rsrp, rssi, pRsvp, sender_id, timestamp)
        self.sensing_data.append(sensing_data)
    
    def send_packet(self, current_time):
        """Send a packet using the selected resource"""
        if current_time < self.next_transmission_time:
            return None
        
        # Select resources if needed
        if not self.current_resource or self.resel_counter <= 0:
            self.current_resource = self.select_resources(current_time)
        
        # Create packet
        packet = Packet(self.id, current_time, self.position)
        
        # Schedule next transmission (100ms cycle)
        self.next_transmission_time = current_time + 100
        
        # Update stats
        self.packets_sent += 1
        
        return (packet, self.current_resource)
    
    def receive_packet(self, packet, resource, collision_occurred, rsrp, rssi):
        """Process a received packet with signal measurements"""
        if hasattr(packet, 'is_attack') and packet.is_attack:
            pRsvp = 20  # Attacker transmission cycle
        else:
            pRsvp = 100  # Normal vehicle transmission cycle
        
        # Add sensing data
        self.add_sensing_data(
            resource.subframe, 
            resource.rb_start, 
            resource.rb_len, 
            rsrp, 
            rssi, 
            pRsvp,
            packet.sender_id, 
            packet.timestamp
        )
        
        # Process reception
        if not collision_occurred:
            self.packets_received += 1
            return True
        else:
            return False
    
    def should_receive_packet(self, sender_position, communication_range):
        """Determine if this vehicle should receive a packet from sender"""
        distance = np.linalg.norm(self.position - sender_position)
        return distance <= communication_range

class ResourcePool:
    """Manages the sidelink resource pool for V2X communication"""
    
    def __init__(self, num_subchannels=5, num_slots=100, subchannel_size=10):
        self.num_subchannels = num_subchannels
        self.num_slots = num_slots
        self.subchannel_size = subchannel_size
        self.total_rbs = num_subchannels * subchannel_size

class Channel:
    """Simulates the wireless channel with realistic path loss calculations"""
    
    def __init__(self):
        # Channel parameters
        self.frequency = 5.9e9  # 5.9 GHz (V2X frequency)
        self.speed_of_light = 3e8
        
    def calculate_path_loss(self, tx_position, rx_position):
        """Calculate path loss using highway propagation model"""
        # Calculate distance
        dx = tx_position[0] - rx_position[0]
        dy = tx_position[1] - rx_position[1]
        distance = math.sqrt(dx**2 + dy**2)
        distance = max(distance, 1.0)  # Minimum 1 meter
        
        # WINNER+ B1 highway model parameters
        d0 = 10  # Reference distance in meters
        n = 3.8  # Path loss exponent for highway
        sigma = 6  # Shadow fading standard deviation in dB
        
        if distance <= d0:
            # Free space path loss for short distances
            path_loss_db = 20 * math.log10(distance) + 20 * math.log10(self.frequency) - 147.55
        else:
            # Two-slope model for longer distances
            path_loss_db = (20 * math.log10(d0) + 
                           10 * n * math.log10(distance/d0) + 
                           20 * math.log10(self.frequency) - 147.55)
        
        # Add shadow fading
        shadow_fading = random.gauss(0, sigma)
        
        return path_loss_db + shadow_fading
    
    def calculate_rsrp(self, tx_power_dbm, path_loss_db):
        """Calculate RSRP (Reference Signal Received Power)"""
        # RSRP is calculated per resource block
        num_rbs = 50  # Total number of resource blocks in the channel
        rb_power_dbm = tx_power_dbm - 10 * math.log10(num_rbs)
        
        # Calculate RSRP
        rsrp_dbm = rb_power_dbm - path_loss_db
        
        # Apply realistic constraints
        rsrp_dbm = min(rsrp_dbm, tx_power_dbm)  # Cannot exceed transmit power
        rsrp_dbm = max(rsrp_dbm, -140)  # Minimum detectable RSRP
        
        return rsrp_dbm
    
    def calculate_rssi(self, rsrp_dbm, interference_dbm=-200):
        """Calculate RSSI (Received Signal Strength Indicator)"""
        # Convert to linear scale
        rsrp_linear = 10**(rsrp_dbm/10) if rsrp_dbm > -200 else 0
        interference_linear = 10**(interference_dbm/10) if interference_dbm > -200 else 0
        noise_linear = 10**(-95/10)  # Thermal noise at -95 dBm
        
        # Total received power
        total_power_linear = rsrp_linear + interference_linear + noise_linear
        
        # Convert back to dB
        if total_power_linear > 0:
            rssi_dbm = 10 * math.log10(total_power_linear)
        else:
            rssi_dbm = -200
        
        return rssi_dbm
    
    def calculate_interference(self, receiver_pos, interfering_transmissions, target_resource):
        """Calculate interference from other transmissions"""
        total_interference = 0
        
        for sender, packet, resource in interfering_transmissions:
            # Check if this transmission uses overlapping resources
            if self._resources_overlap(target_resource, resource):
                # Calculate path loss to interferer
                path_loss = self.calculate_path_loss(sender.position, receiver_pos)
                # Calculate interference power (assuming same transmit power)
                interference_power = 23.0 - path_loss  # 23 dBm transmit power
                # Convert to linear and add
                total_interference += 10**(interference_power/10)
        
        # Convert back to dB
        if total_interference > 0:
            return 10 * math.log10(total_interference)
        else:
            return -200
    
    def _resources_overlap(self, resource1, resource2):
        """Check if two resources overlap in frequency domain"""
        r1_start = resource1.rb_start
        r1_end = r1_start + resource1.rb_len - 1
        r2_start = resource2.rb_start
        r2_end = r2_start + resource2.rb_len - 1
        
        return not (r1_end < r2_start or r1_start > r2_end)

class V2XEnvironment(gym.Env):
    """V2X Environment for PPO-based attacker training"""
    
    def __init__(self, num_vehicles=20, communication_range=320.0, tx_power=23.0):
        super(V2XEnvironment, self).__init__()
        
        # Environment parameters
        self.num_vehicles = num_vehicles
        self.communication_range = communication_range
        self.tx_power = tx_power
        
        # Initialize components
        self.resource_pool = ResourcePool(num_subchannels=5, num_slots=100, subchannel_size=10)
        self.channel = Channel()
        
        # Initialize vehicles
        self.vehicles = []
        self._initialize_vehicles()
        
        # Environment state
        self.current_time = 0
        self.episode_length = 1000  # Episode length in time steps (ms)
        self.step_count = 0
        
        # Attacker parameters
        self.attacker_position = np.array([500.0, 0.0])  # Fixed position
        self.attacker_transmission_cycle = 20  # 20ms transmission cycle
        self.next_attack_time = 0
        
        # Sensing data for attacker
        self.attacker_sensing_data = []
        self.sensing_window_duration = 100  # 100ms sensing window
        
        # Define action and observation spaces
        # Action space: [frame_no, subframe_no, subchannel]
        # frame_no: 0-1023, subframe_no: 1-10, subchannel: 0-4
        self.action_space = spaces.Box(
            low=np.array([0, 1, 0]), 
            high=np.array([1023, 10, self.resource_pool.num_subchannels-1]), 
            dtype=np.int32
        )
        
        # Observation space: resource pool info + sensing data statistics + subchannel usage + time
        obs_dim = 4 + 1 + self.resource_pool.num_subchannels + 1  # 11 dimensions
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        # Statistics
        self.episode_collisions = 0
        self.episode_attacks = 0
        self.episode_successes = 0
        
    def _initialize_vehicles(self):
        """Initialize vehicles with random positions and velocities"""
        lane1_y = 5.0
        lane2_y = 10.0
        highway_length = 1000.0
        
        for i in range(self.num_vehicles):
            # Alternate between lanes
            lane_y = lane1_y if i % 2 == 0 else lane2_y
            
            # Random x position along the highway
            pos_x = random.uniform(0, highway_length)
            
            # Create position and velocity
            position = np.array([pos_x, lane_y])
            velocity = np.array([10.0 if lane_y == lane1_y else -10.0, 0.0])
            
            # Create vehicle
            vehicle = Vehicle(i, position, velocity, self.resource_pool)
            self.vehicles.append(vehicle)
    
    def reset(self):
        """Reset the environment to initial state"""
        # Reset time and counters
        self.current_time = 0
        self.step_count = 0
        self.next_attack_time = 0
        
        # Reset statistics
        self.episode_collisions = 0
        self.episode_attacks = 0
        self.episode_successes = 0
        
        # Reset vehicles
        self._initialize_vehicles()
        
        # Clear sensing data
        self.attacker_sensing_data = []
        
        # Return initial observation
        return self._get_observation()
    
    def step(self, action):
        """Execute one time step within the environment"""
        # Convert action to resource selection
        frame_no = int(action[0])
        subframe_no = int(action[1])
        subchannel = int(action[2])
        
        # Ensure action is within bounds
        frame_no = np.clip(frame_no, 0, 1023)
        subframe_no = np.clip(subframe_no, 1, 10)
        subchannel = np.clip(subchannel, 0, self.resource_pool.num_subchannels - 1)
        
        # Process one time step
        reward = self._process_time_step(frame_no, subframe_no, subchannel)
        
        # Update time and step count
        self.current_time += 1
        self.step_count += 1
        
        # Check if episode is done
        done = self.step_count >= self.episode_length
        
        # Get next observation
        obs = self._get_observation()
        
        # Additional info
        info = {
            'episode_collisions': self.episode_collisions,
            'episode_attacks': self.episode_attacks,
            'episode_successes': self.episode_successes,
            'success_rate': self.episode_successes / max(1, self.episode_attacks)
        }
        
        return obs, reward, done, info
    
    def _process_time_step(self, frame_no, subframe_no, subchannel):
        """Process one time step of the simulation"""
        reward = 0.0
        
        # Update vehicle positions
        for vehicle in self.vehicles:
            vehicle.move(1 / 1000.0)  # 1ms time step
        
        # Collect vehicle transmissions
        vehicle_transmissions = []
        for vehicle in self.vehicles:
            tx_result = vehicle.send_packet(self.current_time)
            vehicle.resel_counter -= 1
            if vehicle.resel_counter <= 0:
                vehicle.current_resource = None
            if tx_result:
                packet, resource = tx_result
                vehicle_transmissions.append((vehicle, packet, resource))
        
        # Process attacker transmission if it's time
        attack_transmission = None
        if self.current_time >= self.next_attack_time:
            attack_transmission = self._create_attack_transmission(frame_no, subframe_no, subchannel)
            self.next_attack_time = self.current_time + self.attacker_transmission_cycle
            self.episode_attacks += 1
        
        # Combine all transmissions
        all_transmissions = vehicle_transmissions.copy()
        if attack_transmission:
            all_transmissions.append(attack_transmission)
        
        # Process transmissions and detect collisions
        if all_transmissions:
            collision_occurred = self._handle_transmissions(all_transmissions, attack_transmission)
            
            # Calculate reward based on collision success
            if attack_transmission and collision_occurred:
                reward = 1.0  # Positive reward for successful attack
                self.episode_successes += 1
            elif attack_transmission:
                reward = -0.1  # Small negative reward for failed attack
        
        return reward
    
    def _create_attack_transmission(self, frame_no, subframe_no, subchannel):
        """Create an attack transmission with given parameters"""
        subframe = SubframeInfo(frame_no, subframe_no)
        resource = ResourceInfo(subframe, subchannel)
        resource.rb_start = subchannel * self.resource_pool.subchannel_size
        resource.rb_len = self.resource_pool.subchannel_size
        
        # Create attack packet
        attack_packet = Packet(
            sender_id=999,  # Special ID for attacker
            timestamp=self.current_time,
            position=self.attacker_position,
            is_attack=True
        )
        
        return (None, attack_packet, resource)  # None for sender since it's not a vehicle object
    
    def _handle_transmissions(self, transmissions, attack_transmission):
        """Handle transmissions and detect collisions"""
        # Group by subframe
        tx_by_subframe = defaultdict(list)
        
        for sender, packet, resource in transmissions:
            sf_idx = (resource.subframe.frame_no * 10 + resource.subframe.subframe_no - 1) % self.resource_pool.num_slots
            tx_by_subframe[sf_idx].append((sender, packet, resource))
        
        attack_collision = False
        
        # Process each subframe
        for sf_idx, sf_transmissions in tx_by_subframe.items():
            # Detect collisions
            rb_usage = defaultdict(list)
            
            for sender, packet, resource in sf_transmissions:
                for rb in range(resource.rb_start, resource.rb_start + resource.rb_len):
                    rb_usage[rb].append((sender, packet, resource))
            
            # Find collided transmissions
            collided_transmissions = set()
            for rb, users in rb_usage.items():
                if len(users) > 1:
                    for sender, packet, resource in users:
                        if packet.is_attack:
                            attack_collision = True
                        if sender is not None:  # Vehicle transmission
                            collided_transmissions.add(sender.id)
            
            # Update collision statistics
            if collided_transmissions:
                self.episode_collisions += len(collided_transmissions)
                
                # Update individual vehicle collision counts
                for sender_id in collided_transmissions:
                    for vehicle in self.vehicles:
                        if vehicle.id == sender_id:
                            vehicle.collisions += 1
                            break
            
            # Process packet reception for vehicles
            for receiver in self.vehicles:
                for sender, packet, resource in sf_transmissions:
                    if sender is not None and sender.id != receiver.id:
                        if receiver.should_receive_packet(sender.position, self.communication_range):
                            collision_occurred = sender.id in collided_transmissions
                            
                            if not collision_occurred:
                                # Calculate signal parameters
                                path_loss = self.channel.calculate_path_loss(sender.position, receiver.position)
                                rsrp = self.channel.calculate_rsrp(self.tx_power, path_loss)
                                
                                # Calculate interference
                                interfering_tx = [(s, p, r) for s, p, r in sf_transmissions 
                                                if s is not None and s.id != sender.id and s.id != receiver.id]
                                interference = self.channel.calculate_interference(receiver.position, interfering_tx, resource)
                                
                                rssi = self.channel.calculate_rssi(rsrp, interference)
                                
                                # Process reception
                                receiver.receive_packet(packet, resource, collision_occurred, rsrp, rssi)
                    
                    # Attacker sensing: collect data from all transmissions
                    if sender is not None:  # Only from vehicle transmissions
                        distance = np.linalg.norm(self.attacker_position - sender.position)
                        if distance <= self.communication_range:
                            self._add_attacker_sensing_data(resource, sender.id, packet.timestamp)
        
        return attack_collision
    
    def _add_attacker_sensing_data(self, resource, sender_id, timestamp):
        """Add sensing data for the attacker"""
        sensing_data = SensingData(
            resource.subframe,
            resource.rb_start,
            resource.rb_len,
            rsrp=0,  # Dummy values for attacker
            rssi=0,
            pRsvp=100,  # Assume 100ms reservation period
            sender_id=sender_id,
            timestamp=timestamp
        )
        self.attacker_sensing_data.append(sensing_data)
        
        # Update sensing window
        self._update_attacker_sensing_window()
    
    def _update_attacker_sensing_window(self):
        """Maintain only recent sensing data for attacker"""
        if not self.attacker_sensing_data:
            return
        
        window_start = self.current_time - self.sensing_window_duration
        self.attacker_sensing_data = [data for data in self.attacker_sensing_data 
                                     if data.timestamp >= window_start]
    
    def _get_observation(self):
        """Get current observation for the attacker"""
        # Resource pool information
        obs = [
            self.resource_pool.num_subchannels,
            self.resource_pool.num_slots,
            self.resource_pool.subchannel_size,
            self.resource_pool.total_rbs
        ]
        
        # Sensing data statistics
        obs.append(len(self.attacker_sensing_data))
        
        # Subchannel usage statistics
        subchannel_usage = [0] * self.resource_pool.num_subchannels
        for data in self.attacker_sensing_data:
            subchannel = data.rb_start // self.resource_pool.subchannel_size
            if 0 <= subchannel < len(subchannel_usage):
                subchannel_usage[subchannel] += 1
        
        obs.extend(subchannel_usage)
        
        # Time-based features
        obs.append(self.current_time % 1000)  # Current time modulo 1000ms
        
        return np.array(obs, dtype=np.float32)
    
    def render(self, mode='human'):
        """Render the environment (optional)"""
        if mode == 'human':
            print(f"Time: {self.current_time}, Attacks: {self.episode_attacks}, "
                  f"Successes: {self.episode_successes}, Collisions: {self.episode_collisions}")
    
    def close(self):
        """Clean up the environment"""
        pass