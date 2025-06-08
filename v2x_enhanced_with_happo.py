import numpy as np
import matplotlib.pyplot as plt
import random
import time
from collections import defaultdict
import logging
import math

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('V2X-MODE4-Enhanced-HAPPO')

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
        self.pRsvp = pRsvp  # Reservation period in ms
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
    
    def __init__(self, vehicle_id, initial_position, initial_velocity, sim):
        self.id = vehicle_id
        self.position = initial_position
        self.velocity = initial_velocity
        self.sim = sim
        
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
            logger.warning(f"Vehicle {self.id}: No candidate resources, selecting random")
            selected_resource = self._select_random_resource(selection_window)
        else:
            selected_resource = self._select_best_resource(candidate_resources)
        
        # Set resource block parameters
        selected_resource.rb_start = selected_resource.subchannel * self.sim.resource_pool.subchannel_size
        selected_resource.rb_len = self.sim.resource_pool.subchannel_size
        
        # Update reselection counter (5-15 cycles)
        self.resel_counter = random.randint(5, 15)
        self.current_resource = selected_resource
        
        logger.debug(f"Vehicle {self.id} selected resource: subframe {selected_resource.subframe}, " +
                     f"subchannel: {selected_resource.subchannel}")
        
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
            for subchannel in range(self.sim.resource_pool.num_subchannels):
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
                        logger.debug(f"Vehicle {self.id}: Excluding resource due to high RSRP: {sensing_data.rsrp:.1f} dBm")
                        break
            
            if not excluded:
                candidate_resources.append(resource)
        
        # Ensure at least 20% of resources remain as candidates
        min_candidates = max(1, int(0.2 * len(selection_window)))
        if len(candidate_resources) < min_candidates:
            logger.debug(f"Vehicle {self.id}: Too few candidates ({len(candidate_resources)}), " +
                        f"selecting {min_candidates} with lowest RSRP")
            candidate_resources = self._get_resources_by_lowest_rsrp(selection_window, min_candidates, current_time)
        
        return candidate_resources
    
    def _would_overlap_in_future(self, resource, sensing_data, current_time):
        """Check if a future resource would overlap with a sensed transmission considering periodicity"""
        # Calculate the time difference between the sensed transmission and the potential future transmission
        sensed_time = sensing_data.timestamp
        resource_time = self._resource_to_time(resource, current_time)
        
        time_diff = resource_time - sensed_time
        
        # Check if the time difference matches the reservation period
        if time_diff > 0 and (time_diff % sensing_data.pRsvp) == 0:
            # Check frequency domain overlap
            resource_rb_start = resource.subchannel * self.sim.resource_pool.subchannel_size
            resource_rb_end = resource_rb_start + self.sim.resource_pool.subchannel_size - 1
            
            sensed_rb_start = sensing_data.rb_start
            sensed_rb_end = sensed_rb_start + sensing_data.rb_len - 1
            
            # Check for overlap in frequency domain
            return not (resource_rb_end < sensed_rb_start or resource_rb_start > sensed_rb_end)
        
        return False
    
    def _resource_to_time(self, resource, current_time):
        """Convert resource subframe to absolute time"""
        # This is a simplified conversion - in practice would need more sophisticated timing
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
        resource_rb_start = resource.subchannel * self.sim.resource_pool.subchannel_size
        resource_rb_end = resource_rb_start + self.sim.resource_pool.subchannel_size - 1
        
        sensed_rb_start = sensing_data.rb_start
        sensed_rb_end = sensed_rb_start + sensing_data.rb_len - 1
        
        return not (resource_rb_end < sensed_rb_start or resource_rb_start > sensed_rb_end)
    
    def _select_random_resource(self, selection_window):
        """Select a random resource from the selection window (fallback)"""
        resource = random.choice(selection_window)
        resource.rb_start = resource.subchannel * self.sim.resource_pool.subchannel_size
        resource.rb_len = self.sim.resource_pool.subchannel_size
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
    
    def should_receive_packet(self, sender_position):
        """Determine if this vehicle should receive a packet from sender"""
        distance = np.linalg.norm(self.position - sender_position)
        return distance <= self.sim.communication_range

class EnhancedAttacker:
    """Enhanced attacker with reception logic and sensing capabilities"""
    
    def __init__(self, attacker_id, initial_position, initial_velocity, sim, happo_agent=None):
        self.id = attacker_id
        self.position = initial_position
        self.velocity = initial_velocity
        self.sim = sim
        self.happo_agent = happo_agent
        
        # Attacker-specific parameters
        self.next_transmission_time = 0
        self.transmission_cycle = 20  # 20ms transmission cycle (more aggressive)
        
        # Attack statistics
        self.attack_packets_sent = 0
        self.attack_success_count = 0
        self.collisions_caused = 0
        
        # Enhanced sensing capabilities
        self.sensing_data = []
        self.sensing_window_duration = 100  # 100ms sensing window for attackers
        
        # Resource pool information for HAPPO
        self.resource_pool_state = {
            'num_subchannels': sim.resource_pool.num_subchannels,
            'num_slots': sim.resource_pool.num_slots,
            'subchannel_size': sim.resource_pool.subchannel_size,
            'total_rbs': sim.resource_pool.total_rbs
        }
        
    def move(self, delta_time):
        """Update attacker position"""
        self.position = self.position + self.velocity * delta_time
    
    def should_receive_packet(self, sender_position):
        """Enhanced reception logic - attackers can receive packets within 320m range"""
        distance = np.linalg.norm(self.position - sender_position)
        return distance <= self.sim.communication_range
    
    def receive_packet(self, packet, resource, collision_occurred):
        """Process received packets for sensing data collection"""
        # Add sensing data for attack analysis (without actual signal measurements)
        self.add_sensing_data(
            resource.subframe,
            resource.rb_start,
            resource.rb_len,
            packet.sender_id,
            packet.timestamp
        )
        
        # Attackers don't process actual reception, only collect sensing data
        return False
    
    def add_sensing_data(self, subframe_info, rb_start, rb_len, sender_id, timestamp):
        """Add sensing data from received transmissions for attack analysis"""
        sensing_data = SensingData(
            subframe_info, 
            rb_start, 
            rb_len, 
            rsrp=0,  # Dummy values since attackers don't need actual signal measurements
            rssi=0, 
            pRsvp=100,  # Assume 100ms reservation period for vehicles
            sender_id=sender_id,
            timestamp=timestamp
        )
        self.sensing_data.append(sensing_data)
        
        # Update sensing window
        self._update_sensing_window()
    
    def _update_sensing_window(self):
        """Maintain only recent sensing data (100ms window)"""
        if not self.sensing_data:
            return
        
        # Get current time from the last sensing data
        current_time = self.sensing_data[-1].timestamp
        window_start = current_time - self.sensing_window_duration
        
        # Remove old sensing data
        self.sensing_data = [data for data in self.sensing_data 
                            if data.timestamp >= window_start]
    
    def get_state_for_happo(self, current_time):
        """Generate state representation for HAPPO agent"""
        # Create state vector based on resource pool information and sensing data
        state = []
        
        # Resource pool information
        state.extend([
            self.resource_pool_state['num_subchannels'],
            self.resource_pool_state['num_slots'],
            self.resource_pool_state['subchannel_size'],
            self.resource_pool_state['total_rbs']
        ])
        
        # Sensing data statistics
        state.append(len(self.sensing_data))  # Number of sensed transmissions
        
        # Resource usage statistics per subchannel
        subchannel_usage = [0] * self.resource_pool_state['num_subchannels']
        for data in self.sensing_data:
            subchannel = data.rb_start // self.resource_pool_state['subchannel_size']
            if 0 <= subchannel < len(subchannel_usage):
                subchannel_usage[subchannel] += 1
        
        state.extend(subchannel_usage)
        
        # Time-based features
        state.append(current_time % 1000)  # Current time modulo 1000ms
        
        return np.array(state, dtype=np.float32)
    
    def send_attack_packet(self, current_time):
        """Send an attack packet using HAPPO-based resource selection"""
        if current_time < self.next_transmission_time:
            return None
        
        # Get state for HAPPO agent
        state = self.get_state_for_happo(current_time)
        
        # Select resource using HAPPO agent or fallback to random
        if self.happo_agent:
            attack_resource = self.happo_agent.select_resource(state, current_time)
        else:
            attack_resource = self._select_random_resource(current_time)
        
        if attack_resource is None:
            return None
        
        # Create attack packet
        attack_packet = Packet(self.id, current_time, self.position, is_attack=True)
        
        # Schedule next attack transmission
        self.next_transmission_time = current_time + self.transmission_cycle
        
        # Update attack statistics
        self.attack_packets_sent += 1
        
        return (attack_packet, attack_resource)
    
    def _select_random_resource(self, current_time):
        """Fallback random resource selection"""
        frame_no = random.randint(0, 1023)
        subframe_no = random.randint(1, 10)
        subchannel = random.randint(0, self.sim.resource_pool.num_subchannels - 1)
        
        subframe = SubframeInfo(frame_no, subframe_no)
        resource = ResourceInfo(subframe, subchannel)
        resource.rb_start = subchannel * self.sim.resource_pool.subchannel_size
        resource.rb_len = self.sim.resource_pool.subchannel_size
        
        return resource
    
    def record_attack_success(self, collision_occurred):
        """Record attack success and provide feedback to HAPPO agent"""
        if collision_occurred:
            self.attack_success_count += 1
            reward = 1.0  # Positive reward for successful attack
        else:
            reward = -0.1  # Small negative reward for failed attack
        
        # Provide feedback to HAPPO agent if available
        if self.happo_agent:
            self.happo_agent.record_reward(reward)
        
        return reward

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

class EnhancedSimulation:
    """Enhanced V2X simulation with HAPPO-based attackers"""
    
    def __init__(self, num_vehicles=20, num_attackers=1, duration=50000, 
                 communication_range=320.0, tx_power=23.0):
        self.num_vehicles = num_vehicles
        self.num_attackers = num_attackers
        self.duration = duration
        self.communication_range = communication_range
        self.tx_power = tx_power  # Transmit power in dBm
        
        # Initialize components
        self.resource_pool = ResourcePool(num_subchannels=5, num_slots=100, subchannel_size=10)
        self.channel = Channel()
        
        # Initialize vehicles and attackers
        self.vehicles = []
        self.attackers = []
        self._initialize_vehicles()
        self._initialize_attackers()
        
        # Simulation state
        self.current_time = 0
        
        # Statistics
        self.collision_count = 0
        self.transmission_count = 0
        self.total_expected_packets = 0
        self.total_received_packets = 0
        self.attack_transmission_count = 0
        self.total_attack_success = 0
        
        # Time-based statistics
        self.collision_stats = defaultdict(int)
        self.transmission_stats = defaultdict(int)
        self.attack_stats = defaultdict(int)
        self.prr_stats = defaultdict(float)
    
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
            vehicle = Vehicle(i, position, velocity, self)
            self.vehicles.append(vehicle)
    
    def _initialize_attackers(self):
        """Initialize enhanced attackers with HAPPO agents"""
        highway_length = 1000.0
        
        for i in range(self.num_attackers):
            attacker_id = self.num_vehicles + i
            position = np.array([highway_length/2, 0])
            velocity = np.array([0.0, 0.0])
            
            # Create HAPPO agent for this attacker
            from happo_agent import HAPPOAgent
            happo_agent = HAPPOAgent(
                state_dim=4 + 1 + self.resource_pool.num_subchannels + 1,  # Resource pool + sensing + subchannel usage + time
                action_dim=3,  # frame_no, subframe_no, subchannel
                resource_pool=self.resource_pool
            )
            
            attacker = EnhancedAttacker(attacker_id, position, velocity, self, happo_agent)
            self.attackers.append(attacker)
    
    def run(self):
        """Run the enhanced simulation"""
        logger.info(f"Starting enhanced V2X MODE4 simulation with HAPPO-based attackers")
        logger.info(f"Vehicles: {self.num_vehicles}, Attackers: {self.num_attackers}, Duration: {self.duration}ms")
        
        time_step = 1  # 1ms time step
        
        while self.current_time < self.duration:
            # Update positions
            for vehicle in self.vehicles:
                vehicle.move(time_step / 1000.0)
            
            for attacker in self.attackers:
                attacker.move(time_step / 1000.0)
            
            # Process transmissions
            self._process_transmissions()
            
            # Record statistics
            if self.current_time % 1000 == 0:
                self._record_statistics()
                logger.info(f"Time: {self.current_time}ms, Transmissions: {self.transmission_count}, " +
                          f"Collisions: {self.collision_count}, Attack Success: {self.total_attack_success}, " +
                          f"PRR: {self._calculate_current_prr():.3f}")
            
            self.current_time += time_step
        
        logger.info("Enhanced simulation completed")
        self._print_results()
    
    def _process_transmissions(self):
        """Process all transmissions for the current time step"""
        transmissions = []
        attack_transmissions = []
        
        # Collect vehicle transmissions
        for vehicle in self.vehicles:
            tx_result = vehicle.send_packet(self.current_time)
            vehicle.resel_counter -= 1
            if vehicle.resel_counter <= 0:
                vehicle.current_resource = None
            if tx_result:
                packet, resource = tx_result
                transmissions.append((vehicle, packet, resource))
                self.transmission_count += 1
        
        # Collect attacker transmissions
        for attacker in self.attackers:
            attack_result = attacker.send_attack_packet(self.current_time)
            if attack_result:
                packet, resource = attack_result
                attack_transmissions.append((attacker, packet, resource))
                self.attack_transmission_count += 1
        
        # Process all transmissions
        all_transmissions = transmissions + attack_transmissions
        if all_transmissions:
            self._handle_transmissions(all_transmissions)
    
    def _handle_transmissions(self, transmissions):
        """Handle transmissions with enhanced collision detection and signal calculations"""
        # Group by subframe
        tx_by_subframe = defaultdict(list)
        
        for sender, packet, resource in transmissions:
            sf_idx = (resource.subframe.frame_no * 10 + resource.subframe.subframe_no - 1) % self.resource_pool.num_slots
            tx_by_subframe[sf_idx].append((sender, packet, resource))
        
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
                        collided_transmissions.add(sender.id)
            
            # Update collision statistics
            if collided_transmissions:
                num_collisions = len(collided_transmissions)
                self.collision_count += num_collisions
                
                # Update individual collision counts and provide feedback to attackers
                for sender_id in collided_transmissions:
                    for vehicle in self.vehicles:
                        if vehicle.id == sender_id:
                            vehicle.collisions += 1
                            break
                    for attacker in self.attackers:
                        if attacker.id == sender_id:
                            attacker.record_attack_success(True)
                            self.total_attack_success += 1
                            break
            
            # Process packet reception for both vehicles and attackers
            for receiver in self.vehicles + self.attackers:
                for sender, packet, resource in sf_transmissions:
                    if sender.id != receiver.id and receiver.should_receive_packet(sender.position):
                        if not isinstance(sender, EnhancedAttacker):
                            self.total_expected_packets += 1
                        
                        collision_occurred = sender.id in collided_transmissions
                        
                        if isinstance(receiver, Vehicle):
                            # For vehicles: calculate signal parameters and process reception
                            if not collision_occurred:
                                path_loss = self.channel.calculate_path_loss(sender.position, receiver.position)
                                rsrp = self.channel.calculate_rsrp(self.tx_power, path_loss)
                                
                                # Calculate interference from other transmissions
                                interfering_tx = [(s, p, r) for s, p, r in sf_transmissions 
                                                if s.id != sender.id and s.id != receiver.id]
                                interference = self.channel.calculate_interference(receiver.position, interfering_tx, resource)
                                
                                rssi = self.channel.calculate_rssi(rsrp, interference)
                                
                                # Process reception
                                success = receiver.receive_packet(packet, resource, collision_occurred, rsrp, rssi)
                                
                                if success and not isinstance(sender, EnhancedAttacker):
                                    self.total_received_packets += 1
                                    if hasattr(sender, 'successful_transmissions'):
                                        sender.successful_transmissions += 1
                        
                        elif isinstance(receiver, EnhancedAttacker):
                            # For attackers: only record sensing data
                            receiver.receive_packet(packet, resource, collision_occurred)
    
    def _calculate_current_prr(self):
        """Calculate current packet reception ratio"""
        if self.total_expected_packets > 0:
            return self.total_received_packets / self.total_expected_packets
        return 0.0
    
    def _record_statistics(self):
        """Record statistics at current time"""
        time_bin = self.current_time // 1000
        self.collision_stats[time_bin] = self.collision_count
        self.transmission_stats[time_bin] = self.transmission_count
        self.attack_stats[time_bin] = self.total_attack_success
        self.prr_stats[time_bin] = self._calculate_current_prr()
    
    def _print_results(self):
        """Print final simulation results"""
        logger.info("\n=========== ENHANCED HAPPO SIMULATION RESULTS ===========")
        logger.info(f"Total simulation time: {self.duration}ms")
        logger.info(f"Total transmissions: {self.transmission_count}")
        logger.info(f"Total collisions: {self.collision_count}")
        logger.info(f"Total expected packets: {self.total_expected_packets}")
        logger.info(f"Total received packets: {self.total_received_packets}")
        
        if self.total_expected_packets > 0:
            prr = self.total_received_packets / self.total_expected_packets
            collision_rate = self.collision_count / self.transmission_count if self.transmission_count > 0 else 0
            logger.info(f"Overall Packet Reception Ratio (PRR): {prr:.4f}")
            logger.info(f"Collision Rate: {collision_rate:.4f}")
        
        # Attack statistics
        if self.num_attackers > 0:
            logger.info(f"\n=========== HAPPO ATTACK STATISTICS ===========")
            logger.info(f"Total attack transmissions: {self.attack_transmission_count}")
            logger.info(f"Total attack successes: {self.total_attack_success}")
            
            for attacker in self.attackers:
                success_rate = attacker.attack_success_count / attacker.attack_packets_sent if attacker.attack_packets_sent > 0 else 0
                sensing_data_count = len(attacker.sensing_data)
                logger.info(f"Attacker {attacker.id}: Success Rate={success_rate:.3f}, " +
                          f"Sensing Data Entries={sensing_data_count}")
        
        # Vehicle statistics
        logger.info("\nVehicle Statistics:")
        for vehicle in self.vehicles:
            collision_rate = vehicle.collisions / vehicle.packets_sent if vehicle.packets_sent > 0 else 0
            logger.info(f"Vehicle {vehicle.id}: Sent={vehicle.packets_sent}, " +
                      f"Received={vehicle.packets_received}, Collisions={vehicle.collisions}, " +
                      f"Collision Rate={collision_rate:.3f}")
        
        self._plot_results()
    
    def _plot_results(self):
        """Generate comprehensive result plots"""
        plt.figure(figsize=(15, 12))
        
        # Plot 1: Transmissions and Collisions over Time
        plt.subplot(2, 3, 1)
        times = list(self.transmission_stats.keys())
        transmissions = list(self.transmission_stats.values())
        collisions = list(self.collision_stats.values())
        
        plt.plot(times, transmissions, 'b-', label='Total Transmissions', linewidth=2)
        plt.plot(times, collisions, 'r-', label='Total Collisions', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Cumulative Count')
        plt.title('Transmissions and Collisions over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Packet Reception Ratio over Time
        plt.subplot(2, 3, 2)
        prr_times = list(self.prr_stats.keys())
        prr_values = list(self.prr_stats.values())
        
        plt.plot(prr_times, prr_values, 'g-', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Packet Reception Ratio')
        plt.title('Packet Reception Ratio over Time')
        plt.ylim([0, 1])
        plt.grid(True, alpha=0.3)
        
        # Plot 3: HAPPO Attack Success over Time
        plt.subplot(2, 3, 3)
        attack_times = list(self.attack_stats.keys())
        attack_successes = list(self.attack_stats.values())
        
        plt.plot(attack_times, attack_successes, 'purple', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Cumulative Attack Successes')
        plt.title('HAPPO Attack Success over Time')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Collision Rate over Time
        plt.subplot(2, 3, 4)
        collision_rates = []
        for i, t in enumerate(times):
            if transmissions[i] > 0:
                rate = collisions[i] / transmissions[i]
            else:
                rate = 0
            collision_rates.append(rate)
        
        plt.plot(times, collision_rates, 'orange', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Collision Rate')
        plt.title('Collision Rate over Time')
        plt.ylim([0, 1])
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Per-Vehicle Collision Rates
        plt.subplot(2, 3, 5)
        vehicle_ids = [v.id for v in self.vehicles]
        vehicle_collision_rates = []
        
        for vehicle in self.vehicles:
            if vehicle.packets_sent > 0:
                rate = vehicle.collisions / vehicle.packets_sent
            else:
                rate = 0
            vehicle_collision_rates.append(rate)
        
        plt.bar(vehicle_ids, vehicle_collision_rates, alpha=0.7, color='red')
        plt.xlabel('Vehicle ID')
        plt.ylabel('Collision Rate')
        plt.title('Per-Vehicle Collision Rates')
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Attacker Sensing Data Distribution
        plt.subplot(2, 3, 6)
        if self.attackers:
            sensing_counts = [len(attacker.sensing_data) for attacker in self.attackers]
            attacker_ids = [attacker.id for attacker in self.attackers]
            
            plt.bar(attacker_ids, sensing_counts, alpha=0.7, color='purple')
            plt.xlabel('Attacker ID')
            plt.ylabel('Sensing Data Entries')
            plt.title('Attacker Sensing Data Collection')
        else:
            plt.text(0.5, 0.5, 'No Attackers', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Attacker Statistics (No Attackers)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('v2x_enhanced_happo_results.png', dpi=300, bbox_inches='tight')
        logger.info("Results plotted and saved to 'v2x_enhanced_happo_results.png'")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Create and run the enhanced simulation with HAPPO-based attackers
    sim = EnhancedSimulation(num_vehicles=20, num_attackers=1, duration=50000)
    sim.run()