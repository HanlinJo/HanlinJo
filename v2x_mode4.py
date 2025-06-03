# import numpy as np
# import matplotlib.pyplot as plt
# import random
# import time
# from collections import defaultdict
# import logging
# import math

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger('V2X-MODE4')

# class Packet:
#     """Represents a V2X packet for transmission"""
    
#     def __init__(self, sender_id, timestamp, position, size=190):
#         self.sender_id = sender_id
#         self.timestamp = timestamp
#         self.position = position
#         self.size = size  # size in bytes (default is 190B as per the original C++ code)


# class SensingData:
#     """Represents sensing data from previous transmissions"""
    
#     def __init__(self, subframe_info, rb_start, rb_len, rsrp, rssi, pRsvp, sender_id):
#         self.subframe_info = subframe_info
#         self.rb_start = rb_start
#         self.rb_len = rb_len
#         self.rsrp = rsrp
#         self.rssi = rssi
#         self.pRsvp = pRsvp
#         self.sender_id = sender_id


# class SubframeInfo:
#     """Represents a specific subframe in the LTE frame structure"""
    
#     def __init__(self, frame_no, subframe_no):
#         self.frame_no = frame_no
#         self.subframe_no = subframe_no
    
#     def __eq__(self, other):
#         if not isinstance(other, SubframeInfo):
#             return False
#         return self.frame_no == other.frame_no and self.subframe_no == other.subframe_no
    
#     def __lt__(self, other):
#         if self.frame_no < other.frame_no:
#             return True
#         elif self.frame_no == other.frame_no:
#             return self.subframe_no < other.subframe_no
#         return False
    
#     def __repr__(self):
#         return f"({self.frame_no}, {self.subframe_no})"


# class ResourceInfo:
#     """Represents a resource allocation for transmission"""
    
#     def __init__(self, subframe, subchannel, rb_start=None, rb_len=None):
#         self.subframe = subframe
#         self.subchannel = subchannel
#         # If not provided, we'll calculate rb_start and rb_len later
#         self.rb_start = rb_start
#         self.rb_len = rb_len


# class Vehicle:
#     """Represents a vehicle with V2X capability"""
    
#     def __init__(self, vehicle_id, initial_position, initial_velocity, sim):
#         self.id = vehicle_id
#         self.position = initial_position
#         self.velocity = initial_velocity
#         self.sim = sim
        
#         # Resource selection parameters
#         self.resel_counter = 0
#         self.prob_resource_keep = random.uniform(0.2, 0.8)
#         self.current_resource = None
#         self.sensing_data = []
#         self.reserved_resources = []
#         self.next_transmission_time = 0
        
#         # Initialize stats
#         self.packets_sent = 0
#         self.packets_received = 0
#         self.collisions = 0
#         self.successful_transmissions = 0
    
#     def move(self, delta_time):
#         """Update vehicle position based on velocity and time delta"""
#         self.position = self.position + self.velocity * delta_time
    
#     def select_resources(self, current_time):
#         """Perform resource selection according to MODE4 algorithm"""
#         if self.resel_counter > 0 and random.random() < self.prob_resource_keep and self.current_resource is not None:
#             # Keep the previous resource
#             self.resel_counter -= 1
#             logger.debug(f"Vehicle {self.id} kept previous resource. Resel counter: {self.resel_counter}")
#             return self.current_resource
        
#         # Need to select a new resource
#         # Step 1: Update the sensing window (last 1000ms)
#         self._update_sensing_window(current_time)
        
#         # Step 2-3: Create the resource selection window (T1, T2)
#         selection_window = self._create_selection_window(current_time)
        
#         # Step 4: Calculate candidate resources
#         candidate_resources = self._calculate_candidate_resources(selection_window)
        
#         if not candidate_resources:
#             logger.warning(f"Vehicle {self.id} couldn't find any candidate resources")
#             # Fallback: select a random resource
#             return self._select_random_resource(selection_window)
        
#         # Step 5-9: Select best resource based on RSSI
#         selected_resource = self._select_best_resource(candidate_resources)
        
#         # Update reselection counter (5-15 cycles)
#         self.resel_counter = random.randint(5, 15)
#         self.current_resource = selected_resource
        
#         logger.debug(f"Vehicle {self.id} selected new resource: {selected_resource.subframe}, " +
#                      f"subchannel: {selected_resource.subchannel}, resel_counter: {self.resel_counter}")
        
#         return selected_resource
    
#     def _update_sensing_window(self, current_time):
#         """Update sensing window by removing old entries (older than 1000ms)"""
#         sensing_window_start = current_time - 1000  # 1000ms sensing window
        
#         # Convert time to subframe
#         frame_no_start = math.floor(sensing_window_start / 10) % 1024
#         subframe_no_start = math.floor(sensing_window_start % 10) + 1
        
#         start_subframe = SubframeInfo(frame_no_start, subframe_no_start)
        
#         # Remove data outside the window
#         self.sensing_data = [data for data in self.sensing_data 
#                             if not (data.subframe_info < start_subframe)]
    
#     def _create_selection_window(self, current_time):
#         """Create the resource selection window based on T1 and T2 parameters"""
#         # T1 = 4 subframes (4ms)
#         # T2 = 100 subframes (100ms)
#         t1_time = current_time + 4  # T1 = 4ms
#         t2_time = current_time + 100  # T2 = 100ms
        
#         # Convert to subframe information
#         t1_frame = math.floor(t1_time / 10) % 1024
#         t1_subframe = math.floor(t1_time % 10) + 1
        
#         t2_frame = math.floor(t2_time / 10) % 1024
#         t2_subframe = math.floor(t2_time % 10) + 1
        
#         start_subframe = SubframeInfo(t1_frame, t1_subframe)
#         end_subframe = SubframeInfo(t2_frame, t2_subframe)
        
#         # Create a list of all possible subframes in the window
#         selection_window = []
        
#         current_frame = start_subframe.frame_no
#         current_subframe = start_subframe.subframe_no
        
#         while True:
#             current_sf = SubframeInfo(current_frame, current_subframe)
#             if current_sf.frame_no > end_subframe.frame_no or (current_sf.frame_no == end_subframe.frame_no and current_sf.subframe_no > end_subframe.subframe_no):
#                 break
                
#             # For each subframe, consider all subchannels
#             for subchannel in range(self.sim.resource_pool.num_subchannels):
#                 selection_window.append(ResourceInfo(current_sf, subchannel))
            
#             # Move to next subframe
#             current_subframe += 1
#             if current_subframe > 10:
#                 current_subframe = 1
#                 current_frame += 1
#                 if current_frame > 1024:
#                     current_frame = 1
        
#         return selection_window
    
#     def _calculate_candidate_resources(self, selection_window):
#         """Calculate list of candidate resources (CSRs)"""
#         # Step 6: For each potential resource, check if it's excluded by sensing
#         candidate_resources = []
        
#         for res in selection_window:
#             excluded = False
            
#             # Check each sensing data to see if this resource would overlap
#             for data in self.sensing_data:
#                 # Check if this resource would be used in the same subframe as a previous transmission
#                 if self._would_overlap(res, data):
#                     if data.rsrp > -110:  # RSRP threshold in dBm
#                         excluded = True
#                         break
            
#             if not excluded:
#                 candidate_resources.append(res)
        
#         # Step 7: Ensure at least 20% of resources remain
#         if len(candidate_resources) < 0.2 * len(selection_window):
#             # Sort by RSRP and take the bottom 20%
#             candidate_resources = self._get_resources_by_lowest_rsrp(selection_window, int(0.2 * len(selection_window)))
        
#         return candidate_resources
    
#     def _would_overlap(self, resource, sensing_data):
#         """Check if the resource would overlap with a sensed transmission"""
#         # Check if the resource is in the same subframe
#         current_subframe = resource.subframe
#         sensed_subframe = sensing_data.subframe_info
        
#         # Check if it's in the same subframe considering periodicity
#         same_subframe = False
        
#         if (current_subframe.subframe_no == sensed_subframe.subframe_no):
#             # Calculate frame difference considering wraparound
#             frame_diff = (current_subframe.frame_no - sensed_subframe.frame_no) % 1024
#             # Check if the difference matches the reservation period
#             if frame_diff % (sensing_data.pRsvp // 10) == 0:
#                 same_subframe = True
        
#         if not same_subframe:
#             return False
        
#         # Check if the resource is on the same subchannel
#         rb_start = resource.subchannel * self.sim.resource_pool.subchannel_size
#         rb_end = rb_start + self.sim.resource_pool.subchannel_size - 1
        
#         sensed_rb_start = sensing_data.rb_start
#         sensed_rb_end = sensed_rb_start + sensing_data.rb_len - 1
        
#         # Check for overlap in frequency domain
#         return not (rb_end < sensed_rb_start or rb_start > sensed_rb_end)
    
#     def _get_resources_by_lowest_rsrp(self, resources, count):
#         """Get resources with lowest RSRP values"""
#         # For each resource, calculate the average RSRP from sensing data
#         resource_rsrp = []
        
#         for res in resources:
#             avg_rsrp = self._calculate_avg_rsrp(res)
#             resource_rsrp.append((res, avg_rsrp))
        
#         # Sort by RSRP (lowest first)
#         resource_rsrp.sort(key=lambda x: x[1])
        
#         # Return the requested number of resources
#         return [r[0] for r in resource_rsrp[:count]]
    
#     def _calculate_avg_rsrp(self, resource):
#         """Calculate average RSRP for a resource based on sensing data"""
#         relevant_rsrp = []
        
#         # Find sensing data relevant to this resource
#         for data in self.sensing_data:
#             if self._would_overlap(resource, data):
#                 relevant_rsrp.append(data.rsrp)
        
#         if not relevant_rsrp:
#             return -140  # Very low RSRP if no data
        
#         return sum(relevant_rsrp) / len(relevant_rsrp)
    
#     def _select_best_resource(self, candidate_resources):
#         """Select the best resource from candidates based on RSSI metric"""
#         # Step 8-9: Calculate metric E (linear average of S-RSSI) for each resource
#         resource_metrics = []
        
#         for res in candidate_resources:
#             avg_rssi = self._calculate_avg_rssi(res)
#             resource_metrics.append((res, avg_rssi))
        
#         # Sort by RSSI (lowest first, best reception conditions)
#         resource_metrics.sort(key=lambda x: x[1])
        
#         # Select randomly from the best 20%
#         num_best = max(1, int(0.2 * len(resource_metrics)))
#         best_resources = resource_metrics[:num_best]
        
#         # Randomly select one from the best resources
#         selected = random.choice(best_resources)
        
#         # Set the rb_start and rb_len based on the subchannel
#         selected[0].rb_start = selected[0].subchannel * self.sim.resource_pool.subchannel_size
#         selected[0].rb_len = self.sim.resource_pool.subchannel_size
        
#         return selected[0]
    
#     def _calculate_avg_rssi(self, resource):
#         """Calculate average RSSI for a resource based on sensing data"""
#         relevant_rssi = []
        
#         # Find sensing data relevant to this resource
#         for data in self.sensing_data:
#             if self._would_overlap(resource, data):
#                 relevant_rssi.append(data.rssi)
        
#         if not relevant_rssi:
#             return -140  # Very low RSSI if no data
        
#         return sum(relevant_rssi) / len(relevant_rssi)
    
#     def _select_random_resource(self, selection_window):
#         """Select a random resource from the selection window (fallback)"""
#         resource = random.choice(selection_window)
#         resource.rb_start = resource.subchannel * self.sim.resource_pool.subchannel_size
#         resource.rb_len = self.sim.resource_pool.subchannel_size
#         return resource
    
#     def add_sensing_data(self, subframe_info, rb_start, rb_len, rsrp, rssi, pRsvp, sender_id):
#         """Add sensing data from a received transmission"""
#         sensing_data = SensingData(subframe_info, rb_start, rb_len, rsrp, rssi, pRsvp, sender_id)
#         self.sensing_data.append(sensing_data)
    
#     def send_packet(self, current_time):
#         """Send a packet using the selected resource"""
#         if current_time < self.next_transmission_time:
#             return None
        
#         # Select resources if needed
#         if not self.current_resource or self.resel_counter <= 0:
#             self.current_resource = self.select_resources(current_time)
        
#         # Create packet
#         packet = Packet(self.id, current_time, self.position)
        
#         # Schedule next transmission
#         self.next_transmission_time = current_time + 100  # 100ms cycle
        
#         # Update stats
#         self.packets_sent += 1
        
#         return (packet, self.current_resource)
    
#     def receive_packet(self, packet, resource, rsrp, rssi):
#         """Process a received packet"""
#         # Store sensing information
#         subframe_info = resource.subframe
#         rb_start = resource.rb_start
#         rb_len = resource.rb_len
        
#         # Add to sensing window
#         self.add_sensing_data(subframe_info, rb_start, rb_len, rsrp, rssi, 100, packet.sender_id)
        
#         # Update stats
#         self.packets_received += 1


# class ResourcePool:
#     """Manages the sidelink resource pool for V2X communication"""
    
#     def __init__(self, num_subchannels=5, num_slots=100, subchannel_size=10):
#         self.num_subchannels = num_subchannels
#         self.num_slots = num_slots
#         self.subchannel_size = subchannel_size
#         self.total_rbs = num_subchannels * subchannel_size
        
#         # Initialize the resource grid as free
#         self.resource_grid = np.zeros((num_slots, self.total_rbs), dtype=np.int32)
    
#     def is_resource_free(self, subframe_idx, rb_start, rb_len):
#         """Check if a resource is free"""
#         if rb_start + rb_len > self.total_rbs:
#             return False
        
#         return np.sum(self.resource_grid[subframe_idx, rb_start:rb_start+rb_len]) == 0
    
#     def allocate_resource(self, subframe_idx, rb_start, rb_len, vehicle_id):
#         """Allocate a resource to a vehicle"""
#         if not self.is_resource_free(subframe_idx, rb_start, rb_len):
#             return False
        
#         self.resource_grid[subframe_idx, rb_start:rb_start+rb_len] = vehicle_id
#         return True
    
#     def free_resource(self, subframe_idx, rb_start, rb_len):
#         """Free an allocated resource"""
#         if rb_start + rb_len <= self.total_rbs:
#             self.resource_grid[subframe_idx, rb_start:rb_start+rb_len] = 0
    
#     def detect_collision(self, subframe_idx, rb_start, rb_len):
#         """Detect if there's a collision on the allocated resource"""
#         if rb_start + rb_len > self.total_rbs:
#             return True
        
#         # Check if there are any non-zero values in the resource grid
#         used_by = set(self.resource_grid[subframe_idx, rb_start:rb_start+rb_len])
#         used_by.discard(0)  # Remove 0 (unused)
        
#         # If more than one vehicle is using the resource, there's a collision
#         return len(used_by) > 1
    
#     def get_colliding_vehicles(self, subframe_idx, rb_start, rb_len):
#         """Get IDs of vehicles with colliding transmissions"""
#         if rb_start + rb_len > self.total_rbs:
#             return set()
        
#         # Get all vehicle IDs using this resource
#         used_by = set(self.resource_grid[subframe_idx, rb_start:rb_start+rb_len])
#         used_by.discard(0)  # Remove 0 (unused)
        
#         return used_by if len(used_by) > 1 else set()


# class Channel:
#     """Simulates the wireless channel including path loss and fading"""
    
#     def __init__(self, baseline_distance=320.0):
#         self.baseline_distance = baseline_distance  # Transmission range in meters
    
#     def calculate_path_loss(self, tx_position, rx_position):
#         """Calculate path loss between transmitter and receiver"""
#         distance = abs(tx_position - rx_position)
        
#         if distance == 0:
#             return 0  # No path loss for same position
        
#         # Simple path loss model (distance-based)
#         # Using free space path loss model: PL = 20*log10(d) + 20*log10(f) - 147.55
#         # Simplified version
#         path_loss_db = 20 * math.log10(distance)
        
#         return path_loss_db
    
#     def calculate_rsrp(self, tx_power, path_loss):
#         """Calculate RSRP (Reference Signal Received Power)"""
#         # RSRP = Tx Power - Path Loss
#         rsrp_dbm = tx_power - path_loss
#         return rsrp_dbm
    
#     def calculate_rssi(self, rsrp, interference=0):
#         """Calculate RSSI (Received Signal Strength Indicator)"""
#         # RSSI = RSRP + Interference + Noise
#         # For simplicity, assuming fixed noise floor of -95 dBm
#         noise_floor_dbm = -95
#         rssi_dbm = 10 * math.log10(10**(rsrp/10) + 10**(interference/10) + 10**(noise_floor_dbm/10))
#         return rssi_dbm
    
#     def is_in_range(self, tx_position, rx_position):
#         """Check if the receiver is within range of the transmitter"""
#         distance = abs(tx_position - rx_position)
#         return distance <= self.baseline_distance


# class Simulation:
#     """Manages the overall V2X simulation"""
    
#     def __init__(self, num_vehicles=20, duration=50000, tx_power=20.0):
#         self.num_vehicles = num_vehicles
#         self.duration = duration  # Simulation duration in ms
#         self.tx_power = tx_power  # Transmission power in dBm
        
#         # Initialize resource pool
#         self.resource_pool = ResourcePool(num_subchannels=5, num_slots=100, subchannel_size=10)
        
#         # Initialize channel
#         self.channel = Channel()
        
#         # Initialize vehicles
#         self.vehicles = []
#         self._initialize_vehicles()
        
#         # Simulation time variables
#         self.current_time = 0
        
#         # Statistics
#         self.collision_count = 0
#         self.transmission_count = 0
#         self.collision_stats = defaultdict(int)  # Time-based collision statistics
#         self.transmission_stats = defaultdict(int)  # Time-based transmission statistics
#         self.prr_stats = defaultdict(float)  # Packet Reception Ratio stats
    
#     def _initialize_vehicles(self):
#         """Initialize vehicles with random positions and velocities"""
#         # Create 2 lanes with opposite directions
#         lane1_y = 5.0
#         lane2_y = 10.0
#         highway_length = 1000.0
        
#         for i in range(self.num_vehicles):
#             # Alternate between lanes
#             lane_y = lane1_y if i % 2 == 0 else lane2_y
            
#             # Random x position along the highway
#             pos_x = random.uniform(0, highway_length)
            
#             # Create position (x, y)
#             position = np.array([pos_x, lane_y])
            
#             # Set velocity based on lane (lane 1: positive direction, lane 2: negative direction)
#             velocity = np.array([10.0 if lane_y == lane1_y else -10.0, 0.0])
            
#             # Create the vehicle
#             vehicle = Vehicle(i, position, velocity, self)
#             self.vehicles.append(vehicle)
    
#     def run(self):
#         """Run the simulation"""
#         logger.info(f"Starting V2X MODE4 simulation with {self.num_vehicles} vehicles")
        
#         time_step = 1  # 1ms time step
        
#         # Main simulation loop
#         while self.current_time < self.duration:
#             # Update vehicle positions
#             for vehicle in self.vehicles:
#                 vehicle.move(time_step / 1000.0)  # Convert to seconds
            
#             # Process transmissions
#             self._process_transmissions()
            
#             # Record statistics every 1000ms
#             if self.current_time % 1000 == 0:
#                 self._record_statistics()
#                 logger.info(f"Simulation time: {self.current_time}ms, Collisions: {self.collision_count}, " +
#                           f"Transmissions: {self.transmission_count}")
            
#             # Increment time
#             self.current_time += time_step
        
#         logger.info("Simulation completed")
#         self._print_results()
    
#     def _process_transmissions(self):
#         """Process packet transmissions for the current time step"""
#         # Collect all transmissions for this time step
#         transmissions = []
        
#         for vehicle in self.vehicles:
#             tx_result = vehicle.send_packet(self.current_time)
#             if tx_result:
#                 packet, resource = tx_result
#                 transmissions.append((vehicle, packet, resource))
        
#         # Process transmissions and detect collisions
#         if transmissions:
#             self._handle_transmissions(transmissions)
    
#     def _handle_transmissions(self, transmissions):
#         """Handle packet transmissions and detect collisions"""
#         # First, collect transmissions by subframe
#         tx_by_subframe = defaultdict(list)
        
#         for sender, packet, resource in transmissions:
#             # Convert to subframe index for resource pool
#             sf_idx = (resource.subframe.frame_no * 10 + resource.subframe.subframe_no - 1) % self.resource_pool.num_slots
#             tx_by_subframe[sf_idx].append((sender, packet, resource))
            
#             # Allocate resource in the pool
#             self.resource_pool.allocate_resource(sf_idx, resource.rb_start, resource.rb_len, sender.id)
            
#             # Increment transmission count
#             self.transmission_count += 1
        
#         # Process each subframe
#         for sf_idx, sf_transmissions in tx_by_subframe.items():
#             # Check for collisions
#             for i, (sender1, packet1, resource1) in enumerate(sf_transmissions):
#                 for j, (sender2, packet2, resource2) in enumerate(sf_transmissions):
#                     if i != j:  # Don't compare a transmission with itself
#                         # Check if resources overlap
#                         rb1_start, rb1_end = resource1.rb_start, resource1.rb_start + resource1.rb_len
#                         rb2_start, rb2_end = resource2.rb_start, resource2.rb_start + resource2.rb_len
                        
#                         if max(rb1_start, rb2_start) < min(rb1_end, rb2_end):
#                             # Resources overlap - collision detected
#                             self.collision_count += 1
#                             sender1.collisions += 1
#                             sender2.collisions += 1
#                             logger.debug(f"Collision detected between Vehicle {sender1.id} and Vehicle {sender2.id}")
            
#             # Process reception for each vehicle
#             for receiver in self.vehicles:
#                 for sender, packet, resource in sf_transmissions:
#                     if sender.id != receiver.id:  # Don't receive own packet
#                         # Check if receiver is in range of sender
#                         if self.channel.is_in_range(sender.position[0], receiver.position[0]):
#                             # Calculate path loss
#                             path_loss = self.channel.calculate_path_loss(sender.position[0], receiver.position[0])
                            
#                             # Calculate RSRP and RSSI
#                             rsrp = self.channel.calculate_rsrp(self.tx_power, path_loss)
                            
#                             # Calculate interference from other transmissions
#                             interference = self._calculate_interference(sender, resource, sf_transmissions, receiver)
                            
#                             # Calculate RSSI with interference
#                             rssi = self.channel.calculate_rssi(rsrp, interference)
                            
#                             # Process packet reception
#                             receiver.receive_packet(packet, resource, rsrp, rssi)
                            
#                             # Increment successful transmission count if no collision
#                             if not self.resource_pool.detect_collision(sf_idx, resource.rb_start, resource.rb_len):
#                                 sender.successful_transmissions += 1
    
#     def _calculate_interference(self, target_sender, target_resource, all_transmissions, receiver):
#         """Calculate interference from other transmissions"""
#         interference_power = 0
        
#         for other_sender, _, other_resource in all_transmissions:
#             # Skip the target sender
#             if other_sender.id == target_sender.id:
#                 continue
            
#             # Check if resources overlap
#             rb1_start, rb1_end = target_resource.rb_start, target_resource.rb_start + target_resource.rb_len
#             rb2_start, rb2_end = other_resource.rb_start, other_resource.rb_start + other_resource.rb_len
            
#             if max(rb1_start, rb2_start) < min(rb1_end, rb2_end):
#                 # Resources overlap - calculate interference
#                 path_loss = self.channel.calculate_path_loss(other_sender.position[0], receiver.position[0])
#                 interference_rsrp = self.channel.calculate_rsrp(self.tx_power, path_loss)
                
#                 # Convert to linear and add
#                 interference_power += 10 ** (interference_rsrp / 10)
        
#         # Convert back to dB
#         return 10 * math.log10(interference_power) if interference_power > 0 else -float('inf')
    
#     def _record_statistics(self):
#         """Record statistics at the current time"""
#         # Record collision and transmission counts
#         time_bin = self.current_time // 1000  # Group by seconds
#         self.collision_stats[time_bin] = self.collision_count
#         self.transmission_stats[time_bin] = self.transmission_count
        
#         # Calculate PRR (Packet Reception Ratio)
#         if self.transmission_count > 0:
#             prr = 1.0 - (self.collision_count / self.transmission_count)
#             self.prr_stats[time_bin] = prr
    
#     def _print_results(self):
#         """Print final simulation results"""
#         logger.info("\n=========== SIMULATION RESULTS ===========")
#         logger.info(f"Total simulation time: {self.duration}ms")
#         logger.info(f"Total transmissions: {self.transmission_count}")
#         logger.info(f"Total collisions: {self.collision_count}")
        
#         if self.transmission_count > 0:
#             prr = 1.0 - (self.collision_count / self.transmission_count)
#             logger.info(f"Overall Packet Reception Ratio (PRR): {prr:.4f}")
        
#         # Per-vehicle statistics
#         logger.info("\nVehicle Statistics:")
#         for vehicle in self.vehicles:
#             logger.info(f"Vehicle {vehicle.id}: Sent={vehicle.packets_sent}, " +
#                       f"Received={vehicle.packets_received}, Collisions={vehicle.collisions}")
        
#         # Plot results
#         self._plot_results()
    
#     def _plot_results(self):
#         """Generate plots of simulation results"""
#         # Plot collision and transmission statistics
#         plt.figure(figsize=(12, 8))
        
#         plt.subplot(2, 1, 1)
#         plt.plot(list(self.transmission_stats.keys()), list(self.transmission_stats.values()), 'b-', label='Transmissions')
#         plt.plot(list(self.collision_stats.keys()), list(self.collision_stats.values()), 'r-', label='Collisions')
#         plt.xlabel('Time (s)')
#         plt.ylabel('Count')
#         plt.title('Transmissions and Collisions over Time')
#         plt.legend()
#         plt.grid(True)
        
#         plt.subplot(2, 1, 2)
#         plt.plot(list(self.prr_stats.keys()), list(self.prr_stats.values()), 'g-')
#         plt.xlabel('Time (s)')
#         plt.ylabel('Packet Reception Ratio')
#         plt.title('Packet Reception Ratio over Time')
#         plt.ylim([0, 1])
#         plt.grid(True)
        
#         plt.tight_layout()
#         plt.savefig('v2x_simulation_results.png')
#         logger.info("Results plotted and saved to 'v2x_simulation_results.png'")


# if __name__ == "__main__":
#     # Set random seed for reproducibility
#     random.seed(42)
#     np.random.seed(42)
    
#     # Create and run the simulation
#     sim = Simulation(num_vehicles=20, duration=500000)
#     sim.run()

import numpy as np
import matplotlib.pyplot as plt
import random
import time
from collections import defaultdict
import logging
import math

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('V2X-MODE4')

class Packet:
    """Represents a V2X packet for transmission"""
    
    def __init__(self, sender_id, timestamp, position, size=190):
        self.sender_id = sender_id
        self.timestamp = timestamp
        self.position = position
        self.size = size  # size in bytes (default is 190B as per the original C++ code)


class SensingData:
    """Represents sensing data from previous transmissions"""
    
    def __init__(self, subframe_info, rb_start, rb_len, rsrp, rssi, pRsvp, sender_id):
        self.subframe_info = subframe_info
        self.rb_start = rb_start
        self.rb_len = rb_len
        self.rsrp = rsrp
        self.rssi = rssi
        self.pRsvp = pRsvp
        self.sender_id = sender_id


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
    
    def __repr__(self):
        return f"({self.frame_no}, {self.subframe_no})"


class ResourceInfo:
    """Represents a resource allocation for transmission"""
    
    def __init__(self, subframe, subchannel, rb_start=None, rb_len=None):
        self.subframe = subframe
        self.subchannel = subchannel
        # If not provided, we'll calculate rb_start and rb_len later
        self.rb_start = rb_start
        self.rb_len = rb_len


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
        self.reserved_resources = []
        self.next_transmission_time = 0
        
        # Initialize stats
        self.packets_sent = 0
        self.packets_received = 0
        self.collisions = 0
        self.successful_transmissions = 0
    
    def move(self, delta_time):
        """Update vehicle position based on velocity and time delta"""
        self.position = self.position + self.velocity * delta_time
    
    def select_resources(self, current_time):
        """Perform resource selection according to MODE4 algorithm"""
        if self.resel_counter > 0 and random.random() < self.prob_resource_keep and self.current_resource is not None:
            # Keep the previous resource
            self.resel_counter -= 1
            logger.debug(f"Vehicle {self.id} kept previous resource. Resel counter: {self.resel_counter}")
            return self.current_resource
        
        # Need to select a new resource
        # Step 1: Update the sensing window (last 1000ms)
        self._update_sensing_window(current_time)
        
        # Step 2-3: Create the resource selection window (T1, T2)
        selection_window = self._create_selection_window(current_time)
        
        # Step 4: Calculate candidate resources
        candidate_resources = self._calculate_candidate_resources(selection_window)
        
        if not candidate_resources:
            logger.warning(f"Vehicle {self.id} couldn't find any candidate resources")
            # Fallback: select a random resource
            return self._select_random_resource(selection_window)
        
        # Step 5-9: Select best resource based on RSSI
        selected_resource = self._select_best_resource(candidate_resources)
        
        # Update reselection counter (5-15 cycles)
        self.resel_counter = random.randint(5, 15)
        self.current_resource = selected_resource
        
        logger.debug(f"Vehicle {self.id} selected new resource: {selected_resource.subframe}, " +
                     f"subchannel: {selected_resource.subchannel}, resel_counter: {self.resel_counter}")
        
        return selected_resource
    
    def _update_sensing_window(self, current_time):
        """Update sensing window by removing old entries (older than 1000ms)"""
        sensing_window_start = current_time - 1000  # 1000ms sensing window
        
        # Convert time to subframe
        frame_no_start = math.floor(sensing_window_start / 10) % 1024
        subframe_no_start = math.floor(sensing_window_start % 10) + 1
        
        start_subframe = SubframeInfo(frame_no_start, subframe_no_start)
        
        # Remove data outside the window
        self.sensing_data = [data for data in self.sensing_data 
                            if not (data.subframe_info < start_subframe)]
    
    def _create_selection_window(self, current_time):
        """Create the resource selection window based on T1 and T2 parameters"""
        # T1 = 4 subframes (4ms)
        # T2 = 100 subframes (100ms)
        t1_time = current_time + 4  # T1 = 4ms
        t2_time = current_time + 100  # T2 = 100ms
        
        # Convert to subframe information
        t1_frame = math.floor(t1_time / 10) % 1024
        t1_subframe = math.floor(t1_time % 10) + 1
        
        t2_frame = math.floor(t2_time / 10) % 1024
        t2_subframe = math.floor(t2_time % 10) + 1
        
        start_subframe = SubframeInfo(t1_frame, t1_subframe)
        end_subframe = SubframeInfo(t2_frame, t2_subframe)
        
        # Create a list of all possible subframes in the window
        selection_window = []
        
        current_frame = start_subframe.frame_no
        current_subframe = start_subframe.subframe_no
        
        while True:
            current_sf = SubframeInfo(current_frame, current_subframe)
            if current_sf.frame_no > end_subframe.frame_no or (current_sf.frame_no == end_subframe.frame_no and current_sf.subframe_no > end_subframe.subframe_no):
                break
                
            # For each subframe, consider all subchannels
            for subchannel in range(self.sim.resource_pool.num_subchannels):
                selection_window.append(ResourceInfo(current_sf, subchannel))
            
            # Move to next subframe
            current_subframe += 1
            if current_subframe > 10:
                current_subframe = 1
                current_frame += 1
                if current_frame > 1024:
                    current_frame = 1
        
        return selection_window
    
    def _calculate_candidate_resources(self, selection_window):
        """Calculate list of candidate resources (CSRs)"""
        # Step 6: For each potential resource, check if it's excluded by sensing
        candidate_resources = []
        
        for res in selection_window:
            excluded = False
            
            # Check each sensing data to see if this resource would overlap
            for data in self.sensing_data:
                # Check if this resource would be used in the same subframe as a previous transmission
                if self._would_overlap(res, data):
                    if data.rsrp > -110:  # RSRP threshold in dBm
                        excluded = True
                        break
            
            if not excluded:
                candidate_resources.append(res)
        
        # Step 7: Ensure at least 20% of resources remain
        if len(candidate_resources) < 0.2 * len(selection_window):
            # Sort by RSRP and take the bottom 20%
            candidate_resources = self._get_resources_by_lowest_rsrp(selection_window, int(0.2 * len(selection_window)))
        
        return candidate_resources
    
    def _would_overlap(self, resource, sensing_data):
        """Check if the resource would overlap with a sensed transmission"""
        # Check if the resource is in the same subframe
        current_subframe = resource.subframe
        sensed_subframe = sensing_data.subframe_info
        
        # Check if it's in the same subframe considering periodicity
        same_subframe = False
        
        if (current_subframe.subframe_no == sensed_subframe.subframe_no):
            # Calculate frame difference considering wraparound
            frame_diff = (current_subframe.frame_no - sensed_subframe.frame_no) % 1024
            # Check if the difference matches the reservation period
            if frame_diff % (sensing_data.pRsvp // 10) == 0:
                same_subframe = True
        
        if not same_subframe:
            return False
        
        # Check if the resource is on the same subchannel
        rb_start = resource.subchannel * self.sim.resource_pool.subchannel_size
        rb_end = rb_start + self.sim.resource_pool.subchannel_size - 1
        
        sensed_rb_start = sensing_data.rb_start
        sensed_rb_end = sensed_rb_start + sensing_data.rb_len - 1
        
        # Check for overlap in frequency domain
        return not (rb_end < sensed_rb_start or rb_start > sensed_rb_end)
    
    def _get_resources_by_lowest_rsrp(self, resources, count):
        """Get resources with lowest RSRP values"""
        # For each resource, calculate the average RSRP from sensing data
        resource_rsrp = []
        
        for res in resources:
            avg_rsrp = self._calculate_avg_rsrp(res)
            resource_rsrp.append((res, avg_rsrp))
        
        # Sort by RSRP (lowest first)
        resource_rsrp.sort(key=lambda x: x[1])
        
        # Return the requested number of resources
        return [r[0] for r in resource_rsrp[:count]]
    
    def _calculate_avg_rsrp(self, resource):
        """Calculate average RSRP for a resource based on sensing data"""
        relevant_rsrp = []
        
        # Find sensing data relevant to this resource
        for data in self.sensing_data:
            if self._would_overlap(resource, data):
                relevant_rsrp.append(data.rsrp)
        
        if not relevant_rsrp:
            return -140  # Very low RSRP if no data
        
        return sum(relevant_rsrp) / len(relevant_rsrp)
    
    def _select_best_resource(self, candidate_resources):
        """Select the best resource from candidates based on RSSI metric"""
        # Step 8-9: Calculate metric E (linear average of S-RSSI) for each resource
        resource_metrics = []
        
        for res in candidate_resources:
            avg_rssi = self._calculate_avg_rssi(res)
            resource_metrics.append((res, avg_rssi))
        
        # Sort by RSSI (lowest first, best reception conditions)
        resource_metrics.sort(key=lambda x: x[1])
        
        # Select randomly from the best 20%
        num_best = max(1, int(0.2 * len(resource_metrics)))
        best_resources = resource_metrics[:num_best]
        
        # Randomly select one from the best resources
        selected = random.choice(best_resources)
        
        # Set the rb_start and rb_len based on the subchannel
        selected[0].rb_start = selected[0].subchannel * self.sim.resource_pool.subchannel_size
        selected[0].rb_len = self.sim.resource_pool.subchannel_size
        
        return selected[0]
    
    def _calculate_avg_rssi(self, resource):
        """Calculate average RSSI for a resource based on sensing data"""
        relevant_rssi = []
        
        # Find sensing data relevant to this resource
        for data in self.sensing_data:
            if self._would_overlap(resource, data):
                relevant_rssi.append(data.rssi)
        
        if not relevant_rssi:
            return -140  # Very low RSSI if no data
        
        return sum(relevant_rssi) / len(relevant_rssi)
    
    def _select_random_resource(self, selection_window):
        """Select a random resource from the selection window (fallback)"""
        resource = random.choice(selection_window)
        resource.rb_start = resource.subchannel * self.sim.resource_pool.subchannel_size
        resource.rb_len = self.sim.resource_pool.subchannel_size
        return resource
    
    def add_sensing_data(self, subframe_info, rb_start, rb_len, rsrp, rssi, pRsvp, sender_id):
        """Add sensing data from a received transmission"""
        sensing_data = SensingData(subframe_info, rb_start, rb_len, rsrp, rssi, pRsvp, sender_id)
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
        
        # Schedule next transmission
        self.next_transmission_time = current_time + 100  # 100ms cycle
        
        # Update stats
        self.packets_sent += 1
        
        return (packet, self.current_resource)
    
    def receive_packet(self, packet, resource, rsrp, rssi):
        """Process a received packet"""
        # Check SINR threshold for successful reception
        sinr = rsrp - rssi  # Simplified SINR calculation
        if sinr < -5:  # Typical SINR threshold for V2X
            self.collisions += 1
            return False

        # Store sensing information
        subframe_info = resource.subframe
        rb_start = resource.rb_start
        rb_len = resource.rb_len

        # Add to sensing window
        self.add_sensing_data(subframe_info, rb_start, rb_len, rsrp, rssi, 100, packet.sender_id)

        # Update stats
        self.packets_received += 1
        return True



class ResourcePool:
    """Manages the sidelink resource pool for V2X communication"""
    
    def __init__(self, num_subchannels=5, num_slots=100, subchannel_size=10):
        self.num_subchannels = num_subchannels
        self.num_slots = num_slots
        self.subchannel_size = subchannel_size
        self.total_rbs = num_subchannels * subchannel_size
        
        # Initialize the resource grid as free
        self.resource_grid = np.zeros((num_slots, self.total_rbs), dtype=np.int32)
    
    def is_resource_free(self, subframe_idx, rb_start, rb_len):
        """Check if a resource is free"""
        if rb_start + rb_len > self.total_rbs:
            return False
        
        return np.sum(self.resource_grid[subframe_idx, rb_start:rb_start+rb_len]) == 0
    
    def allocate_resource(self, subframe_idx, rb_start, rb_len, vehicle_id):
        """Allocate a resource to a vehicle"""
        if not self.is_resource_free(subframe_idx, rb_start, rb_len):
            return False
        
        self.resource_grid[subframe_idx, rb_start:rb_start+rb_len] = vehicle_id
        return True
    
    def free_resource(self, subframe_idx, rb_start, rb_len):
        """Free an allocated resource"""
        if rb_start + rb_len <= self.total_rbs:
            self.resource_grid[subframe_idx, rb_start:rb_start+rb_len] = 0
    
    def detect_collision(self, subframe_idx, rb_start, rb_len):
        """Detect if there's a collision on the allocated resource"""
        if rb_start + rb_len > self.total_rbs:
            return True
        
        # Check if there are any non-zero values in the resource grid
        used_by = set(self.resource_grid[subframe_idx, rb_start:rb_start+rb_len])
        used_by.discard(0)  # Remove 0 (unused)
        
        # If more than one vehicle is using the resource, there's a collision
        return len(used_by) > 1
    
    def get_colliding_vehicles(self, subframe_idx, rb_start, rb_len):
        """Get IDs of vehicles with colliding transmissions"""
        if rb_start + rb_len > self.total_rbs:
            return set()
        
        # Get all vehicle IDs using this resource
        used_by = set(self.resource_grid[subframe_idx, rb_start:rb_start+rb_len])
        used_by.discard(0)  # Remove 0 (unused)
        
        return used_by if len(used_by) > 1 else set()


class Channel:
    """Simulates the wireless channel including path loss and fading"""
    
    def __init__(self, baseline_distance=320.0):
        self.baseline_distance = baseline_distance  # Transmission range in meters
    
    def calculate_path_loss(self, tx_position, rx_position):
        """Calculate path loss between transmitter and receiver"""
        distance = abs(tx_position - rx_position)
        
        if distance == 0:
            return 0
        
        # More realistic path loss model for V2X
        # Using WINNER+ B1 model for urban areas
        d0 = 10  # Reference distance
        n = 3.8  # Path loss exponent
        shadow_fading = random.gauss(0, 6)  # Log-normal shadowing
        
        # Calculate path loss
        if distance <= d0:
            path_loss_db = 20 * math.log10(distance)
        else:
            path_loss_db = 20 * math.log10(d0) + 10 * n * math.log10(distance/d0)
        
        # Add shadow fading
        path_loss_db += shadow_fading
        
        return path_loss_db

    
    def calculate_rsrp(self, tx_power, path_loss):
        """Calculate RSRP (Reference Signal Received Power)"""
        # RSRP = Tx Power - Path Loss
        rsrp_dbm = tx_power - path_loss
        return rsrp_dbm
    
    def calculate_rssi(self, rsrp, interference=0):
        """Calculate RSSI (Received Signal Strength Indicator)"""
        # RSSI = RSRP + Interference + Noise
        # For simplicity, assuming fixed noise floor of -95 dBm
        noise_floor_dbm = -95
        rssi_dbm = 10 * math.log10(10**(rsrp/10) + 10**(interference/10) + 10**(noise_floor_dbm/10))
        return rssi_dbm
    
    def is_in_range(self, tx_position, rx_position):
        """Check if the receiver is within range of the transmitter"""
        distance = abs(tx_position - rx_position)
        return distance <= self.baseline_distance


class Simulation:
    """Manages the overall V2X simulation"""
    
    def __init__(self, num_vehicles=20, duration=50000, tx_power=20.0):
        self.num_vehicles = num_vehicles
        self.duration = duration  # Simulation duration in ms
        self.tx_power = tx_power  # Transmission power in dBm
        
        # Initialize resource pool
        self.resource_pool = ResourcePool(num_subchannels=5, num_slots=100, subchannel_size=10)
        
        # Initialize channel
        self.channel = Channel()
        
        # Initialize vehicles
        self.vehicles = []
        self._initialize_vehicles()
        
        # Simulation time variables
        self.current_time = 0
        
        # Statistics
        self.collision_count = 0
        self.transmission_count = 0
        self.collision_stats = defaultdict(int)  # Time-based collision statistics
        self.transmission_stats = defaultdict(int)  # Time-based transmission statistics
        self.prr_stats = defaultdict(float)  # Packet Reception Ratio stats
    
    def _initialize_vehicles(self):
        """Initialize vehicles with random positions and velocities"""
        # Create 2 lanes with opposite directions
        lane1_y = 5.0
        lane2_y = 10.0
        highway_length = 1000.0
        
        for i in range(self.num_vehicles):
            # Alternate between lanes
            lane_y = lane1_y if i % 2 == 0 else lane2_y
            
            # Random x position along the highway
            pos_x = random.uniform(0, highway_length)
            
            # Create position (x, y)
            position = np.array([pos_x, lane_y])
            
            # Set velocity based on lane (lane 1: positive direction, lane 2: negative direction)
            velocity = np.array([10.0 if lane_y == lane1_y else -10.0, 0.0])
            
            # Create the vehicle
            vehicle = Vehicle(i, position, velocity, self)
            self.vehicles.append(vehicle)
    
    def run(self):
        """Run the simulation"""
        logger.info(f"Starting V2X MODE4 simulation with {self.num_vehicles} vehicles")
        
        time_step = 1  # 1ms time step
        
        # Main simulation loop
        while self.current_time < self.duration:
            # Update vehicle positions
            for vehicle in self.vehicles:
                vehicle.move(time_step / 1000.0)  # Convert to seconds
            
            # Process transmissions
            self._process_transmissions()
            
            # Record statistics every 1000ms
            if self.current_time % 1000 == 0:
                self._record_statistics()
                logger.info(f"Simulation time: {self.current_time}ms, Collisions: {self.collision_count}, " +
                          f"Transmissions: {self.transmission_count}")
            
            # Increment time
            self.current_time += time_step
        
        logger.info("Simulation completed")
        self._print_results()
    
    def _process_transmissions(self):
        """Process packet transmissions for the current time step"""
        # Collect all transmissions for this time step
        transmissions = []
        
        for vehicle in self.vehicles:
            tx_result = vehicle.send_packet(self.current_time)
            if tx_result:
                packet, resource = tx_result
                transmissions.append((vehicle, packet, resource))
        
        # Process transmissions and detect collisions
        if transmissions:
            self._handle_transmissions(transmissions)
    
    def _handle_transmissions(self, transmissions):
        """Handle packet transmissions and detect collisions"""
        # First, collect transmissions by subframe
        tx_by_subframe = defaultdict(list)
        resource_usage = defaultdict(list)  # (subframe, rb) -> [vehicle_ids]
        for sender, packet, resource in transmissions:
            # Convert to subframe index for resource pool
            sf_idx = (resource.subframe.frame_no * 10 + resource.subframe.subframe_no - 1) % self.resource_pool.num_slots
            tx_by_subframe[sf_idx].append((sender, packet, resource))
            for rb in range(resource.rb_start, resource.rb_start + resource.rb_len):
                resource_usage[(sf_idx, rb)].append(sender.id)
            
            self.ntx_packets += 1  # Count transmitted packets
            self.ntx_total += 1  # Count unique packets
            # Increment transmission count
            self.transmission_count += 1

        # Process each subframe
        for sf_idx, sf_transmissions in tx_by_subframe.items():
            # Track collisions per resource block
            rb_collisions = defaultdict(int)

            # First pass: detect resource conflicts
            for sender, packet, resource in sf_transmissions:
                for rb in range(resource.rb_start, resource.rb_start + resource.rb_len):
                    rb_collisions[rb] += 1

            # Process reception for each vehicle
            for receiver in self.vehicles:
                for sender, packet, resource in sf_transmissions:
                    if sender.id != receiver.id:  # Don't receive own packet
                        # Check if receiver is in range
                        if self.channel.is_in_range(sender.position[0], receiver.position[0]):
                            # Calculate path loss
                            path_loss = self.channel.calculate_path_loss(sender.position[0], receiver.position[0])

                            # Calculate RSRP
                            rsrp = self.channel.calculate_rsrp(self.tx_power, path_loss)

                            # Calculate interference from other transmissions
                            interference = self._calculate_interference(sender, resource, sf_transmissions, receiver)

                            # Calculate RSSI with interference
                            rssi = self.channel.calculate_rssi(rsrp, interference)

                            # Check for collision on any resource block
                            has_collision = False
                            for rb in range(resource.rb_start, resource.rb_start + resource.rb_len):
                                if rb_collisions[rb] > 1:
                                    has_collision = True
                                    break
                                
                            if has_collision:
                                self.collision_count += 1
                                sender.collisions += 1
                            else:
                                # Process packet reception
                                success = receiver.receive_packet(packet, resource, rsrp, rssi)
                                if success:
                                    sender.successful_transmissions += 1


            # Process each subframe
            for sf_idx, sf_transmissions in tx_by_subframe.items():
                # Check for collisions
                for i, (sender1, packet1, resource1) in enumerate(sf_transmissions):
                    for j, (sender2, packet2, resource2) in enumerate(sf_transmissions):
                        if i != j:  # Don't compare a transmission with itself
                            # Check if resources overlap
                            rb1_start, rb1_end = resource1.rb_start, resource1.rb_start + resource1.rb_len
                            rb2_start, rb2_end = resource2.rb_start, resource2.rb_start + resource2.rb_len

                            if max(rb1_start, rb2_start) < min(rb1_end, rb2_end):
                                # Resources overlap - collision detected
                                self.collision_count += 1
                                sender1.collisions += 1
                                sender2.collisions += 1
                                logger.debug(f"Collision detected between Vehicle {sender1.id} and Vehicle {sender2.id}")

                # Process reception for each vehicle
                for receiver in self.vehicles:
                    for sender, packet, resource in sf_transmissions:
                        if sender.id != receiver.id:  # Don't receive own packet
                            # Check if receiver is in range of sender
                            if self.channel.is_in_range(sender.position[0], receiver.position[0]):
                                # Calculate path loss
                                path_loss = self.channel.calculate_path_loss(sender.position[0], receiver.position[0])

                                # Calculate RSRP and RSSI
                                rsrp = self.channel.calculate_rsrp(self.tx_power, path_loss)

                                # Calculate interference from other transmissions
                                interference = self._calculate_interference(sender, resource, sf_transmissions, receiver)

                                # Calculate RSSI with interference
                                rssi = self.channel.calculate_rssi(rsrp, interference)

                                # Process packet reception
                                receiver.receive_packet(packet, resource, rsrp, rssi)

                                # Increment successful transmission count if no collision
                                if not self.resource_pool.detect_collision(sf_idx, resource.rb_start, resource.rb_len):
                                    sender.successful_transmissions += 1

    def _calculate_interference(self, target_sender, target_resource, all_transmissions, receiver):
        """Calculate interference from other transmissions"""
        interference_power = 0
        
        for other_sender, _, other_resource in all_transmissions:
            # Skip the target sender
            if other_sender.id == target_sender.id:
                continue
            
            # Check if resources overlap
            rb1_start, rb1_end = target_resource.rb_start, target_resource.rb_start + target_resource.rb_len
            rb2_start, rb2_end = other_resource.rb_start, other_resource.rb_start + other_resource.rb_len
            
            if max(rb1_start, rb2_start) < min(rb1_end, rb2_end):
                # Resources overlap - calculate interference
                path_loss = self.channel.calculate_path_loss(other_sender.position[0], receiver.position[0])
                interference_rsrp = self.channel.calculate_rsrp(self.tx_power, path_loss)
                
                # Convert to linear and add
                interference_power += 10 ** (interference_rsrp / 10)
        
        # Convert back to dB
        return 10 * math.log10(interference_power) if interference_power > 0 else -float('inf')
    
    def _record_statistics(self):
        """Record statistics at the current time"""
        # Record collision and transmission counts
        time_bin = self.current_time // 1000  # Group by seconds
        self.collision_stats[time_bin] = self.collision_count
        self.transmission_stats[time_bin] = self.transmission_count
        
        # Calculate PRR (Packet Reception Ratio)
        if self.transmission_count > 0:
            prr = 1.0 - (self.collision_count / self.transmission_count)
            self.prr_stats[time_bin] = prr
    
    def _print_results(self):
        """Print final simulation results"""
        logger.info("\n=========== SIMULATION RESULTS ===========")
        logger.info(f"Total simulation time: {self.duration}ms")
        logger.info(f"Total transmissions: {self.transmission_count}")
        logger.info(f"Total collisions: {self.collision_count}")
        
        if self.transmission_count > 0:
            prr = 1.0 - (self.collision_count / self.transmission_count)
            logger.info(f"Overall Packet Reception Ratio (PRR): {prr:.4f}")
        
        # Per-vehicle statistics
        logger.info("\nVehicle Statistics:")
        for vehicle in self.vehicles:
            logger.info(f"Vehicle {vehicle.id}: Sent={vehicle.packets_sent}, " +
                      f"Received={vehicle.packets_received}, Collisions={vehicle.collisions}")
        
        # Plot results
        self._plot_results()
    
    def _plot_results(self):
        """Generate plots of simulation results"""
        # Plot collision and transmission statistics
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(list(self.transmission_stats.keys()), list(self.transmission_stats.values()), 'b-', label='Transmissions')
        plt.plot(list(self.collision_stats.keys()), list(self.collision_stats.values()), 'r-', label='Collisions')
        plt.xlabel('Time (s)')
        plt.ylabel('Count')
        plt.title('Transmissions and Collisions over Time')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(list(self.prr_stats.keys()), list(self.prr_stats.values()), 'g-')
        plt.xlabel('Time (s)')
        plt.ylabel('Packet Reception Ratio')
        plt.title('Packet Reception Ratio over Time')
        plt.ylim([0, 1])
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('v2x_simulation_results.png')
        logger.info("Results plotted and saved to 'v2x_simulation_results.png'")


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Create and run the simulation
    sim = Simulation(num_vehicles=20, duration=500000)
    sim.run()