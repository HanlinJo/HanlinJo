import numpy as np
import matplotlib.pyplot as plt
import random
import time
from collections import defaultdict
import logging
import math

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('V2X-MODE4-Simplified')

class Packet:
    """Represents a V2X packet for transmission"""
    
    def __init__(self, sender_id, timestamp, position, size=190):
        self.sender_id = sender_id
        self.timestamp = timestamp
        self.position = position
        self.size = size

class SensingData:
    """Represents sensing data from previous transmissions"""
    
    def __init__(self, subframe_info, rb_start, rb_len, sender_id):
        self.subframe_info = subframe_info
        self.rb_start = rb_start
        self.rb_len = rb_len
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
        self.current_resource = None
        self.sensing_data = []
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
        """Simplified resource selection - random selection for demonstration"""
        # Create selection window (T1=4ms to T2=100ms)
        selection_window = self._create_selection_window(current_time)
        
        # For simplicity, just select a random resource
        # In real implementation, this would use sensing data and RSSI calculations
        selected_resource = random.choice(selection_window)
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
            
            # For each subframe, consider all subchannels
            for subchannel in range(self.sim.resource_pool.num_subchannels):
                selection_window.append(ResourceInfo(current_sf, subchannel))
                
            if current_sf.frame_no > end_subframe.frame_no or (current_sf.frame_no == end_subframe.frame_no and current_sf.subframe_no > end_subframe.subframe_no):
                break
                
            # Move to next subframe
            current_subframe += 1
            if current_subframe > 10:
                current_subframe = 1
                current_frame += 1
                if current_frame > 1024:
                    current_frame = 1
        
        return selection_window
    
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
    
    def receive_packet(self, packet, resource, collision_occurred):
        """Process a received packet - simplified without SINR calculations"""
        # Store sensing information (simplified)
        subframe_info = resource.subframe
        rb_start = resource.rb_start
        rb_len = resource.rb_len
        
        # Add to sensing window
        sensing_data = SensingData(subframe_info, rb_start, rb_len, packet.sender_id)
        self.sensing_data.append(sensing_data)
        
        # If no collision occurred, packet is successfully received
        if not collision_occurred:
            self.packets_received += 1
            return True
        else:
            # Packet lost due to collision
            return False
    
    def should_receive_packet(self, sender_position):
        """Determine if this vehicle should receive a packet from sender"""
        distance = np.linalg.norm(self.position - sender_position)
        return distance <= self.sim.communication_range

class ResourcePool:
    """Manages the sidelink resource pool for V2X communication"""
    
    def __init__(self, num_subchannels=5, num_slots=100, subchannel_size=10):
        self.num_subchannels = num_subchannels
        self.num_slots = num_slots
        self.subchannel_size = subchannel_size
        self.total_rbs = num_subchannels * subchannel_size

class Simulation:
    """Manages the overall V2X simulation with simplified collision detection"""
    
    def __init__(self, num_vehicles=20, duration=50000, communication_range=320.0):
        self.num_vehicles = num_vehicles
        self.duration = duration  # Simulation duration in ms
        self.communication_range = communication_range  # Communication range in meters
        
        # Initialize resource pool
        self.resource_pool = ResourcePool(num_subchannels=5, num_slots=100, subchannel_size=10)
        
        # Initialize vehicles
        self.vehicles = []
        self._initialize_vehicles()
        
        # Simulation time variables
        self.current_time = 0
        
        # Statistics
        self.collision_count = 0
        self.transmission_count = 0
        self.total_expected_packets = 0
        self.total_received_packets = 0
        
        # Time-based statistics
        self.collision_stats = defaultdict(int)
        self.transmission_stats = defaultdict(int)
        self.prr_stats = defaultdict(float)
    
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
            
            # Set velocity based on lane
            velocity = np.array([10.0 if lane_y == lane1_y else -10.0, 0.0])
            
            # Create the vehicle
            vehicle = Vehicle(i, position, velocity, self)
            self.vehicles.append(vehicle)
    
    def run(self):
        """Run the simulation"""
        logger.info(f"Starting simplified V2X MODE4 simulation with {self.num_vehicles} vehicles")
        
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
                logger.info(f"Time: {self.current_time}ms, Transmissions: {self.transmission_count}, " +
                          f"Collisions: {self.collision_count}")
            
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
            vehicle.resel_counter -= 1  # Decrement reselection counter
            if vehicle.resel_counter == 0:
                vehicle.current_resource = None  # Reset resource if counter reaches zero
            if tx_result:
                packet, resource = tx_result
                transmissions.append((vehicle, packet, resource))

        # Process transmissions and detect collisions
        if transmissions:
            self._handle_transmissions(transmissions)
    
    def _handle_transmissions(self, transmissions):
        """Simplified collision detection based purely on resource conflicts"""
        # Group transmissions by subframe
        tx_by_subframe = defaultdict(list)
        
        for sender, packet, resource in transmissions:
            # Convert to subframe index for resource pool
            sf_idx = (resource.subframe.frame_no * 10 + resource.subframe.subframe_no - 1) % self.resource_pool.num_slots
            tx_by_subframe[sf_idx].append((sender, packet, resource))
            
            # Increment transmission count
            self.transmission_count += 1
        
        # Process each subframe
        for sf_idx, sf_transmissions in tx_by_subframe.items():
            # Track resource usage: Maps resource block to list of senders
            rb_usage = defaultdict(list)
            
            # Record all resource block usage
            for sender, packet, resource in sf_transmissions:
                for rb in range(resource.rb_start, resource.rb_start + resource.rb_len):
                    rb_usage[rb].append((sender, packet, resource))
            
            # Detect collisions: any resource block used by multiple vehicles
            collided_transmissions = set()
            
            for rb, users in rb_usage.items():
                if len(users) > 1:  # Collision detected on this resource block
                    logger.debug(f"Collision detected on RB {rb} in subframe {sf_idx}: " +
                               f"{[u[0].id for u in users]}")
                    
                    # Mark all transmissions using this RB as collided
                    for sender, packet, resource in users:
                        collided_transmissions.add(sender.id)
            
            # Update collision statistics
            collision_count_this_subframe = len(collided_transmissions)
            self.collision_count += collision_count_this_subframe
            
            # Update per-vehicle collision counts
            for sender_id in collided_transmissions:
                self.vehicles[sender_id].collisions += 1
            
            # Process packet reception for all vehicles
            for receiver in self.vehicles:
                for sender, packet, resource in sf_transmissions:
                    if sender.id != receiver.id:  # Don't receive own packet
                        # Check if receiver is within communication range
                        if receiver.should_receive_packet(sender.position):
                            self.total_expected_packets += 1
                            
                            # Determine if collision occurred for this transmission
                            collision_occurred = sender.id in collided_transmissions
                            
                            # Process reception
                            success = receiver.receive_packet(packet, resource, collision_occurred)
                            
                            if success:
                                self.total_received_packets += 1
                                sender.successful_transmissions += 1
                            
                            # Log collision details
                            if collision_occurred:
                                logger.debug(f"Packet from Vehicle {sender.id} to Vehicle {receiver.id} " +
                                           f"lost due to resource collision")
    
    def _record_statistics(self):
        """Record statistics at the current time"""
        time_bin = self.current_time // 1000  # Group by seconds
        self.collision_stats[time_bin] = self.collision_count
        self.transmission_stats[time_bin] = self.transmission_count
        
        # Calculate PRR (Packet Reception Ratio)
        if self.total_expected_packets > 0:
            prr = self.total_received_packets / self.total_expected_packets
            self.prr_stats[time_bin] = prr
    
    def _print_results(self):
        """Print final simulation results"""
        logger.info("\n=========== SIMPLIFIED SIMULATION RESULTS ===========")
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

        # Per-vehicle statistics
        logger.info("\nVehicle Statistics:")
        for vehicle in self.vehicles:
            collision_rate = vehicle.collisions / vehicle.packets_sent if vehicle.packets_sent > 0 else 0
            logger.info(f"Vehicle {vehicle.id}: Sent={vehicle.packets_sent}, " +
                      f"Received={vehicle.packets_received}, Collisions={vehicle.collisions}, " +
                      f"Collision Rate={collision_rate:.3f}")

        self._plot_results()
    
    def _plot_results(self):
        """Generate plots of simulation results"""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Transmissions and Collisions over Time
        plt.subplot(2, 2, 1)
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
        plt.subplot(2, 2, 2)
        prr_times = list(self.prr_stats.keys())
        prr_values = list(self.prr_stats.values())
        
        plt.plot(prr_times, prr_values, 'g-', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Packet Reception Ratio')
        plt.title('Packet Reception Ratio over Time')
        plt.ylim([0, 1])
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Collision Rate over Time
        plt.subplot(2, 2, 3)
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
        
        # Plot 4: Per-Vehicle Statistics
        plt.subplot(2, 2, 4)
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
        
        plt.tight_layout()
        plt.savefig('v2x_simplified_results.png', dpi=300, bbox_inches='tight')
        logger.info("Results plotted and saved to 'v2x_simplified_results.png'")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Create and run the simplified simulation
    sim = Simulation(num_vehicles=20, duration=50000)
    sim.run()