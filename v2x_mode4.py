def _handle_transmissions(self, transmissions):
    """Handle packet transmissions and detect collisions"""
    # First, collect transmissions by subframe
    tx_by_subframe = defaultdict(list)
    
    for sender, packet, resource in transmissions:
        # Convert to subframe index for resource pool
        sf_idx = (resource.subframe.frame_no * 10 + resource.subframe.subframe_no - 1) % self.resource_pool.num_slots
        tx_by_subframe[sf_idx].append((sender, packet, resource))
        
        # Increment transmission count
        self.transmission_count += 1
        
        # Count expected receptions (only for vehicles within range)
        for receiver in self.vehicles:
            if receiver.id != sender.id and receiver.should_receive_packet(sender.position[0]):
                self.total_expected_packets += 1
    
    # Process each subframe
    for sf_idx, sf_transmissions in tx_by_subframe.items():
        # Track resource usage
        rb_usage = defaultdict(list)  # Maps RB to list of (sender, packet, resource)
        
        # First pass: record all resource block usage
        for sender, packet, resource in sf_transmissions:
            for rb in range(resource.rb_start, resource.rb_start + resource.rb_len):
                rb_usage[rb].append((sender, packet, resource))
        
        # Second pass: process transmissions and detect collisions
        collision_pairs = set()  # Track unique collision pairs
        
        # Check for collisions in each resource block
        for rb, users in rb_usage.items():
            if len(users) > 1:  # Collision detected
                # Record all pairs of colliding transmissions
                for i in range(len(users)):
                    for j in range(i + 1, len(users)):
                        sender1, _, _ = users[i]
                        sender2, _, _ = users[j]
                        collision_pairs.add(tuple(sorted([sender1.id, sender2.id])))
        
        # Update collision counts
        for sender1_id, sender2_id in collision_pairs:
            self.collision_count += 1
            self.vehicles[sender1_id].collisions += 1
            self.vehicles[sender2_id].collisions += 1
        
        # Process reception for each vehicle
        for receiver in self.vehicles:
            for sender, packet, resource in sf_transmissions:
                if sender.id != receiver.id:  # Don't receive own packet
                    if receiver.should_receive_packet(sender.position[0]):
                        # Check if this transmission was involved in a collision
                        had_collision = any(
                            (sender.id in pair) for pair in collision_pairs
                        )
                        
                        if not had_collision:
                            # Calculate path loss and signal parameters
                            path_loss = self.channel.calculate_path_loss(
                                sender.position[0], receiver.position[0]
                            )
                            rsrp = self.channel.calculate_rsrp(self.tx_power, path_loss)
                            interference = self._calculate_interference(
                                sender, resource, sf_transmissions, receiver
                            )
                            rssi = self.channel.calculate_rssi(rsrp, interference)
                            
                            # Process successful reception
                            receiver.receive_packet(packet, resource, rsrp, rssi)
                            self.total_received_packets += 1