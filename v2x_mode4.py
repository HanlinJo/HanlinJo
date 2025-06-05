def receive_packet(self, packet, resource, rsrp, rssi):
    """Process a received packet"""
    # Check SINR threshold for successful reception
    sinr = rsrp - rssi  # Simplified SINR calculation
    if sinr < -5:  # SINR threshold for highway V2X
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