def _calculate_interference(self, target_sender, target_resource, all_transmissions, receiver):
    """Calculate interference from other transmissions"""
    interference_power_linear = 0
    noise_floor = -95  # Noise floor in dBm
    
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
            
            # Convert to linear domain and add
            interference_power_linear += 10 ** (interference_rsrp/10)
    
    # Add noise floor to interference
    interference_power_linear += 10 ** (noise_floor/10)
    
    # Convert back to dB, ensuring we never return -inf
    return max(-130, 10 * math.log10(interference_power_linear))