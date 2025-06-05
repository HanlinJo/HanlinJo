def calculate_path_loss(self, tx_position, rx_position):
    """Calculate path loss between transmitter and receiver using highway model"""
    distance = abs(tx_position - rx_position)
    
    if distance == 0:
        return 0
    
    # Highway path loss model parameters
    # Based on WINNER+ B1 highway scenario
    d0 = 10  # Reference distance in meters
    n = 2.27  # Path loss exponent for highway (WINNER+ B1)
    sigma = 6  # Shadow fading standard deviation in dB
    
    # Calculate basic path loss
    if distance <= d0:
        path_loss_db = 20 * math.log10(distance)
    else:
        # WINNER+ B1 highway model
        path_loss_db = 22.7 * math.log10(distance) + 41.0 + 20 * math.log10(5.9)  # 5.9 GHz frequency
    
    # Add shadow fading
    shadow_fading = random.gauss(0, sigma)
    path_loss_db += shadow_fading
    
    return path_loss_db

def calculate_rsrp(self, tx_power, path_loss):
    """Calculate RSRP (Reference Signal Received Power)"""
    # RSRP calculation with consideration of resource block power
    num_rbs = 50  # Number of resource blocks in channel
    rb_power = tx_power - 10 * math.log10(num_rbs)  # Power per resource block
    
    # Calculate RSRP
    rsrp_dbm = rb_power - path_loss
    
    # Apply realistic constraints
    rsrp_dbm = min(rsrp_dbm, tx_power)  # Cannot exceed transmit power
    rsrp_dbm = max(rsrp_dbm, -140)  # Minimum detectable RSRP
    
    return rsrp_dbm

def calculate_rssi(self, rsrp, interference=0):
    """Calculate RSSI (Received Signal Strength Indicator)"""
    # Enhanced RSSI calculation with realistic parameters
    thermal_noise = -174  # Thermal noise floor in dBm/Hz
    bandwidth = 10e6  # 10 MHz bandwidth
    noise_figure = 9  # Receiver noise figure in dB
    
    # Calculate noise floor
    noise_floor_dbm = thermal_noise + 10 * math.log10(bandwidth) + noise_figure
    
    # Convert all powers to linear domain for addition
    rsrp_linear = 10 ** (rsrp/10)
    interference_linear = 10 ** (interference/10)
    noise_linear = 10 ** (noise_floor_dbm/10)
    
    # Sum powers and convert back to dB
    total_power = rsrp_linear + interference_linear + noise_linear
    rssi_dbm = 10 * math.log10(total_power)
    
    # Apply realistic constraints
    rssi_dbm = max(rssi_dbm, noise_floor_dbm)  # Cannot be below noise floor
    
    return rssi_dbm