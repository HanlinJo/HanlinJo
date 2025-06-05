import numpy as np

def process_packets():
    resource_pool = []  # Resource pool to track used resources
    total_packets = 0
    collision_count = 0
    
    # Simulate packet transmission
    def send_packet(resource_block):
        nonlocal total_packets, collision_count
        
        # Check for collisions in resource pool
        for existing_block in resource_pool:
            if has_overlap(resource_block, existing_block):
                collision_count += 2  # Increment by 2 for collision
                return
        
        # No collision, add to resource pool
        resource_pool.append(resource_block)
        total_packets += 1  # Increment by 1 for successful transmission

    def has_overlap(block1, block2):
        # Check if two resource blocks overlap
        return any(rb in block2 for rb in block1)

    # Main processing loop
    # Add your packet processing logic here
    
    return total_packets, collision_count

if __name__ == "__main__":
    total, collisions = process_packets()
    print(f"Total packets sent: {total}")
    print(f"Number of collisions: {collisions}")