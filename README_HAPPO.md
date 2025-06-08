# Enhanced V2X Simulation with HAPPO-based Attackers

This project implements an enhanced V2X (Vehicle-to-Everything) communication simulation featuring intelligent attackers that use HAPPO (Heterogeneous-Agent Proximal Policy Optimization) for resource selection based on sensing data and resource pool information.

## Features

### Enhanced Attacker Capabilities
- **Reception Logic**: Attackers can receive packets from senders within 320m range
- **Sensing Data Collection**: Attackers maintain a 100ms sensing data pool for attack analysis
- **Intelligent Resource Selection**: HAPPO-based learning for optimal resource block selection

### HAPPO Agent
- **State Space**: Resource pool information derived from sensing data
- **Action Space**: Resource block selection (frame_no, subframe_no, subchannel)
- **Learning**: Q-learning based approach with epsilon-greedy exploration
- **Pattern Recognition**: Advanced agents can learn from successful attack patterns

### Simulation Enhancements
- **Realistic Channel Model**: WINNER+ B1 highway propagation model
- **Enhanced Collision Detection**: Proper resource block overlap detection
- **Signal Calculations**: RSRP and RSSI calculations for realistic reception
- **Comprehensive Statistics**: Detailed performance metrics and learning statistics

## File Structure

```
├── v2x_enhanced_with_happo.py    # Main simulation with enhanced attackers
├── happo_agent.py                # HAPPO agent implementation
├── run_enhanced_simulation.py    # Simulation runner with configuration options
├── README_HAPPO.md              # This documentation
└── requirements.txt             # Python dependencies
```

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the simulation:
```bash
python run_enhanced_simulation.py
```

## Usage

### Basic Usage

Run simulation with default parameters:
```bash
python run_enhanced_simulation.py
```

### Advanced Configuration

```bash
python run_enhanced_simulation.py \
    --num_vehicles 20 \
    --num_attackers 2 \
    --duration 60000 \
    --agent_type advanced \
    --learning_rate 0.001 \
    --epsilon 0.1 \
    --save_config
```

### Parameters

#### Simulation Parameters
- `--num_vehicles`: Number of legitimate vehicles (default: 20)
- `--num_attackers`: Number of attackers (default: 1)
- `--duration`: Simulation duration in ms (default: 50000)
- `--communication_range`: Communication range in meters (default: 320.0)
- `--tx_power`: Transmission power in dBm (default: 23.0)

#### HAPPO Parameters
- `--agent_type`: Type of HAPPO agent ('basic' or 'advanced')
- `--learning_rate`: Learning rate for HAPPO agent (default: 0.001)
- `--epsilon`: Initial epsilon for exploration (default: 0.1)
- `--epsilon_decay`: Epsilon decay rate (default: 0.995)

#### Output Parameters
- `--output_dir`: Output directory for results (default: 'results')
- `--save_config`: Save simulation configuration
- `--seed`: Random seed for reproducibility (default: 42)

## HAPPO Agent Details

### State Space
The HAPPO agent observes the following state information:
1. **Resource Pool Information**: Number of subchannels, slots, subchannel size, total RBs
2. **Sensing Data Statistics**: Number of sensed transmissions in the last 100ms
3. **Subchannel Usage**: Usage count per subchannel based on sensing data
4. **Temporal Information**: Current time modulo 1000ms for temporal patterns

### Action Space
The agent selects resources by choosing:
1. **Frame Number**: 0-1023 (LTE frame number)
2. **Subframe Number**: 1-10 (subframe within frame)
3. **Subchannel**: 0-(num_subchannels-1) (frequency subchannel)

### Learning Algorithm
- **Q-Learning**: State-action value function learning
- **Epsilon-Greedy**: Exploration vs exploitation balance
- **Experience Replay**: Memory of past state-action-reward transitions
- **Pattern Recognition**: Advanced agents learn from successful attack patterns

## Results and Analysis

The simulation generates comprehensive results including:

### Performance Metrics
- **Attack Success Rate**: Percentage of attacks causing collisions
- **Packet Reception Ratio (PRR)**: Network performance impact
- **Collision Rate**: Overall collision frequency
- **Per-Vehicle Statistics**: Individual vehicle performance

### Learning Statistics
- **Q-table Growth**: Number of learned state-action pairs
- **Exploration Rate**: Current epsilon value
- **Episode Rewards**: Learning progress over time
- **Success Patterns**: Identified successful attack strategies

### Visualization
- Time-series plots of key metrics
- Collision rate analysis
- Attack success progression
- Sensing data distribution

## Comparison with Baseline

The simulation automatically compares HAPPO-based attackers with random resource selection:

```
=========== PERFORMANCE COMPARISON ===========
HAPPO Attack Success Rate: 0.2150
Baseline Attack Success Rate: 0.1820
Improvement: 18.13%

HAPPO Network PRR: 0.8234
Baseline Network PRR: 0.8456
PRR Impact: 2.63%
```

## Advanced Features

### Enhanced Attacker Class
- **Reception Logic**: Process packets from vehicles within range
- **Sensing Window Management**: Maintain recent transmission history
- **State Generation**: Create HAPPO-compatible state representations
- **Reward Processing**: Provide learning feedback based on attack success

### Advanced HAPPO Agent
- **Pattern Recognition**: Learn from successful attack sequences
- **Multi-objective Optimization**: Balance collision success and efficiency
- **Adaptive Exploration**: Dynamic epsilon adjustment based on performance
- **Success Pattern Bias**: Prefer actions similar to past successes

## Research Applications

This simulation framework supports research in:
- **V2X Security**: Analysis of intelligent jamming attacks
- **Resource Allocation**: Impact of malicious resource selection
- **Machine Learning in Wireless**: RL applications in communication systems
- **Network Resilience**: Evaluation of defense mechanisms

## Future Enhancements

Potential improvements include:
- **Multi-Agent HAPPO**: Coordination between multiple attackers
- **Deep RL**: Neural network-based value functions
- **Countermeasures**: Intelligent defense mechanisms
- **Real-world Validation**: Integration with actual V2X hardware

## Contributing

Contributions are welcome! Please focus on:
- Algorithm improvements
- Additional attack strategies
- Defense mechanisms
- Performance optimizations
- Documentation enhancements

## License

This project is provided for research and educational purposes. Please cite appropriately if used in academic work.