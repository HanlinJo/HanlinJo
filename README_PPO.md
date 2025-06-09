# V2X Simulation with PPO-based Intelligent Attackers

This project implements a comprehensive V2X (Vehicle-to-Everything) communication simulation featuring intelligent attackers that use PPO (Proximal Policy Optimization) for resource selection based on sensing data and resource pool information.

## Features

### PPO-based Intelligent Attackers
- **Deep Reinforcement Learning**: Uses PPO algorithm for optimal resource block selection
- **State Space**: Resource pool information, sensing data statistics, subchannel usage, and temporal features
- **Action Space**: Continuous-to-discrete mapping for frame number, subframe number, and subchannel selection
- **Reward Function**: Positive rewards for successful attacks (causing collisions), negative rewards for failures

### Enhanced V2X Environment
- **Realistic Vehicle Behavior**: Proper resource selection using sensing and selection windows
- **Signal Calculations**: RSRP and RSSI calculations with realistic channel models
- **Collision Detection**: Accurate resource block overlap detection
- **Sensing Data Collection**: Vehicles and attackers maintain sensing windows for decision making

### Modular Architecture
- **Environment Module** (`v2x_environment.py`): Complete V2X simulation environment compatible with OpenAI Gym
- **PPO Agent Module** (`ppo_agent.py`): PPO implementation with actor-critic network
- **Training Module** (`train_ppo_attacker.py`): Training script with comprehensive logging and evaluation
- **Evaluation Module** (`evaluate_ppo_attacker.py`): Evaluation and comparison with baseline methods
- **Simulation Runner** (`run_ppo_simulation.py`): Complete simulation with multiple attacker types

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have Python 3.7+ with PyTorch support.

## Usage

### 1. Training a PPO Agent

Train a new PPO agent for V2X resource selection:

```bash
python train_ppo_attacker.py --episodes 1000 --num_vehicles 20 --save_dir models
```

#### Training Parameters:
- `--episodes`: Number of training episodes (default: 1000)
- `--max_steps`: Maximum steps per episode (default: 1000)
- `--update_freq`: Update frequency in episodes (default: 10)
- `--lr`: Learning rate (default: 3e-4)
- `--gamma`: Discount factor (default: 0.99)
- `--eps_clip`: PPO clipping parameter (default: 0.2)

### 2. Evaluating a Trained Agent

Evaluate a trained PPO agent and compare with random baseline:

```bash
python evaluate_ppo_attacker.py --model_path models/ppo_model_final.pth --episodes 100 --compare_random
```

#### Evaluation Parameters:
- `--model_path`: Path to trained PPO model (required)
- `--episodes`: Number of evaluation episodes (default: 100)
- `--compare_random`: Compare with random baseline
- `--output_dir`: Directory to save results (default: evaluation_results)

### 3. Running Complete Simulation

Run a complete V2X simulation comparing different attacker types:

```bash
python run_ppo_simulation.py --compare_all --duration 50000 --output_dir simulation_results
```

#### Simulation Parameters:
- `--num_vehicles`: Number of legitimate vehicles (default: 20)
- `--duration`: Simulation duration in ms (default: 50000)
- `--ppo_model_path`: Path to trained PPO model (trains new if not provided)
- `--compare_all`: Compare PPO, random, and no-attacker scenarios
- `--train_episodes`: Episodes for training new model (default: 500)

## PPO Algorithm Details

### State Space (11 dimensions)
1. **Resource Pool Information** (4 dims): Number of subchannels, slots, subchannel size, total RBs
2. **Sensing Data Statistics** (1 dim): Number of sensed transmissions in the last 100ms
3. **Subchannel Usage** (5 dims): Usage count per subchannel based on sensing data
4. **Temporal Information** (1 dim): Current time modulo 1000ms for temporal patterns

### Action Space (3 dimensions)
1. **Frame Number**: 0-1023 (LTE frame number)
2. **Subframe Number**: 1-10 (subframe within frame)
3. **Subchannel**: 0-4 (frequency subchannel)

### Network Architecture
- **Shared Layers**: Two hidden layers (64 neurons each) with ReLU activation
- **Actor Head**: Outputs action mean with learnable standard deviation
- **Critic Head**: Outputs state value estimate
- **Action Mapping**: Continuous outputs mapped to discrete resource parameters

### Reward Function
- **Successful Attack**: +1.0 (when attack causes collision)
- **Failed Attack**: -0.1 (when attack doesn't cause collision)
- **No Attack**: 0.0 (when not transmitting)

## Results and Analysis

The simulation generates comprehensive results including:

### Performance Metrics
- **Attack Success Rate**: Percentage of attacks causing collisions
- **Episode Rewards**: Cumulative rewards over training/evaluation
- **Collision Statistics**: Total collisions caused by attacks
- **Network Impact**: Effect on legitimate vehicle communications

### Learning Progress
- **Training Curves**: Policy loss, value loss, and entropy loss over training
- **Success Rate Evolution**: Attack success rate improvement over episodes
- **Convergence Analysis**: Training stability and final performance

### Comparison Results
- **PPO vs Random**: Performance improvement over random resource selection
- **PPO vs No Attacker**: Impact assessment on network performance
- **Statistical Significance**: Confidence intervals and significance tests

## Example Results

```
=========== PERFORMANCE COMPARISON ===========
PPO Success Rate: 0.3450
Random Success Rate: 0.1820
Improvement: 89.56%

PPO Reward: 245.3
Random Reward: 98.7
Reward Improvement: 146.6

Network Impact:
- PPO Attacker: 15.2% PRR reduction
- Random Attacker: 8.3% PRR reduction
```

## File Structure

```
├── v2x_environment.py          # V2X simulation environment (Gym-compatible)
├── ppo_agent.py               # PPO agent implementation
├── train_ppo_attacker.py      # Training script
├── evaluate_ppo_attacker.py   # Evaluation and comparison script
├── run_ppo_simulation.py      # Complete simulation runner
├── requirements.txt           # Python dependencies
├── README_PPO.md             # This documentation
└── models/                   # Directory for saved models
    ├── ppo_model_final.pth   # Final trained model
    └── training_config.json  # Training configuration
```

## Advanced Features

### Hyperparameter Tuning
The PPO implementation supports extensive hyperparameter customization:
- Learning rate scheduling
- Entropy coefficient adjustment
- Value function coefficient tuning
- Clipping parameter optimization

### Multi-Agent Extension
The framework can be extended to support multiple PPO agents:
- Cooperative attackers
- Competitive scenarios
- Hierarchical attack strategies

### Real-world Integration
The modular design enables integration with:
- Hardware-in-the-loop simulations
- Real V2X testbeds
- 5G NR sidelink implementations

## Research Applications

This framework supports research in:
- **Adversarial Machine Learning**: RL-based attack strategies
- **V2X Security**: Intelligent jamming and spoofing attacks
- **Network Resilience**: Defense mechanism evaluation
- **Resource Allocation**: Optimal resource selection under adversarial conditions

## Contributing

Contributions are welcome! Areas for improvement:
- Multi-agent PPO implementation
- Additional attack strategies
- Defense mechanisms
- Performance optimizations
- Real-world validation

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{v2x_ppo_simulation,
  title={V2X Simulation with PPO-based Intelligent Attackers},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-repo/v2x-ppo-simulation}}
}
```

## License

This project is provided for research and educational purposes. Please ensure compliance with applicable regulations when conducting V2X security research.