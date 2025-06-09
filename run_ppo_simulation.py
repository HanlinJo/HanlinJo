#!/usr/bin/env python3
"""
Complete V2X Simulation with PPO-based Attackers

This script runs a complete V2X simulation comparing PPO-trained attackers
with random attackers and baseline scenarios without attackers.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import logging
import argparse
import os
from datetime import datetime
import json

from v2x_environment import V2XEnvironment
from ppo_agent import PPOAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'v2x_ppo_simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('V2X-PPO-Simulation')

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='V2X Simulation with PPO-based Attackers')
    
    # Simulation parameters
    parser.add_argument('--num_vehicles', type=int, default=20,
                       help='Number of legitimate vehicles (default: 20)')
    parser.add_argument('--duration', type=int, default=50000,
                       help='Simulation duration in ms (default: 50000)')
    parser.add_argument('--communication_range', type=float, default=320.0,
                       help='Communication range in meters (default: 320.0)')
    parser.add_argument('--tx_power', type=float, default=23.0,
                       help='Transmission power in dBm (default: 23.0)')
    
    # PPO model parameters
    parser.add_argument('--ppo_model_path', type=str, default=None,
                       help='Path to trained PPO model (if None, trains a new model)')
    parser.add_argument('--train_episodes', type=int, default=500,
                       help='Number of training episodes if training new model (default: 500)')
    
    # Comparison parameters
    parser.add_argument('--compare_all', action='store_true',
                       help='Compare PPO, random, and no-attacker scenarios')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='simulation_results',
                       help='Output directory for results (default: simulation_results)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    return parser.parse_args()

class RandomAttacker:
    """Random baseline attacker"""
    
    def __init__(self, action_space):
        self.action_space = action_space
    
    def select_action(self, state, deterministic=False):
        """Select random action"""
        action = np.random.randint(
            low=self.action_space.low,
            high=self.action_space.high + 1,
            size=self.action_space.shape
        )
        return action, 0.0, 0.0

class V2XSimulator:
    """Complete V2X Simulator with different attacker types"""
    
    def __init__(self, num_vehicles=20, duration=50000, communication_range=320.0, tx_power=23.0):
        self.num_vehicles = num_vehicles
        self.duration = duration
        self.communication_range = communication_range
        self.tx_power = tx_power
        
        # Initialize environment
        self.env = V2XEnvironment(
            num_vehicles=num_vehicles,
            communication_range=communication_range,
            tx_power=tx_power
        )
        
        # Set episode length to match simulation duration
        self.env.episode_length = duration
    
    def run_simulation(self, attacker_agent=None, attacker_type="none"):
        """Run simulation with specified attacker type"""
        logger.info(f"Running simulation with {attacker_type} attacker for {self.duration}ms")
        
        # Reset environment
        state = self.env.reset()
        
        # Simulation statistics
        total_reward = 0
        total_attacks = 0
        total_successes = 0
        total_collisions = 0
        
        # Time-based statistics
        time_stats = {
            'rewards': [],
            'success_rates': [],
            'collisions': [],
            'attacks': []
        }
        
        # Run simulation
        for step in range(self.duration):
            if attacker_agent is not None:
                # Use attacker agent
                action, _, _ = attacker_agent.select_action(state, deterministic=True)
            else:
                # No attacker - use dummy action (won't be used)
                action = np.array([0, 1, 0])
            
            # Step environment
            if attacker_agent is not None:
                state, reward, done, info = self.env.step(action)
                total_reward += reward
            else:
                # Simulate without attacker
                state, reward, done, info = self._step_without_attacker()
            
            # Update statistics
            total_attacks = info.get('episode_attacks', 0)
            total_successes = info.get('episode_successes', 0)
            total_collisions = info.get('episode_collisions', 0)
            
            # Record time-based statistics every 1000ms
            if step % 1000 == 0:
                time_stats['rewards'].append(total_reward)
                time_stats['success_rates'].append(total_successes / max(1, total_attacks))
                time_stats['collisions'].append(total_collisions)
                time_stats['attacks'].append(total_attacks)
            
            if done:
                break
        
        # Calculate final statistics
        final_success_rate = total_successes / max(1, total_attacks)
        
        results = {
            'attacker_type': attacker_type,
            'total_reward': total_reward,
            'total_attacks': total_attacks,
            'total_successes': total_successes,
            'total_collisions': total_collisions,
            'success_rate': final_success_rate,
            'time_stats': time_stats,
            'simulation_duration': self.duration
        }
        
        logger.info(f"{attacker_type.capitalize()} simulation completed:")
        logger.info(f"  Total Attacks: {total_attacks}")
        logger.info(f"  Total Successes: {total_successes}")
        logger.info(f"  Success Rate: {final_success_rate:.4f}")
        logger.info(f"  Total Collisions: {total_collisions}")
        logger.info(f"  Total Reward: {total_reward:.3f}")
        
        return results
    
    def _step_without_attacker(self):
        """Simulate one step without attacker"""
        # Process only vehicle transmissions
        reward = 0.0
        
        # Update vehicle positions
        for vehicle in self.env.vehicles:
            vehicle.move(1 / 1000.0)
        
        # Collect vehicle transmissions
        vehicle_transmissions = []
        for vehicle in self.env.vehicles:
            tx_result = vehicle.send_packet(self.env.current_time)
            vehicle.resel_counter -= 1
            if vehicle.resel_counter <= 0:
                vehicle.current_resource = None
            if tx_result:
                packet, resource = tx_result
                vehicle_transmissions.append((vehicle, packet, resource))
        
        # Process transmissions without attacker
        if vehicle_transmissions:
            self.env._handle_transmissions(vehicle_transmissions, None)
        
        # Update time
        self.env.current_time += 1
        self.env.step_count += 1
        
        # Check if done
        done = self.env.step_count >= self.env.episode_length
        
        # Get observation
        state = self.env._get_observation()
        
        # Info without attacker statistics
        info = {
            'episode_attacks': 0,
            'episode_successes': 0,
            'episode_collisions': self.env.episode_collisions
        }
        
        return state, reward, done, info

def train_ppo_agent(env, episodes=500):
    """Train a new PPO agent"""
    logger.info(f"Training new PPO agent for {episodes} episodes...")
    
    # Initialize PPO agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)
    
    # Training loop
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < 1000:  # Limit steps per episode for training
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            agent.store_transition(state, action, reward, log_prob, value, done)
            
            state = next_state
            episode_reward += reward
            step += 1
        
        # Update agent every 10 episodes
        if (episode + 1) % 10 == 0:
            training_stats = agent.update()
            if training_stats and (episode + 1) % 50 == 0:
                logger.info(f"Episode {episode + 1}: Reward={episode_reward:.3f}, "
                          f"Success Rate={info['success_rate']:.3f}")
    
    logger.info("PPO training completed")
    return agent

def plot_comparison_results(results_list, output_dir):
    """Plot comparison results"""
    plt.figure(figsize=(15, 12))
    
    # Extract data for plotting
    attacker_types = [r['attacker_type'] for r in results_list]
    success_rates = [r['success_rate'] for r in results_list]
    total_collisions = [r['total_collisions'] for r in results_list]
    total_rewards = [r['total_reward'] for r in results_list]
    
    # Plot 1: Success Rate Comparison
    plt.subplot(2, 3, 1)
    bars = plt.bar(attacker_types, success_rates, alpha=0.7, 
                   color=['red', 'blue', 'green'][:len(attacker_types)])
    plt.ylabel('Attack Success Rate')
    plt.title('Attack Success Rate Comparison')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, success_rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Plot 2: Total Collisions Comparison
    plt.subplot(2, 3, 2)
    bars = plt.bar(attacker_types, total_collisions, alpha=0.7,
                   color=['red', 'blue', 'green'][:len(attacker_types)])
    plt.ylabel('Total Collisions')
    plt.title('Total Collisions Comparison')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, total_collisions):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(total_collisions)*0.01,
                f'{value}', ha='center', va='bottom')
    
    # Plot 3: Total Reward Comparison (only for attackers)
    plt.subplot(2, 3, 3)
    attacker_results = [r for r in results_list if r['attacker_type'] != 'none']
    if attacker_results:
        attacker_names = [r['attacker_type'] for r in attacker_results]
        attacker_rewards = [r['total_reward'] for r in attacker_results]
        bars = plt.bar(attacker_names, attacker_rewards, alpha=0.7,
                       color=['red', 'blue'][:len(attacker_names)])
        plt.ylabel('Total Reward')
        plt.title('Attacker Reward Comparison')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, attacker_rewards):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(attacker_rewards)*0.01,
                    f'{value:.1f}', ha='center', va='bottom')
    
    # Plot 4-6: Time series for each scenario
    for i, result in enumerate(results_list):
        plt.subplot(2, 3, 4 + i)
        time_points = range(0, len(result['time_stats']['success_rates']))
        plt.plot(time_points, result['time_stats']['success_rates'], 'b-', linewidth=2)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Success Rate')
        plt.title(f'{result["attacker_type"].capitalize()} Success Rate over Time')
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'simulation_comparison.png'), dpi=300, bbox_inches='tight')
    logger.info(f"Comparison results plotted and saved to {output_dir}/simulation_comparison.png")

def save_results(results_list, output_dir):
    """Save simulation results"""
    # Save detailed results
    results_path = os.path.join(output_dir, 'simulation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_list, f, indent=2)
    
    # Create summary
    summary = {}
    for result in results_list:
        summary[result['attacker_type']] = {
            'success_rate': result['success_rate'],
            'total_collisions': result['total_collisions'],
            'total_reward': result['total_reward'],
            'total_attacks': result['total_attacks']
        }
    
    # Calculate improvements if PPO and random both exist
    if 'ppo' in summary and 'random' in summary:
        ppo_sr = summary['ppo']['success_rate']
        random_sr = summary['random']['success_rate']
        improvement = (ppo_sr - random_sr) / max(random_sr, 1e-8) * 100
        summary['improvement'] = {
            'success_rate_improvement_percent': improvement,
            'absolute_improvement': ppo_sr - random_sr
        }
    
    summary_path = os.path.join(output_dir, 'simulation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")
    return summary

def main():
    """Main simulation function"""
    args = parse_arguments()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(args.output_dir, 'simulation_config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Initialize simulator
    simulator = V2XSimulator(
        num_vehicles=args.num_vehicles,
        duration=args.duration,
        communication_range=args.communication_range,
        tx_power=args.tx_power
    )
    
    results = []
    
    # Initialize PPO agent
    ppo_agent = None
    if args.ppo_model_path and os.path.exists(args.ppo_model_path):
        # Load existing model
        state_dim = simulator.env.observation_space.shape[0]
        action_dim = simulator.env.action_space.shape[0]
        ppo_agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)
        ppo_agent.load_model(args.ppo_model_path)
        logger.info(f"Loaded PPO model from {args.ppo_model_path}")
    else:
        # Train new model
        ppo_agent = train_ppo_agent(simulator.env, args.train_episodes)
        # Save trained model
        model_path = os.path.join(args.output_dir, 'trained_ppo_model.pth')
        ppo_agent.save_model(model_path)
    
    # Run PPO simulation
    logger.info("Running simulation with PPO attacker...")
    ppo_results = simulator.run_simulation(ppo_agent, "ppo")
    results.append(ppo_results)
    
    if args.compare_all:
        # Run random attacker simulation
        logger.info("Running simulation with random attacker...")
        random_agent = RandomAttacker(simulator.env.action_space)
        random_results = simulator.run_simulation(random_agent, "random")
        results.append(random_results)
        
        # Run no-attacker simulation
        logger.info("Running simulation without attacker...")
        no_attacker_results = simulator.run_simulation(None, "none")
        results.append(no_attacker_results)
    
    # Plot and save results
    plot_comparison_results(results, args.output_dir)
    summary = save_results(results, args.output_dir)
    
    # Print summary
    logger.info("\n=========== SIMULATION SUMMARY ===========")
    for attacker_type, stats in summary.items():
        if attacker_type != 'improvement':
            logger.info(f"{attacker_type.upper()} Results:")
            logger.info(f"  Success Rate: {stats['success_rate']:.4f}")
            logger.info(f"  Total Collisions: {stats['total_collisions']}")
            logger.info(f"  Total Attacks: {stats['total_attacks']}")
            if 'total_reward' in stats:
                logger.info(f"  Total Reward: {stats['total_reward']:.3f}")
    
    if 'improvement' in summary:
        logger.info(f"\nPPO vs Random Improvement:")
        logger.info(f"  Success Rate Improvement: {summary['improvement']['success_rate_improvement_percent']:.2f}%")
        logger.info(f"  Absolute Improvement: {summary['improvement']['absolute_improvement']:.4f}")
    
    logger.info("Simulation completed successfully!")

if __name__ == "__main__":
    main()