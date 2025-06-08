#!/usr/bin/env python3
"""
Enhanced V2X Simulation Runner with HAPPO-based Attackers

This script runs the enhanced V2X simulation with intelligent attackers
that use HAPPO (Heterogeneous-Agent Proximal Policy Optimization) for
resource selection based on sensing data and resource pool information.
"""

import numpy as np
import random
import logging
import argparse
import json
from datetime import datetime

from v2x_enhanced_with_happo import EnhancedSimulation
from happo_agent import create_happo_agent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'v2x_happo_simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('V2X-HAPPO-Runner')

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced V2X Simulation with HAPPO Attackers')
    
    # Simulation parameters
    parser.add_argument('--num_vehicles', type=int, default=20, 
                       help='Number of legitimate vehicles (default: 20)')
    parser.add_argument('--num_attackers', type=int, default=1, 
                       help='Number of attackers (default: 1)')
    parser.add_argument('--duration', type=int, default=50000, 
                       help='Simulation duration in ms (default: 50000)')
    parser.add_argument('--communication_range', type=float, default=320.0, 
                       help='Communication range in meters (default: 320.0)')
    parser.add_argument('--tx_power', type=float, default=23.0, 
                       help='Transmission power in dBm (default: 23.0)')
    
    # HAPPO parameters
    parser.add_argument('--agent_type', type=str, default='basic', 
                       choices=['basic', 'advanced'],
                       help='Type of HAPPO agent (default: basic)')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                       help='Learning rate for HAPPO agent (default: 0.001)')
    parser.add_argument('--epsilon', type=float, default=0.1, 
                       help='Initial epsilon for exploration (default: 0.1)')
    parser.add_argument('--epsilon_decay', type=float, default=0.995, 
                       help='Epsilon decay rate (default: 0.995)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results', 
                       help='Output directory for results (default: results)')
    parser.add_argument('--save_config', action='store_true', 
                       help='Save simulation configuration')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility (default: 42)')
    
    return parser.parse_args()

def save_configuration(args, output_dir):
    """Save simulation configuration to file"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    config = {
        'simulation_parameters': {
            'num_vehicles': args.num_vehicles,
            'num_attackers': args.num_attackers,
            'duration': args.duration,
            'communication_range': args.communication_range,
            'tx_power': args.tx_power,
            'seed': args.seed
        },
        'happo_parameters': {
            'agent_type': args.agent_type,
            'learning_rate': args.learning_rate,
            'epsilon': args.epsilon,
            'epsilon_decay': args.epsilon_decay
        },
        'timestamp': datetime.now().isoformat()
    }
    
    config_file = os.path.join(output_dir, 'simulation_config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Configuration saved to {config_file}")

def run_simulation_with_happo(args):
    """Run the enhanced simulation with HAPPO attackers"""
    logger.info("Starting Enhanced V2X Simulation with HAPPO Attackers")
    logger.info(f"Configuration: {vars(args)}")
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Save configuration if requested
    if args.save_config:
        save_configuration(args, args.output_dir)
    
    # Create and configure simulation
    sim = EnhancedSimulation(
        num_vehicles=args.num_vehicles,
        num_attackers=args.num_attackers,
        duration=args.duration,
        communication_range=args.communication_range,
        tx_power=args.tx_power
    )
    
    # Configure HAPPO agents for attackers
    for attacker in sim.attackers:
        if attacker.happo_agent:
            # Update agent parameters
            attacker.happo_agent.learning_rate = args.learning_rate
            attacker.happo_agent.epsilon = args.epsilon
            attacker.happo_agent.epsilon_decay = args.epsilon_decay
            
            logger.info(f"Configured HAPPO agent for Attacker {attacker.id}")
    
    # Run simulation
    logger.info("Starting simulation execution...")
    start_time = datetime.now()
    
    try:
        sim.run()
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        logger.info(f"Simulation completed successfully in {execution_time:.2f} seconds")
        
        # Print HAPPO learning statistics
        print_happo_statistics(sim.attackers)
        
        return sim
        
    except Exception as e:
        logger.error(f"Simulation failed with error: {str(e)}")
        raise

def print_happo_statistics(attackers):
    """Print HAPPO learning statistics"""
    logger.info("\n=========== HAPPO LEARNING STATISTICS ===========")
    
    for attacker in attackers:
        if attacker.happo_agent:
            stats = attacker.happo_agent.get_learning_stats()
            
            logger.info(f"\nAttacker {attacker.id} HAPPO Statistics:")
            logger.info(f"  Epsilon (exploration rate): {stats['epsilon']:.4f}")
            logger.info(f"  Q-table size (states): {stats['q_table_size']}")
            logger.info(f"  Total state-action pairs: {stats['total_state_actions']}")
            logger.info(f"  Average episode reward: {stats['average_episode_reward']:.3f}")
            logger.info(f"  Current episode reward: {stats['current_episode_reward']:.3f}")
            
            # Advanced statistics if available
            if hasattr(attacker.happo_agent, 'get_advanced_stats'):
                advanced_stats = attacker.happo_agent.get_advanced_stats()
                logger.info(f"  Success patterns learned: {advanced_stats['success_patterns_count']}")
                logger.info(f"  Action history length: {advanced_stats['action_history_length']}")
                
                if advanced_stats['top_successful_actions']:
                    logger.info("  Top successful actions:")
                    for i, (action, reward) in enumerate(advanced_stats['top_successful_actions']):
                        logger.info(f"    {i+1}. {action} (reward: {reward:.3f})")

def compare_with_baseline(args):
    """Compare HAPPO performance with baseline random selection"""
    logger.info("Running baseline comparison...")
    
    # Run simulation with HAPPO
    logger.info("Running simulation with HAPPO attackers...")
    happo_sim = run_simulation_with_happo(args)
    
    # Run simulation without HAPPO (random selection)
    logger.info("Running baseline simulation with random attackers...")
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    baseline_sim = EnhancedSimulation(
        num_vehicles=args.num_vehicles,
        num_attackers=args.num_attackers,
        duration=args.duration,
        communication_range=args.communication_range,
        tx_power=args.tx_power
    )
    
    # Disable HAPPO for baseline
    for attacker in baseline_sim.attackers:
        attacker.happo_agent = None
    
    baseline_sim.run()
    
    # Compare results
    logger.info("\n=========== PERFORMANCE COMPARISON ===========")
    
    happo_success_rate = happo_sim.total_attack_success / happo_sim.attack_transmission_count if happo_sim.attack_transmission_count > 0 else 0
    baseline_success_rate = baseline_sim.total_attack_success / baseline_sim.attack_transmission_count if baseline_sim.attack_transmission_count > 0 else 0
    
    happo_prr = happo_sim.total_received_packets / happo_sim.total_expected_packets if happo_sim.total_expected_packets > 0 else 0
    baseline_prr = baseline_sim.total_received_packets / baseline_sim.total_expected_packets if baseline_sim.total_expected_packets > 0 else 0
    
    logger.info(f"HAPPO Attack Success Rate: {happo_success_rate:.4f}")
    logger.info(f"Baseline Attack Success Rate: {baseline_success_rate:.4f}")
    logger.info(f"Improvement: {((happo_success_rate - baseline_success_rate) / baseline_success_rate * 100):.2f}%" if baseline_success_rate > 0 else "N/A")
    
    logger.info(f"HAPPO Network PRR: {happo_prr:.4f}")
    logger.info(f"Baseline Network PRR: {baseline_prr:.4f}")
    logger.info(f"PRR Impact: {((baseline_prr - happo_prr) / baseline_prr * 100):.2f}%" if baseline_prr > 0 else "N/A")

def main():
    """Main function"""
    args = parse_arguments()
    
    try:
        if args.num_attackers > 0:
            # Run comparison if attackers are present
            compare_with_baseline(args)
        else:
            # Run single simulation without attackers
            run_simulation_with_happo(args)
            
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()