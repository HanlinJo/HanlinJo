import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse
from datetime import datetime
import os

from v2x_environment import V2XEnvironment
from ppo_agent import PPOAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('PPO-Evaluation')

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate PPO Agent for V2X Attacker')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained PPO model')
    
    # Evaluation parameters
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of evaluation episodes (default: 100)')
    parser.add_argument('--max_steps', type=int, default=1000,
                       help='Maximum steps per episode (default: 1000)')
    
    # Environment parameters
    parser.add_argument('--num_vehicles', type=int, default=20,
                       help='Number of vehicles (default: 20)')
    parser.add_argument('--communication_range', type=float, default=320.0,
                       help='Communication range in meters (default: 320.0)')
    parser.add_argument('--tx_power', type=float, default=23.0,
                       help='Transmission power in dBm (default: 23.0)')
    
    # Comparison parameters
    parser.add_argument('--compare_random', action='store_true',
                       help='Compare with random baseline')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save results (default: evaluation_results)')
    
    return parser.parse_args()

class RandomAttacker:
    """Random baseline attacker for comparison"""
    
    def __init__(self, action_space):
        self.action_space = action_space
    
    def select_action(self, state, deterministic=False):
        """Select random action"""
        action = np.random.randint(
            low=self.action_space.low,
            high=self.action_space.high + 1,
            size=self.action_space.shape
        )
        return action, 0.0, 0.0  # action, log_prob, value

def evaluate_agent(env, agent, num_episodes=100, agent_name="Agent"):
    """Evaluate an agent"""
    logger.info(f"Evaluating {agent_name} for {num_episodes} episodes...")
    
    episode_rewards = []
    episode_success_rates = []
    episode_collisions = []
    episode_attacks = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < env.episode_length:
            action, _, _ = agent.select_action(state, deterministic=True)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            step += 1
        
        episode_rewards.append(episode_reward)
        episode_success_rates.append(info['success_rate'])
        episode_collisions.append(info['episode_collisions'])
        episode_attacks.append(info['episode_attacks'])
        
        if (episode + 1) % 20 == 0:
            logger.info(f"{agent_name} - Episode {episode + 1}/{num_episodes}: "
                      f"Reward={episode_reward:.3f}, Success Rate={info['success_rate']:.3f}")
    
    results = {
        'episode_rewards': episode_rewards,
        'episode_success_rates': episode_success_rates,
        'episode_collisions': episode_collisions,
        'episode_attacks': episode_attacks,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_success_rate': np.mean(episode_success_rates),
        'std_success_rate': np.std(episode_success_rates),
        'mean_collisions': np.mean(episode_collisions),
        'mean_attacks': np.mean(episode_attacks)
    }
    
    logger.info(f"{agent_name} Results:")
    logger.info(f"  Mean Reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    logger.info(f"  Mean Success Rate: {results['mean_success_rate']:.3f} ± {results['std_success_rate']:.3f}")
    logger.info(f"  Mean Collisions: {results['mean_collisions']:.1f}")
    logger.info(f"  Mean Attacks: {results['mean_attacks']:.1f}")
    
    return results

def plot_comparison_results(ppo_results, random_results, output_dir):
    """Plot comparison results between PPO and random agents"""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Success Rate Comparison
    plt.subplot(2, 3, 1)
    plt.hist(ppo_results['episode_success_rates'], alpha=0.7, label='PPO', bins=20)
    plt.hist(random_results['episode_success_rates'], alpha=0.7, label='Random', bins=20)
    plt.xlabel('Success Rate')
    plt.ylabel('Frequency')
    plt.title('Attack Success Rate Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Reward Comparison
    plt.subplot(2, 3, 2)
    plt.hist(ppo_results['episode_rewards'], alpha=0.7, label='PPO', bins=20)
    plt.hist(random_results['episode_rewards'], alpha=0.7, label='Random', bins=20)
    plt.xlabel('Episode Reward')
    plt.ylabel('Frequency')
    plt.title('Episode Reward Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Success Rate over Episodes
    plt.subplot(2, 3, 3)
    episodes = range(len(ppo_results['episode_success_rates']))
    plt.plot(episodes, ppo_results['episode_success_rates'], alpha=0.7, label='PPO')
    plt.plot(episodes, random_results['episode_success_rates'], alpha=0.7, label='Random')
    
    # Add moving averages
    window = 10
    if len(episodes) >= window:
        ppo_ma = np.convolve(ppo_results['episode_success_rates'], np.ones(window)/window, mode='valid')
        random_ma = np.convolve(random_results['episode_success_rates'], np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(episodes)), ppo_ma, 'r-', linewidth=2, label='PPO (MA)')
        plt.plot(range(window-1, len(episodes)), random_ma, 'b-', linewidth=2, label='Random (MA)')
    
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.title('Success Rate over Episodes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Box Plot Comparison
    plt.subplot(2, 3, 4)
    data = [ppo_results['episode_success_rates'], random_results['episode_success_rates']]
    plt.boxplot(data, labels=['PPO', 'Random'])
    plt.ylabel('Success Rate')
    plt.title('Success Rate Box Plot')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Collisions Comparison
    plt.subplot(2, 3, 5)
    plt.hist(ppo_results['episode_collisions'], alpha=0.7, label='PPO', bins=20)
    plt.hist(random_results['episode_collisions'], alpha=0.7, label='Random', bins=20)
    plt.xlabel('Collisions per Episode')
    plt.ylabel('Frequency')
    plt.title('Collisions Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Performance Summary
    plt.subplot(2, 3, 6)
    metrics = ['Success Rate', 'Reward', 'Collisions']
    ppo_values = [ppo_results['mean_success_rate'], 
                  ppo_results['mean_reward'], 
                  ppo_results['mean_collisions']]
    random_values = [random_results['mean_success_rate'], 
                     random_results['mean_reward'], 
                     random_results['mean_collisions']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, ppo_values, width, label='PPO', alpha=0.7)
    plt.bar(x + width/2, random_values, width, label='Random', alpha=0.7)
    
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.title('Performance Summary')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_comparison.png'), dpi=300, bbox_inches='tight')
    logger.info(f"Comparison results plotted and saved to {output_dir}/evaluation_comparison.png")

def save_results(ppo_results, random_results, output_dir):
    """Save evaluation results to file"""
    import json
    
    results = {
        'ppo_agent': {
            'mean_reward': float(ppo_results['mean_reward']),
            'std_reward': float(ppo_results['std_reward']),
            'mean_success_rate': float(ppo_results['mean_success_rate']),
            'std_success_rate': float(ppo_results['std_success_rate']),
            'mean_collisions': float(ppo_results['mean_collisions']),
            'mean_attacks': float(ppo_results['mean_attacks'])
        },
        'random_agent': {
            'mean_reward': float(random_results['mean_reward']),
            'std_reward': float(random_results['std_reward']),
            'mean_success_rate': float(random_results['mean_success_rate']),
            'std_success_rate': float(random_results['std_success_rate']),
            'mean_collisions': float(random_results['mean_collisions']),
            'mean_attacks': float(random_results['mean_attacks'])
        },
        'improvement': {
            'success_rate_improvement': float(ppo_results['mean_success_rate'] - random_results['mean_success_rate']),
            'reward_improvement': float(ppo_results['mean_reward'] - random_results['mean_reward']),
            'relative_success_improvement': float((ppo_results['mean_success_rate'] - random_results['mean_success_rate']) / max(random_results['mean_success_rate'], 1e-8) * 100)
        }
    }
    
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    return results

def main():
    """Main evaluation function"""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize environment
    env = V2XEnvironment(
        num_vehicles=args.num_vehicles,
        communication_range=args.communication_range,
        tx_power=args.tx_power
    )
    env.episode_length = args.max_steps
    
    # Initialize PPO agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    ppo_agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)
    
    # Load trained model
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        return
    
    ppo_agent.load_model(args.model_path)
    logger.info(f"Loaded PPO model from {args.model_path}")
    
    # Evaluate PPO agent
    ppo_results = evaluate_agent(env, ppo_agent, args.episodes, "PPO Agent")
    
    # Compare with random baseline if requested
    if args.compare_random:
        random_agent = RandomAttacker(env.action_space)
        random_results = evaluate_agent(env, random_agent, args.episodes, "Random Agent")
        
        # Plot comparison
        plot_comparison_results(ppo_results, random_results, args.output_dir)
        
        # Save results
        results = save_results(ppo_results, random_results, args.output_dir)
        
        # Print improvement summary
        logger.info("\n=========== PERFORMANCE COMPARISON ===========")
        logger.info(f"PPO Success Rate: {ppo_results['mean_success_rate']:.4f}")
        logger.info(f"Random Success Rate: {random_results['mean_success_rate']:.4f}")
        logger.info(f"Improvement: {results['improvement']['relative_success_improvement']:.2f}%")
        
        logger.info(f"PPO Reward: {ppo_results['mean_reward']:.4f}")
        logger.info(f"Random Reward: {random_results['mean_reward']:.4f}")
        logger.info(f"Reward Improvement: {results['improvement']['reward_improvement']:.4f}")
    
    logger.info("Evaluation completed successfully!")

if __name__ == "__main__":
    main()