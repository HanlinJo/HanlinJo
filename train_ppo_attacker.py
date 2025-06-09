import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse
from datetime import datetime
import os
import json

from v2x_environment import V2XEnvironment
from ppo_agent import PPOAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'ppo_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('PPO-Training')

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train PPO Agent for V2X Attacker')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=1000, 
                       help='Number of training episodes (default: 1000)')
    parser.add_argument('--max_steps', type=int, default=1000, 
                       help='Maximum steps per episode (default: 1000)')
    parser.add_argument('--update_freq', type=int, default=10, 
                       help='Update frequency in episodes (default: 10)')
    
    # Environment parameters
    parser.add_argument('--num_vehicles', type=int, default=20, 
                       help='Number of vehicles (default: 20)')
    parser.add_argument('--communication_range', type=float, default=320.0, 
                       help='Communication range in meters (default: 320.0)')
    parser.add_argument('--tx_power', type=float, default=23.0, 
                       help='Transmission power in dBm (default: 23.0)')
    
    # PPO parameters
    parser.add_argument('--lr', type=float, default=3e-4, 
                       help='Learning rate (default: 3e-4)')
    parser.add_argument('--gamma', type=float, default=0.99, 
                       help='Discount factor (default: 0.99)')
    parser.add_argument('--eps_clip', type=float, default=0.2, 
                       help='PPO clipping parameter (default: 0.2)')
    parser.add_argument('--k_epochs', type=int, default=4, 
                       help='PPO update epochs (default: 4)')
    parser.add_argument('--entropy_coef', type=float, default=0.01, 
                       help='Entropy coefficient (default: 0.01)')
    parser.add_argument('--value_coef', type=float, default=0.5, 
                       help='Value loss coefficient (default: 0.5)')
    
    # Output parameters
    parser.add_argument('--save_dir', type=str, default='models', 
                       help='Directory to save models (default: models)')
    parser.add_argument('--save_freq', type=int, default=100, 
                       help='Model save frequency in episodes (default: 100)')
    parser.add_argument('--eval_freq', type=int, default=50, 
                       help='Evaluation frequency in episodes (default: 50)')
    
    return parser.parse_args()

def evaluate_agent(env, agent, num_episodes=10):
    """Evaluate the trained agent"""
    total_rewards = []
    total_success_rates = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _, _ = agent.select_action(state, deterministic=True)
            state, reward, done, info = env.step(action)
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        total_success_rates.append(info['success_rate'])
    
    return {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_success_rate': np.mean(total_success_rates),
        'std_success_rate': np.std(total_success_rates)
    }

def train_ppo_agent(args):
    """Train PPO agent for V2X attacker"""
    logger.info("Starting PPO training for V2X attacker")
    logger.info(f"Training configuration: {vars(args)}")
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(args.save_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
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
    
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=args.lr,
        gamma=args.gamma,
        eps_clip=args.eps_clip,
        k_epochs=args.k_epochs,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef
    )
    
    # Training statistics
    episode_rewards = []
    episode_success_rates = []
    episode_collisions = []
    training_losses = []
    
    # Training loop
    for episode in range(args.episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < args.max_steps:
            # Select action
            action, log_prob, value = agent.select_action(state)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, log_prob, value, done)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            step += 1
        
        # Store episode statistics
        episode_rewards.append(episode_reward)
        episode_success_rates.append(info['success_rate'])
        episode_collisions.append(info['episode_collisions'])
        
        # Update agent
        if (episode + 1) % args.update_freq == 0:
            training_stats = agent.update()
            if training_stats:
                training_losses.append(training_stats)
                logger.info(f"Episode {episode + 1}: Policy Loss={training_stats['policy_loss']:.4f}, "
                          f"Value Loss={training_stats['value_loss']:.4f}, "
                          f"Entropy Loss={training_stats['entropy_loss']:.4f}")
        
        # Log progress
        if (episode + 1) % 10 == 0:
            recent_rewards = episode_rewards[-10:]
            recent_success_rates = episode_success_rates[-10:]
            logger.info(f"Episode {episode + 1}/{args.episodes}: "
                      f"Avg Reward={np.mean(recent_rewards):.3f}, "
                      f"Avg Success Rate={np.mean(recent_success_rates):.3f}")
        
        # Evaluate agent
        if (episode + 1) % args.eval_freq == 0:
            eval_stats = evaluate_agent(env, agent, num_episodes=5)
            logger.info(f"Evaluation at episode {episode + 1}: "
                      f"Mean Reward={eval_stats['mean_reward']:.3f}, "
                      f"Mean Success Rate={eval_stats['mean_success_rate']:.3f}")
        
        # Save model
        if (episode + 1) % args.save_freq == 0:
            model_path = os.path.join(args.save_dir, f'ppo_model_episode_{episode + 1}.pth')
            agent.save_model(model_path)
    
    # Save final model
    final_model_path = os.path.join(args.save_dir, 'ppo_model_final.pth')
    agent.save_model(final_model_path)
    
    # Final evaluation
    final_eval_stats = evaluate_agent(env, agent, num_episodes=20)
    logger.info(f"Final evaluation: "
              f"Mean Reward={final_eval_stats['mean_reward']:.3f}, "
              f"Mean Success Rate={final_eval_stats['mean_success_rate']:.3f}")
    
    # Plot training results
    plot_training_results(episode_rewards, episode_success_rates, episode_collisions, 
                         training_losses, args.save_dir)
    
    return agent, env

def plot_training_results(episode_rewards, episode_success_rates, episode_collisions, 
                         training_losses, save_dir):
    """Plot training results"""
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Episode Rewards
    plt.subplot(2, 3, 1)
    plt.plot(episode_rewards, alpha=0.7)
    plt.plot(np.convolve(episode_rewards, np.ones(50)/50, mode='valid'), 'r-', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('Episode Rewards over Training')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Success Rates
    plt.subplot(2, 3, 2)
    plt.plot(episode_success_rates, alpha=0.7)
    plt.plot(np.convolve(episode_success_rates, np.ones(50)/50, mode='valid'), 'r-', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.title('Attack Success Rate over Training')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Episode Collisions
    plt.subplot(2, 3, 3)
    plt.plot(episode_collisions, alpha=0.7)
    plt.plot(np.convolve(episode_collisions, np.ones(50)/50, mode='valid'), 'r-', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Collisions')
    plt.title('Collisions per Episode')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Policy Loss
    if training_losses:
        plt.subplot(2, 3, 4)
        policy_losses = [loss['policy_loss'] for loss in training_losses]
        plt.plot(policy_losses)
        plt.xlabel('Update Step')
        plt.ylabel('Policy Loss')
        plt.title('Policy Loss over Training')
        plt.grid(True, alpha=0.3)
    
    # Plot 5: Value Loss
    if training_losses:
        plt.subplot(2, 3, 5)
        value_losses = [loss['value_loss'] for loss in training_losses]
        plt.plot(value_losses)
        plt.xlabel('Update Step')
        plt.ylabel('Value Loss')
        plt.title('Value Loss over Training')
        plt.grid(True, alpha=0.3)
    
    # Plot 6: Entropy Loss
    if training_losses:
        plt.subplot(2, 3, 6)
        entropy_losses = [loss['entropy_loss'] for loss in training_losses]
        plt.plot(entropy_losses)
        plt.xlabel('Update Step')
        plt.ylabel('Entropy Loss')
        plt.title('Entropy Loss over Training')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_results.png'), dpi=300, bbox_inches='tight')
    logger.info(f"Training results plotted and saved to {save_dir}/training_results.png")

def main():
    """Main training function"""
    args = parse_arguments()
    
    try:
        agent, env = train_ppo_agent(args)
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()