import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import logging

logger = logging.getLogger('PPO-Agent')

class PPONetwork(nn.Module):
    """PPO Actor-Critic Network"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PPONetwork, self).__init__()
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy network)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_std = nn.Parameter(torch.ones(action_dim) * 0.5)
        
        # Critic head (value network)
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        """Forward pass through the network"""
        shared_features = self.shared_layers(state)
        
        # Actor output
        action_mean = self.actor_mean(shared_features)
        action_std = F.softplus(self.actor_std)
        
        # Critic output
        value = self.critic(shared_features)
        
        return action_mean, action_std, value
    
    def get_action(self, state, deterministic=False):
        """Get action from the policy"""
        action_mean, action_std, value = self.forward(state)
        
        if deterministic:
            action = action_mean
        else:
            dist = Normal(action_mean, action_std)
            action = dist.sample()
        
        # Convert continuous actions to discrete resource parameters
        # frame_no: 0-1023, subframe_no: 1-10, subchannel: 0-4
        action_discrete = torch.zeros_like(action)
        action_discrete[0] = torch.clamp(action[0] * 512 + 512, 0, 1023)  # frame_no
        action_discrete[1] = torch.clamp(action[1] * 4.5 + 5.5, 1, 10)    # subframe_no
        action_discrete[2] = torch.clamp(action[2] * 2 + 2, 0, 4)         # subchannel
        
        return action_discrete, value
    
    def evaluate_action(self, state, action):
        """Evaluate action for training"""
        action_mean, action_std, value = self.forward(state)
        
        # Convert discrete actions back to continuous for evaluation
        action_continuous = torch.zeros_like(action, dtype=torch.float32)
        action_continuous[:, 0] = (action[:, 0] - 512) / 512  # frame_no
        action_continuous[:, 1] = (action[:, 1] - 5.5) / 4.5  # subframe_no
        action_continuous[:, 2] = (action[:, 2] - 2) / 2      # subchannel
        
        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(action_continuous).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, value.squeeze(), entropy

class PPOAgent:
    """PPO Agent for V2X Resource Selection"""
    
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, 
                 k_epochs=4, entropy_coef=0.01, value_coef=0.5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        # Initialize network
        self.policy = PPONetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
        # Training statistics
        self.training_step = 0
        
        logger.info(f"PPO Agent initialized with state_dim={state_dim}, action_dim={action_dim}")
    
    def select_action(self, state, deterministic=False):
        """Select action using the current policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, value = self.policy.get_action(state_tensor, deterministic)
            
            # Get log probability for training
            action_mean, action_std, _ = self.policy.forward(state_tensor)
            
            # Convert discrete action back to continuous for log_prob calculation
            action_continuous = torch.zeros_like(action, dtype=torch.float32)
            action_continuous[0] = (action[0] - 512) / 512  # frame_no
            action_continuous[1] = (action[1] - 5.5) / 4.5  # subframe_no
            action_continuous[2] = (action[2] - 2) / 2      # subchannel
            
            dist = Normal(action_mean, action_std)
            log_prob = dist.log_prob(action_continuous).sum()
            
            return action.cpu().numpy().astype(int), log_prob.item(), value.item()
    
    def store_transition(self, state, action, reward, log_prob, value, done):
        """Store transition in experience buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def update(self):
        """Update the policy using PPO algorithm"""
        if len(self.states) == 0:
            return {}
        
        # Convert lists to tensors
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(np.array(self.actions))
        old_log_probs = torch.FloatTensor(self.log_probs)
        old_values = torch.FloatTensor(self.values)
        rewards = torch.FloatTensor(self.rewards)
        dones = torch.BoolTensor(self.dones)
        
        # Calculate returns and advantages
        returns = self._calculate_returns(rewards, old_values, dones)
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        for _ in range(self.k_epochs):
            # Evaluate current policy
            log_probs, values, entropy = self.policy.evaluate_action(states, actions)
            
            # Calculate ratios
            ratios = torch.exp(log_probs - old_log_probs)
            
            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Policy loss
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, returns)
            
            # Entropy loss
            entropy_loss = -entropy.mean()
            
            # Total loss
            total_loss = (policy_loss + 
                         self.value_coef * value_loss + 
                         self.entropy_coef * entropy_loss)
            
            # Update network
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
        
        # Clear experience buffer
        self.clear_buffer()
        
        # Update training step
        self.training_step += 1
        
        # Return training statistics
        return {
            'policy_loss': total_policy_loss / self.k_epochs,
            'value_loss': total_value_loss / self.k_epochs,
            'entropy_loss': total_entropy_loss / self.k_epochs,
            'training_step': self.training_step
        }
    
    def _calculate_returns(self, rewards, values, dones):
        """Calculate discounted returns using GAE"""
        returns = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * 0.95 * (1 - dones[t]) * gae  # GAE with lambda=0.95
            returns[t] = gae + values[t]
        
        return returns
    
    def clear_buffer(self):
        """Clear the experience buffer"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def save_model(self, filepath):
        """Save the trained model"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
        logger.info(f"Model loaded from {filepath}")
    
    def get_stats(self):
        """Get training statistics"""
        return {
            'training_step': self.training_step,
            'buffer_size': len(self.states)
        }