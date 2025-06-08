import numpy as np
import random
import math
from collections import deque
import logging

logger = logging.getLogger('HAPPO-Agent')

class HAPPOAgent:
    """
    HAPPO (Heterogeneous-Agent Proximal Policy Optimization) Agent for V2X Resource Selection
    
    This agent learns to select optimal resources (frame_no, subframe_no, subchannel) 
    for attack transmissions based on observed resource pool information and sensing data.
    """
    
    def __init__(self, state_dim, action_dim, resource_pool, learning_rate=0.001, 
                 epsilon=0.1, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.resource_pool = resource_pool
        
        # Learning parameters
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # Q-table for simplified Q-learning (can be replaced with neural networks)
        self.q_table = {}
        
        # Action space bounds
        self.max_frame_no = 1023
        self.max_subframe_no = 10
        self.max_subchannel = resource_pool.num_subchannels - 1
        
        # Reward tracking
        self.episode_rewards = []
        self.current_episode_reward = 0
        
        # State-action history for learning
        self.last_state = None
        self.last_action = None
        
        logger.info(f"HAPPO Agent initialized with state_dim={state_dim}, action_dim={action_dim}")
    
    def _state_to_key(self, state):
        """Convert continuous state to discrete key for Q-table"""
        # Discretize the state for Q-table lookup
        discretized = []
        
        # Resource pool info (already discrete)
        discretized.extend([int(state[i]) for i in range(4)])
        
        # Sensing data count (discrete)
        discretized.append(int(state[4]))
        
        # Subchannel usage (discretize to bins)
        for i in range(5, 5 + self.resource_pool.num_subchannels):
            discretized.append(min(int(state[i] / 5), 10))  # Bin into groups of 5
        
        # Time (discretize to 100ms bins)
        time_bin = int(state[-1] / 100)
        discretized.append(time_bin)
        
        return tuple(discretized)
    
    def select_resource(self, state, current_time):
        """Select resource using epsilon-greedy policy with Q-learning"""
        state_key = self._state_to_key(state)
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Explore: random action
            action = self._get_random_action()
        else:
            # Exploit: best known action
            action = self._get_best_action(state_key)
        
        # Store state-action for learning
        self.last_state = state_key
        self.last_action = action
        
        # Convert action to resource
        resource = self._action_to_resource(action, current_time)
        
        return resource
    
    def _get_random_action(self):
        """Generate random action (frame_no, subframe_no, subchannel)"""
        frame_no = random.randint(0, self.max_frame_no)
        subframe_no = random.randint(1, self.max_subframe_no)
        subchannel = random.randint(0, self.max_subchannel)
        
        return (frame_no, subframe_no, subchannel)
    
    def _get_best_action(self, state_key):
        """Get best action for given state using Q-table"""
        if state_key not in self.q_table:
            # Initialize Q-values for new state
            self.q_table[state_key] = {}
        
        # Find action with highest Q-value
        best_action = None
        best_q_value = float('-inf')
        
        # Check existing actions
        for action, q_value in self.q_table[state_key].items():
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action
        
        # If no actions exist, return random action
        if best_action is None:
            best_action = self._get_random_action()
        
        return best_action
    
    def _action_to_resource(self, action, current_time):
        """Convert action tuple to ResourceInfo object"""
        from v2x_enhanced_with_happo import SubframeInfo, ResourceInfo
        
        frame_no, subframe_no, subchannel = action
        
        # Ensure action is within bounds
        frame_no = max(0, min(frame_no, self.max_frame_no))
        subframe_no = max(1, min(subframe_no, self.max_subframe_no))
        subchannel = max(0, min(subchannel, self.max_subchannel))
        
        # Create resource
        subframe = SubframeInfo(frame_no, subframe_no)
        resource = ResourceInfo(subframe, subchannel)
        resource.rb_start = subchannel * self.resource_pool.subchannel_size
        resource.rb_len = self.resource_pool.subchannel_size
        
        return resource
    
    def record_reward(self, reward):
        """Record reward and update Q-table"""
        self.current_episode_reward += reward
        
        if self.last_state is not None and self.last_action is not None:
            # Update Q-table using Q-learning update rule
            self._update_q_table(self.last_state, self.last_action, reward)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _update_q_table(self, state_key, action, reward):
        """Update Q-table using Q-learning"""
        # Initialize state in Q-table if not exists
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        # Initialize action in Q-table if not exists
        if action not in self.q_table[state_key]:
            self.q_table[state_key][action] = 0.0
        
        # Q-learning update (simplified without next state)
        # Q(s,a) = Q(s,a) + α * (r - Q(s,a))
        current_q = self.q_table[state_key][action]
        self.q_table[state_key][action] = current_q + self.learning_rate * (reward - current_q)
        
        logger.debug(f"Updated Q-value for state {state_key}, action {action}: "
                    f"{current_q:.3f} -> {self.q_table[state_key][action]:.3f}")
    
    def get_learning_stats(self):
        """Get learning statistics"""
        stats = {
            'epsilon': self.epsilon,
            'q_table_size': len(self.q_table),
            'total_state_actions': sum(len(actions) for actions in self.q_table.values()),
            'average_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'current_episode_reward': self.current_episode_reward
        }
        return stats
    
    def end_episode(self):
        """End current episode and reset"""
        self.episode_rewards.append(self.current_episode_reward)
        self.current_episode_reward = 0
        self.last_state = None
        self.last_action = None
        
        # Keep only recent episodes
        if len(self.episode_rewards) > 1000:
            self.episode_rewards = self.episode_rewards[-1000:]

class AdvancedHAPPOAgent(HAPPOAgent):
    """
    Advanced HAPPO Agent with more sophisticated learning mechanisms
    """
    
    def __init__(self, state_dim, action_dim, resource_pool, **kwargs):
        super().__init__(state_dim, action_dim, resource_pool, **kwargs)
        
        # Advanced features
        self.action_history = deque(maxlen=1000)
        self.success_patterns = {}
        self.resource_usage_tracker = {}
        
        # Multi-objective optimization weights
        self.collision_weight = 1.0
        self.efficiency_weight = 0.5
        
    def select_resource(self, state, current_time):
        """Enhanced resource selection with pattern recognition"""
        state_key = self._state_to_key(state)
        
        # Analyze recent patterns
        successful_patterns = self._analyze_success_patterns()
        
        # Epsilon-greedy with pattern bias
        if random.random() < self.epsilon:
            # Explore with pattern bias
            if successful_patterns and random.random() < 0.7:
                action = self._select_from_patterns(successful_patterns)
            else:
                action = self._get_random_action()
        else:
            # Exploit best action
            action = self._get_best_action(state_key)
        
        # Track action
        self.action_history.append((state_key, action, current_time))
        
        # Store for learning
        self.last_state = state_key
        self.last_action = action
        
        return self._action_to_resource(action, current_time)
    
    def _analyze_success_patterns(self):
        """Analyze patterns in successful actions"""
        successful_actions = []
        
        # Look at recent successful actions
        for i, (state, action, timestamp) in enumerate(self.action_history):
            if i < len(self.action_history) - 1:
                # Check if this action led to success (simplified)
                if state in self.q_table and action in self.q_table[state]:
                    if self.q_table[state][action] > 0:
                        successful_actions.append(action)
        
        return successful_actions[-10:]  # Return recent successful actions
    
    def _select_from_patterns(self, patterns):
        """Select action based on successful patterns"""
        if not patterns:
            return self._get_random_action()
        
        # Select from successful patterns with some variation
        base_action = random.choice(patterns)
        frame_no, subframe_no, subchannel = base_action
        
        # Add small random variation
        frame_no = (frame_no + random.randint(-10, 10)) % (self.max_frame_no + 1)
        subframe_no = max(1, min(self.max_subframe_no, subframe_no + random.randint(-1, 1)))
        subchannel = max(0, min(self.max_subchannel, subchannel + random.randint(-1, 1)))
        
        return (frame_no, subframe_no, subchannel)
    
    def record_reward(self, reward):
        """Enhanced reward recording with pattern learning"""
        super().record_reward(reward)
        
        # Update success patterns
        if self.last_action is not None and reward > 0:
            if self.last_action not in self.success_patterns:
                self.success_patterns[self.last_action] = 0
            self.success_patterns[self.last_action] += reward
    
    def get_advanced_stats(self):
        """Get advanced learning statistics"""
        basic_stats = self.get_learning_stats()
        
        advanced_stats = {
            **basic_stats,
            'success_patterns_count': len(self.success_patterns),
            'action_history_length': len(self.action_history),
            'top_successful_actions': sorted(self.success_patterns.items(), 
                                           key=lambda x: x[1], reverse=True)[:5]
        }
        
        return advanced_stats

# Factory function to create appropriate agent
def create_happo_agent(agent_type='basic', **kwargs):
    """Factory function to create HAPPO agents"""
    if agent_type == 'advanced':
        return AdvancedHAPPOAgent(**kwargs)
    else:
        return HAPPOAgent(**kwargs)