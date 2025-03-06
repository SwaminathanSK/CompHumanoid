import gymnasium as gym
import numpy as np
import torch
from typing import Dict, Tuple, Any, Optional
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

class HumanoidVecEnv:
    """
    Vectorized environment wrapper for HumanoidBench environments with
    support for hierarchical policies.
    """
    def __init__(self, config):
        self.config = config
        self.num_envs = config.num_envs
        
        # Create vectorized environment
        self.env = SubprocVecEnv([self._make_env(i) for i in range(self.num_envs)])
        
        # Get action and observation dimensions
        test_env = self._make_env(0)()
        self.config.action_dim = test_env.action_space.shape[0]
        self.config.state_dim = test_env.observation_space.shape[0]
        test_env.close()
        
        print(f"Initialized environment: {config.env_name}")
        print(f"Action space: {self.config.action_dim}, State space: {self.config.state_dim}")
        
        # Initialize hierarchical policy if needed
        self.hierarchical = config.use_hierarchy
        self.low_level_policy = None
        self.mean = None
        self.var = None
        
        if self.hierarchical:
            self._init_hierarchical_policy()
    
    def _make_env(self, rank):
        """
        Create a single environment instance.
        """
        def _init():
            env = gym.make(self.config.env_name)
            env = TimeLimit(env, max_episode_steps=self.config.max_episode_steps)
            env = Monitor(env)
            
            # Set seed for reproducibility
            env.action_space.seed(self.config.seed + rank)
            
            return env
        
        return _init
    
    def _init_hierarchical_policy(self):
        """
        Initialize the low-level policy for hierarchical control.
        """
        assert self.config.policy_path is not None, "Policy path must be provided for hierarchical control"
        assert self.config.mean_path is not None, "Mean path must be provided for hierarchical control"
        assert self.config.var_path is not None, "Var path must be provided for hierarchical control"
        
        # Load policy
        self.low_level_policy = torch.load(self.config.policy_path, map_location=self.config.device)
        self.low_level_policy.eval()
        
        # Load normalization statistics
        self.mean = np.load(self.config.mean_path)
        self.var = np.load(self.config.var_path)
        
        print(f"Loaded hierarchical policy: {self.config.policy_type}")
    
    def reset(self):
        """
        Reset all environments.
        """
        return self.env.reset()
    
    def step(self, actions):
        """
        Step all environments with given actions.
        """
        if self.hierarchical:
            # Convert high-level actions to low-level actions through the policy
            low_level_actions = self._apply_low_level_policy(actions)
            return self.env.step(low_level_actions)
        else:
            # Directly apply actions
            return self.env.step(actions)
    
    def _apply_low_level_policy(self, high_level_actions):
        """
        Apply the low-level policy to convert high-level actions to low-level actions.
        """
        # This implementation will depend on the specific low-level policy interface
        # For now, we'll use a placeholder implementation
        with torch.no_grad():
            # Normalize states
            states = self.env.get_attr("_state")  # This might need adjustment based on actual env implementation
            norm_states = (states - self.mean) / np.sqrt(self.var + 1e-8)
            norm_states = torch.FloatTensor(norm_states).to(self.config.device)
            
            # Combine with high-level actions
            high_level_actions = torch.FloatTensor(high_level_actions).to(self.config.device)
            inputs = torch.cat([norm_states, high_level_actions], dim=1)
            
            # Get low-level actions
            low_level_actions = self.low_level_policy(inputs)
            
            return low_level_actions.cpu().numpy()
    
    def render(self):
        """
        Render the first environment.
        """
        return self.env.render()
    
    def close(self):
        """
        Close all environments.
        """
        self.env.close()