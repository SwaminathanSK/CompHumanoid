import os
import argparse
import torch
from typing import List, Tuple, Optional, Dict

class Config:
    # Environment parameters
    env_name: str = "h1hand-walk-v0"
    max_episode_steps: int = 1000
    seed: int = 0
    
    # RRR diffusion model parameters
    diffusion_steps: int = 1000
    noise_schedule: str = "linear"
    action_dim: int = None  # Will be set after environment creation
    state_dim: int = None   # Will be set after environment creation
    model_dim: int = 192
    dim_mults: Tuple[int, ...] = (1, 2, 4, 8)
    learn_sigma: bool = True
    
    # Training parameters
    batch_size: int = 256
    lr: float = 3e-5
    weight_decay: float = 0.0
    num_envs: int = 4
    max_steps: int = 10000000
    eval_every: int = 100000
    save_every: int = 100000
    
    # Hierarchical policy parameters
    use_hierarchy: bool = False
    policy_path: Optional[str] = None
    mean_path: Optional[str] = None
    var_path: Optional[str] = None
    policy_type: str = None
    
    # Wandb parameters
    wandb_entity: str = "robot-learning"
    wandb_project: str = "humanoid-bench"
    wandb_name: str = None
    
    # Paths
    data_path: str = "./data"
    output_dir: str = "./runs"
    
    # Device
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        if self.wandb_name is None:
            self.wandb_name = f"rrr_{self.env_name}"

def parse_args():
    parser = argparse.ArgumentParser(description="RRR for HumanoidBench")
    
    # Environment arguments
    parser.add_argument("--env_name", type=str, default="h1hand-walk-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=4)
    
    # Model arguments
    parser.add_argument("--model_dim", type=int, default=192)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-5)
    
    # Hierarchical policy arguments
    parser.add_argument("--use_hierarchy", action="store_true")
    parser.add_argument("--policy_path", type=str, default=None)
    parser.add_argument("--mean_path", type=str, default=None)
    parser.add_argument("--var_path", type=str, default=None)
    parser.add_argument("--policy_type", type=str, default=None)
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=10000000)
    
    # Wandb arguments
    parser.add_argument("--wandb_entity", type=str, default="robot-learning")
    
    # Mode arguments
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "collect"])
    
    args = parser.parse_args()
    return Config(**vars(args))