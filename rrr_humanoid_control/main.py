import os
import torch
import numpy as np
import wandb
import gymnasium as gym
import humanoid_bench

from configs.config import Config, parse_args
from models.rrr_diffusion import RRRDiffusionPolicy
from envs.humanoid import HumanoidVecEnv
from utils.trainer import RRRTrainer

def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train(config):
    """Train the RRR diffusion policy on HumanoidBench."""
    # Initialize environment
    env = HumanoidVecEnv(config)
    
    # Initialize model
    model = RRRDiffusionPolicy(config).to(config.device)
    
    # Initialize wandb
    run = wandb.init(
        entity=config.wandb_entity,
        project=config.wandb_project,
        name=config.wandb_name,
        config=vars(config),
        monitor_gym=True,
        save_code=True,
    )
    
    # Initialize trainer
    trainer = RRRTrainer(model, env, config)
    
    # Train model
    trainer.train(total_steps=config.max_steps)
    
    # Close environment
    env.close()
    
    # Close wandb
    wandb.finish()

def evaluate(config):
    """Evaluate a trained model."""
    # Load model config from checkpoint
    checkpoint_path = os.path.join(config.output_dir, "latest", "checkpoint_final.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    loaded_config = checkpoint["config"]
    
    # Update current config with loaded config
    for key, value in vars(loaded_config).items():
        if key not in ["device"]:  # Skip device to use current device
            setattr(config, key, value)
    
    # Initialize environment
    env = HumanoidVecEnv(config)
    
    # Initialize model
    model = RRRDiffusionPolicy(config).to(config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Initialize wandb
    run = wandb.init(
        entity=config.wandb_entity,
        project=config.wandb_project,
        name=f"eval_{config.wandb_name}",
        config=vars(config),
        monitor_gym=True,
    )
    
    # Initialize trainer and evaluate
    trainer = RRRTrainer(model, env, config)
    avg_return, avg_success = trainer.evaluate(num_episodes=50)
    
    # Close environment
    env.close()
    
    # Close wandb
    wandb.finish()
    
    return avg_return, avg_success

def collect_data(config):
    """Collect data for fine-tuning or analysis."""
    # Initialize environment
    env = HumanoidVecEnv(config)
    
    # Create data directory
    os.makedirs(config.data_path, exist_ok=True)
    
    # Initialize random policy
    def random_policy(state):
        return env.env.action_space.sample()
    
    # Collect data
    print("Collecting random data...")
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    
    for _ in range(100):  # Collect 100 episodes
        state, _ = env.reset()
        done = [False]
        episode_steps = 0
        
        while not done[0] and episode_steps < config.max_episode_steps:
            action = random_policy(state)
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store data
            states.append(state[0])
            actions.append(action[0])
            rewards.append(reward[0])
            next_states.append(next_state[0])
            dones.append(done[0])
            
            state = next_state
            episode_steps += 1
    
    # Save collected data
    data_path = os.path.join(config.data_path, f"{config.env_name}_data.npz")
    np.savez(
        data_path,
        states=np.array(states),
        actions=np.array(actions),
        rewards=np.array(rewards),
        next_states=np.array(next_states),
        dones=np.array(dones)
    )
    
    print(f"Data saved to {data_path}")
    print(f"Collected {len(states)} transitions")
    
    # Close environment
    env.close()

def main():
    # Parse arguments
    config = parse_args()
    
    # Set random seed
    set_seed(config.seed)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Run requested mode
    if config.mode == "train":
        train(config)
    elif config.mode == "eval":
        evaluate(config)
    elif config.mode == "collect":
        collect_data(config)
    else:
        raise ValueError(f"Unknown mode: {config.mode}")

if __name__ == "__main__":
    main()