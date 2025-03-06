import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import wandb
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm

class RRRTrainer:
    def __init__(self, model, env, config):
        self.model = model
        self.env = env
        self.config = config
        
        # Create optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
        # Create output directory
        self.run_id = wandb.run.id if wandb.run else "debug"
        self.output_dir = os.path.join(config.output_dir, self.run_id)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Training stats
        self.episode_returns = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.success_rate = deque(maxlen=100)
        self.losses = deque(maxlen=100)
        
        # Buffer for collecting experiences
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.dones = []
    
    def collect_rollouts(self, num_steps=1000):
        """
        Collect rollouts using the current policy.
        """
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.dones = []
        
        # Prepare for rollout
        states, _ = self.env.reset()
        episode_rewards = [0 for _ in range(self.config.num_envs)]
        episode_lengths = [0 for _ in range(self.config.num_envs)]
        success = [0 for _ in range(self.config.num_envs)]
        
        for _ in range(num_steps):
            # Sample actions from the current policy
            with torch.no_grad():
                states_tensor = torch.FloatTensor(states).to(self.config.device)
                actions = self.model.predict_action(states_tensor).cpu().numpy()
            
            # Execute actions in the environment
            next_states, rewards, dones, truncated, infos = self.env.step(actions)
            
            # Update episode stats
            for i in range(self.config.num_envs):
                episode_rewards[i] += rewards[i]
                episode_lengths[i] += 1
                
                # Store experience
                self.states.append(states[i])
                self.actions.append(actions[i])
                self.next_states.append(next_states[i])
                self.rewards.append(rewards[i])
                self.dones.append(dones[i])
                
                # Check for episode termination
                if dones[i] or truncated[i]:
                    self.episode_returns.append(episode_rewards[i])
                    self.episode_lengths.append(episode_lengths[i])
                    
                    # Check for success
                    if "success" in infos[i]:
                        success[i] = infos[i]["success"]
                        self.success_rate.append(success[i])
                    
                    # Reset episode stats
                    episode_rewards[i] = 0
                    episode_lengths[i] = 0
                    success[i] = 0
            
            # Update current states
            states = next_states
        
        # Convert collected data to tensors
        states_tensor = torch.FloatTensor(self.states).to(self.config.device)
        actions_tensor = torch.FloatTensor(self.actions).to(self.config.device)
        
        return states_tensor, actions_tensor
    
    def train_epoch(self, states, actions, batch_size=256):
        """
        Train the model on collected data for one epoch.
        """
        # Create dataset indices
        indices = np.arange(states.shape[0])
        np.random.shuffle(indices)
        
        # Train in batches
        epoch_loss = 0
        num_batches = (states.shape[0] + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            # Get batch indices
            batch_indices = indices[i * batch_size:(i + 1) * batch_size]
            
            # Get batch data
            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Compute loss
            loss = self.model.compute_loss(batch_states, batch_actions)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Update stats
            epoch_loss += loss.item()
        
        # Average loss
        avg_loss = epoch_loss / num_batches
        self.losses.append(avg_loss)
        
        return avg_loss
    
    def evaluate(self, num_episodes=10):
        """
        Evaluate the current policy.
        """
        # Create a separate evaluation environment
        eval_env = self.env.__class__(self.config)
        
        # Stats
        episode_returns = []
        episode_lengths = []
        success_rate = []
        
        for _ in range(num_episodes):
            states, _ = eval_env.reset()
            done = False
            truncated = False
            episode_reward = 0
            episode_length = 0
            
            while not (done or truncated):
                # Sample actions deterministically
                with torch.no_grad():
                    states_tensor = torch.FloatTensor(states).to(self.config.device)
                    actions = self.model.predict_action(states_tensor, deterministic=True).cpu().numpy()
                
                # Execute actions
                states, rewards, done, truncated, info = eval_env.step(actions)
                
                # Update stats
                episode_reward += rewards[0]  # Take first env
                episode_length += 1
            
            # Store episode stats
            episode_returns.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Check for success
            if "success" in info[0]:
                success_rate.append(info[0]["success"])
        
        # Close evaluation environment
        eval_env.close()
        
        # Compute average stats
        avg_return = np.mean(episode_returns)
        avg_length = np.mean(episode_lengths)
        avg_success = np.mean(success_rate) if success_rate else 0
        
        # Log stats
        print(f"Evaluation: Return={avg_return:.2f}, Length={avg_length:.2f}, Success={avg_success:.2f}")
        if wandb.run:
            wandb.log({
                "eval/return": avg_return,
                "eval/length": avg_length,
                "eval/success": avg_success
            })
        
        return avg_return, avg_success
    
    def save_checkpoint(self, step):
        """
        Save a model checkpoint.
        """
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_{step}.pt")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": step,
            "config": self.config,
        }, checkpoint_path)
        
        print(f"Checkpoint saved at {checkpoint_path}")
    
    def train(self, total_steps=10000000):
        """
        Train the model for the specified number of steps.
        """
        start_time = time.time()
        step = 0
        
        # Main training loop
        pbar = tqdm(total=total_steps)
        while step < total_steps:
            # Collect rollouts
            print(f"Collecting rollouts at step {step}...")
            states, actions = self.collect_rollouts(num_steps=1000)
            
            # Train on collected data
            print(f"Training on collected data...")
            loss = self.train_epoch(states, actions, batch_size=self.config.batch_size)
            
            # Update step count
            step += states.shape[0]
            pbar.update(states.shape[0])
            
            # Log stats
            if self.episode_returns:
                print(f"Step {step}: Loss={loss:.4f}, Return={np.mean(self.episode_returns):.2f}, Success={np.mean(self.success_rate) if self.success_rate else 0:.2f}")
                
                # Log to wandb
                if wandb.run:
                    wandb.log({
                        "train/step": step,
                        "train/loss": loss,
                        "train/return": np.mean(self.episode_returns),
                        "train/length": np.mean(self.episode_lengths),
                        "train/success": np.mean(self.success_rate) if self.success_rate else 0,
                    })
            
            # Evaluate
            if step % self.config.eval_every == 0:
                self.evaluate()
            
            # Save checkpoint
            if step % self.config.save_every == 0:
                self.save_checkpoint(step)
        
        # Final evaluation and checkpoint
        self.evaluate()
        self.save_checkpoint(total_steps)
        
        # Log total training time
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")