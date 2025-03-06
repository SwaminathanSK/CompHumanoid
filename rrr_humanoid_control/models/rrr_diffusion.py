import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional

from reduce_reuse_recycle.composable_diffusion.unet import UNetModel_full
from reduce_reuse_recycle.composable_diffusion.gaussian_diffusion import GaussianDiffusion, get_named_beta_schedule, ModelVarType

class RRRDiffusionPolicy(nn.Module):
    """
    A diffusion model policy that follows the Reduce-Reuse-Recycle approach
    for compositional generation of actions.
    """
    def __init__(self, config):
        super().__init__()
        
        # Configuration
        self.config = config
        self.device = config.device
        self.action_dim = config.action_dim
        self.state_dim = config.state_dim
        
        # Create UNet model - action conditioned on state
        self.model = UNetModel_full(
            in_channels=self.action_dim,
            model_channels=config.model_dim,
            out_channels=self.action_dim * (2 if config.learn_sigma else 1),
            num_res_blocks=2,
            attention_resolutions=(16, 8, 4),
            dropout=0.1,
            channel_mult=config.dim_mults,
            num_heads=8,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_fp16=False,
            dataset="humanoid"
        )
        
        # Create diffusion process
        betas = get_named_beta_schedule(
            config.noise_schedule, 
            config.diffusion_steps
        )
        
        self.diffusion = GaussianDiffusion(
            betas=betas,
            model_var_type=ModelVarType.LEARNED_RANGE if config.learn_sigma else ModelVarType.FIXED_SMALL
        )
        
        # State encoder (to condition the diffusion model)
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, config.model_dim * 4)  # Match time embedding dim
        )
    
    def predict_action(self, state, deterministic=False):
        """
        Sample an action conditioned on the current state.
        """
        # Convert state to tensor if needed
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        
        # Ensure proper shape
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            # Encode state
            state_emb = self.state_encoder(state)
            
            # Start with noise
            action_shape = (state.shape[0], self.action_dim)
            action = torch.randn(action_shape, device=self.device)
            
            # Iterative denoising
            if deterministic:
                # Fast deterministic sampling for evaluation
                timesteps = [self.config.diffusion_steps // 10]  # Use fewer steps for speed
                for t in reversed(timesteps):
                    t_tensor = torch.tensor([t] * state.shape[0], device=self.device)
                    model_out = self.model(action, t_tensor, state_emb)
                    
                    # Extract predicted noise
                    if self.config.learn_sigma:
                        model_out, _ = torch.split(model_out, self.action_dim, dim=1)
                    
                    # One step denoising
                    action = self.diffusion._predict_xstart_from_eps(action, t_tensor, model_out)
            else:
                # Gradual sampling using the full diffusion process
                for t in reversed(range(self.config.diffusion_steps)):
                    t_tensor = torch.tensor([t] * state.shape[0], device=self.device)
                    action_noise = torch.randn_like(action) if t > 0 else 0
                    model_out = self.model(action, t_tensor, state_emb)
                    
                    # Extract predicted noise
                    if self.config.learn_sigma:
                        model_out, _ = torch.split(model_out, self.action_dim, dim=1)
                    
                    # One step denoising
                    action = self.diffusion._predict_xstart_from_eps(action, t_tensor, model_out)
                    
                    # Add noise for stochastic sampling
                    if t > 0:
                        noise_strength = (self.config.diffusion_steps - t) / self.config.diffusion_steps
                        action = action + noise_strength * 0.1 * action_noise
        
        return action
    
    def compute_loss(self, states, actions):
        """
        Compute the diffusion model loss for a batch of state-action pairs.
        """
        # Encode states
        state_emb = self.state_encoder(states)
        
        # Sample random timesteps
        t = torch.randint(0, self.config.diffusion_steps, (states.shape[0],), device=self.device)
        
        # Add noise to actions
        noise = torch.randn_like(actions)
        noisy_actions = self.diffusion.q_sample(actions, t, noise=noise)
        
        # Predict noise
        predicted = self.model(noisy_actions, t, state_emb)
        
        if self.config.learn_sigma:
            predicted, _ = torch.split(predicted, self.action_dim, dim=1)
        
        # Compute loss (mean squared error between predicted and original noise)
        loss = F.mse_loss(predicted, noise)
        
        return loss