'''
Simplified Canonical Flow Field Trainer

Simplified version that removes curriculum learning and uses direct frame-to-frame warping.
'''

import sys
import os
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(WORK_DIR)

import modules
import dataio
import loss_functions


@dataclass
class SimplifiedCanonicalFlowConfig:
    """Configuration for simplified canonical flow field training"""
    # Model parameters
    flow_hidden_features: int = 256
    flow_num_layers: int = 4
    flow_activation: str = "sine"
    
    # Training parameters
    num_epochs: int = 200
    learning_rate: float = 1e-4
    batch_size: int = 1
    
    # Loss parameters - Uniform regularization
    lambda_magnitude: float = 0.01  # Uniform magnitude regularization
    lambda_derivative: float = 0.001  # Derivative regularization
    
    # Logging parameters
    steps_til_summary: int = 20
    log_dir: str = f'{WORK_DIR}/logs/canonical_flow_simplified'
    
    # Visualization parameters
    vis_target_frames: list = None
    
    # Data parameters
    video_data_path: str = f'{WORK_DIR}/data/mock_videos/mockvideo_vid_data.npy'
    feature_model_path: str = f'{WORK_DIR}/logs/mock_video_feature_test/checkpoints/model_final.pth'
    
    # Feature model parameters
    feature_out_features: int = 384
    feature_hidden_features: int = 1024
    feature_num_layers: int = 3
    
    def __post_init__(self):
        if self.vis_target_frames is None:
            self.vis_target_frames = [1, 3, 7, 15]


class SimplifiedFlowDataset(Dataset):
    """
    Simplified dataset that loads all frames without sampling or curriculum
    """
    
    def __init__(self, vid_dataset, pixel_coords, config: SimplifiedCanonicalFlowConfig):
        self.vid_dataset = vid_dataset
        self.pixel_coords = pixel_coords
        self.config = config
        self.num_frames = vid_dataset.shape[0]
        
        # Use all frames except frame 0 as targets
        self.target_frames = list(range(1, self.num_frames))
        
    def __len__(self):
        return len(self.target_frames)
    
    def __getitem__(self, idx):
        target_frame = self.target_frames[idx]
        
        # Get all coordinates for frame 0 (source) and target frame
        source_coords = self.pixel_coords[0]  # [H, W, 3]
        target_coords = self.pixel_coords[target_frame]  # [H, W, 3]
        
        return {
            'source_coords': source_coords,
            'target_coords': target_coords,
            'target_frame': target_frame
        }


class SimplifiedCanonicalFlowTrainer:
    """
    Simplified trainer without curriculum learning
    """
    
    def __init__(self, config: SimplifiedCanonicalFlowConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self._setup_data()
        self._setup_models()
        self._setup_optimizer()
        self._setup_logging()
        
    def _setup_data(self):
        """Setup video dataset and coordinate grids"""
        print("Setting up simplified data pipeline...")
        
        # Load video dataset
        self.vid_dataset = dataio.Video(self.config.video_data_path)
        print(f"Video dataset shape: {self.vid_dataset.shape}")
        
        # Generate coordinate grids
        self.pixel_coords = dataio.get_mgrid(self.vid_dataset.shape, dim=3)
        self.pixel_coords = self.pixel_coords.view(
            self.vid_dataset.shape[0],
            self.vid_dataset.shape[1], 
            self.vid_dataset.shape[2],
            3
        )
        
        # Calculate coordinate deltas
        self.dt = self.pixel_coords[1, 0, 0, 0] - self.pixel_coords[0, 0, 0, 0]
        self.dx = self.pixel_coords[0, 1, 0, 1] - self.pixel_coords[0, 0, 0, 1]
        self.dy = self.pixel_coords[0, 0, 1, 2] - self.pixel_coords[0, 0, 0, 2]
        
        # Setup simplified dataset
        self.dataset = SimplifiedFlowDataset(self.vid_dataset, self.pixel_coords, self.config)
        self.dataloader = DataLoader(
            self.dataset,
            shuffle=False,
            batch_size=self.config.batch_size,
            pin_memory=True,
            num_workers=0
        )
        
        print(f"Coordinate deltas - dt: {self.dt}, dx: {self.dx}, dy: {self.dy}")
        print(f"Training on {len(self.dataset)} frame pairs")
        
    def _setup_models(self):
        """Setup feature and flow field models"""
        print("Setting up models...")
        
        # Load pre-trained feature model
        self.feature_model = modules.SingleBVPNet(
            type="sine",
            in_features=3,
            out_features=self.config.feature_out_features,
            mode='mlp',
            hidden_features=self.config.feature_hidden_features,
            num_hidden_layers=self.config.feature_num_layers
        )
        
        self.feature_model.load_state_dict(torch.load(self.config.feature_model_path))
        self.feature_model.eval()
        self.feature_model.to(self.device)
        
        # Initialize flow field model
        self.flow_model = modules.SingleBVPNet(
            type=self.config.flow_activation,
            in_features=3,  # x, y, t_target
            out_features=2,  # dx, dy
            mode='mlp',
            hidden_features=self.config.flow_hidden_features,
            num_hidden_layers=self.config.flow_num_layers
        )
        self.flow_model.to(self.device)
        
        print(f"Feature model parameters: {sum(p.numel() for p in self.feature_model.parameters())}")
        print(f"Flow model parameters: {sum(p.numel() for p in self.flow_model.parameters())}")
        
    def _setup_optimizer(self):
        """Setup optimizer"""
        self.optimizer = torch.optim.Adam(
            self.flow_model.parameters(),
            lr=self.config.learning_rate
        )
        
    def _setup_logging(self):
        """Setup logging directories"""
        os.makedirs(f"{self.config.log_dir}/checkpoints", exist_ok=True)
        
    def _compute_derivative_regularization(self, coords: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Compute derivative regularization for flow smoothness"""
        # coords: [B, H, W, 3], flow: [B, H, W, 2]
        
        # Compute spatial derivatives
        flow_dx = torch.diff(flow, dim=2)  # [B, H, W-1, 2]
        flow_dy = torch.diff(flow, dim=1)  # [B, H-1, W, 2]
        
        # L2 norm of derivatives
        derivative_loss = torch.mean(flow_dx ** 2) + torch.mean(flow_dy ** 2)
        
        return derivative_loss
        
    def _compute_simplified_flow_loss(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Compute simplified flow loss with uniform regularization
        """
        # Move batch to device
        source_coords = batch['source_coords'].to(self.device)  # [B, H, W, 3]
        target_coords = batch['target_coords'].to(self.device)  # [B, H, W, 3]
        target_frame = batch['target_frame']
        
        B, H, W, _ = source_coords.shape
        
        # Create input coordinates for flow model: (x, y, t_target)
        input_coords = torch.cat([
            source_coords[:, :, :, 1:3],  # x, y from frame 0
            target_coords[:, :, :, 0:1],  # t_target
        ], dim=-1)  # [B, H, W, 3]
        
        # Predict flow from frame 0 to target frame
        flow_output = self.flow_model({
            'coords': input_coords.view(B, -1, 3),
            'idx': torch.tensor([0]).to(self.device)
        })
        predicted_flow = flow_output['model_out'].view(B, H, W, 2)  # [B, H, W, 2] (dx, dy)
        
        # Create warped coordinates using grid_sample
        # grid_sample expects normalized coordinates in [-1, 1]
        # and grid format [B, H, W, 2] with (x, y) order
        
        # Get original coordinate bounds for normalization
        x_coords = source_coords[:, :, :, 1]  # [B, H, W]
        y_coords = source_coords[:, :, :, 2]  # [B, H, W]
        
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        # Apply flow to get warped coordinates
        warped_x = x_coords + predicted_flow[:, :, :, 0]  # [B, H, W]
        warped_y = y_coords + predicted_flow[:, :, :, 1]  # [B, H, W]
        
        # Normalize to [-1, 1] for grid_sample
        warped_x_norm = 2 * (warped_x - x_min) / (x_max - x_min) - 1
        warped_y_norm = 2 * (warped_y - y_min) / (y_max - y_min) - 1
        
        # Create grid for sampling [B, H, W, 2] with (x, y) order
        warp_grid = torch.stack([warped_x_norm, warped_y_norm], dim=-1)
        
        # Get features from source frame (frame 0)
        with torch.no_grad():
            source_feat = self.feature_model({
                'coords': source_coords.view(B, -1, 3),
                'idx': torch.tensor([0]).to(self.device)
            })['model_out'].view(B, H, W, self.config.feature_out_features)
        
        # Get features from target frame
        target_feat = self.feature_model({
            'coords': target_coords.view(B, -1, 3),
            'idx': torch.tensor([0]).to(self.device)
        })['model_out'].view(B, H, W, self.config.feature_out_features)
        
        # Warp source features using predicted flow
        # Reshape for grid_sample: [B, C, H, W]
        source_feat_grid = source_feat.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        warped_source_feat = F.grid_sample(
            source_feat_grid,
            warp_grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        
        # Reshape back: [B, H, W, C]
        warped_source_feat = warped_source_feat.permute(0, 2, 3, 1)
        
        # Feature consistency loss (MSE between warped source and target features)
        feature_loss = F.mse_loss(warped_source_feat, target_feat)
        
        # Uniform magnitude regularization
        flow_magnitude = torch.sqrt(torch.mean(predicted_flow ** 2))
        magnitude_loss = flow_magnitude ** 2 * self.config.lambda_magnitude
        
        # Uniform derivative regularization
        derivative_loss = self._compute_derivative_regularization(
            input_coords, predicted_flow
        ) * self.config.lambda_derivative
        
        # Total loss
        total_loss = feature_loss + magnitude_loss + derivative_loss
        
        return {
            'total_loss': total_loss,
            'feature_loss': feature_loss,
            'magnitude_loss': magnitude_loss,
            'derivative_loss': derivative_loss,
            'flow_magnitude': flow_magnitude,
            'target_frame': target_frame
        }
        
    def predict_canonical_flow(self, target_frame: int) -> torch.Tensor:
        """Predict flow from frame 0 to target frame"""
        self.flow_model.eval()
        
        with torch.no_grad():
            H, W = self.vid_dataset.shape[1], self.vid_dataset.shape[2]
            source_coords = self.pixel_coords[0].unsqueeze(0).to(self.device)  # [1, H, W, 3]
            target_coords = self.pixel_coords[target_frame].unsqueeze(0).to(self.device)  # [1, H, W, 3]
            
            # Create input: (x, y, t_target)
            input_coords = torch.cat([
                source_coords[:, :, :, 1:3],  # x, y
                target_coords[:, :, :, 0:1],  # t_target
            ], dim=-1)
            
            # Predict flow
            flow_output = self.flow_model({
                'coords': input_coords.view(1, -1, 3),
                'idx': torch.tensor([0]).to(self.device)
            })
            
            flow_field = flow_output['model_out'].view(H, W, 2)  # [H, W, 2]
            
        self.flow_model.train()
        return flow_field
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.flow_model.train()
        
        epoch_losses = {
            'total_loss': 0.0,
            'feature_loss': 0.0,
            'magnitude_loss': 0.0,
            'derivative_loss': 0.0,
            'flow_magnitude': 0.0
        }
        
        # Accumulate losses across all frames before optimization
        accumulated_total_loss = 0.0
        num_batches = 0
        
        # Zero gradients at the start of epoch
        self.optimizer.zero_grad()
        
        for batch in tqdm(self.dataloader, desc="Training Progress"):
            import pdb; pdb.set_trace()
            # Compute loss for this frame
            losses = self._compute_simplified_flow_loss(batch)
            
            # Accumulate total loss for backward pass
            accumulated_total_loss += losses['total_loss']
            
            # Accumulate losses for logging
            for key in epoch_losses:
                if key in losses:
                    value = losses[key]
                    if isinstance(value, torch.Tensor):
                        epoch_losses[key] += value.item()
                    else:
                        epoch_losses[key] += value
            num_batches += 1
        
        # Perform backward pass on accumulated loss
        if num_batches > 0:
            # Average the accumulated loss
            accumulated_total_loss = accumulated_total_loss / num_batches
            accumulated_total_loss.backward()
            self.optimizer.step()
            
            # Average losses for logging
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
        
        return epoch_losses
        
    def train(self):
        """Main training loop"""
        print(f"Starting simplified canonical flow training for {self.config.num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Flow model parameters: {sum(p.numel() for p in self.flow_model.parameters())}")
        print(f"Uniform regularization - magnitude: {self.config.lambda_magnitude}, derivative: {self.config.lambda_derivative}")
        
        training_losses = []
        
        for epoch in tqdm(range(self.config.num_epochs), desc="Training Progress"):
            losses = self.train_epoch(epoch)
            training_losses.append(losses)
            
            # Log progress
            if epoch % 10 == 0 or epoch in [1, 2, 5]:
                tqdm.write(f"Epoch {epoch}/{self.config.num_epochs}:")
                tqdm.write(f"  Total Loss: {losses['total_loss']:.6f}")
                tqdm.write(f"  Feature Loss: {losses['feature_loss']:.6f}")
                tqdm.write(f"  Magnitude Loss: {losses['magnitude_loss']:.6f}")
                tqdm.write(f"  Derivative Loss: {losses['derivative_loss']:.6f}")
                tqdm.write(f"  Flow Magnitude: {losses['flow_magnitude']:.6f}")
                
            # Test and visualize every 20 epochs
            if epoch % 20 == 0:
                self._test_flow_progression(epoch)
                self._visualize_results(epoch)
                
        print("Simplified canonical flow training completed!")
        
    def _test_flow_progression(self, epoch: int):
        """Test flow magnitudes across different frames"""
        test_frames = [1, 3, 7, 15]
        print(f"\nFlow Progression Test - Epoch {epoch}:")
        
        for target_frame in test_frames:
            if target_frame < self.vid_dataset.shape[0]:
                flow = self.predict_canonical_flow(target_frame)
                magnitude = flow.norm(dim=-1).mean().item()
                coord_ratio = magnitude / self.dx
                print(f"  Flow 0→{target_frame:2d}: magnitude={magnitude:.6f}, coord_ratio={coord_ratio:.2f}")
        print()
        
    def save_model(self, path: str):
        """Save trained flow model"""
        torch.save(self.flow_model.state_dict(), path)
        print(f"Model saved to {path}")
        
    def _visualize_results(self, epoch: int):
        """Generate comprehensive visualization of training progress"""
        print("Generating comprehensive visualizations...")
        
        self.flow_model.eval()
        self.feature_model.eval()
        
        # Filter valid target frames
        valid_targets = [f for f in self.config.vis_target_frames 
                        if f < self.vid_dataset.shape[0]]
        
        if not valid_targets:
            print("No valid target frames for visualization")
            return
            
        num_targets = len(valid_targets)
        fig, axes = plt.subplots(4, num_targets, figsize=(4*num_targets, 16))
        
        if num_targets == 1:
            axes = axes.reshape(-1, 1)
        
        with torch.no_grad():
            for i, target_frame in enumerate(valid_targets):
                # Get flow prediction
                flow_field = self.predict_canonical_flow(target_frame)
                
                # Get coordinates for source and target frames
                source_coords = self.pixel_coords[0].unsqueeze(0).to(self.device)  # [1, H, W, 3]
                target_coords = self.pixel_coords[target_frame].unsqueeze(0).to(self.device)  # [1, H, W, 3]
                
                H, W = source_coords.shape[1], source_coords.shape[2]
                
                # Get features from source frame (frame 0)
                source_feat = self.feature_model({
                    'coords': source_coords.view(1, -1, 3),
                    'idx': torch.tensor([0]).to(self.device)
                })['model_out'].view(1, H, W, self.config.feature_out_features)
                
                # Get features from target frame
                target_feat = self.feature_model({
                    'coords': target_coords.view(1, -1, 3),
                    'idx': torch.tensor([0]).to(self.device)
                })['model_out'].view(1, H, W, self.config.feature_out_features)
                
                # Apply PCA for feature visualization
                try:
                    from train_feat_test import pca
                    [source_feat_pca, target_feat_pca], _ = pca([
                        source_feat.permute(0, 3, 1, 2),
                        target_feat.permute(0, 3, 1, 2)
                    ])
                except ImportError:
                    # Fallback: use first 3 channels as RGB
                    source_feat_pca = source_feat[:, :, :, :3].permute(0, 3, 1, 2)
                    target_feat_pca = target_feat[:, :, :, :3].permute(0, 3, 1, 2)
                    # Normalize to [0, 1]
                    source_feat_pca = (source_feat_pca - source_feat_pca.min()) / (source_feat_pca.max() - source_feat_pca.min())
                    target_feat_pca = (target_feat_pca - target_feat_pca.min()) / (target_feat_pca.max() - target_feat_pca.min())
                
                # Warp source features using predicted flow
                # Get original coordinate bounds for normalization
                x_coords = source_coords[:, :, :, 1]  # [1, H, W]
                y_coords = source_coords[:, :, :, 2]  # [1, H, W]
                
                x_min, x_max = x_coords.min(), x_coords.max()
                y_min, y_max = y_coords.min(), y_coords.max()
                
                # Apply flow to get warped coordinates
                warped_x = x_coords + flow_field.unsqueeze(0)[:, :, :, 0]  # [1, H, W]
                warped_y = y_coords + flow_field.unsqueeze(0)[:, :, :, 1]  # [1, H, W]
                
                # Normalize to [-1, 1] for grid_sample
                warped_x_norm = 2 * (warped_x - x_min) / (x_max - x_min) - 1
                warped_y_norm = 2 * (warped_y - y_min) / (y_max - y_min) - 1
                
                # Create grid for sampling [1, H, W, 2] with (x, y) order
                warp_grid = torch.stack([warped_x_norm, warped_y_norm], dim=-1)
                
                # Warp source features
                warped_source_feat_pca = F.grid_sample(
                    source_feat_pca,
                    warp_grid,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=True
                )
                
                # Compute error between warped source and target features
                error = torch.abs(target_feat_pca - warped_source_feat_pca)
                
                # Calculate flow statistics for titles
                flow_magnitude = flow_field.norm(dim=-1).mean().item()
                flow_max = flow_field.norm(dim=-1).max().item()
                coord_ratio = flow_magnitude / self.dx
                
                # Plot - Row 0: Source features
                axes[0, i].imshow(source_feat_pca[0].permute(1, 2, 0).cpu().numpy())
                axes[0, i].set_title(f"Frame 0 Features (Source)")
                axes[0, i].axis('off')
                
                # Plot - Row 1: Target features (Ground Truth)
                axes[1, i].imshow(target_feat_pca[0].permute(1, 2, 0).cpu().numpy())
                axes[1, i].set_title(f"Frame {target_frame} Features (Target)")
                axes[1, i].axis('off')
                
                # Plot - Row 2: Warped source features
                axes[2, i].imshow(warped_source_feat_pca[0].permute(1, 2, 0).cpu().numpy())
                axes[2, i].set_title(f"Warped Frame 0→{target_frame}\nMag: {flow_magnitude:.4f}, Ratio: {coord_ratio:.1f}")
                axes[2, i].axis('off')
                
                # Plot - Row 3: Error between warped and target
                error_gray = error[0].mean(dim=0).cpu().numpy()  # Average across channels
                im = axes[3, i].imshow(error_gray, cmap='hot', vmin=0, vmax=error_gray.max())
                axes[3, i].set_title(f"Warping Error\nMax: {error_gray.max():.4f}")
                axes[3, i].axis('off')
                
                # Add colorbar for error
                plt.colorbar(im, ax=axes[3, i], fraction=0.046, pad=0.04)
                
        plt.tight_layout()
        
        # Save visualization
        save_path = f'{self.config.log_dir}/checkpoints/epoch_{epoch}_comprehensive_results.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Comprehensive visualization saved to: {save_path}")
        
        # Also create a separate flow field visualization
        self._save_flow_field_analysis(epoch, valid_targets)
        
        self.flow_model.train()
        
    def _save_flow_field_analysis(self, epoch: int, target_frames: list):
        """Save detailed flow field analysis"""
        fig, axes = plt.subplots(2, len(target_frames), figsize=(4*len(target_frames), 8))
        
        if len(target_frames) == 1:
            axes = axes.reshape(-1, 1)
        
        with torch.no_grad():
            for i, target_frame in enumerate(target_frames):
                flow_field = self.predict_canonical_flow(target_frame)
                
                # Flow statistics
                flow_magnitude = flow_field.norm(dim=-1)
                flow_mean = flow_magnitude.mean().item()
                flow_max = flow_magnitude.max().item()
                flow_std = flow_magnitude.std().item()
                coord_ratio = flow_mean / self.dx
                
                # Flow visualization
                flow_vis = self._visualize_flow_as_rgb(flow_field)
                
                # Plot flow field
                axes[0, i].imshow(flow_vis)
                axes[0, i].set_title(f"Flow 0→{target_frame}\nMean: {flow_mean:.4f}, Max: {flow_max:.4f}")
                axes[0, i].axis('off')
                
                # Plot flow magnitude heatmap
                im = axes[1, i].imshow(flow_magnitude.cpu().numpy(), cmap='viridis')
                axes[1, i].set_title(f"Flow Magnitude\nStd: {flow_std:.4f}, Ratio: {coord_ratio:.1f}")
                axes[1, i].axis('off')
                plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # Save flow analysis
        flow_save_path = f'{self.config.log_dir}/checkpoints/epoch_{epoch}_flow_analysis.png'
        plt.savefig(flow_save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Flow analysis saved to: {flow_save_path}")
        
        # Print flow statistics to console
        print(f"Flow Statistics for Epoch {epoch}:")
        for target_frame in target_frames:
            flow_field = self.predict_canonical_flow(target_frame)
            flow_magnitude = flow_field.norm(dim=-1)
            coord_ratio = flow_magnitude.mean().item() / self.dx
            print(f"  Frame 0→{target_frame}: mean={flow_magnitude.mean().item():.6f}, "
                  f"max={flow_magnitude.max().item():.6f}, coord_ratio={coord_ratio:.2f}")
        print()
        
    def _visualize_flow_as_rgb(self, flow: torch.Tensor) -> np.ndarray:
        """Convert flow field to RGB visualization"""
        # Convert flow to polar coordinates
        u = flow[:, :, 0].cpu().numpy()
        v = flow[:, :, 1].cpu().numpy()
        
        magnitude = np.sqrt(u**2 + v**2)
        angle = np.arctan2(v, u)
        
        # Normalize magnitude
        if magnitude.max() > 0:
            magnitude_normalized = magnitude / magnitude.max()
        else:
            magnitude_normalized = magnitude
        
        # Convert to HSV
        h = (angle + np.pi) / (2 * np.pi)
        s = magnitude_normalized
        v = np.ones_like(h)
        
        hsv = np.stack([h, s, v], axis=2)
        
        from matplotlib.colors import hsv_to_rgb
        rgb = hsv_to_rgb(hsv)
        
        return rgb


def main():
    """Main function to run simplified canonical flow training"""
    config = SimplifiedCanonicalFlowConfig(
        num_epochs=200,
        learning_rate=1e-4,
        lambda_magnitude=0.01,  # Uniform magnitude regularization
        lambda_derivative=0.001,  # Uniform derivative regularization
        flow_hidden_features=32,
        flow_num_layers=2,
        vis_target_frames=[1, 3, 7, 15]
    )
    
    trainer = SimplifiedCanonicalFlowTrainer(config)
    trainer.train()
    trainer.save_model(f"{config.log_dir}/checkpoints/simplified_canonical_flow_final.pth")


if __name__ == "__main__":
    main() 