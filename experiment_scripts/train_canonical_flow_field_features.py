'''
Canonical Flow Field Trainer with Feature Consistency

This module implements a simplified trainer that uses frame 0 as the canonical source
frame and learns flow fields to all other frames using feature consistency loss.
This maintains the same feature-based approach as the modularized trainer but with
the simplification of using frame 0 as the only source.

Key features:
1. Frame 0 is the only source frame
2. Train on entire video, no sampling
3. Feature consistency loss using pre-trained feature network
4. Direct prediction from (x, y, target_frame) to (dx, dy)
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
class CanonicalFlowFeatureConfig:
    """Configuration for canonical flow field training with features"""
    # Model parameters
    flow_hidden_features: int = 64
    flow_num_layers: int = 3
    flow_activation: str = "sine"
    
    # Training parameters
    num_epochs: int = 200
    learning_rate: float = 1e-4
    batch_size: int = 1
    sample_fraction: float = 1e-3  # Sample fraction for training
    
    # Loss parameters
    lambda_magnitude: float = 1.0  # Tunable flow magnitude regularization
    
    # Logging parameters
    steps_til_summary: int = 20
    log_dir: str = f'{WORK_DIR}/logs/canonical_flow_features'
    
    # Visualization parameters
    vis_target_frames: list = None  # Will default to [1, 5, 10, 15]
    
    # Data parameters
    video_data_path: str = f'{WORK_DIR}/data/mock_videos/mockvideo_vid_data.npy'
    feature_model_path: str = f'{WORK_DIR}/logs/mock_video_feature_test/checkpoints/model_final.pth'
    
    # Feature model parameters
    feature_out_features: int = 384
    feature_hidden_features: int = 1024
    feature_num_layers: int = 3
    
    def __post_init__(self):
        if self.vis_target_frames is None:
            self.vis_target_frames = [1, 5, 10, 15]


class CanonicalFlowFeatureDataset(Dataset):
    """
    Dataset that returns sampled pixels for all frames (except frame 0)
    """
    
    def __init__(self, vid_dataset, pixel_coords, config: CanonicalFlowFeatureConfig):
        self.vid_dataset = vid_dataset
        self.pixel_coords = pixel_coords
        self.config = config
        self.num_frames = vid_dataset.shape[0]
        
        # We train on frames 1 to N-1 (frame 0 is canonical source)
        self.target_frames = list(range(1, self.num_frames))
        
        # Calculate samples per frame
        total_pixels = vid_dataset.shape[1] * vid_dataset.shape[2]
        self.samples_per_frame = int(total_pixels * config.sample_fraction)
        
    def __len__(self):
        return len(self.target_frames)
    
    def __getitem__(self, idx):
        target_frame = self.target_frames[idx]
        
        # Sample random pixels
        h, w = self.vid_dataset.shape[1], self.vid_dataset.shape[2]
        pixel_indices = torch.randperm(h * w)[:self.samples_per_frame]
        
        # Get coordinates for frame 0 (source) and target frame
        source_coords = self.pixel_coords[0].view(-1, 3)[pixel_indices]  # [N, 3]
        target_coords = self.pixel_coords[target_frame].view(-1, 3)[pixel_indices]  # [N, 3]
        
        # Create input coordinates: (x, y, target_frame_time)
        input_coords = torch.cat([
            source_coords[:, 1:3],  # x, y from frame 0
            target_coords[:, 0:1],  # t_target
        ], dim=1)  # [N, 3]
        
        return {
            'input_coords': input_coords,
            'source_coords': source_coords,
            'target_coords': target_coords,
            'target_frame': target_frame
        }


class CanonicalFlowFeatureTrainer:
    """
    Simplified trainer for learning flow fields from frame 0 to all other frames
    using feature consistency loss with a pre-trained feature network.
    
    The network learns to map from (x, y, t_target) to (dx, dy) where:
    - (x, y) are coordinates in frame 0
    - t_target is the time of the target frame
    - (dx, dy) is the flow from frame 0 to target frame
    """
    
    def __init__(self, config: CanonicalFlowFeatureConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self._setup_data()
        self._setup_models()
        self._setup_optimizer()
        self._setup_logging()
        
    def _setup_data(self):
        """Setup video dataset and coordinate grids"""
        print("Setting up data...")
        
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
        
        # Calculate coordinate deltas for reference
        self.dt = self.pixel_coords[1, 0, 0, 0] - self.pixel_coords[0, 0, 0, 0]
        self.dx = self.pixel_coords[0, 1, 0, 1] - self.pixel_coords[0, 0, 0, 1]
        self.dy = self.pixel_coords[0, 0, 1, 2] - self.pixel_coords[0, 0, 0, 2]
        
        # Calculate coordinate bounds
        self.bounds = self._calculate_coordinate_bounds()
        
        # Setup dataset and dataloader
        self.dataset = CanonicalFlowFeatureDataset(self.vid_dataset, self.pixel_coords, self.config)
        self.dataloader = DataLoader(
            self.dataset,
            shuffle=True,
            batch_size=self.config.batch_size,
            pin_memory=True,
            num_workers=0
        )
        
        print(f"Coordinate deltas - dt: {self.dt}, dx: {self.dx}, dy: {self.dy}")
        print(f"Training on {len(self.dataset)} target frames (1 to {self.vid_dataset.shape[0]-1})")
        print(f"Samples per frame: {self.dataset.samples_per_frame}")
        
    def _calculate_coordinate_bounds(self) -> Dict[str, float]:
        """Calculate min/max bounds for each coordinate dimension"""
        coords = self.pixel_coords
        return {
            't_min': torch.min(coords[:, :, :, 0]).item(),
            't_max': torch.max(coords[:, :, :, 0]).item(),
            'x_min': torch.min(coords[:, :, :, 1]).item(),
            'x_max': torch.max(coords[:, :, :, 1]).item(),
            'y_min': torch.min(coords[:, :, :, 2]).item(),
            'y_max': torch.max(coords[:, :, :, 2]).item(),
        }
        
    def _setup_models(self):
        """Setup feature and flow field models"""
        print("Setting up models...")
        
        # Load pre-trained feature model (same as modularized version)
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
        
        # Initialize canonical flow field model
        # Input: (x, y, t_target) -> Output: (dx, dy)
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
        
    def _create_roi_mask(self, query_coords: torch.Tensor) -> torch.Tensor:
        """
        Create region of interest mask for coordinates within valid bounds
        
        Args:
            query_coords: Warped coordinates [batch, N, 3]
            
        Returns:
            ROI mask [batch, N, feature_dim]
        """
        bounds = self.bounds
        
        # Check if coordinates are within bounds
        mask = torch.ones_like(query_coords[:, :, 0:1]) * (
            (query_coords[:, :, 1:2] >= bounds['x_min']) &
            (query_coords[:, :, 1:2] <= bounds['x_max']) &
            (query_coords[:, :, 2:3] >= bounds['y_min']) &
            (query_coords[:, :, 2:3] <= bounds['y_max']) &
            (query_coords[:, :, 0:1] >= bounds['t_min']) &
            (query_coords[:, :, 0:1] <= bounds['t_max'])
        ).to(self.device)
        
        # Expand to feature dimensions
        mask = mask.repeat(1, 1, self.config.feature_out_features)
        return mask
        
    def _compute_canonical_feature_loss(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Compute canonical flow field loss using feature consistency
        
        Args:
            batch: Batch data from CanonicalFlowFeatureDataset
            
        Returns:
            Dictionary containing loss components
        """
        # Move batch to device
        input_coords = batch['input_coords'].to(self.device)
        source_coords = batch['source_coords'].to(self.device)
        target_coords = batch['target_coords'].to(self.device)
        
        # Predict flow from frame 0 to target frame
        flow_output = self.flow_model({'coords': input_coords, 'idx': torch.tensor([0]).to(self.device)})
        predicted_flow = flow_output['model_out']  # [batch, N, 2] (dx, dy)
        
        # Compute warped coordinates (warp source to target)
        warped_coords = source_coords.clone()
        warped_coords[:, :, 1:3] += predicted_flow  # Update x, y coordinates
        # Keep the target time coordinate
        warped_coords[:, :, 0:1] = target_coords[:, :, 0:1]
        
        # Get features for source, target, and warped coordinates
        with torch.no_grad():
            source_feat = self.feature_model({
                'coords': source_coords,
                'idx': torch.tensor([0]).to(self.device)
            })['model_out']
            
        target_feat = self.feature_model({
            'coords': target_coords,
            'idx': torch.tensor([0]).to(self.device)
        })['model_out']
        
        warped_feat = self.feature_model({
            'coords': warped_coords,
            'idx': torch.tensor([0]).to(self.device)
        })['model_out']
        
        # Create ROI mask for warped coordinates
        roi_mask = self._create_roi_mask(warped_coords)
        
        # Feature consistency loss (warped source vs target)
        feature_loss = loss_functions.image_mse(
            roi_mask,
            {'model_out': warped_feat},
            {'img': target_feat}
        )
        
        # Flow magnitude regularization
        flow_magnitude = torch.sqrt(torch.mean(predicted_flow ** 2))
        magnitude_loss = flow_magnitude ** 2 * self.config.lambda_magnitude
        
        # Total loss
        total_loss = feature_loss['img_loss'] + magnitude_loss
        
        return {
            'total_loss': total_loss,
            'feature_loss': feature_loss['img_loss'],
            'magnitude_loss': magnitude_loss,
            'flow_magnitude': flow_magnitude,
            'roi_ratio': roi_mask.sum() / roi_mask.numel()
        }
        
    def predict_canonical_flow(self, target_frame: int) -> torch.Tensor:
        """
        Predict flow from frame 0 to target frame
        
        Args:
            target_frame: Target frame index
            
        Returns:
            Flow field [H, W, 2] from frame 0 to target frame
        """
        self.flow_model.eval()
        
        with torch.no_grad():
            # Get coordinates for frame 0 and target frame
            h, w = self.vid_dataset.shape[1], self.vid_dataset.shape[2]
            source_coords = self.pixel_coords[0].view(-1, 3).to(self.device)
            target_coords = self.pixel_coords[target_frame].view(-1, 3).to(self.device)
            
            # Create input: (x, y, t_target)
            input_coords = torch.cat([
                source_coords[:, 1:3],  # x, y from frame 0
                target_coords[:, 0:1],  # t_target
            ], dim=1)
            
            # Predict flow
            flow_output = self.flow_model({
                'coords': input_coords,
                'idx': torch.tensor([0]).to(self.device)
            })
            
            flow_field = flow_output['model_out'].view(h, w, 2)  # [H, W, 2]
            
        self.flow_model.train()
        return flow_field
        
    def _visualize_results(self, epoch: int):
        """Generate visualization of training progress"""
        print("Generating visualizations...")
        
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
                source_coords = self.pixel_coords[0, ...].view(1, -1, 3).to(self.device)
                target_coords = self.pixel_coords[target_frame, ...].view(1, -1, 3).to(self.device)
                
                # Get features
                source_feat = self.feature_model({
                    'coords': source_coords,
                    'idx': torch.tensor([0]).to(self.device)
                })['model_out'].view(1, 224, 224, self.config.feature_out_features)
                
                target_feat = self.feature_model({
                    'coords': target_coords,
                    'idx': torch.tensor([0]).to(self.device)
                })['model_out'].view(1, 224, 224, self.config.feature_out_features)
                
                # Apply PCA for feature visualization
                from train_feat_test import pca
                [source_feat_pca, target_feat_pca], _ = pca([
                    source_feat.permute(0, 3, 1, 2),
                    target_feat.permute(0, 3, 1, 2)
                ])
                
                # Warp source features using flow
                grid_warped = self.pixel_coords[0, :, :, 1:3].to(self.device) + flow_field
                # Add batch dimension and reorder for grid_sample (expects [N, H, W, 2] with [x, y] order)
                grid_for_sample = torch.cat([
                    grid_warped[:, :, 1:2],  # x coordinates
                    grid_warped[:, :, 0:1]   # y coordinates  
                ], dim=-1).unsqueeze(0)  # Add batch dimension: [1, H, W, 2]
                
                warped_source_feat_pca = F.grid_sample(
                    input=source_feat_pca,
                    grid=grid_for_sample,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=True
                )
                
                # Compute error between warped source and target features
                error = torch.abs(target_feat_pca - warped_source_feat_pca)
                
                # Visualize flow field
                flow_vis = self._visualize_flow_white_zero(flow_field.permute(2, 0, 1))
                
                # Calculate flow statistics for title
                flow_magnitude = flow_field.norm(dim=-1).mean().item()
                flow_max = flow_field.norm(dim=-1).max().item()
                
                # Plot - Row 0: Source features
                axes[0, i].imshow(source_feat_pca[0, ...].permute(1, 2, 0).cpu().numpy())
                axes[0, i].set_title(f"Frame 0 Features (PCA)")
                axes[0, i].axis('off')
                
                # Plot - Row 1: Target features (Ground Truth)
                axes[1, i].imshow(target_feat_pca[0, ...].permute(1, 2, 0).cpu().numpy())
                axes[1, i].set_title(f"Frame {target_frame} Features (GT)")
                axes[1, i].axis('off')
                
                # Plot - Row 2: Warped source features
                axes[2, i].imshow(warped_source_feat_pca[0, ...].permute(1, 2, 0).cpu().numpy())
                axes[2, i].set_title(f"Frame 0→{target_frame} (Warped)\nMag: {flow_magnitude:.4f}")
                axes[2, i].axis('off')
                
                # Plot - Row 3: Error between warped and target
                error_gray = error[0, ...].mean(dim=0).cpu().numpy()  # Average across RGB channels
                im = axes[3, i].imshow(error_gray, cmap='hot', vmin=0, vmax=error_gray.max())
                axes[3, i].set_title(f"Warping Error\nMax: {error_gray.max():.4f}")
                axes[3, i].axis('off')
                
                # Add colorbar for error
                plt.colorbar(im, ax=axes[3, i], fraction=0.046, pad=0.04)
                
        plt.tight_layout()
        
        # Save to log folder instead of showing
        save_path = f'{self.config.log_dir}/checkpoints/epoch_{epoch}_canonical_features.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        print(f"Visualization saved to: {save_path}")
        
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
                
                # Flow visualization
                flow_vis = self._visualize_flow_white_zero(flow_field.permute(2, 0, 1))
                
                # Plot flow field
                axes[0, i].imshow(flow_vis)
                axes[0, i].set_title(f"Flow 0→{target_frame}\nMean: {flow_mean:.4f}, Max: {flow_max:.4f}")
                axes[0, i].axis('off')
                
                # Plot flow magnitude heatmap
                im = axes[1, i].imshow(flow_magnitude.cpu().numpy(), cmap='viridis')
                axes[1, i].set_title(f"Flow Magnitude\nStd: {flow_std:.4f}")
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
        
    def _visualize_flow_white_zero(self, flow: torch.Tensor) -> np.ndarray:
        """
        Visualize optical flow with white background for zero flow
        
        Args:
            flow: Flow field tensor [2, height, width]
            
        Returns:
            rgb: RGB image with white background for zero flow
        """
        # Convert flow to polar coordinates
        u = flow[0].cpu().numpy()
        v = flow[1].cpu().numpy()
        
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
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.flow_model.train()
        
        epoch_losses = {
            'total_loss': 0.0,
            'feature_loss': 0.0,
            'magnitude_loss': 0.0,
            'flow_magnitude': 0.0,
            'roi_ratio': 0.0
        }
        
        num_batches = 0
        
        for batch in self.dataloader:
            # Compute loss
            losses = self._compute_canonical_feature_loss(batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            self.optimizer.step()
            
            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            num_batches += 1
        
        # Average losses
        if num_batches > 0:
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
        
        return epoch_losses
        
    def train(self):
        """Main training loop"""
        print(f"Starting canonical feature flow training for {self.config.num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Flow model parameters: {sum(p.numel() for p in self.flow_model.parameters())}")
        print(f"Regularization: λ_magnitude = {self.config.lambda_magnitude}")
        print(f"Visualization every {self.config.steps_til_summary} epochs")
        
        # Track training progress
        training_losses = []
        
        for epoch in tqdm(range(self.config.num_epochs), desc="Training Progress"):
            # Train one epoch
            losses = self.train_epoch(epoch)
            training_losses.append(losses)
            
            # Log progress
            if epoch % 10 == 0 or epoch in [1, 2, 5]:
                tqdm.write(f"Epoch {epoch}/{self.config.num_epochs}:")
                tqdm.write(f"  Total Loss: {losses['total_loss']:.6f}")
                tqdm.write(f"  Feature Loss: {losses['feature_loss']:.6f}")
                tqdm.write(f"  Magnitude Loss: {losses['magnitude_loss']:.6f}")
                tqdm.write(f"  Flow Magnitude: {losses['flow_magnitude']:.6f}")
                tqdm.write(f"  ROI Ratio: {losses['roi_ratio']:.3f}")
                
            # Generate visualizations
            if epoch % self.config.steps_til_summary == 0 or epoch in [1, 2, 5]:
                tqdm.write(f"Generating visualization for epoch {epoch}...")
                self._visualize_results(epoch)
                
        print("Canonical feature flow training completed!")
        
        # Plot training curves
        self._plot_training_curves(training_losses)
        
    def _plot_training_curves(self, training_losses):
        """Plot training loss curves"""
        epochs = range(len(training_losses))
        
        # Extract loss components
        total_losses = [loss['total_loss'] for loss in training_losses]
        feature_losses = [loss['feature_loss'] for loss in training_losses]
        magnitude_losses = [loss['magnitude_loss'] for loss in training_losses]
        flow_magnitudes = [loss['flow_magnitude'] for loss in training_losses]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0, 0].plot(epochs, total_losses)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(epochs, feature_losses)
        axes[0, 1].set_title('Feature Consistency Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(epochs, magnitude_losses)
        axes[1, 0].set_title('Magnitude Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(epochs, flow_magnitudes)
        axes[1, 1].set_title('Flow Magnitude')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.config.log_dir}/training_curves.png', dpi=150)
        plt.close()  # Close the figure to free memory
        
        print(f"Training curves saved to {self.config.log_dir}/training_curves.png")
        
    def save_model(self, path: str):
        """Save trained flow model"""
        torch.save(self.flow_model.state_dict(), path)
        print(f"Model saved to {path}")
        
    def load_model(self, path: str):
        """Load trained flow model"""
        self.flow_model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")


def main():
    """Main function to run canonical feature flow training"""
    config = CanonicalFlowFeatureConfig(
        num_epochs=100,
        learning_rate=1e-4,
        steps_til_summary=10,
        lambda_magnitude=1.0,  # Tunable regularization
        flow_hidden_features=64,
        flow_num_layers=3,
        vis_target_frames=[1, 3, 7, 15],
        sample_fraction=1e-3  # Sample fraction for training
    )
    
    trainer = CanonicalFlowFeatureTrainer(config)
    trainer.train()
    trainer.save_model(f"{config.log_dir}/checkpoints/canonical_feature_flow_model_final.pth")


if __name__ == "__main__":
    main() 