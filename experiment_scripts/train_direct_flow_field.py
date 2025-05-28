'''
Direct Flow Field Trainer

This module implements a trainer that directly predicts optical flow between arbitrary
frame pairs, eliminating the need for flow accumulation. The network learns to map
from (x, y, t_source, t_target) to (dx, dy) directly.

Key improvements over adjacent-frame training:
1. Direct prediction eliminates accumulation errors
2. Larger temporal gaps provide stronger training signal
3. More robust to temporal variations
4. Simpler inference pipeline
'''

import sys
import os
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm

# Add parent directory to path for imports
WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(WORK_DIR)

import modules
import dataio
import loss_functions


@dataclass
class DirectFlowTrainingConfig:
    """Configuration for direct flow field training"""
    # Model parameters
    flow_hidden_features: int = 32
    flow_num_layers: int = 2
    flow_activation: str = "sine"
    
    # Training parameters
    num_epochs: int = 100  # Reduced for faster testing
    learning_rate: float = 1e-4
    batch_size: int = 1
    sample_fraction: float = 1e-3  # Higher sampling for better training
    
    # Temporal gap parameters
    min_frame_gap: int = 1
    max_frame_gap: int = 10
    gap_curriculum: bool = True  # Start with small gaps, increase over time
    
    # Loss parameters
    lambda_magnitude: float = 1.0  # Much lower regularization
    lambda_consistency: float = 10.0  # Bidirectional consistency loss
    
    # Logging parameters
    steps_til_summary: int = 10  # Show visualization every 10 epochs
    log_dir: str = f'{WORK_DIR}/logs/direct_flow_field_test'
    
    # Visualization parameters
    vis_source_frame: int = 0
    vis_target_frame: int = 8
    
    # Data parameters
    video_data_path: str = f'{WORK_DIR}/data/mock_videos/mockvideo_vid_data.npy'
    feature_model_path: str = f'{WORK_DIR}/logs/mock_video_feature_test/checkpoints/model_final.pth'
    
    # Feature model parameters
    feature_out_features: int = 384
    feature_hidden_features: int = 1024
    feature_num_layers: int = 3


class DirectFlowDataset(Dataset):
    """
    Dataset for direct flow training that generates random frame pairs
    """
    
    def __init__(self, vid_dataset, pixel_coords, config: DirectFlowTrainingConfig):
        self.vid_dataset = vid_dataset
        self.pixel_coords = pixel_coords
        self.config = config
        
        # Pre-compute valid frame pairs
        self.frame_pairs = self._generate_frame_pairs()
        
        # Calculate samples per epoch
        total_pixels = vid_dataset.shape[1] * vid_dataset.shape[2]
        self.samples_per_epoch = int(total_pixels * config.sample_fraction)
        
    def _generate_frame_pairs(self) -> List[Tuple[int, int]]:
        """Generate all valid frame pairs within gap constraints"""
        pairs = []
        num_frames = self.vid_dataset.shape[0]
        
        for source in range(num_frames):
            for gap in range(self.config.min_frame_gap, self.config.max_frame_gap + 1):
                # Forward pairs
                if source + gap < num_frames:
                    pairs.append((source, source + gap))
                # Backward pairs
                if source - gap >= 0:
                    pairs.append((source, source - gap))
        
        return pairs
    
    def __len__(self):
        return len(self.frame_pairs) * 10  # Multiple epochs per frame pair
    
    def __getitem__(self, idx):
        # Select random frame pair
        pair_idx = idx % len(self.frame_pairs)
        source_frame, target_frame = self.frame_pairs[pair_idx]
        
        # Sample random pixels
        h, w = self.vid_dataset.shape[1], self.vid_dataset.shape[2]
        pixel_indices = torch.randperm(h * w)[:self.samples_per_epoch]
        
        # Get coordinates for sampled pixels
        source_coords = self.pixel_coords[source_frame].view(-1, 3)[pixel_indices]
        target_coords = self.pixel_coords[target_frame].view(-1, 3)[pixel_indices]
        
        # Create input coordinates: (x, y, t_source, t_target)
        input_coords = torch.cat([
            source_coords[:, 1:3],  # x, y from source
            source_coords[:, 0:1],  # t_source
            target_coords[:, 0:1],  # t_target
        ], dim=1)
        
        # Ground truth flow (for reference, not used in self-supervised training)
        gt_flow = target_coords[:, 1:3] - source_coords[:, 1:3]
        
        return {
            'input_coords': input_coords,
            'source_coords': source_coords,
            'target_coords': target_coords,
            'gt_flow': gt_flow,
            'source_frame': source_frame,
            'target_frame': target_frame
        }


class DirectFlowFieldTrainer:
    """
    Trainer for learning direct optical flow fields between arbitrary frame pairs.
    
    This trainer eliminates the need for flow accumulation by directly predicting
    flow from any source frame to any target frame within a specified temporal window.
    """
    
    def __init__(self, config: DirectFlowTrainingConfig):
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
        
        # Calculate coordinate deltas
        self.dt = self.pixel_coords[1, 0, 0, 0] - self.pixel_coords[0, 0, 0, 0]
        self.dx = self.pixel_coords[0, 1, 0, 1] - self.pixel_coords[0, 0, 0, 1]
        self.dy = self.pixel_coords[0, 0, 1, 2] - self.pixel_coords[0, 0, 0, 2]
        
        # Calculate coordinate bounds
        self.bounds = self._calculate_coordinate_bounds()
        
        # Setup dataset and dataloader
        self.dataset = DirectFlowDataset(self.vid_dataset, self.pixel_coords, self.config)
        self.dataloader = DataLoader(
            self.dataset,
            shuffle=True,
            batch_size=self.config.batch_size,
            pin_memory=True,
            num_workers=0
        )
        
        print(f"Coordinate deltas - dt: {self.dt}, dx: {self.dx}, dy: {self.dy}")
        print(f"Generated {len(self.dataset.frame_pairs)} frame pairs")
        
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
        
        # Initialize direct flow field model
        # Input: (x, y, t_source, t_target) -> Output: (dx, dy)
        self.flow_model = modules.SingleBVPNet(
            type=self.config.flow_activation,
            in_features=4,  # x, y, t_source, t_target
            out_features=2,  # dx, dy
            mode='mlp',
            hidden_features=self.config.flow_hidden_features,
            num_hidden_layers=self.config.flow_num_layers
        )
        self.flow_model.to(self.device)
        
        print("Models setup complete")
        
    def _setup_optimizer(self):
        """Setup optimizer"""
        self.optimizer = torch.optim.Adam(
            self.flow_model.parameters(),
            lr=self.config.learning_rate
        )
        
    def _setup_logging(self):
        """Setup logging directories"""
        os.makedirs(f"{self.config.log_dir}/checkpoints", exist_ok=True)
        
    def _get_current_max_gap(self, epoch: int) -> int:
        """Get current maximum frame gap for curriculum learning"""
        if not self.config.gap_curriculum:
            return self.config.max_frame_gap
        
        # Gradually increase max gap over training
        progress = epoch / self.config.num_epochs
        current_max = self.config.min_frame_gap + int(
            progress * (self.config.max_frame_gap - self.config.min_frame_gap)
        )
        return min(current_max, self.config.max_frame_gap)
        
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
        
    def _compute_direct_flow_loss(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Compute direct flow field loss using feature consistency
        
        Args:
            batch: Batch data from DirectFlowDataset
            
        Returns:
            Dictionary containing loss components
        """
        # Move batch to device
        input_coords = batch['input_coords'].to(self.device)
        source_coords = batch['source_coords'].to(self.device)
        target_coords = batch['target_coords'].to(self.device)
        
        # Predict direct flow
        flow_output = self.flow_model({'coords': input_coords, 'idx': torch.tensor([0]).to(self.device)})
        predicted_flow = flow_output['model_out']  # [batch, N, 2] (dx, dy)
        
        # Compute warped coordinates
        warped_coords = source_coords.clone()
        warped_coords[:, :, 1:3] += predicted_flow  # Update x, y
        
        # Get features for source and target
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
        
        # Bidirectional consistency loss (optional)
        consistency_loss = torch.tensor(0.0).to(self.device)
        if self.config.lambda_consistency > 0:
            # Predict reverse flow
            reverse_input = torch.cat([
                target_coords[:, :, 1:3],  # x, y from target
                target_coords[:, :, 0:1],  # t_target
                source_coords[:, :, 0:1],  # t_source
            ], dim=2)
            
            reverse_flow_output = self.flow_model({
                'coords': reverse_input,
                'idx': torch.tensor([0]).to(self.device)
            })
            reverse_flow = reverse_flow_output['model_out']
            
            # Consistency: forward + reverse should be close to zero
            consistency_loss = torch.mean((predicted_flow + reverse_flow) ** 2)
        
        # Flow magnitude regularization
        flow_magnitude = torch.sqrt(torch.mean(predicted_flow ** 2))
        magnitude_loss = flow_magnitude ** 2 * self.config.lambda_magnitude
        
        # Total loss
        total_loss = (feature_loss['img_loss'] + 
                     consistency_loss * self.config.lambda_consistency + 
                     magnitude_loss)
        
        return {
            'total_loss': total_loss,
            'feature_loss': feature_loss['img_loss'],
            'consistency_loss': consistency_loss,
            'magnitude_loss': magnitude_loss,
            'flow_magnitude': flow_magnitude,
            'roi_ratio': roi_mask.sum() / roi_mask.numel()
        }
        
    def predict_direct_flow(self, source_frame: int, target_frame: int) -> torch.Tensor:
        """
        Predict direct flow from source frame to target frame
        
        Args:
            source_frame: Source frame index
            target_frame: Target frame index
            
        Returns:
            Flow field [1, H*W, 2] from source to target
        """
        self.flow_model.eval()
        
        with torch.no_grad():
            # Get all coordinates for source frame
            h, w = self.vid_dataset.shape[1], self.vid_dataset.shape[2]
            source_coords = self.pixel_coords[source_frame].view(-1, 3).to(self.device)
            target_coords = self.pixel_coords[target_frame].view(-1, 3).to(self.device)
            
            # Create input: (x, y, t_source, t_target)
            input_coords = torch.cat([
                source_coords[:, 1:3],  # x, y
                source_coords[:, 0:1],  # t_source
                target_coords[:, 0:1],  # t_target
            ], dim=1).unsqueeze(0)  # Add batch dimension
            
            # Predict flow
            flow_output = self.flow_model({
                'coords': input_coords,
                'idx': torch.tensor([0]).to(self.device)
            })
            
            flow_field = flow_output['model_out']  # [1, H*W, 2]
            
        self.flow_model.train()
        return flow_field
        
    def _visualize_results(self, epoch: int):
        """Generate visualization of training progress"""
        print("Generating visualizations...")
        
        self.flow_model.eval()
        self.feature_model.eval()
        
        source_frame = self.config.vis_source_frame
        target_frame = self.config.vis_target_frame
        
        # Validate frame indices
        num_frames = self.vid_dataset.shape[0]
        if source_frame >= num_frames or target_frame >= num_frames:
            print(f"Warning: Frame indices ({source_frame}, {target_frame}) out of range. Using (0, 1).")
            source_frame, target_frame = 0, min(1, num_frames-1)
        
        print(f"Visualizing direct flow: frame {source_frame} → frame {target_frame}")
        
        with torch.no_grad():
            # Get direct flow prediction
            direct_flow = self.predict_direct_flow(source_frame, target_frame)
            direct_flow_2d = direct_flow.view(1, 224, 224, 2)
            
            # Get coordinates and features
            source_coords = self.pixel_coords[source_frame, ...].view(1, -1, 3).to(self.device)
            target_coords = self.pixel_coords[target_frame, ...].view(1, -1, 3).to(self.device)
            
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
            
            # Warp source features using direct flow
            grid_warped = self.pixel_coords[source_frame, :, :, 1:3].to(self.device) + direct_flow_2d
            warped_source_feat_pca = F.grid_sample(
                input=source_feat_pca,
                grid=torch.cat([
                    grid_warped[:, :, :, 1:2], 
                    grid_warped[:, :, :, 0:1]
                ], dim=-1),
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            )
            
            # Compute error
            error = torch.abs(target_feat_pca - warped_source_feat_pca)
            
            # Visualize flow field with white background for zero flow
            flow_vis = self._visualize_flow_white_zero(direct_flow_2d[0, ...].permute(2, 0, 1))
            
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        frame_gap = abs(target_frame - source_frame)
        direction = "→" if target_frame > source_frame else "←" if target_frame < source_frame else "="
        
        axes[0, 0].imshow(source_feat_pca[0, ...].permute(1, 2, 0).cpu().numpy())
        axes[0, 0].set_title(f"Frame {source_frame} Original Features (PCA)")
        
        axes[0, 1].imshow(warped_source_feat_pca[0, ...].permute(1, 2, 0).cpu().numpy())
        axes[0, 1].set_title(f"Frame {source_frame} Warped Features (PCA)")
        
        axes[0, 2].imshow(flow_vis)
        axes[0, 2].set_title(f"Direct Flow: {source_frame} {direction} {target_frame} (gap={frame_gap})")
        
        axes[1, 0].axis('off')  # Empty
        
        axes[1, 1].imshow(target_feat_pca[0, ...].permute(1, 2, 0).cpu().numpy())
        axes[1, 1].set_title(f"Frame {target_frame} Ground Truth Features (PCA)")
        
        axes[1, 2].imshow(error[0, ...].permute(1, 2, 0).cpu().numpy(), cmap='gray')
        axes[1, 2].set_title("Warping Error")
        
        for ax in axes.flatten():
            ax.axis('off')
            
        plt.tight_layout()
        plt.savefig(f'{self.config.log_dir}/checkpoints/epoch_{epoch}_direct_flow_{source_frame}_to_{target_frame}.png', dpi=150)
        plt.show()
        
        self.flow_model.train()
        
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
            'consistency_loss': 0.0,
            'magnitude_loss': 0.0,
            'flow_magnitude': 0.0,
            'roi_ratio': 0.0
        }
        
        num_batches = 0
        
        # Use tqdm for batch progress only on first few epochs
        dataloader_iter = self.dataloader
        if epoch < 5:  # Show batch progress for first 5 epochs
            dataloader_iter = tqdm(self.dataloader, desc=f"Epoch {epoch} batches", leave=False)
        
        for batch_idx, batch in enumerate(dataloader_iter):
            # Apply curriculum learning for frame gaps
            current_max_gap = self._get_current_max_gap(epoch)
            source_frame = batch['source_frame'][0].item()
            target_frame = batch['target_frame'][0].item()
            
            if abs(target_frame - source_frame) > current_max_gap:
                continue  # Skip this batch if gap is too large
            
            # Compute loss
            losses = self._compute_direct_flow_loss(batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            self.optimizer.step()
            
            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            num_batches += 1
            
            # Print details only for first batch of first few epochs
            if batch_idx == 0 and epoch < 3:
                tqdm.write(f"Epoch {epoch}, Batch {batch_idx}:")
                tqdm.write(f"  Frame pair: {source_frame} → {target_frame}")
                tqdm.write(f"  Feature loss: {losses['feature_loss']:.6f}")
                tqdm.write(f"  Consistency loss: {losses['consistency_loss']:.6f}")
                tqdm.write(f"  Magnitude loss: {losses['magnitude_loss']:.6f}")
                tqdm.write(f"  Flow magnitude: {losses['flow_magnitude']:.6f}")
                tqdm.write(f"  ROI ratio: {losses['roi_ratio']:.3f}")
                tqdm.write(f"  Current max gap: {current_max_gap}")
        
        # Average losses
        if num_batches > 0:
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
        
        return epoch_losses
        
    def train(self):
        """Main training loop"""
        print(f"Starting direct flow training for {self.config.num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Flow model parameters: {sum(p.numel() for p in self.flow_model.parameters())}")
        print(f"Visualization every {self.config.steps_til_summary} epochs")
        
        # Track training progress
        training_losses = []
        
        for epoch in tqdm(range(self.config.num_epochs), desc="Training Progress"):
            # Train one epoch
            losses = self.train_epoch(epoch)
            training_losses.append(losses)
            
            # Log progress every 10 epochs or at key milestones
            if epoch % 10 == 0 or epoch in [1, 2, 5]:
                tqdm.write(f"Epoch {epoch}/{self.config.num_epochs}:")
                tqdm.write(f"  Total Loss: {losses['total_loss']:.6f}")
                tqdm.write(f"  Feature Loss: {losses['feature_loss']:.6f}")
                tqdm.write(f"  Consistency Loss: {losses['consistency_loss']:.6f}")
                tqdm.write(f"  Magnitude Loss: {losses['magnitude_loss']:.6f}")
                tqdm.write(f"  Flow Magnitude: {losses['flow_magnitude']:.6f}")
                tqdm.write(f"  ROI Ratio: {losses['roi_ratio']:.3f}")
                
            # Generate visualizations more frequently
            if epoch % self.config.steps_til_summary == 0 or epoch in [1, 2, 5]:
                tqdm.write(f"Generating visualization for epoch {epoch}...")
                self._visualize_results(epoch)
                
        print("Direct flow training completed!")
        
        # Plot training curves
        self._plot_training_curves(training_losses)
        
    def _plot_training_curves(self, training_losses: List[Dict[str, float]]):
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
        axes[0, 1].set_title('Feature Loss')
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
        plt.show()
        
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
    """Main function to run direct flow training"""
    config = DirectFlowTrainingConfig(
        num_epochs=500,
        learning_rate=1e-4,
        steps_til_summary=50,
        lambda_magnitude=1.0,
        lambda_consistency=5.0,
        gap_curriculum=True,
        min_frame_gap=1,
        max_frame_gap=10,
        vis_source_frame=0,
        vis_target_frame=8
    )
    
    trainer = DirectFlowFieldTrainer(config)
    trainer.train()
    trainer.save_model(f"{config.log_dir}/checkpoints/direct_flow_model_final.pth")


if __name__ == "__main__":
    main() 