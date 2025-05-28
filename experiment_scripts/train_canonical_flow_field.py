'''
Canonical Flow Field Trainer

This module implements a simplified trainer that uses frame 0 as the canonical source
frame and learns flow fields to all other frames. This eliminates the complexity of
frame pair generation and focuses on the core learning objective.

Key simplifications:
1. Frame 0 is the only source frame
2. Train on entire video, no sampling
3. Simple image MSE loss + flow magnitude regularization
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
class CanonicalFlowConfig:
    """Configuration for canonical flow field training"""
    # Model parameters
    flow_hidden_features: int = 64
    flow_num_layers: int = 3
    flow_activation: str = "sine"
    
    # Training parameters
    num_epochs: int = 200
    learning_rate: float = 1e-4
    batch_size: int = 1
    
    # Loss parameters
    lambda_magnitude: float = 1.0  # Tunable flow magnitude regularization
    
    # Logging parameters
    steps_til_summary: int = 20
    log_dir: str = f'{WORK_DIR}/logs/canonical_flow_field'
    
    # Visualization parameters
    vis_target_frames: list = None  # Will default to [1, 5, 10, 15]
    
    # Data parameters
    video_data_path: str = f'{WORK_DIR}/data/mock_videos/mockvideo_vid_data.npy'
    
    def __post_init__(self):
        if self.vis_target_frames is None:
            self.vis_target_frames = [1, 5, 10, 15]


class CanonicalFlowDataset(Dataset):
    """
    Simple dataset that returns all pixels for all frames (except frame 0)
    """
    
    def __init__(self, vid_dataset, pixel_coords):
        self.vid_dataset = vid_dataset
        self.pixel_coords = pixel_coords
        self.num_frames = vid_dataset.shape[0]
        
        # We train on frames 1 to N-1 (frame 0 is canonical source)
        self.target_frames = list(range(1, self.num_frames))
        
    def __len__(self):
        return len(self.target_frames)
    
    def __getitem__(self, idx):
        target_frame = self.target_frames[idx]
        
        # Get all coordinates for frame 0 (source) and target frame
        source_coords = self.pixel_coords[0].view(-1, 3)  # [H*W, 3]
        target_coords = self.pixel_coords[target_frame].view(-1, 3)  # [H*W, 3]
        
        # Create input coordinates: (x, y, target_frame_time)
        # We use target frame time as the "query time"
        input_coords = torch.cat([
            source_coords[:, 1:3],  # x, y from frame 0
            target_coords[:, 0:1],  # t_target
        ], dim=1)  # [H*W, 3]
        
        # Ground truth images - access frames directly from vid attribute
        source_img = torch.from_numpy(self.vid_dataset.vid[0]).view(-1, 1).float()  # [H*W, 1]
        target_img = torch.from_numpy(self.vid_dataset.vid[target_frame]).view(-1, 1).float()  # [H*W, 1]
        
        return {
            'input_coords': input_coords,
            'source_coords': source_coords,
            'target_coords': target_coords,
            'source_img': source_img,
            'target_img': target_img,
            'target_frame': target_frame
        }


class CanonicalFlowFieldTrainer:
    """
    Simplified trainer for learning flow fields from frame 0 to all other frames.
    
    The network learns to map from (x, y, t_target) to (dx, dy) where:
    - (x, y) are coordinates in frame 0
    - t_target is the time of the target frame
    - (dx, dy) is the flow from frame 0 to target frame
    """
    
    def __init__(self, config: CanonicalFlowConfig):
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
        
        # Setup dataset and dataloader
        self.dataset = CanonicalFlowDataset(self.vid_dataset, self.pixel_coords)
        self.dataloader = DataLoader(
            self.dataset,
            shuffle=True,
            batch_size=self.config.batch_size,
            pin_memory=True,
            num_workers=0
        )
        
        print(f"Coordinate deltas - dt: {self.dt}, dx: {self.dx}, dy: {self.dy}")
        print(f"Training on {len(self.dataset)} target frames (1 to {self.vid_dataset.shape[0]-1})")
        
    def _setup_models(self):
        """Setup flow field model"""
        print("Setting up models...")
        
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
        
    def _compute_canonical_flow_loss(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Compute canonical flow field loss using image MSE
        
        Args:
            batch: Batch data from CanonicalFlowDataset
            
        Returns:
            Dictionary containing loss components
        """
        # Move batch to device
        input_coords = batch['input_coords'].to(self.device)
        source_coords = batch['source_coords'].to(self.device)
        source_img = batch['source_img'].to(self.device)
        target_img = batch['target_img'].to(self.device)
        
        # Predict flow from frame 0 to target frame
        flow_output = self.flow_model({'coords': input_coords, 'idx': torch.tensor([0]).to(self.device)})
        predicted_flow = flow_output['model_out']  # [batch, H*W, 2] (dx, dy)
        
        # Compute warped coordinates
        warped_coords = source_coords.clone()
        warped_coords[:, :, 1:3] += predicted_flow  # Update x, y coordinates
        
        # Sample warped image using grid_sample
        h, w = self.vid_dataset.shape[1], self.vid_dataset.shape[2]
        
        # Reshape for grid_sample
        source_img_2d = source_img.view(1, 1, h, w)  # [1, 1, H, W]
        warped_coords_2d = warped_coords[:, :, 1:3].view(1, h, w, 2)  # [1, H, W, 2]
        
        # Grid sample expects coordinates in [-1, 1] range
        # Our coordinates are already normalized, but we need to swap x,y to y,x for grid_sample
        grid = torch.cat([
            warped_coords_2d[:, :, :, 1:2],  # y coordinate
            warped_coords_2d[:, :, :, 0:1],  # x coordinate
        ], dim=-1)
        
        # Sample warped image
        warped_img_2d = F.grid_sample(
            input=source_img_2d,
            grid=grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        
        warped_img = warped_img_2d.view(1, -1, 1)  # [1, H*W, 1] to match target_img shape
        
        # Image MSE loss
        image_loss = F.mse_loss(warped_img, target_img)
        
        # Flow magnitude regularization
        flow_magnitude = torch.sqrt(torch.mean(predicted_flow ** 2))
        magnitude_loss = flow_magnitude ** 2 * self.config.lambda_magnitude
        
        # Total loss
        total_loss = image_loss + magnitude_loss
        
        return {
            'total_loss': total_loss,
            'image_loss': image_loss,
            'magnitude_loss': magnitude_loss,
            'flow_magnitude': flow_magnitude
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
        
        # Filter valid target frames
        valid_targets = [f for f in self.config.vis_target_frames 
                        if f < self.vid_dataset.shape[0]]
        
        if not valid_targets:
            print("No valid target frames for visualization")
            return
            
        num_targets = len(valid_targets)
        fig, axes = plt.subplots(3, num_targets, figsize=(4*num_targets, 12))
        
        if num_targets == 1:
            axes = axes.reshape(-1, 1)
        
        with torch.no_grad():
            for i, target_frame in enumerate(valid_targets):
                # Get flow prediction
                flow_field = self.predict_canonical_flow(target_frame)
                
                # Get images
                source_img = self.vid_dataset.vid[0]
                target_img = self.vid_dataset.vid[target_frame]
                
                # Warp source image
                warped_img = self._warp_image_with_flow(source_img, flow_field)
                
                # Visualize flow field
                flow_vis = self._visualize_flow_white_zero(flow_field.permute(2, 0, 1))
                
                # Plot
                axes[0, i].imshow(source_img, cmap='gray')
                axes[0, i].set_title(f"Frame 0 (Source)")
                axes[0, i].axis('off')
                
                axes[1, i].imshow(warped_img, cmap='gray')
                axes[1, i].set_title(f"Frame 0 → {target_frame} (Warped)")
                axes[1, i].axis('off')
                
                axes[2, i].imshow(flow_vis)
                axes[2, i].set_title(f"Flow 0→{target_frame}")
                axes[2, i].axis('off')
                
        plt.tight_layout()
        plt.savefig(f'{self.config.log_dir}/checkpoints/epoch_{epoch}_canonical_flow.png', dpi=150)
        plt.show()
        
        self.flow_model.train()
        
    def _warp_image_with_flow(self, source_img: np.ndarray, flow_field: torch.Tensor) -> np.ndarray:
        """Warp source image using flow field"""
        # Handle different image shapes
        if len(source_img.shape) == 3:
            # If image has channels, take first channel or convert to grayscale
            if source_img.shape[2] == 3:  # RGB
                source_img = np.mean(source_img, axis=2)  # Convert to grayscale
            else:
                source_img = source_img[:, :, 0]  # Take first channel
        
        h, w = source_img.shape
        
        # Create coordinate grid
        y, x = np.mgrid[0:h, 0:w]
        coords = np.stack([x, y], axis=-1).astype(np.float32)
        
        # Add flow to coordinates
        flow_np = flow_field.cpu().numpy()
        warped_coords = coords + flow_np
        
        # Normalize coordinates to [-1, 1] for grid_sample
        warped_coords[:, :, 0] = 2 * warped_coords[:, :, 0] / (w - 1) - 1  # x
        warped_coords[:, :, 1] = 2 * warped_coords[:, :, 1] / (h - 1) - 1  # y
        
        # Convert to torch tensors
        source_tensor = torch.from_numpy(source_img).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        grid = torch.from_numpy(warped_coords).unsqueeze(0)  # [1, H, W, 2]
        
        # Swap x,y for grid_sample
        grid = torch.cat([grid[:, :, :, 1:2], grid[:, :, :, 0:1]], dim=-1)
        
        # Warp image
        warped_tensor = F.grid_sample(
            source_tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=True
        )
        
        return warped_tensor[0, 0].numpy()
        
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
            'image_loss': 0.0,
            'magnitude_loss': 0.0,
            'flow_magnitude': 0.0
        }
        
        num_batches = 0
        
        for batch in self.dataloader:
            # Compute loss
            losses = self._compute_canonical_flow_loss(batch)
            
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
        print(f"Starting canonical flow training for {self.config.num_epochs} epochs...")
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
                tqdm.write(f"  Image Loss: {losses['image_loss']:.6f}")
                tqdm.write(f"  Magnitude Loss: {losses['magnitude_loss']:.6f}")
                tqdm.write(f"  Flow Magnitude: {losses['flow_magnitude']:.6f}")
                
            # Generate visualizations
            if epoch % self.config.steps_til_summary == 0 or epoch in [1, 2, 5]:
                tqdm.write(f"Generating visualization for epoch {epoch}...")
                self._visualize_results(epoch)
                
        print("Canonical flow training completed!")
        
        # Plot training curves
        self._plot_training_curves(training_losses)
        
    def _plot_training_curves(self, training_losses):
        """Plot training loss curves"""
        epochs = range(len(training_losses))
        
        # Extract loss components
        total_losses = [loss['total_loss'] for loss in training_losses]
        image_losses = [loss['image_loss'] for loss in training_losses]
        magnitude_losses = [loss['magnitude_loss'] for loss in training_losses]
        flow_magnitudes = [loss['flow_magnitude'] for loss in training_losses]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0, 0].plot(epochs, total_losses)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(epochs, image_losses)
        axes[0, 1].set_title('Image MSE Loss')
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
    """Main function to run canonical flow training"""
    config = CanonicalFlowConfig(
        num_epochs=100,
        learning_rate=1e-4,
        steps_til_summary=10,
        lambda_magnitude=1.0,  # Tunable regularization
        flow_hidden_features=64,
        flow_num_layers=3,
        vis_target_frames=[1, 3, 7, 15]
    )
    
    trainer = CanonicalFlowFieldTrainer(config)
    trainer.train()
    trainer.save_model(f"{config.log_dir}/checkpoints/canonical_flow_model_final.pth")


if __name__ == "__main__":
    main() 