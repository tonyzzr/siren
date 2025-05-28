'''
Modularized Dense Flow Field Trainer

This module provides a clean, object-oriented interface for training dense optical flow fields
using neural implicit representations. The trainer learns flow by enforcing feature consistency
between warped and target frames using pre-trained feature networks.
'''

import sys
import os
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

# Add parent directory to path for imports
WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(WORK_DIR)

import modules
import dataio
import loss_functions
from modules import get_subdict


@dataclass
class FlowTrainingConfig:
    """Configuration for dense flow field training"""
    # Model parameters
    flow_hidden_features: int = 256
    flow_num_layers: int = 2
    flow_activation: str = "sine"
    
    # Training parameters
    num_epochs: int = 1000
    learning_rate: float = 1e-4
    batch_size: int = 1
    sample_fraction: float = 38e-4
    
    # Loss parameters
    lambda_magnitude: float = 10
    lambda_flow_smoothness: float = 0.01
    lambda_roi_coverage: float = 1.0
    
    # Logging parameters
    steps_til_summary: int = 100
    log_dir: str = f'{WORK_DIR}/logs/dense_flow_field_test'
    
    # Visualization parameters
    vis_canonical_frame_idx: int = 0 # Source for visualization is the canonical frame
    vis_target_frame_to_plot: int = 1 # Target frame for visualization against canonical
    max_frame_gap: int = 10  # Safety limit for frame gap (mostly for _accumulate_flow_between_frames if used for arbitrary pairs)
    
    # Data parameters
    video_data_path: str = f'{WORK_DIR}/data/mock_videos/mockvideo_vid_data.npy'
    feature_model_path: str = f'{WORK_DIR}/logs/mock_video_feature_test/checkpoints/model_final.pth'
    canonical_frame_idx: int = 0 # The reference frame (e.g., frame 0)
    
    # Feature model parameters
    feature_out_features: int = 384
    feature_hidden_features: int = 1024
    feature_num_layers: int = 3


class DenseFlowFieldTrainer:
    """
    Trainer for learning dense optical flow fields using neural implicit representations.
    
    The trainer uses a self-supervised approach where flow is learned by enforcing
    feature consistency between source and warped target coordinates using a pre-trained
    feature network.
    """
    
    def __init__(self, config: FlowTrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.canonical_frame_time_coord: Optional[float] = None # Will be set in _setup_data
        
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
        self.pixel_coords = dataio.get_mgrid(self.vid_dataset.shape, dim=3) # shape: (num_pixels_total, 3)
        self.pixel_coords = self.pixel_coords.view(
            self.vid_dataset.shape[0], # num_frames
            self.vid_dataset.shape[1], # height
            self.vid_dataset.shape[2], # width
            3 # t, x, y
        )
        
        # Store the normalized time coordinate of the canonical frame
        # get_mgrid normalizes coords to [-1, 1]. So frame 0's time is pixel_coords[0,0,0,0]
        if self.config.canonical_frame_idx >= self.vid_dataset.shape[0]:
            raise ValueError(f"canonical_frame_idx {self.config.canonical_frame_idx} is out of bounds for video with {self.vid_dataset.shape[0]} frames.")
        self.canonical_frame_time_coord = self.pixel_coords[self.config.canonical_frame_idx, 0, 0, 0].item()
        print(f"Canonical frame index: {self.config.canonical_frame_idx}, Canonical frame time coordinate: {self.canonical_frame_time_coord}")

        # Calculate coordinate deltas (dt might be less relevant now for direct flow calculation)
        self.dt = self.pixel_coords[1, 0, 0, 0] - self.pixel_coords[0, 0, 0, 0] if self.vid_dataset.shape[0] > 1 else torch.tensor(0.0)
        self.dx = self.pixel_coords[0, 1, 0, 1] - self.pixel_coords[0, 0, 0, 1] if self.vid_dataset.shape[1] > 1 else torch.tensor(0.0)
        self.dy = self.pixel_coords[0, 0, 1, 2] - self.pixel_coords[0, 0, 0, 2]
        
        # Calculate coordinate bounds
        self.bounds = self._calculate_coordinate_bounds()
        
        # Setup dataloader
        coord_dataset = dataio.Implicit3DWrapper(
            self.vid_dataset,
            sidelength=self.vid_dataset.shape,
            sample_fraction=self.config.sample_fraction
        )
        
        self.dataloader = DataLoader(
            coord_dataset,
            shuffle=True,
            batch_size=self.config.batch_size,
            pin_memory=True,
            num_workers=0
        )
        
        print(f"Coordinate deltas - dt: {self.dt}, dx: {self.dx}, dy: {self.dy}")
        print(f"Coordinate bounds: {self.bounds}")
        
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
        
        # Initialize flow field model
        self.flow_model = modules.SingleBVPNet(
            type=self.config.flow_activation,
            in_features=3,
            out_features=2,  # (dx, dy)
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
        
    def _compute_flow_loss(self, model_input: Dict, gt: Dict, epoch: int) -> Dict[str, torch.Tensor]:
        """
        Compute flow field loss using feature consistency, flow smoothness, and ROI coverage.
        """
        coords_for_flow_model = model_input['coords']
        # For flow smoothness, input to flow_model needs to track gradients
        if self.config.lambda_flow_smoothness > 0:
            coords_for_flow_model = model_input['coords'].clone().detach().requires_grad_(True)
        
        # Get flow prediction
        # Pass only coordinates and necessary idx to flow_model
        flow_model_input_dict = {
            'coords': coords_for_flow_model, 
            'idx': model_input['idx'],
            'preserve_coord_graph': True if self.config.lambda_flow_smoothness > 0 and coords_for_flow_model.requires_grad else False
        }
        flow_output = self.flow_model(flow_model_input_dict)
        flow_field = flow_output['model_out']  # [batch, N, 2] (dx, dy)
        
        # --- Feature Consistency Loss ---
        # Original model_input['coords'] should be used for defining source/target for feature loss
        coords_target_input_actual = model_input['coords'] 
        coords_source_spatial = coords_target_input_actual[:, :, 1:3]
        coords_source_temporal = torch.full_like(coords_target_input_actual[:, :, 0:1], self.canonical_frame_time_coord)
        coords_source_canonical = torch.cat([coords_source_temporal, coords_source_spatial], dim=-1)

        # Warped coordinates for querying target features
        coords_query_spatial_target = coords_source_spatial + flow_field # (x+dx, y+dy)
        coords_query_temporal_target = coords_target_input_actual[:, :, 0:1] # Temporal part is t_i
        coords_query_target = torch.cat([coords_query_temporal_target, coords_query_spatial_target], dim=-1)
        
        feat_input_source_canonical = {'coords': coords_source_canonical, 'idx': model_input['idx']}
        feat_input_query_target = {'coords': coords_query_target, 'idx': model_input['idx']}
        
        with torch.no_grad():
            feat_source_canonical = self.feature_model(feat_input_source_canonical)['model_out']
            
        feature_model_params = OrderedDict(self.feature_model.named_parameters())
        feat_query_target = self.feature_model.net(
            feat_input_query_target['coords'], 
            params=get_subdict(feature_model_params, 'net')
        )
        
        roi_mask = self._create_roi_mask(coords_query_target)
        feature_loss_val = loss_functions.image_mse(
            roi_mask, 
            {'model_out': feat_query_target},
            {'img': feat_source_canonical}
        )['img_loss']
        
        # --- Flow Magnitude Regularization ---
        flow_magnitude_sq_mean = torch.mean(flow_field ** 2) 
        magnitude_loss_val = flow_magnitude_sq_mean * self.config.lambda_magnitude
        
        # --- Flow Smoothness Loss (NEW) ---
        flow_smoothness_loss_val = torch.tensor(0.0, device=self.device)
        if self.config.lambda_flow_smoothness > 0 and coords_for_flow_model.requires_grad:
            dx_flow = flow_field[..., 0]
            dy_flow = flow_field[..., 1]

            # --- Inner DEBUG PRINT --- #
            if epoch % 10 == 0: 
                print(f"    DEBUG Pre-autograd: coords_for_flow_model.requires_grad = {coords_for_flow_model.requires_grad}")
            # --- END Inner DEBUG PRINT --- #

            grad_outputs_dx = torch.ones_like(dx_flow)
            grad_dx_wrt_inputs = torch.autograd.grad(
                outputs=dx_flow,
                inputs=coords_for_flow_model,
                grad_outputs=grad_outputs_dx,
                create_graph=True, 
                retain_graph=True,
                allow_unused=True
            )[0] 
            
            grad_outputs_dy = torch.ones_like(dy_flow)
            grad_dy_wrt_inputs = torch.autograd.grad(
                outputs=dy_flow, 
                inputs=coords_for_flow_model,
                grad_outputs=grad_outputs_dy,
                create_graph=True,
                allow_unused=True
            )[0]
            
            # --- DEBUG PRINT --- # 
            if epoch % 10 == 0: # Match print frequency of other gradient info
                print(f"  DEBUG Smoothness Grads at Epoch {epoch}:")
                print(f"    grad_dx_wrt_inputs is None: {grad_dx_wrt_inputs is None}")
                if grad_dx_wrt_inputs is not None:
                    print(f"    grad_dx_wrt_inputs.abs().mean(): {grad_dx_wrt_inputs.abs().mean().item()}")
                print(f"    grad_dy_wrt_inputs is None: {grad_dy_wrt_inputs is None}")
                if grad_dy_wrt_inputs is not None:
                    print(f"    grad_dy_wrt_inputs.abs().mean(): {grad_dy_wrt_inputs.abs().mean().item()}")
            # --- END DEBUG PRINT --- #
            
            # grad_dx_wrt_inputs and grad_dy_wrt_inputs might contain None if a component of coords_for_flow_model was unused.
            # We need to handle this before indexing.
            d_dx_dx = torch.zeros_like(dx_flow) if grad_dx_wrt_inputs is None else grad_dx_wrt_inputs[..., 1]
            d_dx_dy = torch.zeros_like(dx_flow) if grad_dx_wrt_inputs is None else grad_dx_wrt_inputs[..., 2]
            d_dy_dx = torch.zeros_like(dy_flow) if grad_dy_wrt_inputs is None else grad_dy_wrt_inputs[..., 1]
            d_dy_dy = torch.zeros_like(dy_flow) if grad_dy_wrt_inputs is None else grad_dy_wrt_inputs[..., 2]
            
            smoothness_penalty = torch.mean(d_dx_dx**2 + d_dx_dy**2 + d_dy_dx**2 + d_dy_dy**2)
            flow_smoothness_loss_val = smoothness_penalty * self.config.lambda_flow_smoothness
        
        # --- ROI Coverage Loss (NEW) ---
        # roi_mask is [batch, N, feature_dim]. For ratio, just need one feature dim.
        roi_ratio_val = roi_mask[:, :, 0].sum().float() / roi_mask[:, :, 0].numel()
        roi_coverage_loss_val = (1.0 - roi_ratio_val) * self.config.lambda_roi_coverage
        
        # --- Total Loss ---
        total_loss = (
            feature_loss_val + 
            magnitude_loss_val + 
            flow_smoothness_loss_val +
            roi_coverage_loss_val
        )
        
        return {
            'total_loss': total_loss,
            'feature_loss': feature_loss_val,
            'magnitude_loss': magnitude_loss_val,
            'flow_smoothness_loss': flow_smoothness_loss_val,
            'roi_coverage_loss': roi_coverage_loss_val,
            'flow_magnitude': torch.sqrt(flow_magnitude_sq_mean), # For logging
            'roi_ratio': roi_ratio_val
        }
        
    def _visualize_results(self, epoch_identifier: str, target_frame_to_visualize: int):
        """Generate visualization of training progress for a specific target frame."""
        print(f"Generating visualization for {epoch_identifier}, canonical to target {target_frame_to_visualize}...")
        
        self.flow_model.eval()
        self.feature_model.eval()
        
        # Source frame is always the canonical frame for this visualization
        canonical_frame_idx = self.config.vis_canonical_frame_idx 
        # Target frame is passed as an argument
        
        # Validate frame indices
        num_frames = self.vid_dataset.shape[0]
        if not (0 <= canonical_frame_idx < num_frames and 0 <= target_frame_to_visualize < num_frames):
            print(f"Warning: Frame indices (canonical: {canonical_frame_idx}, target: {target_frame_to_visualize}) out of range [0, {num_frames-1}]. Skipping this visualization.")
            # Restore model states and return if frames are invalid
            self.flow_model.train()
            return

        print(f"Visualizing flow from canonical frame {canonical_frame_idx} to target frame {target_frame_to_visualize}")
        H, W = self.vid_dataset.shape[1], self.vid_dataset.shape[2]

        with torch.no_grad():
            # Get coordinates for canonical and target frames (full frames for visualization)
            coords_canonical_flat = self.pixel_coords[canonical_frame_idx, ...].view(1, -1, 3).to(self.device)
            coords_target_flat = self.pixel_coords[target_frame_to_visualize, ...].view(1, -1, 3).to(self.device)
            
            # Get features for both frames using the feature model
            feat_out_canonical = self.feature_model({
                'coords': coords_canonical_flat, 
                'idx': torch.tensor([0]).to(self.device) # dummy batch idx
            })['model_out'].view(1, H, W, self.config.feature_out_features)
            
            feat_out_target = self.feature_model({
                'coords': coords_target_flat,
                'idx': torch.tensor([0]).to(self.device) # dummy batch idx
            })['model_out'].view(1, H, W, self.config.feature_out_features)
            
            # Get flow from canonical frame to target_frame_to_vis
            # The flow model input should be the coordinates of the *target* frame for which we want to predict flow from canonical
            flow_output_cano_to_target = self.flow_model({
                'coords': coords_target_flat, # (t_target, x, y)
                'idx': torch.tensor([0]).to(self.device)
            })
            flow_cano_to_target_2d = flow_output_cano_to_target['model_out'].view(1, H, W, 2)
            
            # Diagnostic print for raw flow magnitude
            avg_flow_l2_norm = torch.norm(flow_cano_to_target_2d, p=2, dim=-1).mean().item()
            avg_flow_abs_val = torch.mean(torch.abs(flow_cano_to_target_2d)).item()
            print(f"  Visualization raw flow stats (cano_idx={canonical_frame_idx}, target_idx={target_frame_to_visualize}): Avg L2 Norm: {avg_flow_l2_norm:.6f}, Avg Abs Val: {avg_flow_abs_val:.6f}")
            
            # Apply PCA for feature visualization (ensure train_feat_test.pca is available and correct)
            try:
                from train_feat_test import pca # Assuming this is in the PYTHONPATH and handles B,C,H,W format
                [feat_canonical_pca, feat_target_pca], _ = pca([
                    feat_out_canonical.permute(0, 3, 1, 2), # B, Feat, H, W
                    feat_out_target.permute(0, 3, 1, 2)   # B, Feat, H, W
                ])
            except ImportError:
                print("Warning: train_feat_test.pca could not be imported. Using first 3 feature channels for visualization.")
                feat_canonical_pca = feat_out_canonical[..., :3].permute(0, 3, 1, 2) 
                feat_target_pca = feat_out_target[..., :3].permute(0, 3, 1, 2)
            except Exception as e:
                print(f"Error during PCA: {e}. Using first 3 feature channels.")
                feat_canonical_pca = feat_out_canonical[..., :3].permute(0, 3, 1, 2) 
                feat_target_pca = feat_out_target[..., :3].permute(0, 3, 1, 2)

            # Warp canonical features using the predicted flow
            # Spatial coordinates of the canonical frame:
            canonical_spatial_coords_hw = self.pixel_coords[canonical_frame_idx, :, :, 1:3].to(self.device) # H, W, 2 (x,y)
            # Grid sample expects normalized coords [-1, 1]. Our get_mgrid output is already in this range.
            # The flow is an offset in this normalized coordinate space.
            grid_warped = canonical_spatial_coords_hw.unsqueeze(0) - flow_cano_to_target_2d # Changed + to -

            # grid_sample expects grid to be (N, H_out, W_out, 2) with (x,y) order and values in [-1,1]
            # Our canonical_spatial_coords_hw is (H,W,2) with (x,y) pixel_coords[...,1] is x, pixel_coords[...,2] is y.
            # The flow (dx,dy) is also in this space. So grid_warped is correct.
            warped_canonical_feat_pca = F.grid_sample(
                input=feat_canonical_pca, # B, C, H, W
                grid=grid_warped[..., [1,0]], # grid_warped is (y+dy, x+dx), F.grid_sample expects (x,y)
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True # Matches get_mgrid behavior if it was centered on pixels
            )
            
            # Compute error between warped canonical and actual target features
            error_pca = torch.abs(feat_target_pca - warped_canonical_feat_pca)
            
            # Visualize flow field
            flow_vis = self._visualize_flow_white_zero(flow_cano_to_target_2d[0, ...].permute(2, 0, 1)) # expects 2,H,W
            
        # Create 5-panel visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Identifier: {epoch_identifier} - Canonical Frame {canonical_frame_idx} to Target Frame {target_frame_to_visualize}", fontsize=16)
        
        # Panel 1: Canonical Frame Features (PCA)
        axes[0, 0].imshow(feat_canonical_pca[0, ...].permute(1, 2, 0).cpu().numpy())
        axes[0, 0].set_title(f"Canonical Frame {canonical_frame_idx} Features (PCA)")
        
        # Panel 2: Target Frame Features (PCA)
        axes[0, 1].imshow(feat_target_pca[0, ...].permute(1, 2, 0).cpu().numpy())
        axes[0, 1].set_title(f"Target Frame {target_frame_to_visualize} Features (PCA)")
        
        # Panel 3: Warped Canonical Features (PCA)
        axes[0, 2].imshow(warped_canonical_feat_pca[0, ...].permute(1, 2, 0).cpu().numpy())
        axes[0, 2].set_title(f"Warped Canonical (to Target {target_frame_to_visualize}) (PCA)")
        
        # Panel 4: Predicted Flow Field (Canonical -> Target)
        axes[1, 0].imshow(flow_vis)
        axes[1, 0].set_title(f"Predicted Flow ({canonical_frame_idx} -> {target_frame_to_visualize})")
        
        # Panel 5: Error (Absolute Difference in PCA space)
        axes[1, 1].imshow(error_pca[0, ...].permute(1, 2, 0).cpu().numpy(), cmap='gray') # Assuming error is best viewed in grayscale
        axes[1, 1].set_title("Warping Error (PCA Features)")

        axes[1, 2].axis('off') # Empty subplot
        
        for ax in axes.flatten():
            ax.axis('off')
            
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
        save_path = f'{self.config.log_dir}/checkpoints/vis_{epoch_identifier}_cano{canonical_frame_idx}_to_target{target_frame_to_visualize}.png'
        plt.savefig(save_path, dpi=150)
        print(f"Visualization saved to {save_path}")
        # plt.show()
        
        self.flow_model.train()
        
    def _visualize_flow_white_zero(self, flow: torch.Tensor) -> np.ndarray:
        """
        Visualize optical flow with white background for zero flow (traditional style)
        
        Args:
            flow: Flow field tensor [2, height, width]
            
        Returns:
            rgb: RGB image with white background for zero flow
        """
        # Convert flow to polar coordinates (magnitude and angle)
        u = flow[0].cpu().numpy()
        v = flow[1].cpu().numpy()
        
        # Calculate magnitude and angle
        magnitude = np.sqrt(u**2 + v**2)
        angle = np.arctan2(v, u)
        
        # Normalize magnitude to [0, 1]
        if magnitude.max() > 0:
            magnitude_normalized = magnitude / magnitude.max()
        else:
            magnitude_normalized = magnitude
        
        # Convert to HSV (hue is angle, saturation is magnitude, value is 1)
        h = (angle + np.pi) / (2 * np.pi)  # Map [-pi, pi] to [0, 1]
        s = magnitude_normalized  # Saturation based on magnitude (zero flow = white)
        v = np.ones_like(h)  # Full brightness
        
        # Stack HSV channels
        hsv = np.stack([h, s, v], axis=2)
        
        # Convert HSV to RGB using matplotlib
        from matplotlib.colors import hsv_to_rgb
        rgb = hsv_to_rgb(hsv)
        
        return rgb
        
    def _accumulate_flow_between_frames(self, source_frame: int, target_frame: int) -> torch.Tensor:
        """
        Accumulate flow field from source frame to target frame
        
        Args:
            source_frame: Starting frame index
            target_frame: Ending frame index
            
        Returns:
            accumulated_flow: Flow field [1, H*W, 2] from source to target
        """
        # Validate frame indices
        num_frames = self.vid_dataset.shape[0]
        H, W = self.vid_dataset.shape[1], self.vid_dataset.shape[2]

        if not (0 <= source_frame < num_frames and 0 <= target_frame < num_frames):
            raise ValueError(f"Frame indices (source: {source_frame}, target: {target_frame}) out of range [0, {num_frames-1}].")
        
        # max_frame_gap check is less about computational cost now, but can be kept as a semantic check
        if abs(target_frame - source_frame) > self.config.max_frame_gap:
            print(f"Warning: Frame gap {abs(target_frame - source_frame)} exceeds max_frame_gap {self.config.max_frame_gap}. Proceeding anyway.")

        # If source and target are the same, flow is zero
        if source_frame == target_frame:
            return torch.zeros(1, H * W, 2).to(self.device)

        self.flow_model.eval() # Ensure model is in eval mode for this utility

        with torch.no_grad():
            # Coordinates for the source frame (used as input to flow model to get flow from canonical to source)
            coords_source_frame_flat = self.pixel_coords[source_frame, ...].view(1, -1, 3).to(self.device)
            # Coordinates for the target frame (used as input to flow model to get flow from canonical to target)
            coords_target_frame_flat = self.pixel_coords[target_frame, ...].view(1, -1, 3).to(self.device)

            # Predict flow from canonical to source_frame
            # The input to flow_model is (t_source, x, y) and it predicts flow from (t_canonical, x, y) to (t_source, x+dx, y+dy)
            flow_out_cano_to_source = self.flow_model({
                'coords': coords_source_frame_flat,
                'idx': torch.tensor([0]).to(self.device) # dummy batch idx
            })['model_out'] # Shape: [1, H*W, 2]

            # Predict flow from canonical to target_frame
            flow_out_cano_to_target = self.flow_model({
                'coords': coords_target_frame_flat,
                'idx': torch.tensor([0]).to(self.device) # dummy batch idx
            })['model_out'] # Shape: [1, H*W, 2]

            # Flow(A -> B) = Flow(Canonical -> B) - Flow(Canonical -> A)
            accumulated_flow = flow_out_cano_to_target - flow_out_cano_to_source
        
        # self.flow_model.train() # Revert to train mode if changed - usually not needed here as test_flow_accumulation calls it separately
        return accumulated_flow
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.flow_model.train()
        
        # This dictionary will store the scalar loss values for returning and logging
        epoch_metric_values = {
            'total_loss': 0.0, 'feature_loss': 0.0, 'magnitude_loss': 0.0,
            'flow_smoothness_loss': 0.0, 'roi_coverage_loss': 0.0, 'flow_magnitude': 0.0,
            'roi_ratio': 0.0
        }
        
        # Assuming dataloader yields one batch that effectively constitutes an epoch's training data,
        # based on logs showing "Epoch X, Batch 0:" followed by "Epoch X/1000:".
        try:
            model_input, gt = next(iter(self.dataloader))
        except StopIteration: # Should not happen with a persistent DataLoader unless dataset is empty
            print(f"Warning: DataLoader was empty for epoch {epoch}. Skipping training for this epoch.")
            return epoch_metric_values # Return zeroed metrics
        
        # import pdb; pdb.set_trace()

        model_input = {k: v.to(self.device) for k, v in model_input.items()}
        gt = {k: v.to(self.device) for k, v in gt.items()} 
            
        self.optimizer.zero_grad()
        
        # Compute loss (returns a dict of tensors)
        computed_losses = self._compute_flow_loss(model_input, gt, epoch)
        
        # Backward pass on the total loss for this batch/epoch
        computed_losses['total_loss'].backward() 

        # --- GRADIENT PRINTING ---
        if epoch % 10 == 0: # Print for every 10th epoch
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"--- Gradients for flow_model at Epoch {epoch} (LR: {current_lr:.1e}) ---")
            total_params_count = 0
            total_abs_grad_sum = 0.0
            max_abs_grad_overall = 0.0
            min_mean_abs_grad_layer = float('inf') # Min of the mean absolute gradients per layer
            any_grad_found = False

            for name, param in self.flow_model.named_parameters():
                if param.grad is not None:
                    any_grad_found = True
                    grad_abs = torch.abs(param.grad)
                    mean_abs_grad_layer = torch.mean(grad_abs).item()
                    max_abs_grad_layer = torch.max(grad_abs).item()
                    
                    total_params_count += param.numel()
                    total_abs_grad_sum += torch.sum(grad_abs).item() # Sum of all absolute gradient values
                    
                    if max_abs_grad_layer > max_abs_grad_overall:
                        max_abs_grad_overall = max_abs_grad_layer
                    if mean_abs_grad_layer > 0 and mean_abs_grad_layer < min_mean_abs_grad_layer:
                        min_mean_abs_grad_layer = mean_abs_grad_layer
                        
                    print(f"  Param: {name:<50} Shape: {str(param.shape):<20} Grad Mean Abs: {mean_abs_grad_layer:.3e}, Grad Max Abs: {max_abs_grad_layer:.3e}")
                else:
                    print(f"  Param: {name:<50} Shape: {str(param.shape):<20} Grad: None")
            
            if any_grad_found and total_params_count > 0:
                overall_mean_abs_grad = total_abs_grad_sum / total_params_count
                print(f"  STATS: Overall Mean Abs Grad: {overall_mean_abs_grad:.3e}, Overall Max Abs Grad: {max_abs_grad_overall:.3e}", end="")
                if min_mean_abs_grad_layer != float('inf'):
                    print(f", Min Layer Mean Abs Grad: {min_mean_abs_grad_layer:.3e}")
                else:
                    print("") # Newline
            elif not any_grad_found:
                 print("  All parameter gradients are None.")
            print("--- End Gradients ---")
        # --- END GRADIENT PRINTING ---
            
        self.optimizer.step()
            
        # Populate epoch_metric_values from the computed_losses (converting tensors to scalars)
        for key_metric in epoch_metric_values.keys():
            if key_metric in computed_losses and hasattr(computed_losses[key_metric], 'item'):
                epoch_metric_values[key_metric] = computed_losses[key_metric].item()
        
        # This print provides per-epoch feedback as num_batches is 1. batch_idx is effectively 0.
        print(f"Epoch {epoch}, Batch 0 (metrics after step):")
        print(f"  Coords shape: {model_input['coords'].shape}")
        print(f"  GT shape from dataloader: {gt['img'].shape}") # gt['img'] is used by original loss_functions.image_mse
        print(f"  Total Loss: {epoch_metric_values['total_loss']:.6f}")
        print(f"  Feature loss: {epoch_metric_values['feature_loss']:.6f}")
        print(f"  Magnitude loss: {epoch_metric_values['magnitude_loss']:.6f}")
        print(f"  Flow smoothness loss: {epoch_metric_values['flow_smoothness_loss']:.6f}")
        print(f"  ROI coverage loss: {epoch_metric_values['roi_coverage_loss']:.6f}")
        print(f"  Flow magnitude: {epoch_metric_values['flow_magnitude']:.6f}")
        print(f"  ROI ratio: {epoch_metric_values['roi_ratio']:.3f}")
        
        return epoch_metric_values
        
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config.num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Flow model parameters: {sum(p.numel() for p in self.flow_model.parameters())}")
        
        for epoch in range(self.config.num_epochs):
            # Train one epoch
            losses = self.train_epoch(epoch)
            
            # Log progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{self.config.num_epochs}:")
                print(f"  Total Loss: {losses['total_loss']:.6f}")
                print(f"  Feature Loss: {losses['feature_loss']:.6f}")
                print(f"  Magnitude Loss: {losses['magnitude_loss']:.6f}")
                print(f"  Flow smoothness loss: {losses['flow_smoothness_loss']:.6f}")
                print(f"  ROI coverage loss: {losses['roi_coverage_loss']:.6f}")
                print(f"  Flow Magnitude: {losses['flow_magnitude']:.6f}")
                print(f"  ROI Ratio: {losses['roi_ratio']:.3f}")
                
            # Generate visualizations
            if epoch % self.config.steps_til_summary == 0:
                self._visualize_results(str(epoch), self.config.vis_target_frame_to_plot)
                if epoch == 0: # Also run full diagnostics at epoch 0
                    self.generate_diagnostic_visualizations(identifier_prefix="epoch_0_diagnostic")

        print("Training completed!")
        # Generate final diagnostic visualizations
        self.generate_diagnostic_visualizations(identifier_prefix="final_diagnostic")
        
    def save_model(self, path: str):
        """Save trained flow model"""
        torch.save(self.flow_model.state_dict(), path)
        print(f"Model saved to {path}")
        
    def load_model(self, path: str):
        """Load trained flow model"""
        self.flow_model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
        
    def test_flow_accumulation(self, test_cases: list = None):
        """
        Test flow accumulation between different frame pairs
        
        Args:
            test_cases: List of (source_frame, target_frame) tuples to test
        """
        if test_cases is None:
            # Default test cases
            num_frames = min(self.vid_dataset.shape[0], 10)  # Limit to first 10 frames
            test_cases = [
                (0, 1),    # Adjacent frames
                (0, 2),    # Skip 1 frame
                (0, 5),    # Skip 4 frames
                (5, 0),    # Backward
                (3, 3),    # Same frame
            ]
            # Filter valid test cases
            test_cases = [(s, t) for s, t in test_cases if s < num_frames and t < num_frames]
        
        print("Testing flow accumulation...")
        self.flow_model.eval()
        
        with torch.no_grad():
            for source_frame, target_frame in test_cases:
                try:
                    flow = self._accumulate_flow_between_frames(source_frame, target_frame)
                    flow_magnitude = torch.sqrt(torch.sum(flow ** 2, dim=-1)).mean()
                    print(f"  Frame {source_frame} → {target_frame}: "
                          f"avg magnitude = {flow_magnitude:.4f}")
                except Exception as e:
                    print(f"  Frame {source_frame} → {target_frame}: ERROR - {e}")
        
        self.flow_model.train()

    def generate_diagnostic_visualizations(self, identifier_prefix: str):
        """Generates visualizations for a set of predefined target frames across the video timeline."""
        print(f"\nGenerating diagnostic visualizations with prefix: {identifier_prefix}...")
        self.flow_model.eval() # Ensure models are in eval mode for diagnostics
        self.feature_model.eval()

        num_frames = self.vid_dataset.shape[0]
        if num_frames == 0:
            print("No frames in video dataset, skipping diagnostic visualizations.")
            return

        # Define proportions of video timeline to visualize
        proportions = [0.0, 0.25, 0.5, 0.75, 1.0]
        target_frames_to_visualize = []
        for p in proportions:
            frame_idx = int(round(p * (num_frames - 1))) # round to nearest int
            # Ensure frame_idx is within bounds, especially for num_frames=1
            frame_idx = max(0, min(frame_idx, num_frames - 1))
            if frame_idx not in target_frames_to_visualize:
                 target_frames_to_visualize.append(frame_idx)
        
        target_frames_to_visualize.sort() # Ensure they are processed in order

        print(f"Will generate diagnostics for target frames: {target_frames_to_visualize}")

        for target_frame in target_frames_to_visualize:
            full_identifier = f"{identifier_prefix}_target{target_frame}"
            self._visualize_results(epoch_identifier=full_identifier, target_frame_to_visualize=target_frame)
        
        print("Diagnostic visualizations complete.")
        # Restore model state if changed by _visualize_results, though it should handle its own state.
        self.flow_model.train() # Only flow_model needs to be restored
        # self.feature_model.train() # feature_model should stay in eval


def main():
    """Main function to run training with new features"""
    # Create configuration with custom visualization settings
    config = FlowTrainingConfig(
        flow_num_layers=2,
        flow_hidden_features=32,
        flow_activation="sine",# try if relu results in flatter flow field

        num_epochs=3000,
        learning_rate=1e-4,
        steps_til_summary=300,
        lambda_magnitude=1,
        lambda_flow_smoothness=10,
        lambda_roi_coverage=1.0,
        # New visualization parameters for canonical flow
        canonical_frame_idx=0,
        vis_target_frame_to_plot=5,
        max_frame_gap=10,

        sample_fraction=38e-4,
    )
    
    # Create trainer
    trainer = DenseFlowFieldTrainer(config)
    
    # Test flow accumulation before training
    print("Testing flow accumulation with untrained model:")
    trainer.test_flow_accumulation()
    
    # Start training
    trainer.train()
    
    # Test flow accumulation after training
    print("\nTesting flow accumulation with trained model:")
    trainer.test_flow_accumulation()
    
    # Save final model
    trainer.save_model(f"{config.log_dir}/checkpoints/flow_model_final.pth")


if __name__ == "__main__":
    main()
