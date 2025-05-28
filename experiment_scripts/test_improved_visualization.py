"""
Test script for improved canonical flow visualization with ground truth and error analysis
"""

import sys
import os

# Add parent directory to path
WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(WORK_DIR)

from train_canonical_flow_field_features import CanonicalFlowFeatureTrainer, CanonicalFlowFeatureConfig


def test_improved_visualization():
    """Test the improved visualization with ground truth and error analysis"""
    print("🔍 Testing Improved Canonical Flow Visualization")
    print("=" * 55)
    print("New features:")
    print("  ✓ Ground truth target frame features")
    print("  ✓ Error visualization between warped and target")
    print("  ✓ Flow field analysis with statistics")
    print("  ✓ Saves plots to log folder (no plt.show() interruption)")
    print("  ✓ Detailed flow magnitude statistics")
    print()
    
    # Configure for quick test
    config = CanonicalFlowFeatureConfig(
        num_epochs=20,  # Short test
        learning_rate=1e-4,
        steps_til_summary=5,  # Visualize every 5 epochs
        lambda_magnitude=1.0,
        flow_hidden_features=64,
        flow_num_layers=3,
        vis_target_frames=[1, 3, 7, 15],
        sample_fraction=1e-3,  # Sample more pixels for better training
        log_dir=f'{WORK_DIR}/logs/canonical_flow_improved_viz'
    )
    
    print(f"Configuration:")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Visualization every: {config.steps_til_summary} epochs")
    print(f"  Target frames: {config.vis_target_frames}")
    print(f"  Log directory: {config.log_dir}")
    print()
    
    # Create and train
    trainer = CanonicalFlowFeatureTrainer(config)
    trainer.train()
    
    print("\n📊 Final Flow Analysis:")
    print("=" * 30)
    
    # Test final predictions with detailed analysis
    test_frames = [1, 3, 7, 15]
    for target_frame in test_frames:
        if target_frame < trainer.vid_dataset.shape[0]:
            flow = trainer.predict_canonical_flow(target_frame)
            magnitude = flow.norm(dim=-1)
            mean_mag = magnitude.mean().item()
            max_mag = magnitude.max().item()
            std_mag = magnitude.std().item()
            coord_ratio = mean_mag / trainer.dx
            
            print(f"  Flow 0→{target_frame:2d}: mean={mean_mag:.6f}, max={max_mag:.6f}, "
                  f"std={std_mag:.6f}, coord_ratio={coord_ratio:.2f}")
    
    print(f"\n✅ Improved visualization test completed!")
    print(f"📁 Check visualizations in: {config.log_dir}/checkpoints/")
    print(f"   • epoch_X_canonical_features.png (4-row layout with GT and error)")
    print(f"   • epoch_X_flow_analysis.png (flow fields and magnitude heatmaps)")
    print(f"   • training_curves.png (loss curves)")


if __name__ == "__main__":
    test_improved_visualization() 