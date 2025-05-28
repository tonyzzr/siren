"""
Test script for improved canonical flow trainer with comprehensive visualization
"""

import sys
import os

# Add parent directory to path
WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(WORK_DIR)

from train_canonical_flow_improved import ImprovedCanonicalFlowTrainer, ImprovedCanonicalFlowConfig


def test_improved_with_visualization():
    """Test the improved canonical flow trainer with comprehensive visualization"""
    print("🎯 Testing Improved Canonical Flow Trainer with Comprehensive Visualization")
    print("=" * 75)
    print("Features:")
    print("  ✓ Larger network capacity (128 hidden, 4 layers)")
    print("  ✓ Curriculum learning (adjacent frames → full temporal range)")
    print("  ✓ Coordinate scaling (10x) to encourage larger flows")
    print("  ✓ Progressive regularization (0.001 → 0.01)")
    print("  ✓ Temporal consistency loss")
    print("  ✓ Comprehensive visualization:")
    print("    • Source features (PCA)")
    print("    • Target features (Ground Truth)")
    print("    • Warped source features")
    print("    • Error visualization")
    print("    • Flow field analysis")
    print()
    
    # Configure for detailed test
    config = ImprovedCanonicalFlowConfig(
        num_epochs=60,  # Enough epochs to see progression
        learning_rate=1e-4,
        curriculum_epochs=20,  # First 20 epochs: curriculum learning
        lambda_magnitude_start=0.001,  # Very low initial regularization
        lambda_magnitude_end=0.01,     # Gradually increase
        coordinate_scale=10.0,  # Scale coordinates to encourage larger flows
        flow_hidden_features=128,  # Larger network
        flow_num_layers=4,
        sample_fraction=2e-3,  # More samples for better training
        vis_target_frames=[1, 3, 7, 15],
        log_dir=f'{WORK_DIR}/logs/improved_canonical_with_viz'
    )
    
    print(f"Configuration:")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Curriculum epochs: {config.curriculum_epochs}")
    print(f"  Network: {config.flow_num_layers} layers, {config.flow_hidden_features} hidden")
    print(f"  Coordinate scaling: {config.coordinate_scale}x")
    print(f"  Regularization: {config.lambda_magnitude_start} → {config.lambda_magnitude_end}")
    print(f"  Target frames: {config.vis_target_frames}")
    print(f"  Log directory: {config.log_dir}")
    print()
    
    # Create and train
    trainer = ImprovedCanonicalFlowTrainer(config)
    trainer.train()
    
    print("\n📊 Final Analysis:")
    print("=" * 30)
    
    # Test final predictions with detailed analysis
    test_frames = [1, 3, 7, 15]
    print("Final Flow Magnitudes:")
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
    
    # Calculate temporal progression
    flow_1 = trainer.predict_canonical_flow(1)
    flow_15 = trainer.predict_canonical_flow(15)
    mag_1 = flow_1.norm(dim=-1).mean().item()
    mag_15 = flow_15.norm(dim=-1).mean().item()
    progression_ratio = mag_15 / mag_1
    
    print(f"\nTemporal Progression:")
    print(f"  Frame 1 magnitude: {mag_1:.6f}")
    print(f"  Frame 15 magnitude: {mag_15:.6f}")
    print(f"  Progression ratio (15/1): {progression_ratio:.2f}")
    
    if progression_ratio > 2.0:
        print("  ✅ Excellent temporal progression!")
    elif progression_ratio > 1.5:
        print("  ✅ Good temporal progression!")
    elif progression_ratio > 1.2:
        print("  ⚠️  Moderate temporal progression")
    else:
        print("  ❌ Poor temporal progression")
    
    print(f"\n✅ Improved canonical flow training with visualization completed!")
    print(f"📁 Check comprehensive visualizations in: {config.log_dir}/checkpoints/")
    print(f"   • epoch_X_improved_features.png (4-row layout: source, target, warped, error)")
    print(f"   • epoch_X_improved_flow_analysis.png (flow fields and magnitude heatmaps)")
    print(f"   • Visualizations generated every 20 epochs")
    
    # Save final model
    trainer.save_model(f"{config.log_dir}/checkpoints/improved_canonical_flow_final.pth")


if __name__ == "__main__":
    test_improved_with_visualization() 