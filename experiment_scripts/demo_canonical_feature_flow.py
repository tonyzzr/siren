"""
Demo script for Canonical Flow Field Training with Feature Consistency
"""

import sys
import os

# Add parent directory to path
WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(WORK_DIR)

from train_canonical_flow_field_features import CanonicalFlowFeatureTrainer, CanonicalFlowFeatureConfig


def demo_canonical_feature_flow():
    """Demo canonical flow training with feature consistency"""
    print("🎯 Canonical Flow Training with Feature Consistency")
    print("=" * 60)
    print("Key features:")
    print("  ✓ Frame 0 as canonical source")
    print("  ✓ Feature consistency loss (same as modularized version)")
    print("  ✓ Pre-trained feature network")
    print("  ✓ Tunable flow magnitude regularization")
    print("  ✓ Direct prediction (x, y, t_target) → (dx, dy)")
    print()
    
    # Configure training
    config = CanonicalFlowFeatureConfig(
        num_epochs=50,  # Quick demo
        learning_rate=1e-4,
        steps_til_summary=10,
        lambda_magnitude=1.0,  # Balanced regularization
        flow_hidden_features=64,
        flow_num_layers=3,
        vis_target_frames=[1, 3, 7, 15],
        sample_fraction=5e-4,  # Sample 0.05% of pixels per frame
        log_dir=f'{WORK_DIR}/logs/canonical_feature_flow_demo'
    )
    
    print(f"Configuration:")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  λ_magnitude: {config.lambda_magnitude}")
    print(f"  Sample fraction: {config.sample_fraction}")
    print(f"  Visualization frames: {config.vis_target_frames}")
    print(f"  Log directory: {config.log_dir}")
    print()
    
    # Create and train
    trainer = CanonicalFlowFeatureTrainer(config)
    trainer.train()
    
    # Save model
    trainer.save_model(f"{config.log_dir}/checkpoints/canonical_feature_flow_final.pth")
    
    # Test predictions
    print("\n📊 Testing Flow Predictions:")
    print("=" * 40)
    
    test_frames = [1, 3, 7, 15, 20]
    for target_frame in test_frames:
        if target_frame < trainer.vid_dataset.shape[0]:
            flow = trainer.predict_canonical_flow(target_frame)
            magnitude = flow.norm(dim=-1).mean()
            coord_ratio = magnitude / trainer.dx
            print(f"  Flow 0→{target_frame:2d}: magnitude = {magnitude:.6f} (coord ratio: {coord_ratio:.2f})")
    
    print(f"\n✅ Canonical feature flow training completed!")
    print(f"📁 Results saved in: {config.log_dir}")
    print(f"🎨 Visualizations show feature warping quality")


def test_regularization_comparison():
    """Test different regularization values"""
    print("\n🔬 Testing Regularization Effects")
    print("=" * 40)
    
    regularization_values = [0.1, 1.0, 10.0]
    
    for lambda_mag in regularization_values:
        print(f"\nTesting λ_magnitude = {lambda_mag}")
        
        config = CanonicalFlowFeatureConfig(
            num_epochs=20,  # Quick test
            lambda_magnitude=lambda_mag,
            steps_til_summary=1000,  # No visualization during test
            sample_fraction=1e-3,
            log_dir=f'{WORK_DIR}/logs/canonical_feature_reg_test_{lambda_mag}'
        )
        
        trainer = CanonicalFlowFeatureTrainer(config)
        
        # Train briefly
        for epoch in range(config.num_epochs):
            losses = trainer.train_epoch(epoch)
            if epoch == config.num_epochs - 1:  # Final epoch
                print(f"  Final flow magnitude: {losses['flow_magnitude']:.6f}")
                print(f"  Final feature loss: {losses['feature_loss']:.6f}")
                print(f"  ROI ratio: {losses['roi_ratio']:.3f}")


if __name__ == "__main__":
    demo_canonical_feature_flow()
    test_regularization_comparison() 