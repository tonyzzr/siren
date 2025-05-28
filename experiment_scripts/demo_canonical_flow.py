"""
Demo script for Canonical Flow Field Training
"""

import sys
import os

# Add parent directory to path
WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(WORK_DIR)

from train_canonical_flow_field import CanonicalFlowFieldTrainer, CanonicalFlowConfig


def demo_canonical_flow():
    """Demo canonical flow training with different regularization settings"""
    print("🎯 Canonical Flow Training Demo")
    print("=" * 50)
    print("Key features:")
    print("  ✓ Frame 0 as canonical source")
    print("  ✓ Train on entire video")
    print("  ✓ Simple image MSE loss")
    print("  ✓ Tunable flow magnitude regularization")
    print()
    
    # Test with low regularization for visible flow
    config = CanonicalFlowConfig(
        # Fast training settings
        num_epochs=50,
        learning_rate=1e-4,
        
        # Frequent visualization
        steps_til_summary=10,
        
        # Model settings
        flow_hidden_features=64,
        flow_num_layers=3,
        
        # Low regularization for more visible flow
        lambda_magnitude=0.1,  # Much lower than default 1.0
        
        # Visualization frames
        vis_target_frames=[1, 3, 7, 15],
        
        log_dir=f'{WORK_DIR}/logs/canonical_flow_demo'
    )
    
    print(f"Configuration:")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Visualization every: {config.steps_til_summary} epochs")
    print(f"  Target frames: {config.vis_target_frames}")
    print(f"  Regularization: λ_magnitude = {config.lambda_magnitude}")
    print(f"  Model: {config.flow_hidden_features} hidden, {config.flow_num_layers} layers")
    print()
    
    # Create and train
    trainer = CanonicalFlowFieldTrainer(config)
    
    print("Starting training...")
    trainer.train()
    
    # Test predictions
    print("\n" + "=" * 50)
    print("Testing final predictions:")
    
    import torch
    
    test_frames = [1, 3, 7, 15]
    
    for target_frame in test_frames:
        if target_frame < trainer.vid_dataset.shape[0]:
            flow = trainer.predict_canonical_flow(target_frame)
            magnitude = torch.sqrt(torch.sum(flow ** 2, dim=-1)).mean()
            
            # Compare to coordinate scale
            coord_ratio = magnitude / trainer.dx
            
            print(f"  Flow 0→{target_frame}: magnitude = {magnitude:.6f} (ratio to coord: {coord_ratio:.2f})")
    
    # Save model
    model_path = f"{config.log_dir}/checkpoints/canonical_model.pth"
    trainer.save_model(model_path)
    
    print(f"\n✅ Demo completed! Model saved to {model_path}")
    return trainer


def test_regularization_effects():
    """Test different regularization values"""
    print("\n🔬 Testing Regularization Effects")
    print("=" * 50)
    
    import torch
    
    regularization_values = [0.01, 0.1, 1.0, 10.0]
    results = {}
    
    for lambda_mag in regularization_values:
        print(f"\nTesting λ_magnitude = {lambda_mag}")
        
        config = CanonicalFlowConfig(
            num_epochs=20,  # Quick test
            lambda_magnitude=lambda_mag,
            steps_til_summary=1000,  # No visualization during test
            log_dir=f'{WORK_DIR}/logs/canonical_reg_test_{lambda_mag}'
        )
        
        trainer = CanonicalFlowFieldTrainer(config)
        
        # Train briefly
        for epoch in range(config.num_epochs):
            losses = trainer.train_epoch(epoch)
            if epoch == config.num_epochs - 1:  # Final epoch
                print(f"  Final flow magnitude: {losses['flow_magnitude']:.6f}")
                print(f"  Final image loss: {losses['image_loss']:.6f}")
                
                # Test prediction
                flow = trainer.predict_canonical_flow(7)
                magnitude = torch.sqrt(torch.sum(flow ** 2, dim=-1)).mean()
                results[lambda_mag] = {
                    'training_magnitude': losses['flow_magnitude'],
                    'prediction_magnitude': magnitude.item(),
                    'image_loss': losses['image_loss']
                }
    
    print("\n📊 Regularization Comparison:")
    print("λ_magnitude | Train Mag | Pred Mag  | Image Loss")
    print("-" * 45)
    for lambda_mag, result in results.items():
        print(f"{lambda_mag:10.2f} | {result['training_magnitude']:8.6f} | {result['prediction_magnitude']:8.6f} | {result['image_loss']:9.6f}")
    
    return results


def main():
    """Run the demo"""
    print("🚀 Canonical Flow Field Training Demo")
    print("This demo shows simplified flow training with:")
    print("  ✓ Frame 0 as canonical source")
    print("  ✓ Direct image MSE loss")
    print("  ✓ No complex frame pair generation")
    print("  ✓ Tunable regularization")
    print()
    
    # Main demo
    trainer = demo_canonical_flow()
    
    # Test regularization effects
    try:
        test_regularization_effects()
    except Exception as e:
        print(f"Regularization test failed: {e}")
    
    print("\n🎉 All demos completed!")


if __name__ == "__main__":
    main() 