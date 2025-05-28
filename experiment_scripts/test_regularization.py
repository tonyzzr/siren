"""
Test script to demonstrate regularization effects in canonical flow training
"""

import sys
import os
import torch

# Add parent directory to path
WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(WORK_DIR)

from train_canonical_flow_field import CanonicalFlowFieldTrainer, CanonicalFlowConfig


def test_regularization_effects():
    """Test different regularization values"""
    print("🔬 Testing Regularization Effects")
    print("=" * 50)
    
    regularization_values = [0.01, 0.1, 1.0, 10.0]
    results = {}
    
    for lambda_mag in regularization_values:
        print(f"\nTesting λ_magnitude = {lambda_mag}")
        
        config = CanonicalFlowConfig(
            num_epochs=15,  # Quick test
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


if __name__ == "__main__":
    test_regularization_effects() 