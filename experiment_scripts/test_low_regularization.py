"""
Test with very low regularization to check if we can get effective flow magnitudes
"""

import sys
import os

# Add parent directory to path
WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(WORK_DIR)

from train_dense_flow_field_modularized import DenseFlowFieldTrainer, FlowTrainingConfig
import torch
import matplotlib.pyplot as plt


def test_very_low_regularization():
    """Test with very low regularization values"""
    print("=== Testing Very Low Regularization ===")
    
    # Test with extremely low regularization
    config = FlowTrainingConfig(
        num_epochs=200,
        lambda_magnitude=0.1,  # Very low regularization
        learning_rate=1e-4,
        steps_til_summary=50,
        vis_source_frame=7,
        vis_target_frame=2,
        log_dir=f'{WORK_DIR}/logs/low_reg_test'
    )
    
    trainer = DenseFlowFieldTrainer(config)
    
    print("Training with λ_magnitude = 0.1...")
    
    # Track flow magnitudes during training
    flow_magnitudes = []
    
    for epoch in range(config.num_epochs):
        losses = trainer.train_epoch(epoch)
        flow_magnitudes.append(losses['flow_magnitude'])
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Flow magnitude = {losses['flow_magnitude']:.6f}, "
                  f"Feature loss = {losses['feature_loss']:.6f}")
        
        # Generate visualization at key epochs
        if epoch in [50, 100, 150, 199]:
            trainer._visualize_results(epoch)
    
    # Test final flow accumulation
    with torch.no_grad():
        print("\nFinal flow accumulation test:")
        
        # Test different frame gaps
        test_cases = [(0, 1), (0, 5), (7, 2), (2, 7), (0, 10)]
        
        for source, target in test_cases:
            if target < trainer.vid_dataset.shape[0]:
                try:
                    flow = trainer._accumulate_flow_between_frames(source, target)
                    magnitude = torch.sqrt(torch.sum(flow ** 2, dim=-1)).mean()
                    print(f"  Flow {source}→{target}: magnitude = {magnitude:.6f}")
                except Exception as e:
                    print(f"  Flow {source}→{target}: ERROR - {e}")
    
    # Plot flow magnitude evolution
    plt.figure(figsize=(10, 6))
    plt.plot(flow_magnitudes)
    plt.xlabel('Epoch')
    plt.ylabel('Flow Magnitude')
    plt.title('Flow Magnitude Evolution (λ_magnitude = 0.1)')
    plt.grid(True)
    plt.savefig(f'{config.log_dir}/flow_magnitude_evolution.png')
    plt.show()
    
    return trainer


def test_no_regularization():
    """Test with no magnitude regularization"""
    print("\n=== Testing No Regularization ===")
    
    config = FlowTrainingConfig(
        num_epochs=100,
        lambda_magnitude=0.0,  # No regularization
        learning_rate=1e-4,
        steps_til_summary=25,
        vis_source_frame=7,
        vis_target_frame=2,
        log_dir=f'{WORK_DIR}/logs/no_reg_test'
    )
    
    trainer = DenseFlowFieldTrainer(config)
    
    print("Training with λ_magnitude = 0.0 (no regularization)...")
    
    for epoch in range(config.num_epochs):
        losses = trainer.train_epoch(epoch)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Flow magnitude = {losses['flow_magnitude']:.6f}, "
                  f"Feature loss = {losses['feature_loss']:.6f}")
        
        # Generate visualization at key epochs
        if epoch in [25, 50, 75, 99]:
            trainer._visualize_results(epoch)
    
    # Test final flow
    with torch.no_grad():
        flow_7_to_2 = trainer._accumulate_flow_between_frames(7, 2)
        magnitude = torch.sqrt(torch.sum(flow_7_to_2 ** 2, dim=-1)).mean()
        print(f"\nFinal flow 7→2 magnitude: {magnitude:.6f}")
        
        # Check if this is large enough for meaningful alignment
        # Compare to coordinate deltas
        print(f"Coordinate delta scale: {trainer.dx:.6f}")
        print(f"Flow/coordinate ratio: {magnitude / trainer.dx:.3f}")
        
        if magnitude / trainer.dx > 10:
            print("✅ Flow magnitude seems sufficient for alignment")
        else:
            print("⚠ Flow magnitude may still be too small")
    
    return trainer


def compare_feature_consistency():
    """Compare feature consistency with different flow magnitudes"""
    print("\n=== Comparing Feature Consistency ===")
    
    # Test with trained model (low regularization)
    config = FlowTrainingConfig(
        lambda_magnitude=0.1,
        vis_source_frame=7,
        vis_target_frame=2,
        log_dir=f'{WORK_DIR}/logs/feature_consistency_test'
    )
    
    trainer = DenseFlowFieldTrainer(config)
    
    # Train briefly
    for epoch in range(50):
        trainer.train_epoch(epoch)
    
    with torch.no_grad():
        # Get coordinates and features
        source_coords = trainer.pixel_coords[7, ...].view(1, -1, 3).to(trainer.device)
        target_coords = trainer.pixel_coords[2, ...].view(1, -1, 3).to(trainer.device)
        
        # Get features
        source_feat = trainer.feature_model({
            'coords': source_coords,
            'idx': torch.tensor([0]).to(trainer.device)
        })['model_out']
        
        target_feat = trainer.feature_model({
            'coords': target_coords,
            'idx': torch.tensor([0]).to(trainer.device)
        })['model_out']
        
        # Get flow and warp
        flow_7_to_2 = trainer._accumulate_flow_between_frames(7, 2)
        
        # Create warped coordinates
        warped_coords = source_coords.clone()
        warped_coords[:, :, :2] += flow_7_to_2
        
        # Get warped features
        warped_feat = trainer.feature_model({
            'coords': warped_coords,
            'idx': torch.tensor([0]).to(trainer.device)
        })['model_out']
        
        # Compute similarities
        import torch.nn.functional as F
        
        # Original similarity (frame 7 vs frame 2)
        orig_sim = F.cosine_similarity(source_feat, target_feat, dim=-1).mean()
        
        # Warped similarity (warped frame 7 vs frame 2)
        warped_sim = F.cosine_similarity(warped_feat, target_feat, dim=-1).mean()
        
        print(f"Original similarity (7 vs 2): {orig_sim:.4f}")
        print(f"Warped similarity (warped 7 vs 2): {warped_sim:.4f}")
        print(f"Improvement: {warped_sim - orig_sim:.4f}")
        
        if warped_sim > orig_sim:
            print("✅ Flow improves feature alignment")
        else:
            print("⚠ Flow does not improve feature alignment")


def main():
    """Run low regularization tests"""
    print("🔍 Testing Low Regularization Hypothesis\n")
    
    # Test 1: Very low regularization
    trainer1 = test_very_low_regularization()
    
    # Test 2: No regularization
    trainer2 = test_no_regularization()
    
    # Test 3: Feature consistency
    compare_feature_consistency()
    
    print("\n🎯 Summary:")
    print("1. Tested very low regularization (λ = 0.1)")
    print("2. Tested no regularization (λ = 0.0)")
    print("3. Analyzed feature consistency improvement")
    print("4. Generated visualizations for comparison")


if __name__ == "__main__":
    main() 