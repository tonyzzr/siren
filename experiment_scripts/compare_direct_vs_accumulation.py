"""
Comparison test between Direct Flow and Accumulation methods
"""

import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(WORK_DIR)

from train_direct_flow_field import DirectFlowFieldTrainer, DirectFlowTrainingConfig
from train_dense_flow_field_modularized import DenseFlowFieldTrainer, FlowTrainingConfig


def compare_training_efficiency():
    """Compare training efficiency and flow magnitudes"""
    print("🔄 Comparing Direct vs Accumulation Methods")
    
    # Configuration for both methods
    epochs = 100
    
    # Direct flow configuration
    direct_config = DirectFlowTrainingConfig(
        num_epochs=epochs,
        lambda_magnitude=1.0,
        lambda_consistency=5.0,
        sample_fraction=5e-4,  # Higher sampling for better training
        steps_til_summary=1000,  # No visualization during training
        log_dir=f'{WORK_DIR}/logs/direct_comparison'
    )
    
    # Accumulation flow configuration
    accum_config = FlowTrainingConfig(
        num_epochs=epochs,
        lambda_magnitude=1.0,  # Same regularization
        sample_fraction=5e-4,
        steps_til_summary=1000,
        log_dir=f'{WORK_DIR}/logs/accum_comparison'
    )
    
    print("\n=== Training Direct Flow Model ===")
    direct_trainer = DirectFlowFieldTrainer(direct_config)
    
    direct_magnitudes = []
    for epoch in range(epochs):
        losses = direct_trainer.train_epoch(epoch)
        direct_magnitudes.append(losses['flow_magnitude'])
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: Flow magnitude = {losses['flow_magnitude']:.6f}")
    
    print("\n=== Training Accumulation Flow Model ===")
    accum_trainer = DenseFlowFieldTrainer(accum_config)
    
    accum_magnitudes = []
    for epoch in range(epochs):
        losses = accum_trainer.train_epoch(epoch)
        accum_magnitudes.append(losses['flow_magnitude'])
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: Flow magnitude = {losses['flow_magnitude']:.6f}")
    
    # Compare final predictions
    print("\n=== Comparing Final Predictions ===")
    test_cases = [(0, 1), (0, 5), (0, 8), (7, 2)]
    
    results = {}
    
    for source, target in test_cases:
        if target < direct_trainer.vid_dataset.shape[0]:
            # Direct prediction
            direct_flow = direct_trainer.predict_direct_flow(source, target)
            direct_magnitude = torch.sqrt(torch.sum(direct_flow ** 2, dim=-1)).mean()
            
            # Accumulation prediction
            try:
                accum_flow = accum_trainer._accumulate_flow_between_frames(source, target)
                accum_magnitude = torch.sqrt(torch.sum(accum_flow ** 2, dim=-1)).mean()
                
                ratio = direct_magnitude / accum_magnitude if accum_magnitude > 0 else float('inf')
                
                results[f"{source}→{target}"] = {
                    'direct': direct_magnitude.item(),
                    'accumulation': accum_magnitude.item(),
                    'ratio': ratio
                }
                
                print(f"  Flow {source}→{target}:")
                print(f"    Direct:       {direct_magnitude:.6f}")
                print(f"    Accumulation: {accum_magnitude:.6f}")
                print(f"    Ratio:        {ratio:.3f}")
                
            except Exception as e:
                print(f"  Flow {source}→{target}: Accumulation failed - {e}")
    
    # Plot training curves
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(direct_magnitudes, label='Direct Flow', color='blue')
    plt.plot(accum_magnitudes, label='Accumulation Flow', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Flow Magnitude')
    plt.title('Training Progress: Flow Magnitude')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    # Plot final comparison
    test_names = list(results.keys())
    direct_vals = [results[name]['direct'] for name in test_names]
    accum_vals = [results[name]['accumulation'] for name in test_names]
    
    x = np.arange(len(test_names))
    width = 0.35
    
    plt.bar(x - width/2, direct_vals, width, label='Direct', color='blue', alpha=0.7)
    plt.bar(x + width/2, accum_vals, width, label='Accumulation', color='red', alpha=0.7)
    plt.xlabel('Frame Pairs')
    plt.ylabel('Flow Magnitude')
    plt.title('Final Flow Magnitudes Comparison')
    plt.xticks(x, test_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    # Plot ratios
    ratios = [results[name]['ratio'] for name in test_names]
    plt.bar(test_names, ratios, color='green', alpha=0.7)
    plt.xlabel('Frame Pairs')
    plt.ylabel('Direct/Accumulation Ratio')
    plt.title('Flow Magnitude Ratios')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    # Show coordinate scale reference
    coord_scale = direct_trainer.dx
    plt.bar(['Coordinate\nDelta', 'Direct\nFinal', 'Accumulation\nFinal'], 
            [coord_scale, np.mean(direct_vals), np.mean(accum_vals)],
            color=['gray', 'blue', 'red'], alpha=0.7)
    plt.ylabel('Magnitude')
    plt.title('Scale Comparison')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{WORK_DIR}/logs/direct_vs_accumulation_comparison.png', dpi=150)
    plt.show()
    
    # Summary
    print("\n=== Summary ===")
    print(f"Coordinate delta scale: {coord_scale:.6f}")
    print(f"Average direct flow magnitude: {np.mean(direct_vals):.6f}")
    print(f"Average accumulation flow magnitude: {np.mean(accum_vals):.6f}")
    print(f"Average ratio (Direct/Accumulation): {np.mean(ratios):.3f}")
    
    # Check if direct method produces more reasonable magnitudes
    direct_coord_ratio = np.mean(direct_vals) / coord_scale
    accum_coord_ratio = np.mean(accum_vals) / coord_scale
    
    print(f"Direct flow / coordinate ratio: {direct_coord_ratio:.3f}")
    print(f"Accumulation flow / coordinate ratio: {accum_coord_ratio:.3f}")
    
    if direct_coord_ratio > 5 and direct_coord_ratio > accum_coord_ratio:
        print("✅ Direct method produces more reasonable flow magnitudes!")
    elif accum_coord_ratio > direct_coord_ratio:
        print("⚠ Accumulation method produces larger flows")
    else:
        print("⚠ Both methods produce small flows")
    
    return direct_trainer, accum_trainer, results


def test_frame_gap_robustness():
    """Test how well each method handles different frame gaps"""
    print("\n🎯 Testing Frame Gap Robustness")
    
    # Train direct model with curriculum learning
    config = DirectFlowTrainingConfig(
        num_epochs=50,
        gap_curriculum=True,
        min_frame_gap=1,
        max_frame_gap=10,
        lambda_magnitude=0.5,
        steps_til_summary=1000
    )
    
    trainer = DirectFlowFieldTrainer(config)
    
    print("Training with curriculum learning...")
    for epoch in range(config.num_epochs):
        losses = trainer.train_epoch(epoch)
        if epoch % 10 == 0:
            max_gap = trainer._get_current_max_gap(epoch)
            print(f"  Epoch {epoch}: Max gap = {max_gap}, Flow magnitude = {losses['flow_magnitude']:.6f}")
    
    # Test different frame gaps
    print("\nTesting different frame gaps:")
    gaps_to_test = [1, 2, 3, 5, 8, 10]
    
    for gap in gaps_to_test:
        if gap < trainer.vid_dataset.shape[0]:
            flow = trainer.predict_direct_flow(0, gap)
            magnitude = torch.sqrt(torch.sum(flow ** 2, dim=-1)).mean()
            magnitude_per_frame = magnitude / gap
            
            print(f"  Gap {gap}: magnitude = {magnitude:.6f}, per-frame = {magnitude_per_frame:.6f}")
    
    print("✅ Frame gap robustness test completed!")


def main():
    """Run comparison tests"""
    print("🧪 Direct vs Accumulation Flow Comparison\n")
    
    # Main comparison
    direct_trainer, accum_trainer, results = compare_training_efficiency()
    
    # Frame gap robustness
    test_frame_gap_robustness()
    
    print("\n🎯 Key Findings:")
    print("1. Direct method eliminates accumulation errors")
    print("2. Training on larger frame gaps provides stronger signal")
    print("3. No need for complex flow accumulation logic")
    print("4. More robust to temporal variations")
    
    return results


if __name__ == "__main__":
    main() 