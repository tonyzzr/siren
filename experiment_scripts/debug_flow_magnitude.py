"""
Debug script to investigate flow magnitude issues and test different regularization values
"""

import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(WORK_DIR)

from train_canonical_flow_field_features import CanonicalFlowFeatureTrainer, CanonicalFlowFeatureConfig


def test_regularization_effects():
    """Test different regularization values to find optimal flow learning"""
    print("🔬 Debugging Flow Magnitude Issues")
    print("=" * 50)
    print("Testing different λ_magnitude values to find optimal regularization...")
    print()
    
    # Test different regularization values
    lambda_values = [0.01, 0.1, 0.5, 1.0, 5.0]
    results = {}
    
    for lambda_mag in lambda_values:
        print(f"\n🧪 Testing λ_magnitude = {lambda_mag}")
        print("-" * 30)
        
        config = CanonicalFlowFeatureConfig(
            num_epochs=30,  # Longer training
            learning_rate=1e-4,
            steps_til_summary=1000,  # No visualization during test
            lambda_magnitude=lambda_mag,
            flow_hidden_features=64,
            flow_num_layers=3,
            sample_fraction=2e-3,  # More samples for better training
            log_dir=f'{WORK_DIR}/logs/debug_lambda_{lambda_mag}'
        )
        
        trainer = CanonicalFlowFeatureTrainer(config)
        
        # Train and track progress
        epoch_losses = []
        epoch_magnitudes = []
        
        for epoch in range(config.num_epochs):
            losses = trainer.train_epoch(epoch)
            epoch_losses.append(losses)
            
            # Test flow predictions every 5 epochs
            if epoch % 5 == 0 or epoch == config.num_epochs - 1:
                with torch.no_grad():
                    flow_1 = trainer.predict_canonical_flow(1)
                    flow_7 = trainer.predict_canonical_flow(7)
                    flow_15 = trainer.predict_canonical_flow(15)
                    
                    mag_1 = flow_1.norm(dim=-1).mean().item()
                    mag_7 = flow_7.norm(dim=-1).mean().item()
                    mag_15 = flow_15.norm(dim=-1).mean().item()
                    
                    epoch_magnitudes.append({
                        'epoch': epoch,
                        'mag_1': mag_1,
                        'mag_7': mag_7,
                        'mag_15': mag_15
                    })
                    
                    if epoch % 10 == 0:
                        print(f"  Epoch {epoch:2d}: Flow 1={mag_1:.4f}, 7={mag_7:.4f}, 15={mag_15:.4f}")
        
        # Final results
        final_losses = epoch_losses[-1]
        final_mags = epoch_magnitudes[-1]
        
        results[lambda_mag] = {
            'final_feature_loss': final_losses['feature_loss'],
            'final_magnitude_loss': final_losses['magnitude_loss'],
            'final_total_loss': final_losses['total_loss'],
            'flow_mag_1': final_mags['mag_1'],
            'flow_mag_7': final_mags['mag_7'],
            'flow_mag_15': final_mags['mag_15'],
            'temporal_progression': final_mags['mag_15'] / final_mags['mag_1'],  # Should be > 1
            'epoch_magnitudes': epoch_magnitudes
        }
        
        print(f"  Final: Feature={final_losses['feature_loss']:.6f}, "
              f"Flow_1={final_mags['mag_1']:.4f}, Flow_15={final_mags['mag_15']:.4f}")
        print(f"  Temporal ratio (15/1): {results[lambda_mag]['temporal_progression']:.2f}")
    
    # Analysis and recommendations
    print(f"\n📊 Regularization Analysis Results:")
    print("=" * 50)
    print(f"{'λ_mag':<8} {'Feature Loss':<12} {'Flow_1':<8} {'Flow_7':<8} {'Flow_15':<8} {'Ratio_15/1':<10}")
    print("-" * 60)
    
    best_lambda = None
    best_score = 0
    
    for lambda_mag, result in results.items():
        ratio = result['temporal_progression']
        feature_loss = result['final_feature_loss']
        
        # Score: good temporal progression + low feature loss
        score = ratio * (1.0 / (feature_loss + 1e-6))
        
        print(f"{lambda_mag:<8.2f} {feature_loss:<12.6f} {result['flow_mag_1']:<8.4f} "
              f"{result['flow_mag_7']:<8.4f} {result['flow_mag_15']:<8.4f} {ratio:<10.2f}")
        
        if score > best_score:
            best_score = score
            best_lambda = lambda_mag
    
    print(f"\n🎯 Recommendations:")
    print(f"  Best λ_magnitude: {best_lambda}")
    print(f"  Issues identified:")
    
    # Check for common issues
    if all(r['temporal_progression'] < 1.2 for r in results.values()):
        print(f"    ❌ Poor temporal progression across all λ values")
        print(f"    💡 Suggestion: Check if network architecture is too constrained")
    
    if all(r['flow_mag_15'] < 0.01 for r in results.values()):
        print(f"    ❌ Flow magnitudes too small across all λ values")
        print(f"    💡 Suggestion: Reduce regularization further or check coordinate scaling")
    
    if results[1.0]['final_feature_loss'] > 0.01:
        print(f"    ❌ High feature loss indicates poor alignment")
        print(f"    💡 Suggestion: Network may need more capacity or training")
    
    return results, best_lambda


def visualize_temporal_progression(results):
    """Visualize how flow magnitude changes with temporal distance"""
    print(f"\n📈 Creating temporal progression visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Flow magnitude vs temporal distance for different λ values
    temporal_distances = [1, 7, 15]
    
    for lambda_mag, result in results.items():
        magnitudes = [result['flow_mag_1'], result['flow_mag_7'], result['flow_mag_15']]
        axes[0, 0].plot(temporal_distances, magnitudes, 'o-', label=f'λ={lambda_mag}')
    
    axes[0, 0].set_xlabel('Temporal Distance (frames)')
    axes[0, 0].set_ylabel('Flow Magnitude')
    axes[0, 0].set_title('Flow Magnitude vs Temporal Distance')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Feature loss vs λ_magnitude
    lambdas = list(results.keys())
    feature_losses = [results[l]['final_feature_loss'] for l in lambdas]
    
    axes[0, 1].semilogx(lambdas, feature_losses, 'ro-')
    axes[0, 1].set_xlabel('λ_magnitude')
    axes[0, 1].set_ylabel('Final Feature Loss')
    axes[0, 1].set_title('Feature Loss vs Regularization')
    axes[0, 1].grid(True)
    
    # Plot 3: Temporal progression ratio vs λ_magnitude
    ratios = [results[l]['temporal_progression'] for l in lambdas]
    
    axes[1, 0].semilogx(lambdas, ratios, 'go-')
    axes[1, 0].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='No progression')
    axes[1, 0].set_xlabel('λ_magnitude')
    axes[1, 0].set_ylabel('Flow Ratio (15/1)')
    axes[1, 0].set_title('Temporal Progression vs Regularization')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 4: Training progression for best λ
    best_lambda = min(results.keys(), key=lambda x: results[x]['final_feature_loss'])
    epoch_data = results[best_lambda]['epoch_magnitudes']
    
    epochs = [d['epoch'] for d in epoch_data]
    mag_1_history = [d['mag_1'] for d in epoch_data]
    mag_7_history = [d['mag_7'] for d in epoch_data]
    mag_15_history = [d['mag_15'] for d in epoch_data]
    
    axes[1, 1].plot(epochs, mag_1_history, 'b-', label='Frame 1')
    axes[1, 1].plot(epochs, mag_7_history, 'g-', label='Frame 7')
    axes[1, 1].plot(epochs, mag_15_history, 'r-', label='Frame 15')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Flow Magnitude')
    axes[1, 1].set_title(f'Training Progress (λ={best_lambda})')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # Save the analysis
    save_path = f'{WORK_DIR}/logs/flow_magnitude_debug_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Analysis saved to: {save_path}")


def main():
    """Main debug function"""
    results, best_lambda = test_regularization_effects()
    visualize_temporal_progression(results)
    
    print(f"\n🔧 Next Steps:")
    print(f"  1. Use λ_magnitude = {best_lambda} for better flow learning")
    print(f"  2. If temporal progression is still poor, consider:")
    print(f"     • Increasing network capacity (more layers/neurons)")
    print(f"     • Using curriculum learning (start with adjacent frames)")
    print(f"     • Adding temporal consistency losses")
    print(f"     • Checking coordinate normalization")


if __name__ == "__main__":
    main() 