"""
Demo script for Direct Flow Field Training with frequent visualizations
"""

import sys
import os

# Add parent directory to path
WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(WORK_DIR)

from train_direct_flow_field import DirectFlowFieldTrainer, DirectFlowTrainingConfig


def demo_fast_training():
    """Demo fast training with frequent visualizations"""
    print("🚀 Direct Flow Training Demo")
    print("=" * 50)
    
    # Create config for fast demo
    config = DirectFlowTrainingConfig(
        # Fast training settings
        num_epochs=30,
        learning_rate=1e-4,
        sample_fraction=2e-3,  # Higher sampling for faster convergence
        
        # Frequent visualization
        steps_til_summary=5,  # Show results every 5 epochs
        
        # Model settings
        flow_hidden_features=32,
        flow_num_layers=2,
        
        # Loss settings
        lambda_magnitude=0.5,  # Lower regularization for more visible flow
        lambda_consistency=5.0,
        
        # Frame comparison
        vis_source_frame=0,
        vis_target_frame=7,  # Larger gap for more visible flow
        
        # Curriculum learning
        gap_curriculum=True,
        min_frame_gap=1,
        max_frame_gap=8,
        
        log_dir=f'{WORK_DIR}/logs/direct_flow_demo'
    )
    
    print(f"Configuration:")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Visualization every: {config.steps_til_summary} epochs")
    print(f"  Frame comparison: {config.vis_source_frame} → {config.vis_target_frame}")
    print(f"  Sample fraction: {config.sample_fraction}")
    print(f"  Regularization: λ_magnitude={config.lambda_magnitude}, λ_consistency={config.lambda_consistency}")
    print()
    
    # Create and train
    trainer = DirectFlowFieldTrainer(config)
    
    print("Starting training...")
    trainer.train()
    
    # Test final predictions
    print("\n" + "=" * 50)
    print("Testing final predictions:")
    
    test_cases = [(0, 1), (0, 3), (0, 7), (7, 0)]
    
    for source, target in test_cases:
        if target < trainer.vid_dataset.shape[0]:
            flow = trainer.predict_direct_flow(source, target)
            magnitude = torch.sqrt(torch.sum(flow ** 2, dim=-1)).mean()
            
            # Compare to coordinate scale
            coord_ratio = magnitude / trainer.dx
            
            print(f"  Flow {source}→{target}: magnitude = {magnitude:.6f} (ratio to coord: {coord_ratio:.2f})")
    
    # Save model
    model_path = f"{config.log_dir}/checkpoints/demo_model.pth"
    trainer.save_model(model_path)
    
    print(f"\n✅ Demo completed! Model saved to {model_path}")
    return trainer


def demo_comparison_with_accumulation():
    """Quick comparison with accumulation method"""
    print("\n🔄 Quick Comparison with Accumulation Method")
    print("=" * 50)
    
    # Import accumulation trainer
    from train_dense_flow_field_modularized import DenseFlowFieldTrainer, FlowTrainingConfig
    
    # Train direct flow (quick)
    direct_config = DirectFlowTrainingConfig(
        num_epochs=20,
        lambda_magnitude=0.5,
        steps_til_summary=1000,  # No visualization during comparison
        log_dir=f'{WORK_DIR}/logs/direct_comparison_demo'
    )
    
    print("Training direct flow model...")
    direct_trainer = DirectFlowFieldTrainer(direct_config)
    
    for epoch in range(direct_config.num_epochs):
        losses = direct_trainer.train_epoch(epoch)
        if epoch % 10 == 0:
            print(f"  Direct epoch {epoch}: Flow magnitude = {losses['flow_magnitude']:.6f}")
    
    # Train accumulation model (quick)
    accum_config = FlowTrainingConfig(
        num_epochs=20,
        lambda_magnitude=0.5,
        steps_til_summary=1000,
        log_dir=f'{WORK_DIR}/logs/accum_comparison_demo'
    )
    
    print("Training accumulation flow model...")
    accum_trainer = DenseFlowFieldTrainer(accum_config)
    
    for epoch in range(accum_config.num_epochs):
        losses = accum_trainer.train_epoch(epoch)
        if epoch % 10 == 0:
            print(f"  Accum epoch {epoch}: Flow magnitude = {losses['flow_magnitude']:.6f}")
    
    # Compare results
    print("\nComparison results:")
    test_cases = [(0, 1), (0, 5), (7, 2)]
    
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
                
                print(f"  Flow {source}→{target}:")
                print(f"    Direct:       {direct_magnitude:.6f}")
                print(f"    Accumulation: {accum_magnitude:.6f}")
                print(f"    Ratio:        {ratio:.3f}")
                
            except Exception as e:
                print(f"  Flow {source}→{target}: Accumulation failed - {e}")
    
    return direct_trainer, accum_trainer


def main():
    """Run the demo"""
    import torch
    
    print("🎯 Direct Flow Field Training Demo")
    print("This demo shows the improved training with:")
    print("  ✓ Frequent visualizations")
    print("  ✓ Progress bars with tqdm")
    print("  ✓ Optimized training loop")
    print("  ✓ Training curve plots")
    print()
    
    # Main demo
    trainer = demo_fast_training()
    
    # Optional comparison
    try:
        demo_comparison_with_accumulation()
    except Exception as e:
        print(f"Comparison demo failed: {e}")
    
    print("\n🎉 All demos completed!")


if __name__ == "__main__":
    main() 