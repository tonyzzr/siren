"""
Diagnostic script to analyze flow field issues in dense flow field trainer
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(WORK_DIR)

from train_dense_flow_field_modularized import DenseFlowFieldTrainer, FlowTrainingConfig
import torch.nn.functional as F


def test_flow_accumulation_correctness():
    """Test if flow accumulation is mathematically correct"""
    print("=== Testing Flow Accumulation Correctness ===")
    
    config = FlowTrainingConfig(
        num_epochs=1,
        max_frame_gap=10,
        log_dir=f'{WORK_DIR}/logs/flow_diagnosis'
    )
    
    trainer = DenseFlowFieldTrainer(config)
    trainer.flow_model.eval()
    
    with torch.no_grad():
        # Test case: accumulate flow 0→1→2 vs direct 0→2
        print("Testing flow accumulation consistency...")
        
        # Method 1: Direct accumulation 0→2
        try:
            flow_0_to_2_direct = trainer._accumulate_flow_between_frames(0, 2)
            print(f"✓ Direct flow 0→2 magnitude: {torch.sqrt(torch.sum(flow_0_to_2_direct**2, dim=-1)).mean():.6f}")
        except Exception as e:
            print(f"✗ Direct flow 0→2 failed: {e}")
            return False
        
        # Method 2: Step-by-step accumulation 0→1 + 1→2
        try:
            flow_0_to_1 = trainer._accumulate_flow_between_frames(0, 1)
            flow_1_to_2 = trainer._accumulate_flow_between_frames(1, 2)
            flow_0_to_2_manual = flow_0_to_1 + flow_1_to_2
            
            print(f"✓ Manual flow 0→1 magnitude: {torch.sqrt(torch.sum(flow_0_to_1**2, dim=-1)).mean():.6f}")
            print(f"✓ Manual flow 1→2 magnitude: {torch.sqrt(torch.sum(flow_1_to_2**2, dim=-1)).mean():.6f}")
            print(f"✓ Manual flow 0→2 magnitude: {torch.sqrt(torch.sum(flow_0_to_2_manual**2, dim=-1)).mean():.6f}")
            
            # Compare the two methods
            diff = torch.abs(flow_0_to_2_direct - flow_0_to_2_manual).mean()
            print(f"✓ Difference between methods: {diff:.6f}")
            
            if diff < 1e-5:
                print("✅ Flow accumulation is mathematically consistent")
            else:
                print("⚠ Flow accumulation may have issues")
                
        except Exception as e:
            print(f"✗ Manual accumulation failed: {e}")
            return False
        
        # Test backward flow consistency
        try:
            flow_2_to_0 = trainer._accumulate_flow_between_frames(2, 0)
            flow_magnitude_forward = torch.sqrt(torch.sum(flow_0_to_2_direct**2, dim=-1)).mean()
            flow_magnitude_backward = torch.sqrt(torch.sum(flow_2_to_0**2, dim=-1)).mean()
            
            print(f"✓ Forward flow 0→2 magnitude: {flow_magnitude_forward:.6f}")
            print(f"✓ Backward flow 2→0 magnitude: {flow_magnitude_backward:.6f}")
            print(f"✓ Magnitude ratio (should be ~1): {flow_magnitude_forward/flow_magnitude_backward:.3f}")
            
        except Exception as e:
            print(f"✗ Backward flow test failed: {e}")
    
    return True


def test_regularization_effects():
    """Test different regularization strengths"""
    print("\n=== Testing Regularization Effects ===")
    
    regularization_values = [1.0, 10.0, 100.0, 1000.0]
    
    for lambda_mag in regularization_values:
        print(f"\n--- Testing λ_magnitude = {lambda_mag} ---")
        
        config = FlowTrainingConfig(
            num_epochs=50,  # Short training
            lambda_magnitude=lambda_mag,
            learning_rate=1e-4,
            steps_til_summary=1000,  # No visualization
            log_dir=f'{WORK_DIR}/logs/reg_test_{lambda_mag}'
        )
        
        trainer = DenseFlowFieldTrainer(config)
        
        # Train briefly
        for epoch in range(config.num_epochs):
            losses = trainer.train_epoch(epoch)
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Flow magnitude = {losses['flow_magnitude']:.6f}, "
                      f"Magnitude loss = {losses['magnitude_loss']:.6f}")
        
        # Test final flow magnitude
        with torch.no_grad():
            flow_0_to_1 = trainer._accumulate_flow_between_frames(0, 1)
            flow_0_to_5 = trainer._accumulate_flow_between_frames(0, 5)
            
            mag_0_to_1 = torch.sqrt(torch.sum(flow_0_to_1**2, dim=-1)).mean()
            mag_0_to_5 = torch.sqrt(torch.sum(flow_0_to_5**2, dim=-1)).mean()
            
            print(f"  Final flow 0→1 magnitude: {mag_0_to_1:.6f}")
            print(f"  Final flow 0→5 magnitude: {mag_0_to_5:.6f}")
            print(f"  Ratio 0→5 / 0→1: {mag_0_to_5 / mag_0_to_1:.3f}")


def test_coordinate_system_consistency():
    """Test if coordinate systems are consistent in flow accumulation"""
    print("\n=== Testing Coordinate System Consistency ===")
    
    config = FlowTrainingConfig(
        num_epochs=1,
        log_dir=f'{WORK_DIR}/logs/coord_test'
    )
    
    trainer = DenseFlowFieldTrainer(config)
    
    # Check coordinate system setup
    print(f"Video shape: {trainer.vid_dataset.shape}")
    print(f"Pixel coords shape: {trainer.pixel_coords.shape}")
    print(f"dt: {trainer.dt}, dx: {trainer.dx}, dy: {trainer.dy}")
    print(f"Bounds: {trainer.bounds}")
    
    # Test a simple flow prediction
    with torch.no_grad():
        # Get coordinates for frame 0
        frame_0_coords = trainer.pixel_coords[0, ...].view(1, -1, 3).to(trainer.device)
        
        # Predict flow
        flow_output = trainer.flow_model({
            'coords': frame_0_coords,
            'idx': torch.tensor([0]).to(trainer.device)
        })
        flow_field = flow_output['model_out']  # [1, H*W, 2]
        
        print(f"Flow field shape: {flow_field.shape}")
        print(f"Flow field range: [{flow_field.min():.6f}, {flow_field.max():.6f}]")
        
        # Check if flow is in reasonable range compared to coordinate deltas
        flow_magnitude = torch.sqrt(torch.sum(flow_field**2, dim=-1))
        print(f"Flow magnitude stats: mean={flow_magnitude.mean():.6f}, "
              f"max={flow_magnitude.max():.6f}, std={flow_magnitude.std():.6f}")
        
        # Compare to coordinate deltas
        print(f"Coordinate delta scale: dx={trainer.dx:.6f}, dy={trainer.dy:.6f}")
        print(f"Flow/coordinate ratio: {flow_magnitude.mean() / trainer.dx:.3f}")


def test_feature_consistency_across_frames():
    """Test if feature model gives consistent results across frames"""
    print("\n=== Testing Feature Consistency Across Frames ===")
    
    config = FlowTrainingConfig(
        num_epochs=1,
        log_dir=f'{WORK_DIR}/logs/feature_test'
    )
    
    trainer = DenseFlowFieldTrainer(config)
    
    with torch.no_grad():
        # Test feature similarity between nearby and distant frames
        frame_pairs = [(0, 1), (0, 2), (0, 5), (0, 7)]
        
        for source, target in frame_pairs:
            if target >= trainer.vid_dataset.shape[0]:
                continue
                
            # Get coordinates
            source_coords = trainer.pixel_coords[source, ...].view(1, -1, 3).to(trainer.device)
            target_coords = trainer.pixel_coords[target, ...].view(1, -1, 3).to(trainer.device)
            
            # Get features
            source_feat = trainer.feature_model({
                'coords': source_coords,
                'idx': torch.tensor([0]).to(trainer.device)
            })['model_out']
            
            target_feat = trainer.feature_model({
                'coords': target_coords,
                'idx': torch.tensor([0]).to(trainer.device)
            })['model_out']
            
            # Compute feature similarity
            cosine_sim = F.cosine_similarity(source_feat, target_feat, dim=-1).mean()
            l2_distance = torch.norm(source_feat - target_feat, dim=-1).mean()
            
            print(f"Frame {source}→{target}: Cosine similarity = {cosine_sim:.4f}, "
                  f"L2 distance = {l2_distance:.4f}")


def visualize_flow_accumulation_steps():
    """Visualize individual steps in flow accumulation"""
    print("\n=== Visualizing Flow Accumulation Steps ===")
    
    config = FlowTrainingConfig(
        num_epochs=100,  # Some training
        lambda_magnitude=10.0,  # Lower regularization
        vis_source_frame=0,
        vis_target_frame=5,
        log_dir=f'{WORK_DIR}/logs/flow_steps_viz'
    )
    
    trainer = DenseFlowFieldTrainer(config)
    
    # Train briefly
    print("Training briefly with lower regularization...")
    for epoch in range(config.num_epochs):
        losses = trainer.train_epoch(epoch)
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: Flow magnitude = {losses['flow_magnitude']:.6f}")
    
    # Visualize step-by-step accumulation
    with torch.no_grad():
        source_frame, target_frame = 0, 5
        
        # Get individual step flows
        step_flows = []
        accumulated_flows = []
        
        current_flow = torch.zeros(1, 224*224, 2).to(trainer.device)
        
        for step in range(source_frame, target_frame):
            # Get coordinates for current step
            step_coords = trainer.pixel_coords[step, ...].view(1, -1, 3).to(trainer.device)
            
            # Predict flow for this step
            flow_output = trainer.flow_model({
                'coords': step_coords,
                'idx': torch.tensor([0]).to(trainer.device)
            })
            step_flow = flow_output['model_out']
            
            step_flows.append(step_flow.clone())
            current_flow += step_flow
            accumulated_flows.append(current_flow.clone())
            
            step_magnitude = torch.sqrt(torch.sum(step_flow**2, dim=-1)).mean()
            accum_magnitude = torch.sqrt(torch.sum(current_flow**2, dim=-1)).mean()
            
            print(f"  Step {step}→{step+1}: magnitude = {step_magnitude:.6f}")
            print(f"  Accumulated 0→{step+1}: magnitude = {accum_magnitude:.6f}")
        
        # Compare with direct accumulation
        direct_flow = trainer._accumulate_flow_between_frames(source_frame, target_frame)
        direct_magnitude = torch.sqrt(torch.sum(direct_flow**2, dim=-1)).mean()
        final_accum_magnitude = torch.sqrt(torch.sum(current_flow**2, dim=-1)).mean()
        
        print(f"\nFinal comparison:")
        print(f"  Manual accumulation magnitude: {final_accum_magnitude:.6f}")
        print(f"  Direct accumulation magnitude: {direct_magnitude:.6f}")
        print(f"  Difference: {abs(final_accum_magnitude - direct_magnitude):.6f}")
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Visualize some steps
        for i, (step_idx, step_flow) in enumerate(zip([0, 2, 4], [step_flows[0], step_flows[2], step_flows[4]])):
            if i < 3:
                flow_2d = step_flow.view(224, 224, 2).cpu()
                flow_vis = trainer._visualize_flow_white_zero(flow_2d.permute(2, 0, 1))
                axes[0, i].imshow(flow_vis)
                axes[0, i].set_title(f"Step {step_idx}→{step_idx+1}")
                axes[0, i].axis('off')
        
        # Visualize accumulated flows
        for i, (accum_idx, accum_flow) in enumerate(zip([1, 3, 5], [accumulated_flows[0], accumulated_flows[2], accumulated_flows[4]])):
            if i < 3:
                flow_2d = accum_flow.view(224, 224, 2).cpu()
                flow_vis = trainer._visualize_flow_white_zero(flow_2d.permute(2, 0, 1))
                axes[1, i].imshow(flow_vis)
                axes[1, i].set_title(f"Accumulated 0→{accum_idx}")
                axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{config.log_dir}/flow_accumulation_steps.png', dpi=150)
        plt.show()


def main():
    """Run all diagnostic tests"""
    print("🔍 Diagnosing Dense Flow Field Issues\n")
    
    tests = [
        test_coordinate_system_consistency,
        test_flow_accumulation_correctness,
        test_feature_consistency_across_frames,
        test_regularization_effects,
        visualize_flow_accumulation_steps,
    ]
    
    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"✗ Test {test_func.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "="*60 + "\n")
    
    print("🎯 Diagnostic Summary:")
    print("1. Check flow accumulation mathematical consistency")
    print("2. Test different regularization strengths")
    print("3. Verify coordinate system setup")
    print("4. Analyze feature model behavior across frames")
    print("5. Visualize step-by-step flow accumulation")


if __name__ == "__main__":
    main() 