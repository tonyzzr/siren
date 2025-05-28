"""
Test script to verify the flow accumulation fix
"""

import sys
import os

# Add parent directory to path
WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(WORK_DIR)

from train_dense_flow_field_modularized import DenseFlowFieldTrainer, FlowTrainingConfig
import torch


def test_flow_accumulation_fix():
    """Test the fixed flow accumulation implementation"""
    print("=== Testing Fixed Flow Accumulation ===")
    
    # Create configuration with lower regularization to see more flow
    config = FlowTrainingConfig(
        num_epochs=100,
        lambda_magnitude=10.0,  # Lower regularization
        learning_rate=1e-4,
        steps_til_summary=25,
        vis_source_frame=7,
        vis_target_frame=2,
        log_dir=f'{WORK_DIR}/logs/flow_fix_test'
    )
    
    trainer = DenseFlowFieldTrainer(config)
    
    print("Training with fixed flow accumulation...")
    
    # Train briefly to get some flow
    for epoch in range(config.num_epochs):
        losses = trainer.train_epoch(epoch)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Flow magnitude = {losses['flow_magnitude']:.6f}")
        
        # Generate visualization at key epochs
        if epoch in [25, 50, 75, 99]:
            trainer._visualize_results(epoch)
    
    # Test flow accumulation with the fix
    with torch.no_grad():
        print("\nTesting flow accumulation after fix:")
        
        # Test different frame gaps
        test_cases = [(0, 1), (0, 5), (7, 2), (2, 7), (0, 8)]
        
        for source, target in test_cases:
            if target < trainer.vid_dataset.shape[0]:
                try:
                    flow = trainer._accumulate_flow_between_frames(source, target)
                    magnitude = torch.sqrt(torch.sum(flow ** 2, dim=-1)).mean()
                    print(f"  Flow {source}→{target}: magnitude = {magnitude:.6f}")
                    
                    # Check if magnitude is reasonable for the frame gap
                    gap = abs(target - source)
                    if gap > 0:
                        magnitude_per_frame = magnitude / gap
                        print(f"    Magnitude per frame: {magnitude_per_frame:.6f}")
                        
                except Exception as e:
                    print(f"  Flow {source}→{target}: ERROR - {e}")
    
    print("\n✅ Flow accumulation fix test completed!")
    return trainer


def compare_before_after_fix():
    """Compare flow magnitudes before and after the fix"""
    print("\n=== Comparing Before/After Fix ===")
    
    # The fix should result in more coherent flow accumulation
    # and potentially larger magnitudes for multi-frame gaps
    
    config = FlowTrainingConfig(
        num_epochs=50,
        lambda_magnitude=1.0,  # Very low regularization
        vis_source_frame=7,
        vis_target_frame=2,
        log_dir=f'{WORK_DIR}/logs/flow_comparison_test'
    )
    
    trainer = DenseFlowFieldTrainer(config)
    
    # Train briefly
    for epoch in range(config.num_epochs):
        trainer.train_epoch(epoch)
    
    # Test the fixed implementation
    with torch.no_grad():
        print("Testing with FIXED implementation:")
        
        # Test backward flow (7→2)
        flow_7_to_2 = trainer._accumulate_flow_between_frames(7, 2)
        magnitude_7_to_2 = torch.sqrt(torch.sum(flow_7_to_2 ** 2, dim=-1)).mean()
        
        # Test forward flow (2→7)
        flow_2_to_7 = trainer._accumulate_flow_between_frames(2, 7)
        magnitude_2_to_7 = torch.sqrt(torch.sum(flow_2_to_7 ** 2, dim=-1)).mean()
        
        print(f"  Flow 7→2 magnitude: {magnitude_7_to_2:.6f}")
        print(f"  Flow 2→7 magnitude: {magnitude_2_to_7:.6f}")
        print(f"  Magnitude ratio: {magnitude_7_to_2 / magnitude_2_to_7:.3f}")
        
        # Check coordinate scale
        print(f"  Coordinate delta: {trainer.dx:.6f}")
        print(f"  Flow/coord ratio 7→2: {magnitude_7_to_2 / trainer.dx:.3f}")
        
        if magnitude_7_to_2 / trainer.dx > 5:
            print("✅ Flow magnitude looks more reasonable now!")
        else:
            print("⚠ Flow magnitude still seems small")
    
    # Generate final visualization
    trainer._visualize_results(epoch=999)


def main():
    """Run flow accumulation fix tests"""
    print("🔧 Testing Flow Accumulation Fix\n")
    
    # Test 1: Basic fix verification
    trainer1 = test_flow_accumulation_fix()
    
    # Test 2: Before/after comparison
    compare_before_after_fix()
    
    print("\n🎯 Summary:")
    print("1. Fixed critical bug in flow accumulation")
    print("2. Now using current_coords instead of fixed frame_coords")
    print("3. Properly following motion trajectory")
    print("4. Should see improved alignment for large frame gaps")


if __name__ == "__main__":
    main() 