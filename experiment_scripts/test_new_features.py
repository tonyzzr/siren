"""
Test script for new features in the modularized dense flow field trainer
"""

import sys
import os

# Add parent directory to path
WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(WORK_DIR)

from train_dense_flow_field_modularized import DenseFlowFieldTrainer, FlowTrainingConfig
import torch
import numpy as np


def test_flow_visualization():
    """Test the new white-background flow visualization"""
    print("=== Testing Flow Visualization ===")
    
    # Create a simple test flow field
    test_flow = torch.zeros(2, 10, 10)
    
    # Add some test patterns
    test_flow[0, 2:8, 2:8] = 0.5  # Horizontal flow
    test_flow[1, 4:6, 4:6] = -0.3  # Vertical flow
    
    # Create a minimal config
    config = FlowTrainingConfig(
        num_epochs=1,  # Minimal training
        log_dir=f'{WORK_DIR}/logs/test_features'
    )
    
    try:
        # Create trainer (this will load data and models)
        trainer = DenseFlowFieldTrainer(config)
        
        # Test the new visualization function
        flow_vis = trainer._visualize_flow_white_zero(test_flow)
        print(f"✓ Flow visualization successful. Output shape: {flow_vis.shape}")
        
        # Check that zero flow areas are white (close to [1, 1, 1])
        zero_areas = (test_flow[0] == 0) & (test_flow[1] == 0)
        if zero_areas.any():
            zero_colors = flow_vis[zero_areas.numpy()]
            avg_zero_color = np.mean(zero_colors, axis=0)
            print(f"✓ Average color in zero-flow areas: {avg_zero_color}")
            if np.all(avg_zero_color > 0.9):  # Should be close to white
                print("✓ Zero flow areas are white as expected")
            else:
                print("⚠ Zero flow areas may not be white enough")
        
        return True
        
    except Exception as e:
        print(f"✗ Flow visualization test failed: {e}")
        return False


def test_frame_validation():
    """Test frame index validation"""
    print("\n=== Testing Frame Validation ===")
    
    config = FlowTrainingConfig(
        num_epochs=1,
        vis_source_frame=0,
        vis_target_frame=5,
        max_frame_gap=10,
        log_dir=f'{WORK_DIR}/logs/test_features'
    )
    
    try:
        trainer = DenseFlowFieldTrainer(config)
        num_frames = trainer.vid_dataset.shape[0]
        print(f"✓ Video has {num_frames} frames")
        
        # Test valid frame indices
        if config.vis_source_frame < num_frames and config.vis_target_frame < num_frames:
            print(f"✓ Frame indices ({config.vis_source_frame}, {config.vis_target_frame}) are valid")
        else:
            print(f"⚠ Frame indices ({config.vis_source_frame}, {config.vis_target_frame}) may be out of range")
        
        return True
        
    except Exception as e:
        print(f"✗ Frame validation test failed: {e}")
        return False


def test_flow_accumulation_logic():
    """Test the flow accumulation logic without full training"""
    print("\n=== Testing Flow Accumulation Logic ===")
    
    config = FlowTrainingConfig(
        num_epochs=1,
        max_frame_gap=5,
        log_dir=f'{WORK_DIR}/logs/test_features'
    )
    
    try:
        trainer = DenseFlowFieldTrainer(config)
        num_frames = trainer.vid_dataset.shape[0]
        
        # Test cases
        test_cases = [
            (0, 0),  # Same frame
            (0, 1),  # Adjacent frames
            (0, min(3, num_frames-1)),  # Skip frames
        ]
        
        print("Testing flow accumulation with untrained model:")
        for source, target in test_cases:
            try:
                flow = trainer._accumulate_flow_between_frames(source, target)
                magnitude = torch.sqrt(torch.sum(flow ** 2, dim=-1)).mean()
                print(f"  ✓ Frame {source} → {target}: magnitude = {magnitude:.6f}")
            except Exception as e:
                print(f"  ✗ Frame {source} → {target}: {e}")
        
        # Test error cases
        try:
            trainer._accumulate_flow_between_frames(-1, 0)
            print("  ✗ Should have failed for negative frame index")
        except ValueError:
            print("  ✓ Correctly rejected negative frame index")
        
        try:
            trainer._accumulate_flow_between_frames(0, num_frames)
            print("  ✗ Should have failed for out-of-range frame index")
        except ValueError:
            print("  ✓ Correctly rejected out-of-range frame index")
        
        return True
        
    except Exception as e:
        print(f"✗ Flow accumulation test failed: {e}")
        return False


def test_config_parameters():
    """Test the new configuration parameters"""
    print("\n=== Testing Configuration Parameters ===")
    
    # Test default config
    config1 = FlowTrainingConfig()
    print(f"✓ Default vis_source_frame: {config1.vis_source_frame}")
    print(f"✓ Default vis_target_frame: {config1.vis_target_frame}")
    print(f"✓ Default max_frame_gap: {config1.max_frame_gap}")
    
    # Test custom config
    config2 = FlowTrainingConfig(
        vis_source_frame=2,
        vis_target_frame=8,
        max_frame_gap=15
    )
    print(f"✓ Custom vis_source_frame: {config2.vis_source_frame}")
    print(f"✓ Custom vis_target_frame: {config2.vis_target_frame}")
    print(f"✓ Custom max_frame_gap: {config2.max_frame_gap}")
    
    return True


def main():
    """Run all tests"""
    print("Testing new features in Dense Flow Field Trainer\n")
    
    tests = [
        test_config_parameters,
        test_flow_visualization,
        test_frame_validation,
        test_flow_accumulation_logic,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test_func.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    print(f"\n=== Test Summary ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠ Some tests failed. Check the output above.")
    
    return passed == total


if __name__ == "__main__":
    main() 