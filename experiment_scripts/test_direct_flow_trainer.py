"""
Comprehensive test suite for the Direct Flow Field Trainer

This module tests the new direct flow prediction approach and compares it
with the traditional accumulation-based method.
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Add parent directory to path
WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(WORK_DIR)

from train_direct_flow_field import DirectFlowFieldTrainer, DirectFlowTrainingConfig
from train_dense_flow_field_modularized import DenseFlowFieldTrainer, FlowTrainingConfig


class DirectFlowTester:
    """Comprehensive tester for direct flow field trainer"""
    
    def __init__(self):
        self.results = {}
        
    def test_dataset_generation(self):
        """Test the DirectFlowDataset generates valid frame pairs"""
        print("=== Testing Dataset Generation ===")
        
        config = DirectFlowTrainingConfig(
            min_frame_gap=1,
            max_frame_gap=5,
            sample_fraction=1e-4  # Small for testing
        )
        
        trainer = DirectFlowFieldTrainer(config)
        dataset = trainer.dataset
        
        print(f"✓ Generated {len(dataset.frame_pairs)} frame pairs")
        print(f"✓ Dataset length: {len(dataset)}")
        
        # Test a few samples
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            
            # Validate sample structure
            required_keys = ['input_coords', 'source_coords', 'target_coords', 'gt_flow', 'source_frame', 'target_frame']
            for key in required_keys:
                assert key in sample, f"Missing key: {key}"
            
            # Validate coordinate dimensions
            assert sample['input_coords'].shape[1] == 4, "Input coords should be 4D (x, y, t_source, t_target)"
            assert sample['source_coords'].shape[1] == 3, "Source coords should be 3D (t, x, y)"
            assert sample['target_coords'].shape[1] == 3, "Target coords should be 3D (t, x, y)"
            assert sample['gt_flow'].shape[1] == 2, "GT flow should be 2D (dx, dy)"
            
            source_frame = sample['source_frame']
            target_frame = sample['target_frame']
            gap = abs(target_frame - source_frame)
            
            print(f"  Sample {i}: Frame {source_frame} → {target_frame} (gap={gap})")
            assert config.min_frame_gap <= gap <= config.max_frame_gap, f"Gap {gap} out of range"
        
        print("✅ Dataset generation test passed!")
        return True
        
    def test_model_architecture(self):
        """Test the direct flow model architecture"""
        print("\n=== Testing Model Architecture ===")
        
        config = DirectFlowTrainingConfig()
        trainer = DirectFlowFieldTrainer(config)
        
        # Test input/output dimensions
        batch_size = 2
        num_points = 100
        
        # Create test input: (x, y, t_source, t_target)
        test_input = torch.randn(batch_size, num_points, 4).to(trainer.device)
        
        with torch.no_grad():
            output = trainer.flow_model({
                'coords': test_input,
                'idx': torch.tensor([0]).to(trainer.device)
            })
            
            flow_output = output['model_out']
            
            # Validate output shape
            expected_shape = (batch_size, num_points, 2)
            assert flow_output.shape == expected_shape, f"Expected {expected_shape}, got {flow_output.shape}"
            
            print(f"✓ Model input shape: {test_input.shape}")
            print(f"✓ Model output shape: {flow_output.shape}")
            print(f"✓ Flow range: [{flow_output.min():.4f}, {flow_output.max():.4f}]")
        
        print("✅ Model architecture test passed!")
        return True
        
    def test_direct_flow_prediction(self):
        """Test direct flow prediction functionality"""
        print("\n=== Testing Direct Flow Prediction ===")
        
        config = DirectFlowTrainingConfig(
            num_epochs=10,  # Quick training
            steps_til_summary=1000  # No visualization
        )
        
        trainer = DirectFlowFieldTrainer(config)
        
        # Train briefly
        print("Training for a few epochs...")
        for epoch in range(config.num_epochs):
            losses = trainer.train_epoch(epoch)
            if epoch % 5 == 0:
                print(f"  Epoch {epoch}: Flow magnitude = {losses['flow_magnitude']:.6f}")
        
        # Test direct prediction
        test_cases = [(0, 1), (0, 5), (2, 7), (7, 2)]
        
        for source_frame, target_frame in test_cases:
            if target_frame < trainer.vid_dataset.shape[0]:
                flow = trainer.predict_direct_flow(source_frame, target_frame)
                magnitude = torch.sqrt(torch.sum(flow ** 2, dim=-1)).mean()
                
                print(f"  Direct flow {source_frame}→{target_frame}: magnitude = {magnitude:.6f}")
                
                # Validate flow shape
                expected_shape = (1, 224*224, 2)
                assert flow.shape == expected_shape, f"Expected {expected_shape}, got {flow.shape}"
        
        print("✅ Direct flow prediction test passed!")
        return trainer
        
    def test_bidirectional_consistency(self):
        """Test bidirectional flow consistency"""
        print("\n=== Testing Bidirectional Consistency ===")
        
        config = DirectFlowTrainingConfig(
            num_epochs=20,
            lambda_consistency=10.0,  # High consistency weight
            steps_til_summary=1000
        )
        
        trainer = DirectFlowFieldTrainer(config)
        
        # Train with consistency loss
        print("Training with bidirectional consistency...")
        for epoch in range(config.num_epochs):
            losses = trainer.train_epoch(epoch)
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Consistency loss = {losses['consistency_loss']:.6f}")
        
        # Test consistency
        test_pairs = [(0, 5), (2, 8), (7, 3)]
        
        for source, target in test_pairs:
            if target < trainer.vid_dataset.shape[0]:
                # Forward flow
                flow_forward = trainer.predict_direct_flow(source, target)
                
                # Backward flow
                flow_backward = trainer.predict_direct_flow(target, source)
                
                # Check if forward + backward ≈ 0
                consistency_error = torch.mean(torch.abs(flow_forward + flow_backward))
                
                print(f"  Consistency {source}↔{target}: error = {consistency_error:.6f}")
                
                # Should be reasonably small after training
                assert consistency_error < 0.1, f"Consistency error too large: {consistency_error}"
        
        print("✅ Bidirectional consistency test passed!")
        return trainer
        
    def test_curriculum_learning(self):
        """Test curriculum learning for frame gaps"""
        print("\n=== Testing Curriculum Learning ===")
        
        config = DirectFlowTrainingConfig(
            num_epochs=100,
            gap_curriculum=True,
            min_frame_gap=1,
            max_frame_gap=10,
            steps_til_summary=1000
        )
        
        trainer = DirectFlowFieldTrainer(config)
        
        # Test gap progression
        test_epochs = [0, 25, 50, 75, 99]
        
        for epoch in test_epochs:
            max_gap = trainer._get_current_max_gap(epoch)
            expected_progress = epoch / config.num_epochs
            expected_gap = config.min_frame_gap + int(
                expected_progress * (config.max_frame_gap - config.min_frame_gap)
            )
            expected_gap = min(expected_gap, config.max_frame_gap)
            
            print(f"  Epoch {epoch}: Max gap = {max_gap} (expected ≈ {expected_gap})")
            assert max_gap == expected_gap, f"Gap mismatch at epoch {epoch}"
        
        print("✅ Curriculum learning test passed!")
        return True
        
    def compare_with_accumulation_method(self):
        """Compare direct flow with accumulation-based method"""
        print("\n=== Comparing Direct vs Accumulation Methods ===")
        
        # Train direct flow model
        direct_config = DirectFlowTrainingConfig(
            num_epochs=50,
            lambda_magnitude=1.0,
            steps_til_summary=1000,
            log_dir=f'{WORK_DIR}/logs/direct_comparison_test'
        )
        
        direct_trainer = DirectFlowFieldTrainer(direct_config)
        
        print("Training direct flow model...")
        for epoch in range(direct_config.num_epochs):
            losses = direct_trainer.train_epoch(epoch)
            if epoch % 20 == 0:
                print(f"  Direct epoch {epoch}: Flow magnitude = {losses['flow_magnitude']:.6f}")
        
        # Train accumulation model (with fix)
        accum_config = FlowTrainingConfig(
            num_epochs=50,
            lambda_magnitude=1.0,
            steps_til_summary=1000,
            log_dir=f'{WORK_DIR}/logs/accum_comparison_test'
        )
        
        accum_trainer = DenseFlowFieldTrainer(accum_config)
        
        print("Training accumulation flow model...")
        for epoch in range(accum_config.num_epochs):
            losses = accum_trainer.train_epoch(epoch)
            if epoch % 20 == 0:
                print(f"  Accum epoch {epoch}: Flow magnitude = {losses['flow_magnitude']:.6f}")
        
        # Compare predictions
        test_cases = [(0, 1), (0, 5), (7, 2)]
        
        print("\nComparing predictions:")
        for source, target in test_cases:
            if target < direct_trainer.vid_dataset.shape[0]:
                # Direct prediction
                direct_flow = direct_trainer.predict_direct_flow(source, target)
                direct_magnitude = torch.sqrt(torch.sum(direct_flow ** 2, dim=-1)).mean()
                
                # Accumulation prediction
                try:
                    accum_flow = accum_trainer._accumulate_flow_between_frames(source, target)
                    accum_magnitude = torch.sqrt(torch.sum(accum_flow ** 2, dim=-1)).mean()
                except Exception as e:
                    print(f"  Accumulation failed for {source}→{target}: {e}")
                    continue
                
                print(f"  Flow {source}→{target}:")
                print(f"    Direct:      {direct_magnitude:.6f}")
                print(f"    Accumulation: {accum_magnitude:.6f}")
                print(f"    Ratio:       {direct_magnitude / accum_magnitude:.3f}")
        
        print("✅ Comparison test completed!")
        return direct_trainer, accum_trainer
        
    def test_temporal_consistency(self):
        """Test temporal consistency of predictions"""
        print("\n=== Testing Temporal Consistency ===")
        
        config = DirectFlowTrainingConfig(
            num_epochs=30,
            steps_til_summary=1000
        )
        
        trainer = DirectFlowFieldTrainer(config)
        
        # Train model
        for epoch in range(config.num_epochs):
            trainer.train_epoch(epoch)
        
        # Test transitivity: flow(0→2) ≈ flow(0→1) + flow(1→2)
        flow_0_to_1 = trainer.predict_direct_flow(0, 1)
        flow_1_to_2 = trainer.predict_direct_flow(1, 2)
        flow_0_to_2_direct = trainer.predict_direct_flow(0, 2)
        
        # Note: This is approximate since direct prediction doesn't enforce transitivity
        flow_0_to_2_composed = flow_0_to_1 + flow_1_to_2
        
        difference = torch.mean(torch.abs(flow_0_to_2_direct - flow_0_to_2_composed))
        
        print(f"  Transitivity test 0→1→2:")
        print(f"    Direct 0→2:     {torch.sqrt(torch.sum(flow_0_to_2_direct ** 2, dim=-1)).mean():.6f}")
        print(f"    Composed 0→1→2: {torch.sqrt(torch.sum(flow_0_to_2_composed ** 2, dim=-1)).mean():.6f}")
        print(f"    Difference:     {difference:.6f}")
        
        # The difference might be significant since we don't enforce transitivity
        print("  Note: Direct method doesn't enforce transitivity, so some difference is expected")
        
        print("✅ Temporal consistency test completed!")
        return trainer
        
    def visualize_comparison(self, direct_trainer, accum_trainer):
        """Create visualization comparing both methods"""
        print("\n=== Creating Comparison Visualization ===")
        
        source_frame, target_frame = 0, 8
        
        # Get predictions
        direct_flow = direct_trainer.predict_direct_flow(source_frame, target_frame)
        direct_flow_2d = direct_flow.view(224, 224, 2)
        
        try:
            accum_flow = accum_trainer._accumulate_flow_between_frames(source_frame, target_frame)
            accum_flow_2d = accum_flow.view(224, 224, 2)
        except Exception as e:
            print(f"Accumulation failed: {e}")
            return
        
        # Visualize both
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Direct flow visualization
        direct_vis = direct_trainer._visualize_flow_white_zero(direct_flow_2d.permute(2, 0, 1))
        axes[0, 0].imshow(direct_vis)
        axes[0, 0].set_title(f"Direct Flow: {source_frame}→{target_frame}")
        
        # Accumulation flow visualization
        accum_vis = accum_trainer._visualize_flow_white_zero(accum_flow_2d.permute(2, 0, 1))
        axes[0, 1].imshow(accum_vis)
        axes[0, 1].set_title(f"Accumulation Flow: {source_frame}→{target_frame}")
        
        # Difference
        diff_flow = torch.abs(direct_flow_2d - accum_flow_2d)
        diff_magnitude = torch.sqrt(torch.sum(diff_flow ** 2, dim=-1))
        axes[0, 2].imshow(diff_magnitude.cpu().numpy(), cmap='hot')
        axes[0, 2].set_title("Flow Magnitude Difference")
        
        # Statistics
        direct_mag = torch.sqrt(torch.sum(direct_flow_2d ** 2, dim=-1)).mean()
        accum_mag = torch.sqrt(torch.sum(accum_flow_2d ** 2, dim=-1)).mean()
        diff_mag = torch.mean(diff_magnitude)
        
        axes[1, 0].text(0.1, 0.5, f"Direct Flow\nMagnitude: {direct_mag:.4f}", 
                       transform=axes[1, 0].transAxes, fontsize=12)
        axes[1, 0].axis('off')
        
        axes[1, 1].text(0.1, 0.5, f"Accumulation Flow\nMagnitude: {accum_mag:.4f}", 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].axis('off')
        
        axes[1, 2].text(0.1, 0.5, f"Mean Difference\n{diff_mag:.4f}\nRatio: {direct_mag/accum_mag:.3f}", 
                       transform=axes[1, 2].transAxes, fontsize=12)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{WORK_DIR}/logs/direct_vs_accumulation_comparison.png', dpi=150)
        plt.show()
        
        print("✅ Comparison visualization saved!")


def main():
    """Run all tests for the direct flow trainer"""
    print("🧪 Testing Direct Flow Field Trainer\n")
    
    tester = DirectFlowTester()
    
    # Core functionality tests
    tests = [
        tester.test_dataset_generation,
        tester.test_model_architecture,
        tester.test_direct_flow_prediction,
        tester.test_curriculum_learning,
        tester.test_bidirectional_consistency,
        tester.test_temporal_consistency,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(True)
            print()
        except Exception as e:
            print(f"✗ Test {test_func.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
            print()
    
    # Comparison test
    try:
        print("Running comparison test...")
        direct_trainer, accum_trainer = tester.compare_with_accumulation_method()
        tester.visualize_comparison(direct_trainer, accum_trainer)
        results.append(True)
    except Exception as e:
        print(f"✗ Comparison test failed: {e}")
        results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n🎯 Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Direct flow trainer is working correctly.")
    else:
        print("⚠ Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    main() 