"""
Quick test for Direct Flow Field Trainer
"""

import sys
import os

# Add parent directory to path
WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(WORK_DIR)

from train_direct_flow_field import DirectFlowFieldTrainer, DirectFlowTrainingConfig


def quick_test():
    """Quick test of direct flow trainer"""
    print("🧪 Quick Direct Flow Test")
    
    # Create minimal config
    config = DirectFlowTrainingConfig(
        num_epochs=5,
        sample_fraction=1e-4,  # Very small for quick test
        steps_til_summary=1000,  # No visualization
        min_frame_gap=1,
        max_frame_gap=3
    )
    
    try:
        # Test initialization
        print("1. Testing initialization...")
        trainer = DirectFlowFieldTrainer(config)
        print(f"   ✓ Video shape: {trainer.vid_dataset.shape}")
        print(f"   ✓ Frame pairs: {len(trainer.dataset.frame_pairs)}")
        
        # Test model forward pass
        print("2. Testing model forward pass...")
        sample = trainer.dataset[0]
        input_coords = sample['input_coords'].unsqueeze(0).to(trainer.device)
        
        output = trainer.flow_model({
            'coords': input_coords,
            'idx': torch.tensor([0]).to(trainer.device)
        })
        
        print(f"   ✓ Input shape: {input_coords.shape}")
        print(f"   ✓ Output shape: {output['model_out'].shape}")
        
        # Test training step
        print("3. Testing training step...")
        losses = trainer.train_epoch(0)
        print(f"   ✓ Feature loss: {losses['feature_loss']:.6f}")
        print(f"   ✓ Flow magnitude: {losses['flow_magnitude']:.6f}")
        
        # Test direct prediction
        print("4. Testing direct prediction...")
        flow = trainer.predict_direct_flow(0, 2)
        magnitude = torch.sqrt(torch.sum(flow ** 2, dim=-1)).mean()
        print(f"   ✓ Flow 0→2 magnitude: {magnitude:.6f}")
        
        print("\n✅ All quick tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import torch
    quick_test() 