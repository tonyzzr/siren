"""
Example usage of the modularized Dense Flow Field Trainer

This script demonstrates how to use the DenseFlowFieldTrainer class
with different configurations and training scenarios.
"""

import sys
import os

# Add parent directory to path
WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(WORK_DIR)

from train_dense_flow_field_modularized import DenseFlowFieldTrainer, FlowTrainingConfig


def example_basic_training():
    """Basic training example with default configuration"""
    print("=== Basic Training Example ===")
    
    # Create default configuration
    config = FlowTrainingConfig()
    
    # Create and run trainer
    trainer = DenseFlowFieldTrainer(config)
    trainer.train()
    
    # Save the trained model
    trainer.save_model(f"{config.log_dir}/checkpoints/basic_model.pth")


def example_custom_training():
    """Custom training example with modified hyperparameters"""
    print("=== Custom Training Example ===")
    
    # Create custom configuration
    config = FlowTrainingConfig(
        # Model parameters
        flow_hidden_features=32,  # Larger model
        flow_num_layers=4,
        flow_activation="relu",   # Different activation
        
        # Training parameters
        num_epochs=500,           # Fewer epochs
        learning_rate=5e-4,       # Higher learning rate
        sample_fraction=1e-3,     # More samples per batch
        
        # Loss parameters
        lambda_magnitude=50.0,    # Less magnitude regularization
        
        # Logging
        steps_til_summary=50,     # More frequent visualization
        log_dir=f'{WORK_DIR}/logs/custom_flow_field_test'
    )
    
    # Create and run trainer
    trainer = DenseFlowFieldTrainer(config)
    trainer.train()
    
    # Save the trained model
    trainer.save_model(f"{config.log_dir}/checkpoints/custom_model.pth")


def example_resume_training():
    """Example of resuming training from a checkpoint"""
    print("=== Resume Training Example ===")
    
    # Create configuration
    config = FlowTrainingConfig(
        num_epochs=200,
        log_dir=f'{WORK_DIR}/logs/resumed_flow_field_test'
    )
    
    # Create trainer
    trainer = DenseFlowFieldTrainer(config)
    
    # Load pre-trained model (if exists)
    checkpoint_path = f"{config.log_dir}/checkpoints/checkpoint_model.pth"
    if os.path.exists(checkpoint_path):
        trainer.load_model(checkpoint_path)
        print("Resumed from checkpoint")
    else:
        print("No checkpoint found, starting from scratch")
    
    # Continue training
    trainer.train()
    
    # Save final model
    trainer.save_model(f"{config.log_dir}/checkpoints/final_model.pth")


def example_evaluation_only():
    """Example of using the trainer for evaluation/visualization only"""
    print("=== Evaluation Only Example ===")
    
    # Create configuration
    config = FlowTrainingConfig(
        log_dir=f'{WORK_DIR}/logs/evaluation_flow_field_test'
    )
    
    # Create trainer
    trainer = DenseFlowFieldTrainer(config)
    
    # Load trained model
    model_path = f"{WORK_DIR}/logs/dense_flow_field_test/checkpoints/flow_model_final.pth"
    if os.path.exists(model_path):
        trainer.load_model(model_path)
        
        # Generate visualizations without training
        trainer._visualize_results(epoch=0)
        print("Evaluation completed")
    else:
        print(f"Model not found at {model_path}")


def example_hyperparameter_sweep():
    """Example of running multiple experiments with different hyperparameters"""
    print("=== Hyperparameter Sweep Example ===")
    
    # Define hyperparameter combinations to test
    hyperparams = [
        {'lambda_magnitude': 10.0, 'learning_rate': 1e-4, 'flow_hidden_features': 16},
        {'lambda_magnitude': 100.0, 'learning_rate': 1e-4, 'flow_hidden_features': 16},
        {'lambda_magnitude': 1000.0, 'learning_rate': 1e-4, 'flow_hidden_features': 16},
        {'lambda_magnitude': 100.0, 'learning_rate': 5e-4, 'flow_hidden_features': 32},
    ]
    
    for i, params in enumerate(hyperparams):
        print(f"\n--- Experiment {i+1}/{len(hyperparams)} ---")
        print(f"Parameters: {params}")
        
        # Create configuration with current hyperparameters
        config = FlowTrainingConfig(
            num_epochs=100,  # Shorter training for sweep
            steps_til_summary=50,
            log_dir=f'{WORK_DIR}/logs/sweep_exp_{i+1}',
            **params
        )
        
        # Run training
        trainer = DenseFlowFieldTrainer(config)
        trainer.train()
        
        # Save model
        trainer.save_model(f"{config.log_dir}/checkpoints/sweep_model_{i+1}.pth")


def example_custom_frame_comparison():
    """Example of training with custom frame comparison for visualization"""
    print("=== Custom Frame Comparison Example ===")
    
    # Create configuration with larger frame gap for visualization
    config = FlowTrainingConfig(
        # Training parameters
        num_epochs=1000,           # Shorter training for demo
        learning_rate=1e-4,
        steps_til_summary=100,     # More frequent visualization
        
        # Custom frame comparison
        vis_source_frame=0,       # Start from frame 0
        vis_target_frame=8,       # Compare with frame 8 (larger gap)
        max_frame_gap=15,         # Allow larger gaps
        
        # Logging
        log_dir=f'{WORK_DIR}/logs/custom_frame_comparison_test'
    )
    
    # Create and run trainer
    trainer = DenseFlowFieldTrainer(config)
    
    # Test flow accumulation
    print("Testing flow accumulation with different frame gaps:")
    trainer.test_flow_accumulation([
        (0, 1), (0, 3), (0, 8), (8, 0), (5, 5)
    ])
    
    trainer.train()
    
    # Save the trained model
    trainer.save_model(f"{config.log_dir}/checkpoints/custom_frame_model.pth")


def example_flow_visualization_comparison():
    """Example comparing different frame gaps and flow visualizations"""
    print("=== Flow Visualization Comparison Example ===")
    
    # Test different frame gaps
    frame_gaps = [(0, 1), (0, 3), (0, 7), (3, 8)]
    
    for i, (source_frame, target_frame) in enumerate(frame_gaps):
        print(f"\n--- Testing frame gap: {source_frame} → {target_frame} ---")
        
        config = FlowTrainingConfig(
            num_epochs=100,  # Short training for demo
            steps_til_summary=25,
            vis_source_frame=source_frame,
            vis_target_frame=target_frame,
            log_dir=f'{WORK_DIR}/logs/flow_viz_gap_{abs(target_frame-source_frame)}'
        )
        
        trainer = DenseFlowFieldTrainer(config)
        trainer.train()
        trainer.save_model(f"{config.log_dir}/checkpoints/gap_{abs(target_frame-source_frame)}_model.pth")


def example_backward_flow():
    """Example of backward flow (from later frame to earlier frame)"""
    print("=== Backward Flow Example ===")
    
    config = FlowTrainingConfig(
        num_epochs=1000,
        steps_til_summary=100,
        # Backward flow: from frame 7 to frame 2
        vis_source_frame=7,
        vis_target_frame=2,
        max_frame_gap=10,
        log_dir=f'{WORK_DIR}/logs/backward_flow_test',
        lambda_magnitude=0.001,
    )
    
    trainer = DenseFlowFieldTrainer(config)
    
    # Test both forward and backward flows
    print("Testing forward and backward flow accumulation:")
    trainer.test_flow_accumulation([
        (2, 7),  # Forward
        (7, 2),  # Backward (should be opposite)
        (0, 5),  # Forward
        (5, 0),  # Backward
    ])
    
    trainer.train()
    trainer.save_model(f"{config.log_dir}/checkpoints/backward_flow_model.pth")


def example_evaluation_with_custom_frames():
    """Example of evaluation with custom frame comparison"""
    print("=== Evaluation with Custom Frames Example ===")
    
    # Create configuration for evaluation
    config = FlowTrainingConfig(
        vis_source_frame=1,
        vis_target_frame=9,
        log_dir=f'{WORK_DIR}/logs/evaluation_custom_frames'
    )
    
    # Create trainer
    trainer = DenseFlowFieldTrainer(config)
    
    # Load trained model if available
    model_path = f"{WORK_DIR}/logs/dense_flow_field_test/checkpoints/flow_model_final.pth"
    if os.path.exists(model_path):
        trainer.load_model(model_path)
        
        # Test flow accumulation with trained model
        print("Testing flow accumulation with trained model:")
        trainer.test_flow_accumulation([
            (1, 9), (9, 1), (0, 5), (5, 10), (3, 3)
        ])
        
        # Generate visualizations without training
        trainer._visualize_results(epoch=0)
        print("Evaluation completed")
    else:
        print(f"Model not found at {model_path}")


def main():
    """Main function to run examples"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dense Flow Field Trainer Examples")
    parser.add_argument('--example', type=str, default='basic',
                       choices=['basic', 'custom', 'resume', 'eval', 'sweep', 
                               'custom_frames', 'flow_viz', 'backward', 'eval_custom'],
                       help='Which example to run')
    
    args = parser.parse_args()
    
    if args.example == 'basic':
        example_basic_training()
    elif args.example == 'custom':
        example_custom_training()
    elif args.example == 'resume':
        example_resume_training()
    elif args.example == 'eval':
        example_evaluation_only()
    elif args.example == 'sweep':
        example_hyperparameter_sweep()
    elif args.example == 'custom_frames':
        example_custom_frame_comparison()
    elif args.example == 'flow_viz':
        example_flow_visualization_comparison()
    elif args.example == 'backward':
        example_backward_flow()
    elif args.example == 'eval_custom':
        example_evaluation_with_custom_frames()
    else:
        print(f"Unknown example: {args.example}")


if __name__ == "__main__":
    main() 