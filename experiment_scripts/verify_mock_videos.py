import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
import os

# Add the parent directory to the Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment_scripts.prepare_mock_video import load_mock_video, visualize_flow

def display_video_info(data_dict):
    """Display information about the loaded video file."""
    motion_type = data_dict['motion_type']
    flow_fields = data_dict['flow_fields']
    video_frames = data_dict['video_frames']
    source_image = data_dict['source_image']
    
    print(f"Motion Type: {motion_type}")
    print(f"Number of frames: {len(video_frames)}")
    print(f"Video shape: {video_frames.shape}")
    print(f"Flow fields shape: {flow_fields.shape}")
    print(f"Source image shape: {source_image.shape}")
    
    # Calculate average and max flow magnitudes
    flow_magnitudes = torch.sqrt(flow_fields[:, 0]**2 + flow_fields[:, 1]**2)
    avg_magnitude = torch.mean(flow_magnitudes).item()
    max_magnitude = torch.max(flow_magnitudes).item()
    
    print(f"Average flow magnitude: {avg_magnitude:.2f} pixels")
    print(f"Maximum flow magnitude: {max_magnitude:.2f} pixels")
    
    return {
        'motion_type': motion_type,
        'num_frames': len(video_frames),
        'avg_magnitude': avg_magnitude,
        'max_magnitude': max_magnitude
    }

def view_video_interactive(data_dict, start_frame=0):
    """
    Interactive viewing of a mock video with flow fields.
    Use left/right arrow keys to navigate through frames.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button
    
    motion_type = data_dict['motion_type']
    flow_fields = data_dict['flow_fields']
    video_frames = data_dict['video_frames']
    source_image = data_dict['source_image']
    
    num_frames = len(video_frames)
    current_frame = start_frame
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Motion Type: {motion_type} - Frame {current_frame+1}/{num_frames}", fontsize=16)
    
    # Set titles
    axes[0].set_title("Original Image")
    axes[1].set_title("Flow Field")
    axes[2].set_title("Warped Frame")
    
    # Display images
    im1 = axes[0].imshow(source_image, cmap='gray')
    axes[0].axis('off')
    
    flow_vis = visualize_flow(flow_fields[current_frame])
    im2 = axes[1].imshow(flow_vis)
    axes[1].axis('off')
    
    im3 = axes[2].imshow(video_frames[current_frame], cmap='gray')
    axes[2].axis('off')
    
    # Add slider for frame navigation
    ax_slider = plt.axes([0.25, 0.02, 0.65, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, num_frames-1, valinit=current_frame, valstep=1)
    
    def update(val):
        frame = int(slider.val)
        fig.suptitle(f"Motion Type: {motion_type} - Frame {frame+1}/{num_frames}", fontsize=16)
        flow_vis = visualize_flow(flow_fields[frame])
        im2.set_array(flow_vis)
        im3.set_array(video_frames[frame])
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()

def compare_motion_types(video_dir="data/mock_videos"):
    """Compare statistics across different motion types."""
    video_dir = Path(video_dir)
    results = []
    
    for file_path in sorted(video_dir.glob("*.pt")):
        data_dict = load_mock_video(file_path)
        info = display_video_info(data_dict)
        results.append(info)
        print("----------")
    
    # Create a summary table for easy comparison
    print("\nSummary:")
    print("Motion Type         | Frames | Avg Magnitude | Max Magnitude")
    print("-" * 60)
    for info in results:
        print(f"{info['motion_type']:<20} | {info['num_frames']:<6} | {info['avg_magnitude']:<13.2f} | {info['max_magnitude']:<12.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify and visualize mock videos")
    parser.add_argument("--video", default="rotation_cw", help="Motion type to visualize")
    parser.add_argument("--compare", action="store_true", help="Compare all motion types")
    parser.add_argument("--interactive", action="store_true", help="Interactive visualization")
    args = parser.parse_args()
    
    if args.compare:
        compare_motion_types()
    else:
        file_path = f"data/mock_videos/{args.video}.pt"
        data_dict = load_mock_video(file_path)
        display_video_info(data_dict)
        
        if args.interactive:
            view_video_interactive(data_dict)
        else:
            print("\nTo view interactively, use: --interactive")
            print("To compare all motion types, use: --compare")
            print("Available motion types:")
            print("  - translation_h: Horizontal translation")
            print("  - translation_v: Vertical translation")
            print("  - zoom_in: Zoom in effect")
            print("  - zoom_out: Zoom out effect")  
            print("  - rotation_cw: Clockwise rotation")
            print("  - rotation_ccw: Counter-clockwise rotation") 