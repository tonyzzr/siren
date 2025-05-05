import torch
import numpy as np
from skimage import data, transform
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for saving animations

def get_camera_image(size=(224, 224)):
    """Load the cameraman image and resize it to the specified size."""
    img = data.camera()
    img = transform.resize(img, size, mode='reflect', anti_aliasing=True)
    return img

def create_translation_flow(height, width, frames, direction='horizontal', magnitude=2.0):
    """
    Create dense flow fields for translation motion.
    
    Args:
        height, width: Dimensions of the image
        frames: Number of frames to generate
        direction: 'horizontal' or 'vertical'
        magnitude: Pixels to move per frame
        
    Returns:
        flow_fields: Tensor of shape [frames, 2, height, width]
    """
    flow_fields = torch.zeros(frames, 2, height, width)
    
    # For translation, the flow should accumulate over frames
    # to create continuous motion (e.g., frame 1: move 2px, frame 2: move 4px)
    for f in range(frames):
        # Compute cumulative offset
        cumulative_magnitude = magnitude * (f + 1)
        
        # Create the flow field (channel 0 is horizontal, channel 1 is vertical)
        flow_idx = 0 if direction == 'horizontal' else 1
        flow_fields[f, flow_idx] = cumulative_magnitude
    
    return flow_fields

def create_scaling_flow(height, width, frames, scale_factor=1.02):
    """
    Create dense flow fields for scaling motion (zoom in/out).
    
    Args:
        height, width: Dimensions of the image
        frames: Number of frames to generate
        scale_factor: Factor to scale by each frame (>1 for zoom in, <1 for zoom out)
        
    Returns:
        flow_fields: Tensor of shape [frames, 2, height, width]
    """
    flow_fields = torch.zeros(frames, 2, height, width)
    
    # Create coordinate grid
    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing='ij'
    )
    
    # Center coordinates - avoid in-place operations
    y_center, x_center = height / 2, width / 2
    y_coords_centered = y_coords.clone() - y_center
    x_coords_centered = x_coords.clone() - x_center
    
    # For each frame, calculate the flow based on the scale factor
    for f in range(frames):
        # Calculate the current scale (compounding effect)
        current_scale = scale_factor ** (f + 1)
        
        # Calculate flow as the difference between scaled and original positions
        scaling_factor = current_scale - 1
        
        # Horizontal flow (x direction)
        flow_fields[f, 0] = scaling_factor * x_coords_centered
        
        # Vertical flow (y direction)
        flow_fields[f, 1] = scaling_factor * y_coords_centered
    
    return flow_fields

def create_rotation_flow(height, width, frames, angle_per_frame=2.0):
    """
    Create dense flow fields for rotation motion.
    
    Args:
        height, width: Dimensions of the image
        frames: Number of frames to generate
        angle_per_frame: Degrees to rotate per frame (positive for clockwise)
        
    Returns:
        flow_fields: Tensor of shape [frames, 2, height, width]
    """
    flow_fields = torch.zeros(frames, 2, height, width)
    
    # Create coordinate grid
    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing='ij'
    )
    
    # Center coordinates - avoid in-place operations 
    y_center, x_center = height / 2, width / 2
    y_coords_centered = y_coords.clone() - y_center
    x_coords_centered = x_coords.clone() - x_center
    
    # For each frame, calculate the flow based on the rotation
    for f in range(frames):
        # Convert angle to radians
        angle_rad = np.radians(angle_per_frame * (f + 1))
        
        # Calculate new positions after rotation
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)
        
        # Calculate rotated positions (pure rotation matrix)
        x_rot = x_coords_centered * cos_theta - y_coords_centered * sin_theta
        y_rot = x_coords_centered * sin_theta + y_coords_centered * cos_theta
        
        # Flow is the difference between rotated and original positions
        # For pure rotation without zoom effects
        flow_fields[f, 0] = x_rot - x_coords_centered  # Horizontal flow
        flow_fields[f, 1] = y_rot - y_coords_centered  # Vertical flow
    
    return flow_fields

def apply_flow_to_image(image, flow_field):
    """
    Apply a dense flow field to an image using grid_sample.
    
    Args:
        image: Input image tensor [height, width]
        flow_field: Flow field tensor [2, height, width]
        
    Returns:
        warped_image: The transformed image [height, width]
    """
    height, width = image.shape
    
    # Create base sampling grid (normalized coordinates)
    y_coords, x_coords = torch.meshgrid(
        torch.linspace(-1, 1, height),
        torch.linspace(-1, 1, width),
        indexing='ij'
    )
    
    # Convert flow from pixel space to normalized space
    flow_x_normalized = flow_field[0] * (2.0 / width)
    flow_y_normalized = flow_field[1] * (2.0 / height)
    
    # Add flow to the coordinates (negate flow since grid_sample moves in opposite direction)
    sample_x = x_coords - flow_x_normalized
    sample_y = y_coords - flow_y_normalized
    
    # Stack coordinates for grid_sample [height, width, 2]
    grid = torch.stack([sample_x, sample_y], dim=-1)
    
    # Reshape image for grid_sample: [1, 1, height, width]
    img_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Apply grid_sample
    warped_img = torch.nn.functional.grid_sample(
        img_tensor, 
        grid.unsqueeze(0),  # [1, height, width, 2]
        mode='bilinear', 
        padding_mode='zeros',
        align_corners=True
    )
    
    return warped_img.squeeze().numpy()

def generate_mock_video(motion_type, num_frames=30, output_path=None):
    """
    Generate a mock video with the specified motion type.
    
    Args:
        motion_type: One of 'translation_h', 'translation_v', 'zoom_in', 'zoom_out', 'rotation_cw', 'rotation_ccw'
        num_frames: Number of frames to generate
        output_path: Path to save the output .pt file
        
    Returns:
        data_dict: Dictionary containing the flow fields and video frames
    """
    # Get the cameraman image
    img = get_camera_image(size=(224, 224))
    height, width = img.shape
    
    # Create flow fields based on motion type
    if motion_type == 'translation_h':
        flow_fields = create_translation_flow(height, width, num_frames, 'horizontal', 2.0)
    elif motion_type == 'translation_v':
        flow_fields = create_translation_flow(height, width, num_frames, 'vertical', 2.0)
    elif motion_type == 'zoom_in':
        flow_fields = create_scaling_flow(height, width, num_frames, 1.02)
    elif motion_type == 'zoom_out':
        flow_fields = create_scaling_flow(height, width, num_frames, 0.98)
    elif motion_type == 'rotation_cw':
        flow_fields = create_rotation_flow(height, width, num_frames, 2.0)
    elif motion_type == 'rotation_ccw':
        flow_fields = create_rotation_flow(height, width, num_frames, -2.0)
    else:
        raise ValueError(f"Unknown motion type: {motion_type}")
    
    # Apply flow fields to the image to create video frames
    video_frames = []
    for i in range(num_frames):
        warped_frame = apply_flow_to_image(img, flow_fields[i])
        video_frames.append(warped_frame)
    
    # Convert list of frames to tensor
    video_tensor = torch.tensor(np.stack(video_frames), dtype=torch.float32)
    
    # Create data dictionary
    data_dict = {
        'motion_type': motion_type,
        'flow_fields': flow_fields,
        'video_frames': video_tensor,
        'source_image': torch.tensor(img, dtype=torch.float32)
    }
    
    # Save to file if output path is specified
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(data_dict, output_path)
        print(f"Saved mock video with {motion_type} motion to {output_path}")
    
    return data_dict

def generate_all_mock_videos(output_dir="data/mock_videos"):
    """Generate all types of mock videos and save them to the specified directory."""
    motion_types = [
        'translation_h', 'translation_v', 
        'zoom_in', 'zoom_out', 
        'rotation_cw', 'rotation_ccw'
    ]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for motion_type in motion_types:
        output_path = output_dir / f"{motion_type}.pt"
        generate_mock_video(motion_type, num_frames=30, output_path=output_path)
    
    print(f"Generated all mock videos in {output_dir}")

def visualize_flow(flow):
    """
    Visualize a single optical flow field using HSV color coding.
    
    Args:
        flow: Optical flow field tensor [2, height, width]
        
    Returns:
        rgb: RGB image representing the flow field
    """
    # Convert flow to polar coordinates (magnitude and angle)
    u = flow[0].numpy()
    v = flow[1].numpy()
    
    # Calculate magnitude and angle
    magnitude = np.sqrt(u**2 + v**2)
    angle = np.arctan2(v, u)
    
    # Normalize magnitude to [0, 1]
    norm = Normalize()
    norm.autoscale(magnitude)
    magnitude_normalized = norm(magnitude)
    
    # Convert to HSV (hue is angle, saturation is 1, value is magnitude)
    h = (angle + np.pi) / (2 * np.pi)  # Map [-pi, pi] to [0, 1]
    s = np.ones_like(h)
    v = magnitude_normalized
    
    # Stack channels
    hsv = np.stack([h, s, v], axis=2)
    
    # Convert HSV to RGB
    rgb = cm.hsv(h)
    # Scale by magnitude/saturation
    rgb[..., 0:3] *= np.expand_dims(magnitude_normalized, -1)
    
    return rgb

def load_mock_video(file_path):
    """
    Load a mock video file.
    
    Args:
        file_path: Path to the mock video file
        
    Returns:
        data_dict: Dictionary containing the flow fields and video frames
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return torch.load(file_path)

def visualize_mock_video(data_dict, output_path=None, fps=10):
    """
    Visualize a mock video showing original frames, flow fields, and resulting frames.
    
    Args:
        data_dict: Dictionary containing the flow fields and video frames
        output_path: Path to save the visualization (directory for individual frames)
        fps: Frames per second for the animation
    """
    motion_type = data_dict['motion_type']
    flow_fields = data_dict['flow_fields']
    video_frames = data_dict['video_frames']
    source_image = data_dict['source_image']
    
    num_frames = len(video_frames)
    
    # If output_path is a directory, save individual frames
    if output_path:
        output_dir = Path(output_path)
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize each frame separately
    for frame in range(num_frames):
        # Create figure with 1x3 subplots (original, flow, warped)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Motion Type: {motion_type} - Frame {frame+1}/{num_frames}", fontsize=16)
        
        # Set titles
        axes[0].set_title("Original Image")
        axes[1].set_title("Flow Field")
        axes[2].set_title("Warped Frame")
        
        # Display source image in the first subplot
        axes[0].imshow(source_image, cmap='gray')
        axes[0].axis('off')
        
        # Display flow field
        flow_vis = visualize_flow(flow_fields[frame])
        axes[1].imshow(flow_vis)
        axes[1].axis('off')
        
        # Display warped frame
        warped_frame = video_frames[frame].numpy()
        axes[2].imshow(warped_frame, cmap='gray')
        axes[2].axis('off')
        
        # Add colorbar for flow field
        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap=cm.hsv)
        sm.set_array([])
        fig.colorbar(sm, cax=cax, orientation='vertical', label='Flow Direction')
        
        plt.tight_layout()
        
        # Save the frame if output_path is specified
        if output_path:
            frame_path = output_dir / f"{motion_type}_frame_{frame:03d}.png"
            plt.savefig(frame_path)
            if frame == 0 or frame == num_frames // 2 or frame == num_frames - 1:
                print(f"Saved frame {frame+1}/{num_frames} to {frame_path}")
        
        plt.close()
    
    if output_path:
        print(f"Saved {num_frames} frames for {motion_type} to {output_dir}")

def visualize_all_mock_videos(input_dir="data/mock_videos", output_dir="visualizations"):
    """
    Visualize all mock videos in the input directory.
    
    Args:
        input_dir: Directory containing mock video files
        output_dir: Directory to save visualizations
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for file_path in input_dir.glob("*.pt"):
        motion_type = file_path.stem
        motion_output_dir = output_dir / motion_type
        
        try:
            data_dict = load_mock_video(file_path)
            print(f"Visualizing {motion_type} motion...")
            visualize_mock_video(data_dict, motion_output_dir, fps=10)
        except Exception as e:
            print(f"Error visualizing {motion_type}: {str(e)}")

def save_sample_frames(motion_type="rotation_cw", frames=[0, 15, 29], output_dir="visualizations/samples"):
    """
    Save sample frames from a mock video for quick inspection.
    
    Args:
        motion_type: Type of motion to visualize
        frames: List of frame indices to save
        output_dir: Directory to save sample frames
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = f"data/mock_videos/{motion_type}.pt"
    data_dict = load_mock_video(file_path)
    
    flow_fields = data_dict['flow_fields']
    video_frames = data_dict['video_frames']
    source_image = data_dict['source_image']
    
    # Create a single figure with all sample frames
    num_samples = len(frames)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    fig.suptitle(f"Sample Frames for {motion_type} Motion", fontsize=16)
    
    for i, frame_idx in enumerate(frames):
        if num_samples == 1:
            ax_row = axes
        else:
            ax_row = axes[i]
            
        # Set titles for the first row
        if i == 0:
            ax_row[0].set_title("Original Image")
            ax_row[1].set_title("Flow Field")
            ax_row[2].set_title("Warped Frame")
        
        # Add frame number
        ax_row[0].set_ylabel(f"Frame {frame_idx+1}", fontsize=12)
        
        # Display source image
        ax_row[0].imshow(source_image, cmap='gray')
        ax_row[0].axis('off')
        
        # Display flow field
        flow_vis = visualize_flow(flow_fields[frame_idx])
        ax_row[1].imshow(flow_vis)
        ax_row[1].axis('off')
        
        # Display warped frame
        warped_frame = video_frames[frame_idx].numpy()
        ax_row[2].imshow(warped_frame, cmap='gray')
        ax_row[2].axis('off')
    
    # Add colorbar for flow field
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cm.hsv)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, orientation='vertical', label='Flow Direction')
    
    plt.tight_layout()
    
    # Save the figure
    output_path = output_dir / f"{motion_type}_samples.png"
    plt.savefig(output_path)
    plt.close()
    
    print(f"Saved sample frames for {motion_type} to {output_path}")
    
    return output_path

if __name__ == "__main__":
    # Uncomment to generate mock videos
    # generate_all_mock_videos()
    
    # Uncomment to visualize existing mock videos (all frames)
    # visualize_all_mock_videos()
    
    # Default behavior - generate videos if they don't exist
    if not Path("data/mock_videos").exists() or len(list(Path("data/mock_videos").glob("*.pt"))) == 0:
        print("Generating mock videos...")
        generate_all_mock_videos()
    
    # Create sample visualizations for each motion type
    print("Creating sample visualizations...")
    motion_types = [
        'translation_h', 'translation_v', 
        'zoom_in', 'zoom_out', 
        'rotation_cw', 'rotation_ccw'
    ]
    
    output_dir = Path("visualizations/samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for motion_type in motion_types:
        try:
            save_sample_frames(motion_type, frames=[0, 10, 20, 29])
        except Exception as e:
            print(f"Error saving sample frames for {motion_type}: {str(e)}")
