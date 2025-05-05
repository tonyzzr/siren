"""
Fix scikit-video FFmpeg path issue and NumPy compatibility issues
Run this script before importing skvideo in your project.
"""
import os
import subprocess
import re
import shutil

def fix_skvideo_numpy_types():
    """
    Fix scikit-video NumPy type issues by replacing deprecated types.
    """
    try:
        import skvideo
        
        # Find the abstract.py file
        abstract_path = os.path.join(os.path.dirname(skvideo.__file__), 'io', 'abstract.py')
        
        if not os.path.exists(abstract_path):
            print(f"Cannot find abstract.py at {abstract_path}")
            return False
        
        # Create a backup
        backup_path = abstract_path + '.bak'
        if not os.path.exists(backup_path):
            shutil.copy2(abstract_path, backup_path)
            print(f"Created backup at {backup_path}")
        
        # Read the file
        with open(abstract_path, 'r') as f:
            content = f.read()
        
        # Replace np.float and np.int with float and int
        content = re.sub(r'np\.float\(', 'float(', content)
        content = re.sub(r'np\.int\(', 'int(', content)
        
        # Write back the patched file
        with open(abstract_path, 'w') as f:
            f.write(content)
        
        print(f"Successfully patched {abstract_path}")
        return True
    
    except Exception as e:
        print(f"Error patching scikit-video NumPy types: {str(e)}")
        return False

def fix_skvideo_ffmpeg():
    """
    Fix scikit-video FFmpeg path issue by correctly setting the path.
    This is needed because skvideo has issues with FFmpeg path detection.
    """
    try:
        import skvideo
        
        # First verify FFmpeg is actually installed
        try:
            ffmpeg_path = subprocess.check_output(['which', 'ffmpeg']).decode().strip()
            ffprobe_path = subprocess.check_output(['which', 'ffprobe']).decode().strip()
            
            # FFmpeg is installed, get the path (directory only)
            ffmpeg_dir = os.path.dirname(ffmpeg_path)
            
            # Set FFmpeg path in skvideo
            skvideo.setFFmpegPath(ffmpeg_dir)
            
            print(f"Successfully set FFmpeg path to: {ffmpeg_dir}")
            print(f"FFmpeg application: {skvideo._FFMPEG_APPLICATION}")
            print(f"HAS_FFMPEG status: {skvideo._HAS_FFMPEG}")
            
            return True
        except subprocess.CalledProcessError:
            print("FFmpeg is not installed in the system. Please install FFmpeg.")
            return False
    except ImportError:
        print("scikit-video is not installed. Please install scikit-video.")
        return False

def fix_skvideo():
    """Fix all scikit-video issues"""
    numpy_fix = fix_skvideo_numpy_types()
    ffmpeg_fix = fix_skvideo_ffmpeg()
    return numpy_fix and ffmpeg_fix

if __name__ == "__main__":
    fix_skvideo() 