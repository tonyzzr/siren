"""
Patch scikit-video abstract.py to fix NumPy deprecated float type issues.
"""
import os
import shutil
import re

def patch_skvideo_abstract():
    """
    Patch scikit-video abstract.py file to replace deprecated NumPy float types.
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
        print(f"Error patching scikit-video: {str(e)}")
        return False

if __name__ == "__main__":
    patch_skvideo_abstract() 