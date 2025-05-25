import os
import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Update .pth files with superpoints from .npy files.")
parser.add_argument("--pth_dir", required=True, help="Path to the directory containing .pth files.")
parser.add_argument("--scene_dir", required=True, help="Path to the directory containing scene directories with .npy files.")
args = parser.parse_args()

pth_dir = args.pth_dir
scene_dir = args.scene_dir

# Build dictionaries for .pth files and scene directories
pth_files = {os.path.splitext(f)[0]: os.path.join(pth_dir, f) for f in os.listdir(pth_dir) if f.endswith('.pth')}
scene_dirs = {d: os.path.join(scene_dir, d) for d in os.listdir(scene_dir) if os.path.isdir(os.path.join(scene_dir, d))}

# Process files
for scene_name, pth_path in pth_files.items():
    if scene_name in scene_dirs:
        scene_path = scene_dirs[scene_name]

        # Find .npy files with 'superpoints.npy'
        npy_files = [os.path.join(scene_path, f) for f in os.listdir(scene_path) if f.endswith('superpoints.npy')]

        if len(npy_files) > 0:
            npy_path = npy_files[0]  
            superpoint = np.load(npy_path)

            # Load and update the .pth file
            data = torch.load(pth_path, weights_only=False)
            data['superpoints'] = superpoint
            torch.save(data, pth_path)
            print(f"Updated {pth_path} with superpoint from {npy_path}")
        else:
            print(f"No .npy file found in directory: {scene_path}")
    else:
        print(f"No corresponding scene directory found for {pth_path}")