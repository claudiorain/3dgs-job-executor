"""
opacity_utils.py - Simple utility for Taming 3DGS opacity conversion
"""

import numpy as np
from plyfile import PlyData, PlyElement

def convert_taming_opacity(ply_path):
    """Convert Taming sigmoid opacities to Three.js logit format (replaces original file)"""
    
    # Load PLY
    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex'].data
    vertex_array = np.array(vertices.tolist(), dtype=vertices.dtype)
    
    # Get sigmoid opacities
    sigmoid_opacities = vertex_array['opacity']
    
    # Convert to logit (inverse sigmoid)
    epsilon = 1e-7
    clamped = np.clip(sigmoid_opacities, epsilon, 1.0 - epsilon)
    logit_opacities = np.log(clamped / (1.0 - clamped))
    
    # Update and save back to same file
    vertex_array['opacity'] = logit_opacities.astype(np.float32)
    el = PlyElement.describe(vertex_array, 'vertex')
    PlyData([el]).write(ply_path)
    
    print(f"✅ Converted {len(sigmoid_opacities)} opacities in {ply_path} (file replaced)")