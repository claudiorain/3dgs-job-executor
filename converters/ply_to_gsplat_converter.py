# You can use this to convert a .ply file to a .splat file programmatically in python
# Alternatively you can drag and drop a .ply file into the viewer at https://antimatter15.com/splat

from plyfile import PlyData
import numpy as np
from io import BytesIO


def process_ply_to_splat(ply_file_path, convert_taming_opacity=False):
    """
    Convert PLY to SPLAT format with optional Taming opacity conversion
    
    Args:
        ply_file_path: Path to PLY file
        convert_taming_opacity: If True, applies inverse sigmoid to opacity values
    
    Returns:
        bytes: SPLAT format data
    """
    plydata = PlyData.read(ply_file_path)
    vert = plydata["vertex"]
    
    # 🎯 OPACITY CONVERSION LOGIC
    if convert_taming_opacity:
        print("🔄 Applying Taming→Three.js opacity conversion during PLY→SPLAT...")
        
        # Convert sigmoid opacities to logit for Three.js compatibility
        original_opacities = np.array([v["opacity"] for v in vert])
        
        # Apply inverse sigmoid (logit)
        epsilon = 1e-7
        clamped = np.clip(original_opacities, epsilon, 1.0 - epsilon)
        logit_opacities = np.log(clamped / (1.0 - clamped))
        
        print(f"   Original opacity range: [{original_opacities.min():.3f}, {original_opacities.max():.3f}]")
        print(f"   Converted opacity range: [{logit_opacities.min():.3f}, {logit_opacities.max():.3f}]")
        
        # Create opacity lookup for fast access during processing
        opacity_lookup = {i: logit_opacities[i] for i in range(len(logit_opacities))}
    else:
        opacity_lookup = None
    
    # Sort indices by volume (same as original logic)
    if convert_taming_opacity:
        # For sorting, we need to use the converted opacities
        sorted_indices = np.argsort(
            -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"])
            / (1 + np.exp(-logit_opacities))
        )
    else:
        # Original sorting logic
        sorted_indices = np.argsort(
            -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"])
            / (1 + np.exp(-vert["opacity"]))
        )
    
    buffer = BytesIO()
    
    for idx in sorted_indices:
        v = plydata["vertex"][idx]
        
        # Position (unchanged)
        position = np.array([v["x"], v["y"], v["z"]], dtype=np.float32)
        
        # Scales (unchanged)
        scales = np.exp(
            np.array(
                [v["scale_0"], v["scale_1"], v["scale_2"]],
                dtype=np.float32,
            )
        )
        
        # Rotation (unchanged)
        rot = np.array(
            [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
            dtype=np.float32,
        )
        
        # Color calculation with opacity handling
        SH_C0 = 0.28209479177387814
        
        # 🎯 USE CONVERTED OPACITY IF AVAILABLE
        if convert_taming_opacity and opacity_lookup is not None:
            # Use converted logit opacity
            opacity_value = opacity_lookup[idx]
            alpha = 1 / (1 + np.exp(-opacity_value))
        else:
            # Use original opacity
            alpha = 1 / (1 + np.exp(-v["opacity"]))
        
        color = np.array(
            [
                0.5 + SH_C0 * v["f_dc_0"],
                0.5 + SH_C0 * v["f_dc_1"],
                0.5 + SH_C0 * v["f_dc_2"],
                alpha,  # Use processed alpha
            ]
        )
        
        # Write to buffer (same as original)
        buffer.write(position.tobytes())
        buffer.write(scales.tobytes())
        buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
        buffer.write(
            ((rot / np.linalg.norm(rot)) * 128 + 128)
            .clip(0, 255)
            .astype(np.uint8)
            .tobytes()
        )

    if convert_taming_opacity:
        print(f"   ✅ Processed {len(sorted_indices)} gaussians with opacity conversion")
    
    return buffer.getvalue()


def save_splat_file(splat_data, output_path):
    with open(output_path, "wb") as f:
        f.write(splat_data)