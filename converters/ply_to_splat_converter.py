import math
import struct
import os
import numpy as np
from io import BytesIO

# Costante per spherical harmonics
SH_C0 = 0.28209479177387814

def process_ply_to_splat(ply_file_path, convert_taming_opacity=False):
    """
    Convert PLY to SPLAT format with optional Taming opacity conversion
    Basato sul codice originale con fix NaN e gestione errori robusta
    
    Args:
        ply_file_path: Path to PLY file
        convert_taming_opacity: If True, applies inverse sigmoid to opacity values
    
    Returns:
        bytes: SPLAT format data
    """
    try:
        # Import plyfile con fallback se non disponibile
        try:
            from plyfile import PlyData
        except ImportError:
            raise ImportError("plyfile is required. Install with: pip install plyfile")
        
        print(f"🔄 Starting PLY to SPLAT conversion...")
        print(f"📂 Input: {ply_file_path}")
        print(f"🎯 Taming opacity conversion: {'ON' if convert_taming_opacity else 'OFF'}")
        
        # Leggi il file PLY
        plydata = PlyData.read(ply_file_path)
        vert = plydata["vertex"]
        
        print(f"📊 PLY contains {len(vert):,} vertices")
        
        # 🎯 OPACITY CONVERSION LOGIC (dal tuo codice originale)
        opacity_lookup = None
        if convert_taming_opacity:
            print("🔄 Applying Taming→Three.js opacity conversion during PLY→SPLAT...")
            
            # Convert sigmoid opacities to logit for Three.js compatibility
            original_opacities = np.array([safe_float(v["opacity"], 0.0) for v in vert])
            
            # Apply inverse sigmoid (logit) - ESATTO dal tuo codice
            epsilon = 1e-7
            clamped = np.clip(original_opacities, epsilon, 1.0 - epsilon)
            logit_opacities = np.log(clamped / (1.0 - clamped))
            
            print(f"   Original opacity range: [{original_opacities.min():.3f}, {original_opacities.max():.3f}]")
            print(f"   Converted opacity range: [{logit_opacities.min():.3f}, {logit_opacities.max():.3f}]")
            
            # Create opacity lookup for fast access during processing
            opacity_lookup = {i: logit_opacities[i] for i in range(len(logit_opacities))}
        
        # Sort indices by volume (stesso identico codice originale)
        print("🔄 Sorting gaussians by volume...")
        
        try:
            if convert_taming_opacity and opacity_lookup is not None:
                # For sorting, we need to use the converted opacities
                logit_opacities = np.array([opacity_lookup[i] for i in range(len(vert))])
                sorted_indices = np.argsort(
                    -np.exp(safe_array([v["scale_0"] for v in vert]) + 
                           safe_array([v["scale_1"] for v in vert]) + 
                           safe_array([v["scale_2"] for v in vert]))
                    / (1 + np.exp(-logit_opacities))
                )
            else:
                # Original sorting logic
                sorted_indices = np.argsort(
                    -np.exp(safe_array([v["scale_0"] for v in vert]) + 
                           safe_array([v["scale_1"] for v in vert]) + 
                           safe_array([v["scale_2"] for v in vert]))
                    / (1 + np.exp(-safe_array([v["opacity"] for v in vert])))
                )
        except Exception as e:
            print(f"⚠️  Error in sorting, using original order: {e}")
            sorted_indices = np.arange(len(vert))
        
        buffer = BytesIO()
        error_count = 0
        
        print(f"🔄 Converting {len(sorted_indices):,} gaussians to SPLAT format...")
        
        for idx_num, idx in enumerate(sorted_indices):
            try:
                v = plydata["vertex"][idx]
                
                # Position (unchanged)
                position = np.array([
                    safe_float(v["x"], 0.0),
                    safe_float(v["y"], 0.0), 
                    safe_float(v["z"], 0.0)
                ], dtype=np.float32)
                
                # Scales (unchanged) - con gestione sicura di exp
                scales = np.array([
                    safe_exp(safe_float(v["scale_0"], 0.0)),
                    safe_exp(safe_float(v["scale_1"], 0.0)),
                    safe_exp(safe_float(v["scale_2"], 0.0))
                ], dtype=np.float32)
                
                # Rotation (unchanged) - con normalizzazione sicura
                rot_raw = np.array([
                    safe_float(v["rot_0"], 1.0),
                    safe_float(v["rot_1"], 0.0),
                    safe_float(v["rot_2"], 0.0),
                    safe_float(v["rot_3"], 0.0)
                ], dtype=np.float32)
                
                # Normalizza quaternione sicuro
                rot_norm = np.linalg.norm(rot_raw)
                if rot_norm < 1e-9:
                    rot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                else:
                    rot = rot_raw / rot_norm
                
                # 🎯 COLOR CALCULATION WITH OPACITY HANDLING (identico al tuo codice)
                
                # USE CONVERTED OPACITY IF AVAILABLE
                if convert_taming_opacity and opacity_lookup is not None:
                    # Use converted logit opacity
                    opacity_value = opacity_lookup[idx]
                    alpha = 1 / (1 + np.exp(-safe_float(opacity_value, 0.0)))
                else:
                    # Use original opacity
                    alpha = 1 / (1 + np.exp(-safe_float(v["opacity"], 0.0)))
                
                # Color con SH (identico al tuo codice)
                color = np.array([
                    0.5 + SH_C0 * safe_float(v["f_dc_0"], 0.0),
                    0.5 + SH_C0 * safe_float(v["f_dc_1"], 0.0),
                    0.5 + SH_C0 * safe_float(v["f_dc_2"], 0.0),
                    alpha,  # Use processed alpha
                ])
                
                # Debug per i primi 5 vertex quando Taming è attivo
                if convert_taming_opacity and idx_num < 5:
                    print(f"🔧 Vertex {idx}: opacity_raw={safe_float(v['opacity'], 0.0):.3f} → "
                          f"logit={opacity_lookup[idx]:.3f} → alpha={alpha:.3f}")
                
                # Write to buffer (identico al tuo codice)
                buffer.write(position.tobytes())
                buffer.write(scales.tobytes())
                buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
                buffer.write(
                    ((rot / np.linalg.norm(rot)) * 128 + 128)
                    .clip(0, 255)
                    .astype(np.uint8)
                    .tobytes()
                )
                
                # Progress ogni 10k
                if (idx_num + 1) % 10000 == 0:
                    progress = ((idx_num + 1) / len(sorted_indices)) * 100
                    print(f"📊 Progress: {idx_num + 1:,}/{len(sorted_indices):,} ({progress:.1f}%)")
                
            except Exception as e:
                error_count += 1
                if error_count <= 10:
                    print(f"⚠️  Error processing vertex {idx}: {e}")
                continue
        
        if error_count > 10:
            print(f"⚠️  Total {error_count} vertices had errors and were skipped")
        
        result_data = buffer.getvalue()
        file_size_mb = len(result_data) / (1024 * 1024)
        
        print(f"✅ SPLAT conversion completed successfully!")
        print(f"📊 Final stats:")
        print(f"   - Output splats: {len(sorted_indices) - error_count:,}")
        print(f"   - File size: {file_size_mb:.1f}MB") 
        print(f"   - Errors handled: {error_count}")
        
        if convert_taming_opacity:
            print(f"   ✅ Processed with Taming opacity conversion")
        
        return result_data
        
    except Exception as e:
        print(f"❌ SPLAT conversion failed: {e}")
        import traceback
        traceback.print_exc()
        raise e

def safe_float(value, default=0.0):
    """Converte un valore a float, gestendo NaN e infiniti"""
    try:
        if value is None:
            return default
        float_val = float(value)
        if math.isnan(float_val) or math.isinf(float_val):
            return default
        return float_val
    except (ValueError, TypeError, OverflowError):
        return default

def safe_exp(x, default=1.0):
    """Exponential sicuro che gestisce overflow"""
    try:
        safe_x = safe_float(x, 0.0)
        if abs(safe_x) > 100:  # Previeni overflow
            safe_x = math.copysign(100, safe_x)
        return math.exp(safe_x)
    except (ValueError, TypeError, OverflowError):
        return default

def safe_array(values):
    """Converte lista di valori in array numpy sicuro"""
    try:
        safe_values = [safe_float(v, 0.0) for v in values]
        return np.array(safe_values, dtype=np.float32)
    except Exception:
        return np.zeros(len(values), dtype=np.float32)

def save_splat_file(splat_data, output_path):
    """
    Salva i dati SPLAT su file
    Mantiene la stessa interfaccia della funzione originale
    """
    try:
        with open(output_path, "wb") as f:
            f.write(splat_data)
        print(f"💾 SPLAT file saved: {output_path}")
    except Exception as e:
        print(f"❌ Failed to save SPLAT file: {e}")
        raise e

