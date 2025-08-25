import os
import tempfile
import subprocess
import base64
import numpy as np

def process_ply_to_ksplat(
    ply_file_path,
    convert_taming_opacity=False,
    sh_degree=2,
    compression_level=1
):
    """
    Converte un PLY in KSPLAT usando lo script ufficiale create-ksplat.js
    di GaussianSplats3D.
    """
    src_ply = ply_file_path

    # (opzionale) Applica conversione TAMING → logit
    if convert_taming_opacity:
        from plyfile import PlyData, PlyElement
        ply = PlyData.read(ply_file_path)
        v = ply["vertex"]
        if "opacity" not in v.data.dtype.names:
            raise RuntimeError("Nel PLY manca il campo 'opacity'.")

        opac = np.asarray(v["opacity"], dtype=np.float64)
        eps = 1e-7
        opac = np.clip(opac, eps, 1.0 - eps)
        logit = np.log(opac / (1.0 - opac))

        arr = v.data.copy()
        arr["opacity"] = logit

        vert_elem = PlyElement.describe(arr, "vertex")
        ply_new = PlyData([vert_elem], text=ply.text)

        tmp_dir = tempfile.TemporaryDirectory()
        tmp_ply = os.path.join(tmp_dir.name, "tmp_opacity_logit.ply")
        ply_new.write(tmp_ply)
        src_ply = tmp_ply

    # Output temporaneo KSPLAT
    tmp_dir_out = tempfile.TemporaryDirectory()
    out_ksplat = os.path.join(tmp_dir_out.name, "tmp_out.ksplat")

    # Comando Node per eseguire lo script ufficiale
    script_path = "/code/gaussian-splats-3d/util/create-ksplat.js"
    cmd = [
        "node",
        script_path,
        os.path.abspath(src_ply),
        os.path.abspath(out_ksplat),
        str(int(compression_level)),   # lascia gli argomenti come li passi già ora
        "1",                           # ...
        "0,0,0",
        "5.0",
        "256",
        str(int(sh_degree)),]
    subprocess.check_call(cmd, cwd="/code/gaussian-splats-3d")

    with open(out_ksplat, "rb") as f:
        return f.read()

def save_splat_file(data_bytes, output_path):
    """Scrive i bytes del KSPLAT su disco."""
    with open(output_path, "wb") as f:
        f.write(data_bytes)
    print(f"💾 File salvato: {output_path}")
