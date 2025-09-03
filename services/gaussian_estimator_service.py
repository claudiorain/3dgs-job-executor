import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

# importa funzioni ausiliarie
from utils.image_hftr import compute_hftr, norm_hftr

def estimate_gaussians_from_stats(
    stats: Dict[str, Any],
    images_dir: Optional[str] = None,
    use_points: bool = True,
    use_features: bool = True,
    use_track: bool = True,
    use_reproj: bool = True,
    use_hf: bool = True,
    w_points: float = 0.4,
    w_features: float = 0.4,
    w_track: float = 0.2,
    w_reproj: float = 0.15,
    w_hf: float = 0.15,
    g_min: int = 100_000,
    g_max: int = 5_000_000,
    beta: float = 1.15,
) -> Dict[str, Any]:
    """
    Stima il numero di gaussiane finali usando le statistiche ricostruzione + eventuali immagini.
    Restituisce dict con suggerimenti parametri, debug, input.
    """

    # --- estrazione valori base ---
    n_points = stats.get("generated_points", None)
    avg_features = stats.get("avg_features", None)
    feature_consistency = stats.get("feature_consistency", None)
    avg_track_length = stats.get("avg_track_length", None)
    median_track_length = stats.get("median_track_length", None)
    reproj_err = stats.get("median_reproj_error_px", None)

    # --- normalizzazioni ---
    norm_points = 0.0
    if n_points:
        norm_points = np.clip((np.log10(n_points) - 4.3) / (5.7 - 4.3), 0, 1)

    norm_features = 0.0
    if avg_features:
        norm_features = np.clip((np.log10(avg_features) - 3.3) / (4.2 - 3.3), 0, 1)

    norm_track = 0.0
    if avg_track_length:
        norm_track = np.clip((np.log10(avg_track_length) - 0.3) / (1.1 - 0.3), 0, 1)

    norm_reproj = 0.0
    if reproj_err:
        # errore basso = qualità alta
        norm_reproj = np.clip(1.0 - (reproj_err / 50.0), 0, 1)

    norm_hf = 0.0
    hftr_val = None
    if use_hf and images_dir:
        hftr_val = compute_hftr(images_dir, sample_k=20, short_side=720, agg="blend")
        if hftr_val is not None:
            norm_hf = norm_hftr(hftr_val, lo=0.05, hi=0.35)

    # --- attivazione componenti ---
    components = {
        "points":   (use_points,  w_points,  norm_points),
        "features": (use_features,w_features,norm_features),
        "track":    (use_track,   w_track,   norm_track),
        "reproj":   (use_reproj,  w_reproj,  norm_reproj),
        "hf":       (use_hf,      w_hf,      norm_hf),
    }

    active = {k: v for k,v in components.items() if v[0]}
    if not active:
        # fallback se tutto disattivo
        g_scene = int(np.sqrt(g_min * g_max))
        return {
            "suggested_params": {"budget": g_scene, "cap_max": int(beta*g_scene), "beta": beta},
            "estimation": {"g_scene": g_scene, "score": None, "components": {}},
            "inputs": stats | {"images_dir": images_dir, "hftr_raw": hftr_val}
        }

    total_w = sum(v[1] for v in active.values())
    score = sum((v[1]/total_w)*v[2] for v in active.values())

    # --- stima finale in log-space ---
    log_gmin, log_gmax = np.log10(g_min), np.log10(g_max)
    log_est = log_gmin + score*(log_gmax-log_gmin)
    g_scene = int(round(10**log_est))

    # --- output ---
    debug_components = {}
    for k,(enabled,w,val) in components.items():
        debug_components[k] = {
            "enabled": enabled,
            "weight_base": w,
            "weight_eff": (w/total_w if enabled else 0.0),
            "value_norm": round(val,4),
        }

    return {
        "suggested_params": {
            "budget": g_scene,
            "cap_max": int(beta*g_scene),
            "beta": beta
        },
        "estimation": {
            "g_scene": g_scene,
            "score": round(score,4),
            "components": debug_components
        },
        "inputs": stats | {
            "images_dir": images_dir,
            "hftr_raw": hftr_val
        }
    }
