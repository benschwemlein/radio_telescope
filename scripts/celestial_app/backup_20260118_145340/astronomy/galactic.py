import numpy as np
from .coordinates import gal_to_eq_matrix_j2000

def build_milky_way_band_equatorial(radius=1.0, half_width_deg=10.0, n=1600, m=33, seed=7):
    """Generate Milky Way band points in equatorial coordinates"""
    rng = np.random.default_rng(seed)
    R_g2e = gal_to_eq_matrix_j2000()
    
    l = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False).astype(np.float32)
    hw = np.deg2rad(half_width_deg).astype(np.float32)
    b_vals = np.linspace(-hw, hw, m).astype(np.float32)
    
    pts = []
    alpha = []
    
    for b in b_vals:
        x = np.cos(b) * np.cos(l)
        y = np.cos(b) * np.sin(l)
        z = np.sin(b) * np.ones_like(l)
        
        g = np.stack([x, y, z], axis=1).astype(np.float32)
        
        noise = rng.normal(0.0, 1.0, size=g.shape).astype(np.float32)
        noise = noise / (np.linalg.norm(noise, axis=1, keepdims=True) + 1e-12)
        g = g + 0.008 * noise
        g = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-12)
        
        e = (R_g2e @ g.T).T.astype(np.float32)
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
        e = radius * e
        
        edge = abs(float(b) / float(hw)) if float(hw) > 0.0 else 0.0
        a = (1.0 - edge) ** 1.8
        
        pts.append(e)
        alpha.append(np.full(n, a, dtype=np.float32))
    
    P = np.concatenate(pts, axis=0).astype(np.float32)
    A = np.concatenate(alpha, axis=0).astype(np.float32)
    return P, A