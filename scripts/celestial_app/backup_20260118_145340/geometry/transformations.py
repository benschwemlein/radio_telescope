import numpy as np

def rotz_deg(deg: float) -> np.ndarray:
    """Rotation matrix around Z axis by given degrees"""
    a = np.deg2rad(deg)
    ca, sa = np.cos(a), np.sin(a)
    return np.array([
        [ca, -sa, 0.0],
        [sa,  ca, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)