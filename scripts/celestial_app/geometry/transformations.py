
import numpy as np

def normalize_vector(vec: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """
    Normalize a vector with numerical stability.
    
    Args:
        vec: Input vector to normalize
        epsilon: Small value to prevent division by zero
    
    Returns:
        Normalized unit vector
    """
    return vec / (np.linalg.norm(vec) + epsilon)

def rotz_deg(deg: float) -> np.ndarray:
    """Rotation matrix around Z axis by given degrees"""
    a = np.deg2rad(deg)
    ca, sa = np.cos(a), np.sin(a)
    return np.array([
        [ca, -sa, 0.0],
        [sa,  ca, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

