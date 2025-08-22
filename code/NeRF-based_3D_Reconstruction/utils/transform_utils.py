import numpy as np
from scipy.spatial.transform import Rotation as R

def load_sim3(filepath):
    with open(filepath, 'r') as f:
        
        vals = list(map(float, f.read().strip().split()))
        scale = vals[0]
        quat = vals[1:5]  # x, y, z, w
        t = vals[5:]

        trans = quaternion_to_transform_matrix(quat, t)
        return scale, trans[:3,:3], trans[:3,3]

def sim3_matrix(scale, rotation, translation):
    """
    Make a general sim3 matrix (scale -> rotation -> translation)
    """
    T = np.eye(4)
    T[:3, :3] = scale * rotation 
    T[:3, 3] = translation    
    return T

def inverse_sim3(scale, R, t):    
    R_inv = R.T
    s_inv = 1.0 / scale
    t_inv = -s_inv * R_inv @ t    
    return s_inv, R_inv, t_inv

def quaternion_to_transform_matrix(q, t):
    """
    Converts a 4x1 quaternion and a 3x1 translation vector to a 4x4 transformation matrix.
    
    Args:
    - quaternion (np.ndarray): 4x1 array representing the quaternion (w, x, y, z)
    - t (np.ndarray): 3x1 array representing the translation vector (tx, ty, tz)
    
    Returns:
    - transform_matrix (np.ndarray): 4x4 transformation matrix
    """
    # Ensure the quaternion is a numpy array
    q = np.asarray(q).flatten()
    t = np.asarray(t).flatten()

    # Normalize the quaternion
    q = q / np.linalg.norm(q)
    
    # Extract the components of the quaternion
    w, x, y, z = q
    
    # Compute the rotation matrix from the quaternion
    rotation_matrix = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])
    
    # Create the 4x4 transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = t
    
    return transform_matrix