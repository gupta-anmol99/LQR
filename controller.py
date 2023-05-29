import numpy as np


def apply_state_controller(K, x):
    # feedback controller
    u = -np.dot(K, x)  # u = -Kx
    if u > 0:
        return 1, u  # if force_dem > 0 -> move cart right
    else:
        return 0, u  # if force_dem <= 0 -> move cart left
