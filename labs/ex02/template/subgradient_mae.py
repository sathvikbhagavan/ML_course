import numpy as np


def compute_subgradient_mae(y, tx, w):
    """Compute a subgradient of the MAE at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the subgradient of the MAE at w.
    """
    e = y - tx@w
    mae_term = np.zeros((y.shape[0], 1))
    mae_term[e > 0.0] = 1.0
    mae_term[e < 0.0] = -1.0
    # mae_term[e == 0.0] = 1.0
    mae_term = mae_term.reshape((1, y.shape[0]))/y.shape[0]
    return (-mae_term@tx).T[:, 0]
