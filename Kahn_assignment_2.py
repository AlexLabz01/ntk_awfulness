import numpy as np
from itertools import combinations
from matplotlib import pyplot as plt

def lay_comp(inpts, width_out, CW):

    width_in = inpts.shape[0]
    var = CW / width_in
    std = np.sqrt(var)

    W = np.random.normal(loc=0.0, scale=std, size=(width_out, width_in))
    out = np.dot(W, inpts)
    return out, W

def net_output(inpt, width, CW, num_layers):

    outputs = []
    weights = []

    current = inpt


    for _ in range(num_layers - 1):
        z, W = lay_comp(current, width, CW)
        weights.append(W)
        outputs.append(z)
        current = np.sin(z)


    z, W = lay_comp(current, 1, CW)
    weights.append(W)
    outputs.append(z)

    return z, outputs, weights


def compute_H_next(H_l, W, sigma, sigma0, lambda_b, lambda_W):
    """
    Compute H^{(l+1)} given the previous NTK H^{(l)}, weights W,
    activations sigma (forward pass), sigma0 (backward pass), and regularization.

    Parameters:
        H_l:      (n_l, n_l, α, α)
        W:        (n_{l+1}, n_l)
        sigma:    (n_l, α)
        sigma0:   (n_l, α)
        lambda_b: scalar
        lambda_W: scalar

    Returns:
        H^{(l+1)}: (n_{l+1}, n_{l+1}, α, α)
    """

    n_l, _, alpha_dim, _ = H_l.shape
    n_lplus1 = W.shape[0]

    # Compute M_diag (α x α)
    M_diag = lambda_b + lambda_W * (sigma.T @ sigma) / n_l  # shape: (α, α)

    # Reshape and weight H_l
    # H_l: (n_l, n_l, α, α)
    # sigma0: (n_l, α) so broadcasted sigma0[i, α1] * sigma0[j, α2]
    M = H_l * sigma0[:, None, :, None] * sigma0[None, :, None, :]  # shape: (n_l, n_l, α, α)

    # Reshape to: (α^2, n_l, n_l)
    M_reshaped = M.transpose(2, 3, 0, 1).reshape(alpha_dim * alpha_dim, n_l, n_l)

    # Apply W to both sides: W (n_{l+1} x n_l)
    WM = np.einsum('ik,bkj->bij', W, M_reshaped)  # shape: (α^2, n_{l+1}, n_l)
    WM = np.einsum('bij,jm->bim', WM, W.T)  # shape: (α^2, n_{l+1}, n_{l+1})

    # Reshape back to: (n_{l+1}, n_{l+1}, α, α)
    H_next = WM.reshape(alpha_dim, alpha_dim, n_lplus1, n_lplus1).transpose(2, 3, 0, 1)

    # Add M_diag to diagonals for each (α1, α2)
    diag_indices = np.arange(n_lplus1)
    H_next[diag_indices, diag_indices, :, :] += M_diag

    return H_next


def initialize_H(inpt, n, lambda_W):
    """
    Initialize the first-layer NTK tensor H.

    Args:
        inpt: np.ndarray of shape (2, alpha) -- input vectors (dim=2)
        n: int -- width of first hidden layer
        lambda_W: float -- scaling factor (weight variance)

    Returns:
        H: np.ndarray of shape (n, n, alpha, alpha)
    """
    alpha = inpt.shape[1]

    # Compute Gram matrix (alpha, alpha)
    gram = (lambda_W / 2.0) * (inpt.T @ inpt)

    # Initialize H tensor (n, n, alpha, alpha)
    H = np.zeros((n, n, alpha, alpha))

    # Fill diagonal blocks with Gram matrix
    diag_indices = np.arange(n)
    H[diag_indices, diag_indices, :, :] = gram

    return H



def compute_H(inpt, width, depth):
    _, outputs, weights = net_output(inpt, width, 1, depth)

    H = initialize_H(inpt, width,1)

    for l in range(1, depth):
        H = compute_H_next(H, weights[l], np.sin(outputs[l]), np.cos(outputs[l]), 0, 1)
    return H, outputs

def compute_next_prediction(prev_vecs, labels, lr, NTK_full_train, num_train):
    prev_vecs_reshaped = prev_vecs.reshape(-1)
    train_vecs = prev_vecs_reshaped[:num_train]
    diff = train_vecs - labels
    return prev_vecs - lr * np.dot(NTK_full_train, diff)

def compute_tth_pred(inpt_vecs, labels, lr, num_train, CW, width, depth, t):
    NTK_full_array, outputs = compute_H(inpt_vecs, width, depth)
    z_prev = outputs[-1]
    NTK_full = NTK_full_array[-1]
    num_vecs = inpt_vecs.shape[1]
    NTK_full = NTK_full.reshape(num_vecs, num_vecs)
    NTK_full_train = NTK_full[:, :num_train]
    for i in range(t):
        z_next = compute_next_prediction(z_prev, labels, lr, NTK_full_train, num_train)
        z_prev = z_next
    return z_prev

def main():
    n_inputs = 105
    inpt_angles = np.random.uniform(low=0, high=2 * np.pi, size=n_inputs)
    inpt_vecs = np.zeros((2, n_inputs))
    for i in range(n_inputs):
        inpt_vecs[0][i] += np.cos(inpt_angles[i])
        inpt_vecs[1][i] += np.sin(inpt_angles[i])



    width = 105
    depth = 20
    lr=0.001
    num_train = 3
    labels = np.random.uniform(low=-5, high=5, size=num_train) # same specs as wiki animation
    H, outputs = compute_H(inpt_vecs, width, depth)
    CW=1
    out = compute_tth_pred(inpt_vecs, labels, lr, num_train, CW, width, depth, 500_000)


    plt.scatter(inpt_angles, out, color='r', label='predictions')
    plt.scatter(inpt_angles[:num_train], labels, color='b', label='training points')
    plt.xlabel("angle (rad)")
    plt.ylabel("output")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
