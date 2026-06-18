import numpy as np

import numpy as np


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



def compute_H_next_loops(H_l, W, sigma, sigma0, lambda_b, lambda_W):
    n_l, _, alpha_dim, _ = H_l.shape
    n_lplus1 = W.shape[0]

    H_next = np.zeros((n_lplus1, n_lplus1, alpha_dim, alpha_dim))

    # Precompute M_diag
    M_diag = np.zeros((alpha_dim, alpha_dim))
    for a1 in range(alpha_dim):
        for a2 in range(alpha_dim):
            s = 0.0
            for j in range(n_l):
                s += sigma[j, a1] * sigma[j, a2]
            M_diag[a1, a2] = lambda_b + lambda_W * s / n_l

    # Compute explicitly
    for i1 in range(n_lplus1):
        for i2 in range(n_lplus1):
            for a1 in range(alpha_dim):
                for a2 in range(alpha_dim):
                    val = 0.0
                    if i1 == i2:
                        val += M_diag[a1, a2]
                    for j1 in range(n_l):
                        for j2 in range(n_l):
                            val += (W[i1, j1] * W[i2, j2] *
                                    sigma0[j1, a1] * sigma0[j2, a2] *
                                    H_l[j1, j2, a1, a2])
                    H_next[i1, i2, a1, a2] = val

    return H_next


def main():
    np.random.seed(41)

    n_l = 15
    n_lplus1 = 11
    alpha_dim = 10

    # Random input tensors
    H_l = np.random.randn(n_l, n_l, alpha_dim, alpha_dim)
    W = np.random.randn(n_lplus1, n_l)
    sigma = np.random.randn(n_l, alpha_dim)
    sigma0 = np.random.randn(n_l, alpha_dim)
    lambda_b = 1.0
    lambda_W = 0.5

    # Compute both outputs
    H_next_vec = compute_H_next(H_l, W, sigma, sigma0, lambda_b, lambda_W)
    H_next_loop = compute_H_next_loops(H_l, W, sigma, sigma0, lambda_b, lambda_W)

    # Compare
    are_close = np.allclose(H_next_vec, H_next_loop, atol=1e-8)
    print("Are the vectorized and looped results close?:", are_close)

    # Percent difference
    diff = np.abs(H_next_vec - H_next_loop)
    denom = np.maximum(np.abs(H_next_loop), 1e-12)
    percent_diff = 100 * diff / denom

    max_percent = np.max(percent_diff)
    max_index = np.unravel_index(np.argmax(percent_diff), percent_diff.shape)

    print(f"Max percent difference: {max_percent:.6f}% at index {max_index}")
    print(f"Loop value: {H_next_loop[max_index]:.6e}, Vectorized value: {H_next_vec[max_index]:.6e}")


if __name__ == "__main__":
    main()
