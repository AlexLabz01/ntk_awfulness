import numpy as np
from itertools import combinations_with_replacement
from matplotlib import pyplot as plt

def lay_comp(inpt, width, CW):
    # Preconditions:
    # width must be the same length as the shape of the inpt vector
    var = CW / width
    std = np.sqrt(var)
    # Sampling covariance matrix from normal distribution (mean 0, std, matrix_size)
    W = np.random.normal(loc=0.0, scale=std, size=(width, inpt.size))
    return np.dot(W, inpt)

def compute_new_theta_K(theta_prev, C_W, C_b, lambda_b, lambda_W, K):
    # decided to rewrite code from compute_new_K_vectorized for efficiency to avoid redundant steps, rather than calling it
    diag = np.diag(K)
    # diag[:, None] + diag[None, :] is the matrix who's i j element
    # is K_ii + K_jj.
    exp_factor = np.exp(-0.5 * (diag[:, None] + diag[None, :]))
    sinh_term = np.sinh(K)
    cosh_term = np.cosh(K)

    K_new = C_b + C_W * exp_factor * sinh_term
    K_new = 0.5 * (K_new + K_new.T)

    K_cos = C_b + C_W * exp_factor * cosh_term
    K_cos = 0.5 * (K_cos + K_cos.T)

    theta_new = lambda_b + lambda_W * K_new + C_W * K_cos * theta_prev
    return theta_new, K_new

def compute_new_K_sin(
    K_prev: torch.Tensor,
    C_W: float,
    C_b: float = 0.0,
) -> torch.Tensor:

    diag = torch.diagonal(K_prev)                       # K_aa
    exp_factor = torch.exp(-0.5 * (diag[:, None] + diag[None, :]))
    sinh_term = torch.sinh(K_prev)

    K_next = C_b + C_W * exp_factor * sinh_term
    K_next = 0.5 * (K_next + K_next.T)                  # enforce symmetry

    return K_next

def compute_l_layer_theta(inpt_vecs, C_W, C_b, depth, lambda_b, lambda_W):
    # inpt_vecs is a matrix who's columns are input vectors to the neural network
    K_list = []
    K_prev = C_b + (C_W / inpt_vecs.shape[0]) * np.dot(np.transpose(inpt_vecs), inpt_vecs)
    K_list.append(K_prev)

    theta_list = []
    theta_prev = lambda_b + (lambda_W / inpt_vecs.shape[0]) * np.dot(np.transpose(inpt_vecs), inpt_vecs)
    theta_list.append(theta_prev)

    for dep in range(depth - 1):
        theta_new, K_new = compute_new_theta_K(theta_prev, C_W, C_b, lambda_b, lambda_W, K_prev)
        K_list.append(K_new)
        theta_list.append(theta_new)
        K_prev = K_new
        theta_prev = theta_new

    return theta_list, K_list

def compute_next_prediction(prev_vecs, labels, lr, NTK_full_train, num_train):
    # prev vecs is a vector of 1d outputs of the neural net, each entry corresponding to a specific alpha (inpt vec num in training set)
    # labels is a vector of labels, each entry corresponding to a specific alpha
    train_vecs = prev_vecs[:num_train]
    return prev_vecs - (2 * lr / 3) * np.dot(NTK_full_train, train_vecs - labels)

def compute_tth_pred(inpt_vecs, labels, C_W, C_b, depth, lambda_b, lambda_W, lr, num_train, t):
    # inpt_vecs is a matrix who's columns are the input vectors to the neural network
    # returns the final predictions
    z_prev = np.zeros(inpt_vecs.shape[1])
    NTK_full_array, _ = compute_l_layer_theta(inpt_vecs, C_W, C_b, depth, lambda_b, lambda_W)
    NTK_full = NTK_full_array[-1]
    NTK_full_train = NTK_full[:, :num_train]
    for i in range(t):
        z_next = compute_next_prediction(z_prev, labels, lr, NTK_full_train, num_train)
        z_prev = z_next
    return z_prev

def compute_tth_pred_per_step(inpt_vecs, labels, C_W, C_b, depth, lambda_b, lambda_W, lr, num_train, t):
    # Same as compute_tth_pred but records predictions at each time step for animation
    z_prev = np.zeros(inpt_vecs.shape[1])
    NTK_full_array, _ = compute_l_layer_theta(inpt_vecs, C_W, C_b, depth, lambda_b, lambda_W)
    NTK_full = NTK_full_array[-1]
    NTK_full_train = NTK_full[:, :num_train]

    all_preds = []
    for i in range(t):
        z_next = compute_next_prediction(z_prev, labels, lr, NTK_full_train, num_train)
        all_preds.append(z_next.copy())
        z_prev = z_next
    return np.array(all_preds)
