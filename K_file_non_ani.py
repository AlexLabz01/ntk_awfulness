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


def compute_new_K(K_prev, C_W, C_b):
    'computes new K given prev K using 5.1'
    K_new = np.zeros_like(K_prev)

    for alpha, beta in list(combinations_with_replacement(range(K_prev.shape[0]), 2)):

        new_entry = C_b + C_W * (np.exp(-0.5 * (K_prev[alpha, alpha] + K_prev[beta, beta]))) * np.sinh(
            K_prev[alpha, beta])
        K_new[alpha, beta] = new_entry

        if alpha != beta:
            K_new[beta, alpha] = new_entry

    return K_new


def compute_new_K_vectorized(K_prev, C_W, C_b):
    'does same thing as compute_new_K, but avoids for loops and is completely vectorized'
    diag = np.diag(K_prev)

    # diag[:, None] + diag[None, :] is the matrix who's i j element
    # is K_ii + K_jj.
    exp_factor = np.exp(-0.5 * (diag[:, None] + diag[None, :]))
    sinh_term = np.sinh(K_prev)

    K_new = C_b + C_W * exp_factor * sinh_term
    K_new = 0.5 * (K_new + K_new.T)

    return K_new


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
    return prev_vecs - ( lr) * np.dot(NTK_full_train, train_vecs - labels)


def compute_tth_pred(inpt_vecs, labels, C_W, C_b, depth, lambda_b, lambda_W, lr, num_train, t):
    # inpt_vecs is a matrix who's columns are the input vectors to the neural network
    z_prev = np.zeros(inpt_vecs.shape[1])
    NTK_full_array, _ = compute_l_layer_theta(inpt_vecs, C_W, C_b, depth, lambda_b, lambda_W)
    NTK_full = NTK_full_array[-1]
    NTK_full_train = NTK_full[:, :num_train]
    for i in range(t):
        z_next = compute_next_prediction(z_prev, labels, lr, NTK_full_train, num_train)
        z_prev = z_next
    return z_prev


if __name__ == "__main__":
    np.random.seed(45)
    # inpt_vecs = np.transpose(np.array([[0.3, 0.5], [0.1, 0.6]]))
    # n_features = 2
    N = 3
    angles = np.linspace(0.5, 2 * np.pi, N, endpoint=False)
    x1 = np.cos(angles)
    x2 = np.sin(angles)
    train_vecs = np.vstack([x1, x2])  # shape (2, N), columns = training vectors
    labels = np.random.uniform(-5, 5, size=N)

    # Test data: different angles on the unit circle
    M = 200
    test_angles = np.linspace(0, 2 * np.pi, M, endpoint=False)
    test_x1 = np.cos(test_angles)
    test_x2 = np.sin(test_angles)
    test_vecs = np.vstack([test_x1, test_x2])  # shape (2, M)

    inpt_vecs = np.concatenate([train_vecs, test_vecs], axis=1)
    inpt_angles = np.concatenate([angles, test_angles])
    print("Size[0] of inpt_vecs: ", inpt_vecs.shape[0])

    C_W = 1.0
    C_b = 0.0
    lambda_b = 0.0
    lambda_W = 1
    depth = 3
    num_train = N
    t = 2000
    lr = 0.02

    outputs = compute_tth_pred(inpt_vecs, labels, C_W, C_b, depth, lambda_b, lambda_W, lr, num_train, t)

    plt.scatter(inpt_angles, outputs, color='r', label='predictions')
    plt.scatter(inpt_angles[:num_train], labels, color='b', label='training points')
    plt.xlabel("angle (rad)")
    plt.ylabel("output")
    plt.legend()
    plt.show()

    # C_W = 1.0
    # C_b = 0.0
    # lambda_W = 1.0
    # lambda_b = 0.0
    # depth = 4
    # v1 = [0.3, 0.5]
    # v2 = [0.1, 0.6]
    # inpt_vecs = np.column_stack((v1, v2))
    # theta_list, K_list = compute_l_layer_theta(inpt_vecs, C_W, C_b, depth, lambda_b, lambda_W)
    # for layer in range(len(K_list)):
    #     print("Layer " + str(layer + 1) + " K matrix:")
    #     print(K_list[layer])
    #     print("Layer " + str(layer + 1) + " theta matrix:")
    #     print(theta_list[layer])

