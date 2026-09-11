import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml

def compute_new_theta_K(theta_prev,C_W,C_b,lambda_b,lambda_W,K_prev):

    diag = torch.diagonal(K_prev)

    exp_factor = torch.exp(-0.5 * (diag[:, None] + diag[None, :]))

    sinh_term = torch.sinh(K_prev)
    cosh_term = torch.cosh(K_prev)

    K_new = C_b + C_W * exp_factor * sinh_term
    K_new = 0.5 * (K_new + K_new.T)

    K_cos = C_b + C_W * exp_factor * cosh_term
    K_cos = 0.5 * (K_cos + K_cos.T)

    theta_new = (lambda_b + lambda_W * K_new + C_W * K_cos * theta_prev)

    # theta_new = 0.5 * (theta_new + theta_new.T)

    return theta_new, K_new


def compute_l_layer_theta(inpt_vecs,C_W,C_b,depth,lambda_b,lambda_W):



    """
    inpt_vecs has shape of input dimension x number of inputs
    """

    input_dimension = inpt_vecs.shape[0]

    gram_matrix = inpt_vecs.T @ inpt_vecs

    K_prev = (C_b + (C_W / input_dimension) * gram_matrix)

    theta_prev = (lambda_b + (lambda_W / input_dimension) * gram_matrix)

    K_prev = 0.5 * (K_prev + K_prev.T)
    theta_prev = 0.5 * (theta_prev + theta_prev.T)

    K_list = [K_prev]
    theta_list = [theta_prev]

    for _ in range(depth - 1):
        theta_new, K_new = compute_new_theta_K(theta_prev=theta_prev, C_W=C_W, C_b=C_b, lambda_b=lambda_b,
            lambda_W=lambda_W,
            K_prev=K_prev,
        )

        theta_list.append(theta_new)
        K_list.append(K_new)

        theta_prev = theta_new
        K_prev = K_new

    return theta_list, K_list



def get_mnist_inputs(images_per_class):
    mnist = fetch_openml(
        "mnist_784",
        as_frame=False,
        cache=False,
    )

    labels = mnist.target.astype("int64")
    selected_indices = []

    for digit in range(10):
        digit_indices = (labels == digit).nonzero()[0][:images_per_class]
        selected_indices.extend(digit_indices)

    X = mnist.data[selected_indices].astype("float32")

    return torch.tensor(X).T / 255.0


def compute_XII(H, eta):
    n = H.shape[0]
    I = np.eye(n)

    # Get M tensor in n^2 by n^2 form
    M = np.kron(H, I) + np.kron(I, H) - eta * np.kron(H, H)

    #inverting M when in n^2 by n^2 form
    X_flat = np.linalg.inv(M)

    # get back the desired n by n by n by n tensor from n^2 by n^2 matrix
    X = X_flat.reshape(n, n, n, n)


    return X


def check_XII_tensor(X, H, eta, tolerance=1e-10):
    n = H.shape[0]
    I = np.eye(n)

    # getting tensor M via inefficient einsums
    M = (np.einsum('ac,bd->abcd', H, I) + np.einsum('ac,bd->abcd', I, H)
        - eta * np.einsum('ac,bd->abcd', H, H)
    )

    lhs = np.einsum('ijmn,mnkl->ijkl', X, M)

    rhs = np.einsum('ik,jl->ijkl', I, I)

    #takes max of all entries in the tensor
    max_err = np.max(np.abs(lhs - rhs))
    return max_err < tolerance, max_err


if __name__ == "__main__":
    C_W = 1.0
    C_b = 0.0
    lambda_W = 1.0
    lambda_b = 0.0
    layer_to_check = 2
    images_per_class = 3
    eta = 0.3

    inpt_vecs = get_mnist_inputs(images_per_class)
    theta_list, _ = compute_l_layer_theta( inpt_vecs, C_W, C_b, layer_to_check, lambda_b, lambda_W, )

    H = theta_list[layer_to_check - 1].numpy()

    print("H shape:", H.shape)

    print("Smallest H eigenvalue:", np.linalg.eigvalsh(H)[0])

    X = compute_XII(H, eta)

    print("Done computing XII")

    success, error = check_XII_tensor(X, H, eta)

    if success:
        print("tensor equation works within reasonable accuracy")
    else: print("tensor equation not within reasonable accuracy")

    print("Max Error:", error)