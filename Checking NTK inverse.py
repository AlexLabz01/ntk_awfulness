import torch
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml

# Trying with float.64
#torch.set_default_dtype(torch.float64)

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


@torch.no_grad()
def check_theta_dimensions(inpt_vecs_full,matrix_dimensions,layer,C_W,C_b,lambda_b,lambda_W):

    results = []

    for dimension in matrix_dimensions:
        theta_list, _ = compute_l_layer_theta(inpt_vecs_full[:, :dimension], C_W, C_b, layer, lambda_b, lambda_W)

        theta = theta_list[layer - 1]
        smallest_eigenvalue = torch.linalg.eigvalsh(theta)[0].item()
        theta_inverse = torch.linalg.inv(theta)
        identity = torch.eye(dimension)

        inverse_error = torch.max(torch.abs(theta @ theta_inverse - identity)).item()

        results.append((dimension, smallest_eigenvalue, inverse_error))

    return results


def print_theta_results(results):
    print("dimension, smallest eigenvalue, inverse error")

    for dimension, smallest_eigenvalue, inverse_error in results:
        print(dimension, smallest_eigenvalue, inverse_error)

def plot_smallest_eigenvalues(results, layer):
    dimensions = [result[0] for result in results]
    eigenvalues = [result[1] for result in results]

    plt.plot(dimensions, eigenvalues, marker="o")
    plt.axhline(0)

    plt.xlabel("number of inputs")
    plt.ylabel("Smallest eigenvalue")
    plt.title(f"Layer {layer} theta smallest eigenvalue")

    plt.grid()
    plt.tight_layout()
    plt.savefig(f"theta_layer_{layer}_smallest_eigenvalue.png", dpi=200)
    plt.show()


def plot_inverse_errors(results, layer):
    dimensions = [result[0] for result in results]
    errors = [result[2] for result in results]

    plt.semilogy(dimensions, errors, marker="o")

    plt.xlabel("input number")
    plt.ylabel("Maximum error to identity")
    plt.title(f"Layer {layer} theta inverse precision")

    plt.grid()
    plt.tight_layout()
    plt.savefig(f"theta_layer_{layer}_inverse_error.png", dpi=200)
    plt.show()

@torch.no_grad()
def plot_eigenvalue_spectrum(inpt_vecs,layer,C_W,C_b,lambda_b,lambda_W):


    theta_list, _ = compute_l_layer_theta(inpt_vecs, C_W, C_b, layer, lambda_b, lambda_W)

    theta = theta_list[layer - 1]
    eigenvalues = torch.linalg.eigvalsh(theta)

    plt.semilogy(range(1, len(eigenvalues) + 1), eigenvalues, marker="o")

    plt.xlabel("eigenvalue index")
    plt.ylabel("eigenvalue")
    plt.title(f"Layer {layer} theta eigenvalue spectrum")

    plt.grid()
    plt.tight_layout()
    plt.savefig(f"theta_layer_{layer}_eigenvalue_spectrum.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(44)

    C_W = 1.0
    C_b = 0.0
    lambda_W = 1.0
    lambda_b = 0.0

    layer_to_check = 5e

    matrix_dimensions = [100]

    maximum_dimension = max(matrix_dimensions)

    mnist = fetch_openml("mnist_784", as_frame=False, cache=False)

    labels = mnist.target.astype("int64")

    selected_indices = []

    for digit in range(10):
        digit_indices = (labels == digit).nonzero()[0][:10]
        selected_indices.extend(digit_indices)

    X = mnist.data[selected_indices].astype("float32")

    inpt_vecs_full = torch.tensor(X).T / 255.0 #Need to swap axes for my code

    results = check_theta_dimensions(inpt_vecs_full, matrix_dimensions, layer_to_check, C_W, C_b, lambda_b, lambda_W)

    print_theta_results(results)
    plot_eigenvalue_spectrum(inpt_vecs_full, layer_to_check, C_W, C_b, lambda_b, lambda_W)