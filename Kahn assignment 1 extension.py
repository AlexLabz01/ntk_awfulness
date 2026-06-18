import numpy as np
from itertools import combinations
from matplotlib import pyplot as plt

def lay_comp(inpt, width, CW):
    # Preconditions:
    # width must be the same length as the shape of the inpt vector

    var = CW / width
    std = np.sqrt(var)

    # Sampling covariance matrix from normal distribution (mean 0, std, matrix_size)
    W = np.random.normal(loc=0.0, scale=std, size=(width, inpt.size))
    return np.dot(W, inpt)

def net_output(inpt, width, CW, num_layers):
    # Loop that computes the z's for each layers and changes current_output to be the
    # entire neural net output by the end
    #First step does not apply activation function
    current_output = lay_comp(inpt, width, CW)

    #In between steps all use activation function
    for _ in range(num_layers - 2):
        current_output = lay_comp(np.sin(current_output), width, CW)

    # Final step no activation function
    final_output = lay_comp(current_output, width, CW)
    return final_output

def estimate_connected(inpt, width, CW, num_layers, ensemble_size):
    sum_z2 = np.zeros(width)
    sum_z2z2 = 0  # scalar accumulator for sum over unordered pairs

    for _ in range(ensemble_size):
        z = net_output(inpt, width, CW, num_layers)
        z2 = z**2  # squared activations, shape (width,)
        sum_z2 += z2

        sum_all = np.sum(z2)**2        # (sum_i z_i^2)^2
        sum_diag = np.sum(z2**2)       # sum_i z_i^4
        sum_pairs = (sum_all - sum_diag) / 2  # sum over unordered pairs i < j
        sum_z2z2 += sum_pairs

    G2 = np.mean(sum_z2 / ensemble_size)  # average over ensemble and neurons
    total_pairs = width * (width - 1) // 2 # number of unordered pairs
    G4 = sum_z2z2 / (ensemble_size * total_pairs)  # average over ensemble and pairs

    connected_correlator_normalized = width * (G4 - G2 ** 2) / ( G2 ** 2)
    return connected_correlator_normalized

def plot_connected_vs_depth(width=50, CW=0.99, ensemble_size=75, max_depth=100):
    inpt = np.random.randn(width)
    depths = list(range(4, max_depth + 1))
    conn_values = []

    for L in depths:
        print("Running ensemble for depth {}".format(L))
        conn = estimate_connected(inpt, width, CW, L, ensemble_size)
        conn_values.append(conn)

    # plotting the connected correlator vs depth as a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(depths, conn_values, label=f"$C_W$ = {CW}", s=25)
    plt.xlabel("Depth (L)")
    plt.ylabel(r"$G_4^{(L)} - G_2^{(L)\,2} / G_2^{(L)}$")
    plt.title("Connected 4-point correlator (multiplied by factor) vs Depth")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_connected_vs_depth(width=200, CW=1, ensemble_size=1000, max_depth=50)