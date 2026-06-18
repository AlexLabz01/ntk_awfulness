from K_file import compute_tth_pred as mean_out
from Kahn_assignment_2 import compute_tth_pred as single_out
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    np.random.seed(45)
    n_inputs = 105
    num_train = 3
    train_angles = np.array([2* np.pi / 3 -0.2, 4 * np.pi / 3 -0.2, 2 * np.pi -0.2])
    #train_angles = np.array([np.pi / 2 - 0.2, np.pi -0.2, 3 * np.pi / 2 - 0.2, 2 * np.pi - 0.2])
    test_angles = np.linspace(0, 2 * np.pi, num=n_inputs-num_train)
    inpt_angles = np.concatenate((train_angles, test_angles))
    inpt_vecs = np.zeros((2, n_inputs))
    for i in range(n_inputs):
        inpt_vecs[0][i] += np.cos(inpt_angles[i])
        inpt_vecs[1][i] += np.sin(inpt_angles[i])

    C_W = 1.0
    C_b = 0.0
    lambda_b = 0.0
    lambda_W = 1
    depth = 10
    t = int(10_000_000)
    labels = np.random.uniform(low=-5, high=5, size=num_train) #same specs as wiki animation
    lr = 0.02
    width = 205
    mean_outputs = mean_out(inpt_vecs, labels, C_W, C_b, depth, lambda_b, lambda_W, lr, num_train, t)
    single_reg_outputs = single_out(inpt_vecs, labels, lr, num_train, C_W, width, depth, t)

    plt.scatter(inpt_angles, single_reg_outputs, color='r', label='ensemble predictions', s=1)
    plt.scatter(inpt_angles, mean_outputs, color='g', label='mean predictions', s=10)
    for i in range(5):
        single_reg_outputs = single_out(inpt_vecs, labels, lr, num_train, C_W, width, depth, t)
        plt.scatter(inpt_angles, single_reg_outputs, color='r', s=1)
    plt.scatter(inpt_angles[:num_train], labels, color='b', label='training points')
    plt.xlabel("angle (rad)")
    plt.ylabel("output")
    plt.legend()
    plt.savefig("Grand_Finale_finished_version.png")