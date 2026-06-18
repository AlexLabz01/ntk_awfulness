from K_file import compute_tth_pred_per_step as mean_preds_steps
from Test_three import train_sin_net as trainer
from Test_three import predict_sin_net as predictor
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import torch
from matplotlib.animation import PillowWriter

if __name__ == '__main__':
    np.random.seed(45)
    N = 3  # number of training points
    angles = np.linspace(0.5, 2 * np.pi, N, endpoint=False)
    x1 = np.cos(angles)
    x2 = np.sin(angles)
    train_vecs = np.vstack([x1, x2])  # shape (2, N), columns = training vectors
    labels = np.random.uniform(-5, 5, size=N)

    # Test data: different angles on the unit circle
    M = 200  # number of test points
    test_angles = np.linspace(0, 2 * np.pi, M, endpoint=False)
    test_x1 = np.cos(test_angles)
    test_x2 = np.sin(test_angles)
    test_vecs = np.vstack([test_x1, test_x2])  # shape (2, M)

    inpt_vecs = np.concatenate([train_vecs, test_vecs], axis=1)
    inpt_angles = np.concatenate([angles, test_angles])

    width = 5
    C_W = 1.0
    C_b = 0.0
    lambda_b = 0.0
    lambda_W = 1
    depth = 2
    t = 6000
    lr = 0.04
    ensemble_size = 50

    # NTK predictions
    mean_outputs_all = mean_preds_steps(inpt_vecs, labels, C_W, C_b, depth, lambda_b, lambda_W, lr, N, t)

    # grad descent ensemble predictions
    ensemble_all_preds = []
    for e in range(ensemble_size):
        model, ensemble_preds_steps = trainer(train_vecs, labels,
                                              hidden_dim=width,
                                              num_hidden_layers=depth - 1,
                                              lr=lr,
                                              t=t,
                                              test_vecs=inpt_vecs)
        ensemble_all_preds.append(ensemble_preds_steps)
    ensemble_all_preds = np.array(ensemble_all_preds)

    # ---------- Animation ----------
    fig, ax = plt.subplots()

    # NTK scatter
    ntk_scatter = ax.scatter([], [], color='g', label='Mean NTK predictions', s=0.3)

    # Ensemble scatters: only the first one gets the label for the legend
    ensemble_scatters = [ax.scatter([], [], color='r', s=0.1, alpha=0.3, label="Ensemble predictions")]
    ensemble_scatters += [ax.scatter([], [], color='r', s=0.1, alpha=0.3) for _ in range(1, ensemble_size)]

    # Training points
    train_scatter = ax.scatter(angles, labels, color='b', label='training points')

    ax.set_xlim(0, 2 * np.pi)
    y_min = min(np.min(labels), np.min(mean_outputs_all))
    y_max = max(np.max(labels), np.max(mean_outputs_all))
    ax.set_ylim(y_min - 1, y_max + 1)
    ax.set_xlabel("angle (rad)")
    ax.set_ylabel("output")
    ax.legend()


    def init():
        ntk_scatter.set_offsets(np.c_[[], []])
        for s in ensemble_scatters:
            s.set_offsets(np.c_[[], []])
        return [ntk_scatter] + ensemble_scatters


    def update(frame):
        ntk_scatter.set_offsets(np.c_[inpt_angles, mean_outputs_all[frame]])
        for i, s in enumerate(ensemble_scatters):
            s.set_offsets(np.c_[inpt_angles, ensemble_all_preds[i, frame]])
        ax.set_title(f"Step {frame + 1}")
        return [ntk_scatter] + ensemble_scatters


    # Compute fps for a 10-second GIF
    fps = 2 * t / 3.0  # t = total number of frames

    ani = animation.FuncAnimation(fig, update, frames=t,
                                  init_func=init, blit=True, repeat=False)

    # Save GIF with exactly 10 seconds duration
    ani.save('gradient_NTK_animation.gif', writer='pillow', fps=fps)

    plt.show()

