from matplotlib import pyplot as plt
from matplotlib import animation
import torch
import torch.nn as nn
import numpy as np

# ---------- Neural Network Definition ----------
class SinNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden_layers):
        super(SinNet, self).__init__()
        layers = []
        # first layer
        layers.append(nn.Linear(input_dim, hidden_dim, bias=False))
        # rest of hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
        # output layer (scalar output)
        layers.append(nn.Linear(hidden_dim, 1, bias=False))
        self.layers = nn.ModuleList(layers)

        # Making sure weights are sampled from N(0, 1/width)
        with torch.no_grad():
            self.layers[0].weight.normal_(0.0, 0.5 ** 0.5)
            for layer in self.layers[1:-1]:  # hidden layers
                layer.weight.normal_(0.0, (1/hidden_dim)**0.5)
            # output layer variance 1/hidden_dim
            self.layers[-1].weight.normal_(0.0, (1/hidden_dim)**0.5)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.sin(layer(x))  # applying sin activation function
        x = self.layers[-1](x)  # final layer no activation
        return x

# ---------- Training Function ----------
def train_sin_net(train_vecs, labels, hidden_dim, num_hidden_layers,
                  lr, t, device='cpu', test_vecs=None):
    """
    train_vecs is a 2D numpy array whos columns are training vectors
    """
    # Must transpose train_vecs to N x d because PyTorch wants the number of ensemble as size[0]
    if train_vecs.shape[0] < train_vecs.shape[1]:
        train_vecs = train_vecs.T
    N, d = train_vecs.shape
    labels = labels.reshape(-1, 1)  # transform to a 2D array with a single column (this is what pytorch wants)

    x_train = torch.tensor(train_vecs, dtype=torch.float32, device=device)  # torch supports GPU functionality but for now just using cpu as default
    y_train = torch.tensor(labels, dtype=torch.float32, device=device)

    # make neural net model
    model = SinNet(input_dim=d, hidden_dim=hidden_dim,
                   num_hidden_layers=num_hidden_layers).to(device)

    # make loss
    criterion = nn.MSELoss()

    # --- custom learning rates per layer = lr / width ---
    param_groups = []
    for layer in model.layers:
        divisor = layer.in_features  # width of the incoming layer
        layer_lr = lr / divisor
        param_groups.append({"params": layer.parameters(), "lr": layer_lr})

    optimizer = torch.optim.SGD(param_groups)

    # store predictions if test_vecs provided
    all_preds = []
    if test_vecs is not None:
        if test_vecs.shape[0] < test_vecs.shape[1]:
            test_vecs = test_vecs.T
        x_test = torch.tensor(test_vecs, dtype=torch.float32, device=device)

    # grad descent loop
    for step in range(t):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # record predictions at this step
        if test_vecs is not None:
            with torch.no_grad():
                preds = model(x_test).cpu().numpy().flatten()
            all_preds.append(preds)

    return model, np.array(all_preds)  # now fully trained model and per-step predictions

# ---------- Prediction Function ----------
def predict_sin_net(model, test_vecs, device='cpu'):
    """
    test_vecs is a matrix whose columns are all the test vectors
    """
    if test_vecs.shape[0] < test_vecs.shape[1]:  # torch expects number of data points as size[0] of the test_vecs matrix
        test_vecs = test_vecs.T
    x_test = torch.tensor(test_vecs, dtype=torch.float32, device=device)
    with torch.no_grad():  # we dont want to store any gradients, the network is trained already
        preds = model(x_test).cpu().numpy().flatten()
    return preds

# ---------- Example Usage ----------
if __name__ == "__main__":
    np.random.seed(45)

    # getting training data
    N = 3
    angles = np.linspace(0.5, 2 * np.pi, N, endpoint=False)
    x1 = np.cos(angles)
    x2 = np.sin(angles)
    train_vecs = np.vstack([x1, x2])
    labels = np.random.uniform(-5, 5, size=N)

    # getting test data
    M = 202
    test_angles = np.linspace(0, 2 * np.pi, M, endpoint=False)
    test_x1 = np.cos(test_angles)
    test_x2 = np.sin(test_angles)
    test_vecs = np.vstack([test_x1, test_x2])

    # combining training and test vectors for plotting and making predictions
    inpt_vecs = np.concatenate([train_vecs, test_vecs], axis=1)
    inpt_angles = np.concatenate([angles, test_angles])

    print("Labels:", labels)

    # training the network (now recording predictions for animation)
    model, all_preds = train_sin_net(train_vecs, labels,
                                     hidden_dim=150,  # width of hidden layers
                                     num_hidden_layers=3,  # number of hidden layers
                                     lr=0.01,
                                     t=100,  # large number of steps
                                     test_vecs=inpt_vecs)

    # ---------- Animation ----------
    fig, ax = plt.subplots()
    # predictions as scatter
    pred_scatter = ax.scatter([], [], color='r', label='ensemble predictions')
    # training points
    scatter = ax.scatter(angles, labels, color='b', label='training points')

    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(np.min(labels) - 1, np.max(labels) + 1)
    ax.set_xlabel("angle (rad)")
    ax.set_ylabel("output")
    ax.legend()


    def init():
        pred_scatter.set_offsets(np.c_[[], []])  # empty at start
        return pred_scatter,


    def update(frame):
        y = all_preds[frame]
        # update scatter points: x = inpt_angles, y = predictions
        pred_scatter.set_offsets(np.c_[inpt_angles, y])
        ax.set_title(f"Step {frame + 1}")
        return pred_scatter,


    ani = animation.FuncAnimation(fig, update, frames=len(all_preds),
                                  init_func=init, blit=True, interval=100)

    # Save animation as GIF (Windows-friendly)
    ani.save('training_animation.gif', writer='pillow', fps=10)
    plt.show()

