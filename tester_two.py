import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

# ---------- Neural Network Definition ----------
class SinNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden_layers):
        super(SinNet, self).__init__()
        layers = []

        # first hidden layer
        layers.append(nn.Linear(input_dim, hidden_dim, bias=False))

        # hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))

        # output layer (scalar output)
        layers.append(nn.Linear(hidden_dim, 1, bias=False))

        self.layers = nn.ModuleList(layers)

        # Custom weight initialization: N(0, 1/width)
        with torch.no_grad():
            self.layers[0].weight.normal_(0.0, 0.5 ** 0.5)
            for layer in self.layers[1:-1]:  # hidden layers
                layer.weight.normal_(0.0, (1/hidden_dim)**0.5)
            self.layers[-1].weight.normal_(0.0, (1 / hidden_dim) ** 0.5)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.sin(layer(x))
        x = self.layers[-1](x)
        return x


# ---------- Training Function ----------
def train_sin_net(train_vecs, labels, hidden_dim, num_hidden_layers, lr, t):
    """
    GPU-optimized training function.
    Per-layer learning rates, automatic device selection, large t-friendly.
    """
    # Select GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Prepare training data
    if train_vecs.shape[0] < train_vecs.shape[1]:
        train_vecs = train_vecs.T
    N, d = train_vecs.shape
    labels = labels.reshape(-1, 1)

    x_train = torch.tensor(train_vecs, dtype=torch.float32, device=device)
    y_train = torch.tensor(labels, dtype=torch.float32, device=device)

    # Initialize model on GPU
    model = SinNet(input_dim=d, hidden_dim=hidden_dim,
                   num_hidden_layers=num_hidden_layers).to(device)

    # MSE loss
    criterion = nn.MSELoss()

    # Per-layer learning rates
    param_groups = [{"params": layer.weight, "lr": lr / layer.out_features}
                    for layer in model.layers]
    optimizer = torch.optim.SGD(param_groups)

    # Training loop
    for step in range(t):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Print progress and GPU memory every 10k steps
        if step % 10000 == 0:
            print(f"Step {step}, Loss {loss.item():.4f}, "
                  f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

    return model


# ---------- Prediction Function ----------
def predict_sin_net(model, test_vecs):
    """
    Predict using GPU if model is on GPU.
    Returns NumPy array.
    """
    if test_vecs.shape[0] < test_vecs.shape[1]:
        test_vecs = test_vecs.T

    device = next(model.parameters()).device
    x_test = torch.tensor(test_vecs, dtype=torch.float32, device=device)

    with torch.no_grad():
        preds = model(x_test).cpu().numpy().flatten()
    return preds


# ---------- Main ----------
if __name__ == "__main__":
    np.random.seed(45)

    # Training data: 2D points on unit circle
    N = 3
    angles = np.linspace(0.5, 2*np.pi, N, endpoint=False)
    x1 = np.cos(angles)
    x2 = np.sin(angles)
    train_vecs = np.vstack([x1, x2])
    labels = np.random.uniform(-5, 5, size=N)

    # Test data: 202 points on unit circle
    M = 202
    test_angles = np.linspace(0, 2*np.pi, M, endpoint=False)
    test_x1 = np.cos(test_angles)
    test_x2 = np.sin(test_angles)
    test_vecs = np.vstack([test_x1, test_x2])

    # Combine train + test for plotting
    inpt_vecs = np.concatenate([train_vecs, test_vecs], axis=1)
    inpt_angles = np.concatenate([angles, test_angles])

    print("Labels:", labels)

    # Train the network
    model = train_sin_net(train_vecs, labels,
                          hidden_dim=2000,      # width of hidden layers
                          num_hidden_layers=2,  # number of hidden layers
                          lr=0.2,
                          t=2000)            # large number of steps

    # Predict on combined set
    preds = predict_sin_net(model, inpt_vecs)

    # Plot results
    plt.scatter(inpt_angles, preds, color='r', label='ensemble predictions')
    plt.scatter(angles, labels, color='b', label='training points')
    plt.xlabel("angle (rad)")
    plt.ylabel("output")
    plt.legend()
    plt.savefig("Grand_Finale_finished_version.png")
