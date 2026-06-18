import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

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
            for layer in self.layers[1:]:  # hidden layers
                layer.weight.normal_(0.0, (1/hidden_dim)**0.5)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.sin(layer(x))  #applying sin activation function
        x = self.layers[-1](x)  # final layer no activation
        return x

# ---------- Training Function ----------
def train_sin_net(train_vecs, labels, hidden_dim, num_hidden_layers,
                  lr, t, device='cpu'):
    """
    train_vecs is a 2D numpy array whos columns are training vectors
    """
    # Must transpose train_vecs to N x d because PyTorch wants the the number of ensemble as size[0]
    if train_vecs.shape[0] < train_vecs.shape[1]:
        train_vecs = train_vecs.T
    N, d = train_vecs.shape
    labels = labels.reshape(-1, 1) #transform to a 2D array with a single column (this is what pytorch wants)

    x_train = torch.tensor(train_vecs, dtype=torch.float64, device=device) #torch supports GPU functionality but for now just using cpu as default
    y_train = torch.tensor(labels, dtype=torch.float64, device=device)

    # make neural net model
    model = SinNet(input_dim=d, hidden_dim=hidden_dim,
                   num_hidden_layers=num_hidden_layers).to(device).double()

    # make loss and optimizer
    criterion = nn.MSELoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    #In the case of layer specific learning rates, lambda_W^(l)
    param_groups = []
    for layer in model.layers:
        divisor = layer.in_features  # width of the incoming layer
        layer_lr = lr / divisor
        param_groups.append({"params": layer.parameters(), "lr": layer_lr})
    optimizer = torch.optim.SGD(param_groups)

    # grad descent loop
    for step in range(t):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    return model  # now fully trained model

# ---------- Prediction Function ----------
def predict_sin_net(model, test_vecs, device='cpu'):
    """
    test_vecs is a matrix whose columns are all the test vectors
    """
    if test_vecs.shape[0] < test_vecs.shape[1]: #torch expects number of data points as size[0] of the test_vecs matrix
        test_vecs = test_vecs.T
    x_test = torch.tensor(test_vecs, dtype=torch.float64, device=device)
    with torch.no_grad(): #we dont want to store any gradients, the network is trained already
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
    labels = np.sin(np.random.uniform(0, 2 * np.pi, size=N))

    # getting test data
    M = 200
    test_angles = np.linspace(0, 2 * np.pi, M, endpoint=False)
    test_x1 = np.cos(test_angles)
    test_x2 = np.sin(test_angles)
    test_vecs = np.vstack([test_x1, test_x2])

    # combining training and test vectors for plotting and making predictions
    inpt_vecs = np.concatenate([train_vecs, test_vecs], axis=1)
    inpt_angles = np.concatenate([angles, test_angles])

    print("Labels:", labels)

    # training the network
    model = train_sin_net(train_vecs, labels,
                          hidden_dim=200,  # width of hidden layers
                          num_hidden_layers=14,  # number of hidden layers
                          lr=0.01,
                          t=100)  # large number of steps

    # make predictions for test and training vecs
    preds = predict_sin_net(model, inpt_vecs)

    #plotting
    plt.scatter(inpt_angles, preds, color='r', label='ensemble predictions')
    plt.scatter(angles, labels, color='b', label='training points')
    plt.xlabel("angle (rad)")
    plt.ylabel("output")
    plt.legend()
    plt.savefig("Grand_Finale_finished_version.png")
