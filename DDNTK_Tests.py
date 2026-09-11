import jax
import jax.numpy as jnp

from jax.experimental.jet import jet
from jax.flatten_util import ravel_pytree
from functools import partial
import matplotlib.pyplot as plt


# helpers for initializing and training an ensemble of sin neural nets accoridng to textbook's treatment
def initialize_sin_network(key, input_dim, hidden_dim, num_hidden_layers):
    """
    Initializes a sin neural net with every weight in layer l being independently sampled from
    N(0, 1 / n_{l-1}),
    where n_{l-1} is the input dimension of that layer (width of layer before it, or input vec's dim if l=1)
    """

    layer_dimensions = ([input_dim] + [hidden_dim] * num_hidden_layers + [1])

    number_of_weight_matrices = len(layer_dimensions) - 1

    # unfortunately if we want to save the params on a run we have to give
    # them indiv keys, theres no one general key for all operations at once
    # like there is in a lot of numpy modules
    layer_keys = jax.random.split(key, number_of_weight_matrices)

    weight_matrices = []

    for layer_index in range(number_of_weight_matrices):
        input_width = layer_dimensions[layer_index]
        output_width = layer_dimensions[layer_index + 1]

        # from textbook we have that the variance of each weight is 1 / input_width
        standard_deviation = jnp.sqrt(1.0 / input_width)

        #using basic stats knowledge can just multiply the normal distrib by sigma to get a mean 0 standard deviation sigma
        # gaussian random var
        W = standard_deviation * jax.random.normal(layer_keys[layer_index], shape=(output_width, input_width))
        weight_matrices.append(W)

    return tuple(weight_matrices)



def initialize_ensemble(key, ensemble_size, input_dim, hidden_dim, num_hidden_layers):
    """
    Initializes the ensemble of neural nets

    returns two things:

    theta_ensemble:
        shape (ensemble_size, P) with each row containing one network's
        flattened paramw

    unravel:
        the unravel function shared by every ensemble member,
        since every member has the same architecture (dims of each layer) we only need the one

    """
    # as usual make keys for every object since jax has this limitation
    network_keys = jax.random.split(key, ensemble_size)

    flattened_parameters = []

    unravel = None

    for network_index in range(ensemble_size):
        params = initialize_sin_network(network_keys[network_index], input_dim, hidden_dim, num_hidden_layers)

        theta, current_unravel = ravel_pytree(params)

        flattened_parameters.append(theta)

        #Just collecting the unravel function at the start of loop
        # (which is common about all network indices because it only depends on weight dims)
        if network_index == 0:
            unravel = current_unravel

    #axis=0 means the flattened_param vecs will be rows in the theta ensemble matrix, as needed
    theta_ensemble = jnp.stack(flattened_parameters, axis=0)

    return theta_ensemble, unravel

#Training helpers

def network_outputs(theta, xs, unravel):
    """
    Compute one network's outputs on all N inputs.

    Returns an array with shape (N,).
    """
    return jax.vmap(lambda x: f_flat(theta, x, unravel))(xs)


def squared_error_loss(theta, xs, ys, unravel):
    """
    Uses the textbook formula:

    L(theta) = 1/2 sum_alpha (f_theta(x_alpha) - y_alpha)^2
    """
    predictions = network_outputs(theta, xs, unravel)

    residuals = predictions - ys

    return 0.5 * jnp.sum(residuals**2)


# grad vec of the loss with respect to the flattened theta params which are in place argnums=0
loss_grad = jax.grad(squared_error_loss, argnums=0)

#jax.jit just makes things more efficiently from what I see online, need to wrap it in partial for it to work and declare
# static non vector/array like params like unravel (only unravel actually)
@partial(jax.jit, static_argnums=(4,))
def train_ensemble_one_step(theta_ensemble, xs, ys, metric_diag, unravel, learning_rate):
    """
    Tensorial gradient descent as in the textbook with the lambda hyperparams encoded in metric_diag

    theta_ensemble has shape (M, P)
    """

    def update_one_network(theta):
        gradient = loss_grad(theta, xs, ys, unravel)

        #Can multiply by diagonal G matrix to distribute the lambda hyper parameters which will set to the book's specifications
        updated_theta = (theta - learning_rate * metric_diag * gradient)

        return updated_theta

    # Map the update rowwise in theta_ensemble, noting that jax.vmap distributes the specified function along its
    # 0^th axis which for the theta_enseble matrix is the row axis
    updated_ensemble = jax.vmap(update_one_network)(theta_ensemble)

    return updated_ensemble


def mean_ddntk_over_ensemble(theta_ensemble, xs, metric_diag, unravel):
    """
    This uses the funcions below this one to compute the ddntk's over each member of the ensemble and then
    compute the (elementwise) mean of the ensemble

    he computation is done one network at a time thought, this way
    not all the ddntk's need of each ensemble member need be stored at once
    else the complexity would pick up a factor of ensemble size

    theta_ensemble has shape (ensemble_size, param_number)
    """

    ensemble_size = theta_ensemble.shape[0]
    number_of_inputs = xs.shape[0]

    ddntk_shape = (number_of_inputs, number_of_inputs, number_of_inputs, number_of_inputs)

    ddntk_I_sum = jnp.zeros(ddntk_shape, dtype=theta_ensemble.dtype) #just preserving the data type in case of higher bit floats in the future

    ddntk_II_sum = jnp.zeros(ddntk_shape, dtype=theta_ensemble.dtype)

    #each iteration of the loop destroys local loop variables so the one at a time computation
    # will save space complexity (nerd for memory)
    for network_index in range(ensemble_size):
        theta = theta_ensemble[network_index]

        ddntk_I, ddntk_II = multi_input_ddntk(theta, xs, metric_diag, unravel)

        ddntk_I_sum = ddntk_I_sum + ddntk_I
        ddntk_II_sum = ddntk_II_sum + ddntk_II

    mean_ddntk_I = ddntk_I_sum / ensemble_size
    mean_ddntk_II = ddntk_II_sum / ensemble_size

    return mean_ddntk_I, mean_ddntk_II









#Calculating DDntk's from a specific neural network part of code

# function needed to be defined for jax to do forward pass
def apply_fn(weights, x):
    """
    function needed to be defined for jax to do the forward pass. Jax
    library will use advanced methods to trace this function in terms of elementary operations
    they have defined and which they already know how to take derivatives of for later backprop and other uses
    """
    for W in weights[:-1]:
        x = jnp.sin(W @ x)

    return (weights[-1] @ x).squeeze()

def f_flat(theta, x, unravel):
    """
    the neural network function
    technically this function gets traced by Jax as well
    but this func calls apply_fn where the actual important operations are
    that will get traced
    """
    weight_params = unravel(theta)
    return jnp.asarray(apply_fn(weight_params, x)).squeeze()

#the flattened gradient vector (which takes in input vec x and parameters theta)
grad_f = jax.grad(f_flat, argnums=0) #argnums=0 means it takes the deriv in the theta (0^th) direction


def make_weight_metric_diag(weight_params, Lambda_W):
    """
    making the diagonal metric corresponding to
        lambda_{W_ij^(l), W_kl^(l)} = delta_ik delta_jl Lambda_W / n_{l-1}
    """
    #useful to do since there is a one to one correspondance of the lambda tensor's diagonal elements and the number of weight params
    metric_tree = jax.tree_util.tree_map(lambda W: jnp.full_like(W, Lambda_W / W.shape[-1]), weight_params)

    metric_diag, _ = ravel_pytree(metric_tree) # metric_diag component per each flattened (unravelled) weight parameter
    return metric_diag


def third_diagonal_contraction(theta, x0, direction, unravel):
    """
    Using third directional derivative in a given direction for the diagonal contractions
    (so each of the three direc derivs w.r.t the same direction each time, hence diagonal)
    which can be used to compute non diagonal contractions as well
    """
    zero = jnp.zeros_like(theta)

    _, (_, _, third) = jet(lambda th: f_flat(th, x0, unravel),(theta,),((direction, zero, zero),), factorial_scaled=True)

    return third


def third_mixed_contraction(theta, x0, v1, v2, v3, unravel):
    """
    computes the the off diagonal third derivative contractions using four diagonal
    third directional derivatives in the right directions
    this formula is simple enough to check by using the multilinearity property of directional
    derivatives and seeing that all the right terms cancel and ur left with 24 of the same term.

    Note that this formula assumes order of differentiation doesn't matter (our nn's are C^infty though so this is fine)
    as this is how we collect and cancel like terms
    """
    p_ppp = third_diagonal_contraction(theta, x0, v1 + v2 + v3, unravel)

    p_ppm = third_diagonal_contraction(theta, x0, v1 + v2 - v3, unravel)

    p_pmp = third_diagonal_contraction(theta, x0, v1 - v2 + v3, unravel)

    p_pmm = third_diagonal_contraction(theta, x0, v1 - v2 - v3, unravel)

    return (p_ppp - p_ppm - p_pmp + p_pmm) / 24.0

# using @jax.jit helps with efficiency of code in many ways, can't hurt to have it
# also need to keep the third argument static as jax typically requires vector/array inputs (for differentiability reasons)
# so this tells jax that this is not a true variable to be considered and is just a static parameter we need to make our code work
@partial(jax.jit, static_argnums=(3,))
def multi_input_ddntk(theta, xs, metric_diag, unravel):
    """
    computing the ddNTK's assuming
    theta is shape (P,) where P is the number of params in the nn,
    xs are the inputs of shape (N, vec_dim), and
    metric_diag is shape (P,) where for each weight W_ij^(l), the entry is
    Lambda_W / n_{l-1}
    """


    # computing g_delta = grad_theta f_theta(x_delta) since we'll need to take directional deriv in the direction of gradient vector
    gradients = jax.vmap(lambda x: grad_f(theta, x, unravel))(xs)

    # computing v_delta = G g_delta
    # where G is the diagonal metric matrix which means elementwise multiplication betweem G and g_delta
    #technically this will be the direction vector but its just the gradients scaled by the (scaled) lambda tensors from textbook
    directions = gradients * metric_diag[None, :] #make G row vector to make elementwise multiplication happen

    # First ddNTK
    # just using jax's efficient vmap to store the ddNTK tensor

    map_v3 = jax.vmap(partial(third_mixed_contraction, unravel=unravel), in_axes=(None, None, None, None, 0)) #uisng partial to create new callable version of third_mixed_contraction with unravel filled in
    map_v2_v3 = jax.vmap(map_v3, in_axes=(None, None, None, 0, None)) # is done recursively because thats how they did it online lol
    map_v1_v2_v3 = jax.vmap(map_v2_v3, in_axes=(None, None, 0, None, None))
    map_x0_v1_v2_v3 = jax.vmap(map_v1_v2_v3, in_axes=(None, 0, None, None, None))

    ddntk_I = map_x0_v1_v2_v3(theta, xs, directions, directions, directions)

    # Now we compute all the hessian vector products for the second ddNTK

    def hessian_vector_product(x_out, direction):
        _, Hv = jax.jvp(lambda th: grad_f(th, x_out, unravel),(theta,),(direction,))
        return Hv

    def all_directions_for_output(x_out):
        return jax.vmap(lambda direction: hessian_vector_product(x_out,direction))(directions)

    Hv = jax.vmap(all_directions_for_output)(xs)

    # assembling piececs to compuite the final ddNTK_II contraction
    # of  (H_d1 v_d3)^T G (H_d2 v_d4)
    ddntk_II = jnp.einsum("acp,p,bdp->abcd", Hv, metric_diag, Hv, optimize=True)

    return ddntk_I, ddntk_II



def max_percent_difference(history):
    x = jnp.asarray(history)

    differences = (200.0 * jnp.abs(x[:, None] - x[None, :]) / (jnp.abs(x[:, None]) + jnp.abs(x[None, :])))

    return differences





if __name__ == "__main__":

    # Network and ensemble settings

    RANDOM_SEED = 44
    ENSEMBLE_SIZE = 10

    input_dim = 2
    hidden_dim = 50
    num_hidden_layers = 2

    # Number of training updates
    T = 500

    learning_rate = 0.01
    Lambda_W = 1.0

    #training test vecs

    #input vec(s)
    xs = jnp.array([
        [1.0, -0.3],
        [0.7, 0.4]
    ])

    # labels
    ys = jnp.array([
        0.5,
        0.7
    ])

    #The index to graph
    ddntk_index = (0, 0, 0, 0)


    # initialization

    key = jax.random.PRNGKey(RANDOM_SEED)

    theta_ensemble, unravel = initialize_ensemble(key, ENSEMBLE_SIZE, input_dim, hidden_dim, num_hidden_layers)

    # The metric is the same for every ensemble member because
    # every member has the same architecture.
    first_params = unravel(theta_ensemble[0])

    metric_diag = make_weight_metric_diag(first_params, Lambda_W)

    # data collection

    mean_ddntk_I_history = []
    mean_ddntk_II_history = []
    mean_loss_history = []

    # t=0 is initialization and t=T is after T updates.
    for t in range(T + 1):

        # Compute the ensemble-averaged DDNTKs at the current time.
        mean_ddntk_I, mean_ddntk_II = (mean_ddntk_over_ensemble(theta_ensemble, xs, metric_diag, unravel))

        mean_ddntk_I_history.append(mean_ddntk_I[ddntk_index[0], ddntk_index[1], ddntk_index[2], ddntk_index[3]])
        mean_ddntk_II_history.append(mean_ddntk_II[ddntk_index[0], ddntk_index[1], ddntk_index[2], ddntk_index[3]])

        # Compute the loss of every ensemble member.
        losses_for_all_networks = jax.vmap(lambda theta: squared_error_loss(theta, xs, ys, unravel))(theta_ensemble)

        # Average the losses over the ensemble.
        mean_loss = jnp.mean(losses_for_all_networks)

        mean_loss_history.append(mean_loss)

        # Do not update after the final measurement at t=T.
        if t < T:
            theta_ensemble = train_ensemble_one_step(theta_ensemble, xs, ys, metric_diag, unravel, learning_rate)

    #insert code here for theory pred

























    # plot friendly var names cause the order i did this makes it easier to make this var change now
    loss_values = mean_loss_history

    #ddntk_I_values = mean_ddntk_I_history #altering code to output mean ddNTK's
    ddntk_I_values = [jnp.mean(x) for x in mean_ddntk_I_history]

    #ddntk_II_values = mean_ddntk_II_history
    ddntk_II_values = [jnp.mean(x) for x in mean_ddntk_II_history]

    timesteps = range(T + 1)

    print("Maximum ddNTK-I percent difference:", max_percent_difference(mean_ddntk_I_history))

    print("Maximum ddNTK-II percent difference:", max_percent_difference(mean_ddntk_II_history))

    # plotting time

    figure, axes = plt.subplots(1,2, figsize=(12, 4.5))

    # Mean loss plot
    axes[0].plot(timesteps, loss_values, marker="o", color="tab:blue")

    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("Mean ensemble loss")
    axes[0].set_title("Mean training loss vs. timestep")
    axes[0].set_xticks(list(timesteps))
    axes[0].grid(True, alpha=0.3)

    # DDNTK entry plot
    axes[1].plot(timesteps, ddntk_I_values, marker="o", label="Mean ddNTK-I")

    axes[1].plot(timesteps, ddntk_II_values, marker="s", label="Mean ddNTK-II")


    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("DDNTK entry value")

    axes[1].set_title(f"Mean DDNTK entry {ddntk_index} vs. timestep")

    axes[1].set_xticks(list(timesteps))
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()
































# if __name__ == "__main__":
#
#     # Here are the weight matrices for you, this is a better spot for all the inputs and weights
#     W1 = jnp.array([
#         [0.2, -0.1],
#         [0.3, 0.25],
#         [0.4, 0.3],
#     ])
#
#     W2 = jnp.array([
#         [-0.35, 0.3, 0.6],
#         [0.8, -0.25, 0.55],
#         [-0.65, 0.45, -0.4]
#     ])
#
#     W3 = jnp.array([
#         [0.4, -0.2, 0.15]
#     ])
#
#     params = (W1, W2, W3)
#
#     # flattened vector with weights and unravel function to unflatten back into list of weight matrices (2D arrays)
#     theta0, unravel = ravel_pytree(params)
#
#
#     Lambda_W = 1.0
#
#     xs = jnp.array([
#         [1.0, 0.0]
#     ])
#
#     metric_diag = make_weight_metric_diag(params, Lambda_W)
#
#     # These are the ddNTK's, forward pass output (the z^{(L)}'s) and the weight matrices which you (Yoni) mentioned
#     # you wanted during the meeting
#     ddntk_I, ddntk_II = multi_input_ddntk(theta0, xs, metric_diag, unravel)
#     network_outputs = jax.vmap(lambda x: f_flat(theta0, x, unravel))(xs)
#
#
#     #Print statements for comparing outputs
#     print("input vector(s):")
#     print(xs)
#     print("\nweight matrices:")
#     print(unravel(theta0))
#     print("\nnetwork_outputs:")
#     print(network_outputs)
#     print("\nddNTK_I:")
#     print(ddntk_I)
#     print("\nddNTK_II:")
#     print(ddntk_II)