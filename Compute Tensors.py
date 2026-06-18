import numpy as np


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
def compute_XIII(H, eta):
    n = H.shape[0]
    I = np.eye(n)

    # Get M tensor in n^3  by n^3 form
    M_flat = (
            np.kron(H, np.kron(I, I)) + np.kron(I, np.kron(H, I)) + np.kron(I, np.kron(I, H))
            - eta * (np.kron(H, np.kron(H, I)) + np.kron(H, np.kron(I, H)) + np.kron(I, np.kron(H, H)))
            + (eta ** 2) * np.kron(H, np.kron(H, H))
    )
    #invert
    M_inv_flat = np.linalg.inv(M_flat)

    # get back n,n,n,n,n,n tensor by flattening the n^3 by n^3 matrix
    X_III = M_inv_flat.reshape(n, n, n, n, n, n)
    return X_III

def compute_Y_I(H, X_II, X_III):
    return np.einsum("abde,cf->abcdef", X_II, H) - np.einsum("cg,abgdef->abcdef", H, X_III)

def compute_Y_II(H, X_II, X_III):
    return np.einsum('abde, cf->abdecf', X_II, H) - np.einsum('be, acde->acde', H, X_II)

def compute_Y_III(H, X_II, X_III):
    return np.einsum('ad,be,cf->abcdef', H, H, H) - np.einsum('bg,agde,cf->abcdef', H, X_II, H) - np.einsum('ad, cg, bgef->abcdef', H, H, X_II) + np.einsum('ci, bigh, aghdef->abcdef', H, X_II, X_III)

def compute_Y_IV(H, X_II, X_III):
    return np.einsum('ad, bcef->abcdef', H, X_II) - np.einsum('bcgh, aghdef->abcdef', X_II, X_III)

def compute_Z_A(Y_II):
    return Y_II

def compute_Z_B(Y_II, X_II, eta):
    return Y_II + (eta / 2) * X_II

def compute_Z_IA(Y_III, Y_IV, eta):
    return -Y_III - (eta / 2) * Y_IV

def compute_Z_IB(Y_I, Y_III, Y_IV, X_III, eta):
    return -Y_III - (eta / 2) * Y_IV - (eta / 2) * Y_I - (eta ** 2 / 6) * X_III

def compute_Z_IIA(Y_III):
    return -Y_III

def compute_Z_IIB(Y_I, Y_III, Y_IV, eta):
    return -Y_III -np.einsum('abcdef->bacdef', Y_III) - np.einsum('abcdef->acbdfe', Y_III) - eta * Y_IV - eta * Y_I

def compute_Z_tensors(H: np.ndarray, eta: float):
    """
    Given H (shape n x n) and eta (scalar), compute and return only the Z tensors.

    Returns:
      Z_A   : shape (n,n,n,n)
      Z_B   : shape (n,n,n,n)
      Z_IA  : shape (n,n,n,n,n,n)
      Z_IB  : shape (n,n,n,n,n,n)
      Z_IIA : shape (n,n,n,n,n,n)
      Z_IIB : shape (n,n,n,n,n,n)
    """
    X_II = compute_XII(H, eta)
    X_III = compute_XIII(H, eta)

    Y_I = compute_Y_I(H, X_II, X_III)
    Y_II = compute_Y_II(H, X_II, X_III)
    Y_III = compute_Y_III(H, X_II, X_III)
    Y_IV = compute_Y_IV(H, X_II, X_III)

    Z_A = compute_Z_A(Y_II)
    Z_B = compute_Z_B(Y_II, X_II, eta)

    Z_IA = compute_Z_IA(Y_III, Y_IV, eta)
    Z_IB = compute_Z_IB(Y_I, Y_III, Y_IV, X_III, eta)

    Z_IIA = compute_Z_IIA(Y_III)
    Z_IIB = compute_Z_IIB(Y_I, Y_III, Y_IV, eta)

    return Z_A, Z_B, Z_IA, Z_IB, Z_IIA, Z_IIB