import numpy as np

def compute_XIII(H, eta):
    """
    Compute the inverse tensor X_III defined by:
      δ_{α1}^{α7} δ_{α2}^{α8} δ_{α3}^{α9}
      = Σ_{α4,α5,α6} X_III^{α1,α2,α3,α4,α5,α6} * M_{α4,α5,α6}^{α7,α8,α9}
    where M is constructed from H and η as described.
    """
    n = H.shape[0]
    I = np.eye(n)

    # Build M_flat using Kronecker products
    M_flat = (
            np.kron(H, np.kron(I, I)) + np.kron(I, np.kron(H, I)) + np.kron(I, np.kron(I, H))
            - eta * (np.kron(H, np.kron(H, I)) + np.kron(H, np.kron(I, H)) + np.kron(I, np.kron(H, H)))
            + (eta ** 2) * np.kron(H, np.kron(H, H))
    )

    # Invert the matrix (use pinv if it might be singular)
    M_inv_flat = np.linalg.inv(M_flat)

    # Reshape into 6D tensor
    X_III = M_inv_flat.reshape(n, n, n, n, n, n)
    return X_III


def check_XIII(H, eta, atol=1e-6):
    """
    Verify that the tensor X_III satisfies:
        δ_{abc}^{pqr} = Σ_{def} X_III^{abcdef} * M_{def}^{pqr}
    """

    n = H.shape[0]
    I = np.eye(n)

    # Build the 6-index tensor M_{def}^{pqr}
    # term1: H_{dp} δ_{eq} δ_{fr}
    term1 = np.einsum('dp,eq,fr->defpqr', H, I, I)
    term2 = np.einsum('dp,eq,fr->defpqr', I, H, I)
    term3 = np.einsum('dp,eq,fr->defpqr', I, I, H)
    M = term1 + term2 + term3

    # -η * (HHδ + HδH + δHH)
    M -= eta * (
        np.einsum('dp,eq,fr->defpqr', H, H, I)
        + np.einsum('dp,eq,fr->defpqr', H, I, H)
        + np.einsum('dp,eq,fr->defpqr', I, H, H)
    )

    # +η² * HHH
    M += eta**2 * np.einsum('dp,eq,fr->defpqr', H, H, H)

    # Flatten (d,e,f) as rows, (p,q,r) as cols
    M_flat = M.reshape(n**3, n**3)

    # Invert to get X_III
    X_flat = np.linalg.inv(M_flat)
    X_III = X_flat.reshape(n, n, n, n, n, n)

    # Check X_III * M ≈ Identity tensor
    lhs = np.einsum('abcdef,defpqr->abcpqr', X_III, M)
    rhs = np.einsum('ap,bq,cr->abcpqr', I, I, I)

    success = np.allclose(lhs, rhs, atol=atol)
    return success, np.linalg.norm(lhs - rhs)


# ---------------------- TEST DRIVER ----------------------
if __name__ == "__main__":
    np.random.seed(0)

    # Adjustable parameters
    n = 30           # matrix size
    eta = 0.2       # coupling constant
    atol = 1e-6     # numerical tolerance

    # Create symmetric matrix H
    A = np.random.randn(n, n)
    H = 0.5 * (A + A.T)
    test_accuracy = False
    test_computation = not test_accuracy

    if test_accuracy:
        print(f"Currently testing the max error of X_IIIM with identity for  n={n}, eta={eta}, atol={atol} ...  :)")
        success, error_norm = check_XIII(H, eta, atol)

        if success:
            print("X_III within error tolerance... hell yeah")
        else:
            print("X_III not within tolerance... :(")
        print(f" = {error_norm:.2e}")

    else:
        print(f"Currently computing X_III for  n={n}, eta={eta}, atol={atol} ...  :)")
        tensor = compute_XIII(H, eta)
        print("Done computing X_III... :)")
