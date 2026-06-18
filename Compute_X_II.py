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


def check_XII_tensor(X, H, eta, tolerance=1e-10):
    n = H.shape[0]
    I = np.eye(n)

    # getting tensor M via inefficient einsums
    M = (np.einsum('ac,bd->abcd', H, I) + np.einsum('ac,bd->abcd', I, H)
        - eta * np.einsum('ac,bd->abcd', H, H)
    )

    lhs = np.einsum('ijmn,mnkl->ijkl', X, M)

    rhs = np.einsum('ik,jl->ijkl', I, I)

    #takes max of all entries in the tensor
    max_err = np.max(np.abs(lhs - rhs))
    return max_err < tolerance, max_err


if __name__ == "__main__":
    n = 100
    A = np.random.randn(n, n)
    H = (A + A.T) / 2  # symmetric
    eta = 0.3

    X = compute_XII(H, eta)
    print("Done computing XII")

    # success, error = check_XII_tensor(X, H, eta)
    # if success:
    #     print("Tensor equation works within reasonable accuracy")
    # else:
    #     print("Tensor equation not within reasonable accuracy")
    # print("Max Error: " +  str(error))
