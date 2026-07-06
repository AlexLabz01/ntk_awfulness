import os
import numpy as np
import torch
from typing import Optional
import itertools
from typing import Dict
from typing import Tuple
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")  # needed for Killarney / SLURM jobs before importing pyplot
from matplotlib import pyplot as plt
#X,Y,Z tensors
def compute_XII(H: torch.Tensor, eta: float) -> torch.Tensor:
    """
    X_II = inv( kron(H,I) + kron(I,H) - eta*kron(H,H) ) reshaped to (n,n,n,n)
    """
    n = H.shape[0]
    I = torch.eye(n, dtype=H.dtype, device=H.device)

    # M is (n^2 x n^2)
    M = torch.kron(H, I) + torch.kron(I, H) - eta * torch.kron(H, H)

    X_flat = torch.linalg.inv(M)  # (n^2 x n^2)
    return X_flat.reshape(n, n, n, n)


def compute_XIII(H: torch.Tensor, eta: float) -> torch.Tensor:
    """
    X_III = inv(M_flat) reshaped to (n,n,n,n,n,n), where M_flat is (n^3 x n^3)
    """
    n = H.shape[0]
    I = torch.eye(n, dtype=H.dtype, device=H.device)

    M_flat = (
        torch.kron(H, torch.kron(I, I))
        + torch.kron(I, torch.kron(H, I))
        + torch.kron(I, torch.kron(I, H))
        - eta
        * (
            torch.kron(H, torch.kron(H, I))
            + torch.kron(H, torch.kron(I, H))
            + torch.kron(I, torch.kron(H, H))
        )
        + (eta**2) * torch.kron(H, torch.kron(H, H))
    )

    M_inv_flat = torch.linalg.inv(M_flat)  # (n^3 x n^3)
    return M_inv_flat.reshape(n, n, n, n, n, n)


def compute_Y_I(H: torch.Tensor, X_II: torch.Tensor, X_III: torch.Tensor) -> torch.Tensor:
    """
    Y1[a,b,c,d,e,f] = X_II[a,b,d,e] * H[c,f] - sum_g H[c,g] * X_III[a,b,g,d,e,f]
    """
    term1 = torch.einsum("abde,cf->abcdef", X_II, H)
    term2 = torch.einsum("cg,abgdef->abcdef", H, X_III)
    return term1 - term2


def compute_Y_II(H: torch.Tensor, X_II: torch.Tensor) -> torch.Tensor:
    """
    (matches the paper’s Y2 definition)

    Y2[a,b,c,d] = H[a,c]*H[b,d] - sum_e H[b,e] * X_II[a,e,c,d]
    """
    term1 = torch.einsum("ac,bd->abcd", H, H)
    term2 = torch.einsum("be,aecd->abcd", H, X_II)
    return term1 - term2


def compute_Y_III(H: torch.Tensor, X_II: torch.Tensor, X_III: torch.Tensor) -> torch.Tensor:
    """
    Y3[a,b,c,d,e,f] =
        H[a,d]H[b,e]H[c,f]
      - sum_g H[b,g] X_II[a,g,d,e] H[c,f]
      - sum_g H[a,d] H[c,g] X_II[b,g,e,f]
      + sum_{i,g,h} H[c,i] X_II[b,i,g,h] X_III[a,g,h,d,e,f]
    """
    return (
        torch.einsum("ad,be,cf->abcdef", H, H, H)
        - torch.einsum("bg,agde,cf->abcdef", H, X_II, H)
        - torch.einsum("ad,cg,bgef->abcdef", H, H, X_II)
        + torch.einsum("ci,bigh,aghdef->abcdef", H, X_II, X_III)
    )


def compute_Y_IV(H: torch.Tensor, X_II: torch.Tensor, X_III: torch.Tensor) -> torch.Tensor:
    """
    Y4[a,b,c,d,e,f] = H[a,d] * X_II[b,c,e,f] - sum_{g,h} X_II[b,c,g,h] * X_III[a,g,h,d,e,f]
    """
    return torch.einsum("ad,bcef->abcdef", H, X_II) - torch.einsum("bcgh,aghdef->abcdef", X_II, X_III)


def compute_Z_A(Y_II: torch.Tensor) -> torch.Tensor:
    return Y_II


def compute_Z_B(Y_II: torch.Tensor, X_II: torch.Tensor, eta: float) -> torch.Tensor:
    return Y_II + (eta / 2.0) * X_II


def compute_Z_IA(Y_III: torch.Tensor, Y_IV: torch.Tensor, eta: float) -> torch.Tensor:
    return -Y_III - (eta / 2.0) * Y_IV


def compute_Z_IB(Y_I: torch.Tensor, Y_III: torch.Tensor, Y_IV: torch.Tensor, X_III: torch.Tensor, eta: float) -> torch.Tensor:
    return -Y_III - (eta / 2.0) * Y_IV - (eta / 2.0) * Y_I - (eta**2 / 6.0) * X_III


def compute_Z_IIA(Y_III: torch.Tensor) -> torch.Tensor:
    return -Y_III


def compute_Z_IIB(Y_I: torch.Tensor, Y_III: torch.Tensor, Y_IV: torch.Tensor, eta: float) -> torch.Tensor:
    # -Y_III - Y_III^{bacdef} - Y_III^{acbdfe} - eta*Y_IV - eta*Y_I
    Y_III_bacdef = Y_III.permute(1, 0, 2, 3, 4, 5)
    Y_III_acbdfe = Y_III.permute(0, 2, 1, 3, 5, 4)
    return -Y_III - Y_III_bacdef - Y_III_acbdfe - eta * Y_IV - eta * Y_I


def compute_Z_tensors(H: torch.Tensor, eta: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Given H_lower (n x n) and eta (scalar), compute and return only the Z tensors.

    Important convention:
      - X_II and X_III are computed using the *lower-index* H (the input H).
      - The Y tensors use *raised-index* H, i.e. H_upper = inv(H_lower).
        (So we pass H_upper into compute_Y_I / Y_II / Y_III / Y_IV.)

    Returns:
      Z_A   : (n,n,n,n)
      Z_B   : (n,n,n,n)
      Z_IA  : (n,n,n,n,n,n)
      Z_IB  : (n,n,n,n,n,n)
      Z_IIA : (n,n,n,n,n,n)
      Z_IIB : (n,n,n,n,n,n)
    """
    # X tensors use the normal (lower-index) H
    X_II = compute_XII(H, eta)
    X_III = compute_XIII(H, eta)

    # Y tensors use raised-index H (H_upper)
    H_upper = torch.linalg.inv(H)

    Y_I = compute_Y_I(H_upper, X_II, X_III)
    Y_II = compute_Y_II(H_upper, X_II)
    Y_III = compute_Y_III(H_upper, X_II, X_III)
    Y_IV = compute_Y_IV(H_upper, X_II, X_III)

    Z_A = compute_Z_A(Y_II)
    Z_B = compute_Z_B(Y_II, X_II, eta)

    Z_IA = compute_Z_IA(Y_III, Y_IV, eta)
    Z_IB = compute_Z_IB(Y_I, Y_III, Y_IV, X_III, eta)

    Z_IIA = compute_Z_IIA(Y_III)
    Z_IIB = compute_Z_IIB(Y_I, Y_III, Y_IV, eta)

    return Z_A, Z_B, Z_IA, Z_IB, Z_IIA, Z_IIB

#expectations (by Yigal)

#expectation helper functions (Yigal)
def s2_expval(M):
    """
    M should be a square matrix

    """
    diag = torch.diagonal(M)
    K = torch.exp(-0.5 * (diag.unsqueeze(0) + diag.unsqueeze(1))) * torch.sinh(M)
    return K


def c2_expval(M):
    """
    M should be a square matrix

    """
    diag = torch.diagonal(M)
    K = torch.exp(-0.5 * (diag.unsqueeze(0) + diag.unsqueeze(1))) * torch.cosh(M)
    return K


def s2c1_expval(M):
    """
    M should be a square matrix

    """
    diag = torch.diagonal(M)

    prefactor = torch.exp(-0.5 * (diag.unsqueeze(1).unsqueeze(2) + diag.unsqueeze(0).unsqueeze(2)
                                  + diag.unsqueeze(0).unsqueeze(1)))

    Kij = M.unsqueeze(2)
    Kia = M.unsqueeze(1)
    Kja = M.unsqueeze(0)

    result = prefactor * (torch.exp(Kij) * torch.cosh(Kia - Kja) - torch.exp(-Kij) * torch.cosh(Kia + Kja))/2

    return result


def s1c1z1_expval(M):
    """
    M should be a square matrix

    """
    diag = torch.diagonal(M)

    prefactor = torch.exp(-0.5 * (diag.unsqueeze(1).unsqueeze(2) + diag.unsqueeze(0).unsqueeze(2)))

    Kij = M.unsqueeze(2)
    Kki = M.T.unsqueeze(1)
    Kkj = M.unsqueeze(0)

    result = prefactor * (Kki * torch.cosh(Kij) - Kkj * torch.sinh(Kij))

    return result


def c1z2_expval(M):
    """
    M should be a square matrix

    """
    diag = torch.diagonal(M)

    prefactor = torch.exp(-0.5 * diag).unsqueeze(1).unsqueeze(2)

    Kjk = M.unsqueeze(0)
    Kij = M.unsqueeze(2)
    Kik = M.unsqueeze(1)

    result = prefactor * (Kjk - Kij * Kik)

    return result


def s2z2_expval(M):
    """
    M should be a square matrix
    """
    diag = torch.diagonal(M)

    prefactor = torch.exp(-0.5 * (diag.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                                  + diag.unsqueeze(0).unsqueeze(2).unsqueeze(3)))

    Kij = M.unsqueeze(2).unsqueeze(3)
    Kkl = M.unsqueeze(0).unsqueeze(0)

    Kki = M.T.unsqueeze(1).unsqueeze(3)
    Kkj = M.T.unsqueeze(0).unsqueeze(3)

    Kli = M.T.unsqueeze(1).unsqueeze(2)
    Klj = M.T.unsqueeze(0).unsqueeze(2)

    result = prefactor * ((Kkl - Kki * Kli - Kkj * Klj) * torch.sinh(Kij) + (Kki * Klj + Kkj * Kli) * torch.cosh(Kij))

    return result


def c2z2_expval(M):
    """
    M should be a square matrix

    """
    diag = torch.diagonal(M)

    prefactor = torch.exp(-0.5 * (diag.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                                  + diag.unsqueeze(0).unsqueeze(2).unsqueeze(3)))

    Kij = M.unsqueeze(2).unsqueeze(3)
    Kkl = M.unsqueeze(0).unsqueeze(0)

    Kki = M.T.unsqueeze(1).unsqueeze(3)
    Kkj = M.T.unsqueeze(0).unsqueeze(3)

    Kli = M.T.unsqueeze(1).unsqueeze(2)
    Klj = M.T.unsqueeze(0).unsqueeze(2)

    result = prefactor * ((Kkl - Kki * Kli - Kkj * Klj) * torch.cosh(Kij) + (Kki * Klj + Kkj * Kli) * torch.sinh(Kij))

    return result


def s2c2_expval(M):
    """
    M should be a square matrix

    """
    diag = torch.diagonal(M)

    prefactor = torch.exp(-0.5 * (diag.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                                  + diag.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                                  + diag.unsqueeze(0).unsqueeze(1).unsqueeze(3)
                                  + diag.unsqueeze(0).unsqueeze(1).unsqueeze(2)))

    Kij = M.unsqueeze(2).unsqueeze(3)
    Kab = M.unsqueeze(0).unsqueeze(0)

    Kia = M.unsqueeze(1).unsqueeze(3)
    Kib = M.unsqueeze(1).unsqueeze(2)

    Kja = M.unsqueeze(0).unsqueeze(3)
    Kjb = M.unsqueeze(0).unsqueeze(2)

    term1 = torch.exp(Kij) * (torch.exp(-Kab) * torch.cosh(Kia + Kib - Kja - Kjb)
                              + torch.exp(Kab) * torch.cosh(Kia - Kib - Kja + Kjb))

    term2 = -torch.exp(-Kij) * (torch.exp(-Kab) * torch.cosh(Kia + Kib + Kja + Kjb)
                                + torch.exp(Kab) * torch.cosh(Kia - Kib + Kja - Kjb))

    return prefactor * (term1 + term2)/4


def sk_expval(K, k):
    """
    K should be a square matrix

    k is the number of indices
    """
    n = K.shape[0]

    eps = torch.tensor([[1, *c] for c in itertools.product([-1, 1], repeat=k - 1)], dtype=K.dtype, device=K.device)
    eps_sum = eps.sum(dim=1)
    sign = (-1) ** (eps_sum // 2)

    result = torch.zeros(*([n] * k), device=K.device, dtype=K.dtype)
    for I_tuple in itertools.product(range(n), repeat=k):
        I = torch.tensor(I_tuple, device=K.device)
        K_II = K[I][:, I]

        quad = (eps @ K_II * eps).sum(dim=1)
        val = (sign * torch.exp(-0.5 * quad)).sum() / (2 ** (k - 1))
        result[I_tuple] = val

    return result


def cr_expval(K, r):
    """
    K should be a square matrix

    r is the number of indices
    """
    n = K.shape[0]

    eps = torch.tensor([[1, *c] for c in itertools.product([-1, 1], repeat=r - 1)], dtype=K.dtype, device=K.device)

    result = torch.zeros(*([n] * r), device=K.device, dtype=K.dtype)
    for I_tuple in itertools.product(range(n), repeat=r):
        I = torch.tensor(I_tuple, device=K.device)
        K_II = K[I][:, I]

        quad = (eps @ K_II * eps).sum(dim=1)
        val = torch.exp(-0.5 * quad).sum() / (2 ** (r - 1))
        result[I_tuple] = val

    return result








#m tensors

def compute_m_NTK(
    H_beta_alpha1: torch.Tensor,   # shape (test+train, training_size)
    Htilde_lower: torch.Tensor,    # shape (training_size, training_size)
    y_labels: torch.Tensor,        # shape (n_out, training_size)
    training_size: int,
) -> torch.Tensor:
    ts = training_size

    Htilde_upper = torch.linalg.inv(Htilde_lower[:ts, :ts])   # (ts, ts)
    H_beta_train = H_beta_alpha1[:, :ts]                      # (test+train, ts)


    m = torch.einsum("ba,ac,ic->ib", H_beta_train, Htilde_upper, y_labels)
    return m



def compute_m_delta_NTK(
    n_L: float,            # coefficient n_L in the formula
    A: torch.Tensor,       # A[delta, alpha1, alpha2, alpha3] stored as (delta_size, delta_size, delta_size, delta_size)
    B: torch.Tensor,       # B[delta, alpha1, alpha2, alpha3] stored as (delta_size, delta_size, delta_size, delta_size)
    Htilde_lower: torch.Tensor,  # Htilde_lower[alpha, alpha] (at least training_size x training_size)
    y: torch.Tensor,       # y[i, alpha4] (usually i = output neuron index, alpha4 over training set)
    training_size: int,    # all alpha indices run over {0, ..., training_size-1}
) -> torch.Tensor:
    """
    m[i, delta] = sum_{alpha1, alpha2, alpha3, alpha4}
                  ( A[delta, alpha1, alpha2, alpha3]
                    + B[delta, alpha2, alpha1, alpha3]
                    + n_L * B[delta, alpha3, alpha1, alpha2] )
                  * Htilde_upper[alpha1, alpha2]
                  * Htilde_upper[alpha3, alpha4]
                  * y[i, alpha4]
    where Htilde_upper is the inverse of the training-block of Htilde_lower.
    Only the alpha indices are summed over the training set; delta ranges over the full train+test set.
    """
    ts = training_size

    # alpha indices are training-only
    A_train = A[:, :ts, :ts, :ts]   # (delta, alpha1, alpha2, alpha3)
    B_train = B[:, :ts, :ts, :ts]   # (delta, alpha1, alpha2, alpha3)

    # y is usually already (i, training_size), but slice safely just in case
    y_train = y[:, :ts]             # (i, alpha4)

    # Htilde_upper is the inverse of the training block of Htilde_lower
    H_train = Htilde_lower[:ts, :ts]
    Htilde_upper = torch.linalg.inv(H_train)  # (alpha, alpha)

    # Build the bracketed tensor:
    # A[delta,a1,a2,a3] + B[delta,a2,a1,a3] + n_L * B[delta,a3,a1,a2]
    term_A = A_train
    term_B_a2a1a3 = B_train.permute(0, 2, 1, 3)       # (delta, alpha2, alpha1, alpha3)
    term_B_a3a1a2 = B_train.permute(0, 3, 1, 2)  # (delta, alpha3, alpha1, alpha2)
    bracket = term_A + term_B_a2a1a3 + n_L * term_B_a3a1a2

    # Contract:
    # delta = d, alpha1 = a, alpha2 = b, alpha3 = c, alpha4 = e
    m = torch.einsum("dabc,ab,ce,ie->id", bracket, Htilde_upper, Htilde_upper, y_train)

    return m



def compute_m_dNTK(
    n_L: float,                 # coefficient n_L in the formula
    z_tensor_A: torch.Tensor,   # Z_A[a1, a2, a3, a4] (upper indices are just notation, not an inverse)
    z_tensor_B: torch.Tensor,   # Z_B[a1, a2, a3, a4] (upper indices are just notation, not an inverse)
    P_tensor: torch.Tensor,     # P[i0,i1,i2,i3] stored full over delta_size on every axis (Model A)
    Q_tensor: torch.Tensor,     # Q[i0,i1,i2,i3] stored full over delta_size on every axis (Model A)
    y_labels: torch.Tensor,     # y[i, a4] (a4 is training)
    training_size: int,
) -> torch.Tensor:
    """
    m[i, delta] = - sum_{a1, a2, a3, a4} [
                    2 * ( P[delta,a1,a2,a3] + Q[delta,a1,a2,a3] + n_L * Q[delta,a2,a1,a3] ) * Z_B[a1,a2,a3,a4]
                    + ( n_L * P[a1,delta,a2,a3] + Q[a1,delta,a2,a3] + Q[a1,a2,delta,a3] ) * Z_A[a1,a2,a3,a4]
                    + ( P[a1,delta,a2,a3] + n_L * Q[a1,delta,a2,a3] + Q[a1,a2,delta,a3] ) * Z_A[a1,a2,a4,a3]
                  ] * y[i, a4]

    Model A:
      - P and Q are full 4-index tensors over the full delta_size on every axis.
      - The alpha indices (a1,a2,a3,a4) are training-only and are the only ones sliced to training_size.
      - delta is not sliced and can be a test index.
    """
    ts = training_size

    # alpha indices are training-only
    y_train = y_labels[:, :ts]                  # (i, a4)
    Z_A_train = z_tensor_A[:ts, :ts, :ts, :ts]  # (a1,a2,a3,a4)
    Z_B_train = z_tensor_B[:ts, :ts, :ts, :ts]  # (a1,a2,a3,a4)

    # -------------------------
    # Part 1: Z_B term
    # uses P[delta,a1,a2,a3], Q[delta,a1,a2,a3], Q[delta,a2,a1,a3]
    # -------------------------
    P_Dabc = P_tensor[:, :ts, :ts, :ts]         # (delta, a1, a2, a3)
    Q_Dabc = Q_tensor[:, :ts, :ts, :ts]         # (delta, a1, a2, a3)
    Q_Dbac = Q_Dabc.permute(0, 2, 1, 3)         # (delta, a2, a1, a3)

    bracket_ZB = 2.0 * (P_Dabc + Q_Dabc + n_L * Q_Dbac)
    part_ZB = torch.einsum("dabc,abce,ie->id", bracket_ZB, Z_B_train, y_train)

    # -------------------------
    # Part 2: Z_A term (a1,a2,a3,a4)
    # uses P[a1,delta,a2,a3], Q[a1,delta,a2,a3], Q[a1,a2,delta,a3]
    # -------------------------
    P_aDbc = P_tensor[:ts, :, :ts, :ts]         # (a1, delta, a2, a3)
    Q_aDbc = Q_tensor[:ts, :, :ts, :ts]         # (a1, delta, a2, a3)

    Q_abDc = Q_tensor[:ts, :ts, :, :ts]         # (a1, a2, delta, a3)
    Q_abDc_as_aDbc = Q_abDc.permute(0, 2, 1, 3) # (a1, delta, a2, a3)

    bracket_ZA = n_L * P_aDbc + Q_aDbc + Q_abDc_as_aDbc
    part_ZA = torch.einsum("adbc,abce,ie->id", bracket_ZA, Z_A_train, y_train)

    # -------------------------
    # Part 3: Z_A term with last two indices swapped (a1,a2,a4,a3)
    # uses P[a1,delta,a2,a3], Q[a1,delta,a2,a3], Q[a1,a2,delta,a3]
    # -------------------------
    Z_A_swap_last = Z_A_train.permute(0, 1, 3, 2)  # (a1, a2, a4, a3)

    bracket_ZA_swapped = P_aDbc + n_L * Q_aDbc + Q_abDc_as_aDbc
    part_ZA_swapped = torch.einsum("adbc,abec,ie->id", bracket_ZA_swapped, Z_A_swap_last, y_train)

    return -(part_ZB + part_ZA + part_ZA_swapped)

def compute_m_ddNTK_I(
    n_L: float,                  # coefficient n_L in the formula
    z_tensor_IA: torch.Tensor,   # Z_IA[a1,a2,a3,a4,a5,a6] stored full; all indices are alpha-type
    z_tensor_IB: torch.Tensor,   # Z_IB[a1,a2,a3,a4,a5,a6] stored full; all indices are alpha-type
    R_tensor: torch.Tensor,      # R[i0,i1,i2,i3] stored full over delta_size on every axis (Model A)
    K_matrix: torch.Tensor,      # K[a5,a6] stored full, but a5/a6 are training in the sums
    labels_y: torch.Tensor,      # y[i,a] where a is training
    training_size: int,
) -> torch.Tensor:
    """
    m[i, delta] = - sum_{a1,a2,a3,a4,a5,a6} [
                    R[delta,a1,a2,a3] * ( Z_IB[a1,a2,a3,a4,a5,a6]
                                        + Z_IB[a2,a3,a1,a5,a6,a4]
                                        + Z_IB[a3,a1,a2,a6,a4,a5] )
                    + R[a1,delta,a2,a3] * Z_IA[a1,a2,a3,a4,a5,a6]
                    + R[a1,a2,a3,delta] * Z_IA[a1,a2,a3,a5,a6,a4]
                    + R[a1,a3,delta,a2] * Z_IA[a1,a2,a3,a6,a4,a5]
                  ]
                  * ( y[i,a4] * ( sum_j y[j,a5]*y[j,a6] + n_L*K[a5,a6] )
                      + y[i,a5] * K[a6,a4]
                      + y[i,a6] * K[a4,a5] )

    Model A:
      - R is a full 4-index tensor over the full delta_size index set on every axis.
      - All a1..a6 are training-only and are the only indices sliced to training_size.
      - delta is not sliced and can be a test index.
    """
    ts = training_size

    # alpha indices are training-only
    y_train = labels_y[:, :ts]            # (i, a4)
    K_train = K_matrix[:ts, :ts]          # (a5, a6)

    Z_IA = z_tensor_IA[:ts, :ts, :ts, :ts, :ts, :ts]
    Z_IB = z_tensor_IB[:ts, :ts, :ts, :ts, :ts, :ts]

    # build sum_j y[j,a5]*y[j,a6]
    yy_56 = torch.einsum("ja,jb->ab", y_train, y_train)  # (a5,a6)

    # build Gamma(i,a4,a5,a6)
    term_a = torch.einsum("id,ef->idef", y_train, (yy_56 + n_L * K_train))
    term_b = torch.einsum("ie,fd->idef", y_train, K_train)  # y[i,a5] * K[a6,a4]
    term_c = torch.einsum("if,de->idef", y_train, K_train)  # y[i,a6] * K[a4,a5]
    gamma = term_a + term_b + term_c  # (i,a4,a5,a6)

    # -------------------------
    # IB block: R[delta,a1,a2,a3] times three permutations of Z_IB
    # -------------------------
    R_Dabc = R_tensor[:, :ts, :ts, :ts]  # (delta, a1, a2, a3)

    ib1 = torch.einsum("Dabc,abcdef,idef->iD", R_Dabc, Z_IB, gamma)

    Z_IB_bca_efd = Z_IB.permute(1, 2, 0, 4, 5, 3)  # (a2,a3,a1,a5,a6,a4)
    ib2 = torch.einsum("Dabc,bcaefd,idef->iD", R_Dabc, Z_IB_bca_efd, gamma)

    Z_IB_cab_fde = Z_IB.permute(2, 0, 1, 5, 3, 4)  # (a3,a1,a2,a6,a4,a5)
    ib3 = torch.einsum("Dabc,cabfde,idef->iD", R_Dabc, Z_IB_cab_fde, gamma)

    ib_total = ib1 + ib2 + ib3

    # -------------------------
    # IA block: three different placements of delta inside R, with matching Z_IA permutations
    # -------------------------

    # IA term 1: R[a1,delta,a2,a3] * Z_IA[a1,a2,a3,a4,a5,a6]
    R_aDbc = R_tensor[:ts, :, :ts, :ts]  # (a1, delta, a2, a3)
    ia1 = torch.einsum("aDbc,abcdef,idef->iD", R_aDbc, Z_IA, gamma)

    # IA term 2: R[a1,a2,a3,delta] * Z_IA[a1,a2,a3,a5,a6,a4]
    R_abcD = R_tensor[:ts, :ts, :ts, :]  # (a1, a2, a3, delta)
    Z_IA_abcefd = Z_IA.permute(0, 1, 2, 4, 5, 3)  # (a1,a2,a3,a5,a6,a4)
    ia2 = torch.einsum("abcD,abcefd,idef->iD", R_abcD, Z_IA_abcefd, gamma)

    # IA term 3: R[a1,a3,delta,a2] * Z_IA[a1,a2,a3,a6,a4,a5]
    R_acDb = R_tensor[:ts, :ts, :, :ts]  # (a1, a3, delta, a2)
    Z_IA_abcfde = Z_IA.permute(0, 1, 2, 5, 3, 4)  # (a1,a2,a3,a6,a4,a5)
    ia3 = torch.einsum("acDb,abcfde,idef->iD", R_acDb, Z_IA_abcfde, gamma)

    ia_total = ia1 + ia2 + ia3

    # overall minus sign in the formula
    return -(ib_total + ia_total)

def compute_m_ddNTK_II(
    n_L: float,                   # coefficient n_L in the formula
    z_tensor_IIB: torch.Tensor,   # Z_IIB[a1,a2,a3,a4,a5,a6] (stored full; all six indices are alpha-type)
    z_tensor_IIA: torch.Tensor,   # Z_IIA[a1,a2,a3,a4,a5,a6] (stored full; all six indices are alpha-type)
    S_tensor: torch.Tensor,       # S[i0,i1,i2,i3] stored full over delta_size on every axis (Model A)
    T_tensor: torch.Tensor,       # T[i0,i1,i2,i3] stored full over delta_size on every axis (Model A)
    U_tensor: torch.Tensor,       # U[i0,i1,i2,i3] stored full over delta_size on every axis (Model A)
    K_matrix: torch.Tensor,       # K[a5,a6] stored full, but a5/a6 are training in the sums
    labels_y: torch.Tensor,       # y[i,a] where a is training
    training_size: int,
) -> torch.Tensor:
    """
    m[i, delta] = - sum_{a1,a2,a3,a4,a5,a6} [
                    S[delta,a1,a2,a3] * Z_IIB[a1,a2,a3,a4,a5,a6]
                  + T[delta,a1,a2,a3] * Z_IIB[a2,a3,a1,a5,a6,a4]
                  + U[delta,a1,a2,a3] * Z_IIB[a3,a1,a2,a6,a4,a5]
                  + S[a1,a2,delta,a3] * Z_IIA[a1,a2,a3,a5,a6,a4]
                  + T[a1,delta,a3,a2] * Z_IIA[a1,a2,a3,a4,a5,a6]
                  + U[a1,a3,a2,delta] * Z_IIA[a1,a2,a3,a6,a4,a5]
                  ]
                  * ( y[i,a4] * ( sum_j y[j,a5]*y[j,a6] + n_L*K[a5,a6] )
                      + y[i,a5] * K[a6,a4]
                      + y[i,a6] * K[a4,a5] )

    Model A:
      - S, T, U are full 4-index tensors over the full delta_size index set on every axis.
      - All a1..a6 are training-only and are the only indices sliced to training_size.
      - delta is not sliced and can be a test index.
    """
    ts = training_size

    # alpha indices are training-only
    y_train = labels_y[:, :ts]            # (i, a4)
    K_train = K_matrix[:ts, :ts]          # (a5, a6)

    Z_IIB = z_tensor_IIB[:ts, :ts, :ts, :ts, :ts, :ts]
    Z_IIA = z_tensor_IIA[:ts, :ts, :ts, :ts, :ts, :ts]

    # build sum_j y[j,a5]*y[j,a6]
    yy_56 = torch.einsum("ja,jb->ab", y_train, y_train)  # (a5,a6)

    # build Gamma(i,a4,a5,a6)
    term_a = torch.einsum("id,ef->idef", y_train, (yy_56 + n_L * K_train))
    term_b = torch.einsum("ie,fd->idef", y_train, K_train)  # y[i,a5] * K[a6,a4]
    term_c = torch.einsum("if,de->idef", y_train, K_train)  # y[i,a6] * K[a4,a5]
    gamma = term_a + term_b + term_c  # (i,a4,a5,a6)

    # -------------------------
    # IIB block: delta is in the first slot of S,T,U
    # -------------------------
    S_Dabc = S_tensor[:, :ts, :ts, :ts]  # (delta, a1, a2, a3)
    T_Dabc = T_tensor[:, :ts, :ts, :ts]  # (delta, a1, a2, a3)
    U_Dabc = U_tensor[:, :ts, :ts, :ts]  # (delta, a1, a2, a3)

    iib1 = torch.einsum("Dabc,abcdef,idef->iD", S_Dabc, Z_IIB, gamma)

    Z_IIB_bca_efd = Z_IIB.permute(1, 2, 0, 4, 5, 3)  # (a2,a3,a1,a5,a6,a4)
    iib2 = torch.einsum("Dabc,bcaefd,idef->iD", T_Dabc, Z_IIB_bca_efd, gamma)

    Z_IIB_cab_fde = Z_IIB.permute(2, 0, 1, 5, 3, 4)  # (a3,a1,a2,a6,a4,a5)
    iib3 = torch.einsum("Dabc,cabfde,idef->iD", U_Dabc, Z_IIB_cab_fde, gamma)

    # -------------------------
    # IIA block: delta appears in different slots of S,T,U
    # -------------------------

    # S[a1,a2,delta,a3] with Z_IIA[a1,a2,a3,a5,a6,a4]
    S_abDc = S_tensor[:ts, :ts, :, :ts]          # (a1, a2, delta, a3)
    Z_IIA_abcefd = Z_IIA.permute(0, 1, 2, 4, 5, 3)  # (a1,a2,a3,a5,a6,a4)
    iia1 = torch.einsum("abDc,abcefd,idef->iD", S_abDc, Z_IIA_abcefd, gamma)

    # T[a1,delta,a3,a2] with Z_IIA[a1,a2,a3,a4,a5,a6]
    T_aDcb = T_tensor[:ts, :, :ts, :ts]          # (a1, delta, a3, a2)
    iia2 = torch.einsum("aDcb,abcdef,idef->iD", T_aDcb, Z_IIA, gamma)

    # U[a1,a3,a2,delta] with Z_IIA[a1,a2,a3,a6,a4,a5]
    U_acbD = U_tensor[:ts, :ts, :ts, :]          # (a1, a3, a2, delta)
    Z_IIA_abcfde = Z_IIA.permute(0, 1, 2, 5, 3, 4)  # (a1,a2,a3,a6,a4,a5)
    iia3 = torch.einsum("acbD,abcfde,idef->iD", U_acbD, Z_IIA_abcfde, gamma)

    return -(iib1 + iib2 + iib3 + iia1 + iia2 + iia3)

#Tensor Recursion Relations
def compute_new_theta_K(
    theta_prev: torch.Tensor,
    C_W: float,
    C_b: float,
    lambda_b: float,
    lambda_W: float,
    K: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # decided to rewrite code from compute_new_K_vectorized for efficiency to avoid redundant steps, rather than calling it
    diag = torch.diagonal(K)
    # diag[:, None] + diag[None, :] is the matrix who's i j element
    # is K_ii + K_jj.
    exp_factor = torch.exp(-0.5 * (diag[:, None] + diag[None, :]))
    sinh_term = torch.sinh(K)
    cosh_term = torch.cosh(K)

    K_new = C_b + C_W * exp_factor * sinh_term
    K_new = 0.5 * (K_new + K_new.T)

    K_cos = C_b + C_W * exp_factor * cosh_term
    K_cos = 0.5 * (K_cos + K_cos.T)

    theta_new = lambda_b + lambda_W * K_new + C_W * K_cos * theta_prev
    return theta_new, K_new

def compute_new_K(
    K_prev: torch.Tensor,
    C_W: float,
    C_b: float = 0.0,
) -> torch.Tensor:

    diag = torch.diagonal(K_prev)                       # K_aa
    exp_factor = torch.exp(-0.5 * (diag[:, None] + diag[None, :]))
    sinh_term = torch.sinh(K_prev)

    K_next = C_b + C_W * exp_factor * sinh_term
    K_next = 0.5 * (K_next + K_next.T)                  # making sure theres symmetry

    return K_next

def raise_vertex_indices(
    V_lower: torch.Tensor,   # V_lower[b1,b2,b3,b4]
    Ginv: torch.Tensor,      # Ginv[a,b]
) -> torch.Tensor:
    return torch.einsum("a1b1,a2b2,a3b3,a4b4,b1b2b3b4->a1a2a3a4",
                        Ginv, Ginv, Ginv, Ginv, V_lower)


def lower_vertex_indices(
    V_raised: torch.Tensor,  # V_raised[a1,a2,a3,a4]
    G: torch.Tensor,         # G[a,b]
) -> torch.Tensor:
    return torch.einsum("b1a1,b2a2,b3a3,b4a4,a1a2a3a4->b1b2b3b4",
                        G, G, G, G, V_raised)


def V_update_raised(
    V_prev: torch.Tensor,         # V_prev[p,q,r,s]  (v raised/inverse)
    G: torch.Tensor,              # G[p,q] (same as K)
    E_sigma4: torch.Tensor,       # E_sigma4[a,b,c,d]  = E[sigma_a sigma_b sigma_c sigma_d]
    E_sigma2: torch.Tensor,       # E_sigma2[a,b]      = E[sigma_a sigma_b]
    E_zz_sigma2: torch.Tensor,    # E_zz_sigma2[p,q,a,b] = E[z_p z_q sigma_a sigma_b]
    Cw_next: float,               # C_W^{(l+1)}
    n_l: int,
    n_lm1: int,
) -> torch.Tensor:


    ratio = n_l / n_lm1
    cw2 = Cw_next * Cw_next

    # 4 point correlator part
    connected_sigma4 = E_sigma4 - torch.einsum("ab,cd->abcd", E_sigma2, E_sigma2)
    term1 = cw2 * connected_sigma4

    # centered zz-sigma-sigma part
    centered_zz_sigma2 = E_zz_sigma2 - torch.einsum("pq,ab->pqab", G, E_sigma2)

    # vertex part
    term2_core = torch.einsum("pqrs,pqab,rscd->abcd", V_prev, centered_zz_sigma2, centered_zz_sigma2)
    term2 = (ratio * cw2 / 4.0) * term2_core

    return term1 + term2

#The other recursions

def U_update_abcd(U_l: torch.Tensor, H_l: torch.Tensor, E_ddpp_abcd: torch.Tensor, E_p_ad: torch.Tensor,
    Cw_next: float, n_l: int, n_lm1: int, E_p_bc: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    U^{(l+1)}_{a d b c}
      = (Cw_next)^2 * E_ddpp[a,b,c,d] * H[a,b] * H[a,c] * H[b,d]
        + (n_l/n_{l-1}) * (Cw_next)^2 * E_p_ad[a,d] * E_p_bc[b,c] * U_l[a,d,b,c]
    """
    if E_p_bc is None:
        E_p_bc = E_p_ad

    cw2 = Cw_next ** 2
    ratio = n_l / n_lm1

    term1 = cw2 * torch.einsum("abcd,ab,ac,bd->adbc", E_ddpp_abcd, H_l, H_l, H_l)

    term2 = (ratio * cw2) * torch.einsum("ad,bc,adbc->adbc", E_p_ad, E_p_bc, U_l)

    return term1 + term2

def S_update_abcd(
    S_l: torch.Tensor,            # S^{(l)}_{a b c d}
    B: torch.Tensor,              # B_{a b c d}
    H: torch.Tensor,              # H_{i j}
    E_pppp_abcd: torch.Tensor,    # <σ'_a σ'_b σ'_c σ'_d>
    E_p_ab: torch.Tensor,         # <σ'_a σ'_b>
    E_dd_ab: torch.Tensor,        # <σ''_a σ''_b>
    E_ddpp_abcd: torch.Tensor,    # <σ''_a σ''_b σ'_c σ'_d>
    Cw_next: float,
    Lambda_next: float,
    n_l: int,
    n_lm1: int,
    E_p_cd: Optional[torch.Tensor] = None,  # <σ'_c σ'_d> shape (D,D) in (c,d); default uses E_p_ab
) -> torch.Tensor:
    """
    Term structure:
      1)  Cw*Lambda * <σ'_a σ'_b σ'_c σ'_d> * H_{ac} H_{bd}
      2)  (n_l/n_{l-1})*Cw * <σ'_c σ'_d> * [ Lambda*<σ'_a σ'_b> + Cw*H_{ab}*<σ''_a σ''_b> ] * B_{abcd}
      3)  (Cw)^2 * <σ''_a σ''_b σ'_c σ'_d> * H_{ab} H_{ac} H_{bd}
      4)  (n_l/n_{l-1})*(Cw)^2 * <σ'_a σ'_b> * <σ'_c σ'_d> * S^{(l)}_{abcd}
    """
    if E_p_cd is None:
        E_p_cd = E_p_ab

    ratio = n_l / n_lm1
    cw2 = Cw_next * Cw_next

    term1 = (Cw_next * Lambda_next) * torch.einsum("abcd,ac,bd->abcd", E_pppp_abcd, H, H)

    M_ab = Lambda_next * E_p_ab + Cw_next * (H * E_dd_ab)
    term2 = (ratio * Cw_next) * torch.einsum("cd,ab,abcd->abcd",E_p_cd, M_ab, B)

    term3 = cw2 * torch.einsum("abcd,ab,ac,bd->abcd",E_ddpp_abcd,H, H, H)

    term4 = (ratio * cw2) * torch.einsum("ab,cd,abcd->abcd", E_p_ab, E_p_cd, S_l)

    return term1 + term2 + term3 + term4


def R_update_zabc(
    R_l: torch.Tensor,
    H: torch.Tensor,
    B: torch.Tensor,
    P: torch.Tensor,
    E_ddsp_zabc: torch.Tensor,      # <σ''_z σ_a σ'_b σ'_c>
    E_p: torch.Tensor,              # <σ'_i σ'_j>
    E_dd_za: torch.Tensor,          # <σ''_z σ_a>
    E_3pppp_zabc: torch.Tensor,     # <σ'''_z σ'_a σ'_b σ'_c>
    E_3p_za: torch.Tensor,          # <σ'''_z σ'_a>
    E_dddd_za: torch.Tensor,        # <σ''_z σ''_a>
    Cw: float,
    Lambda: float,
    n_l: int,
    n_lm1: int,
) -> torch.Tensor:
    """
    Term structure:
      1)  Lambda*Cw * <σ''_z σ_a σ'_b σ'_c> * H_{z b} * H_{z c}
      2)  (n_l/n_{l-1})*Cw * <σ'_b σ'_c> * Lambda*<σ''_z σ_a> * B_{z z b c}
      3)  (n_l/n_{l-1})*Cw * <σ'_b σ'_c> * Lambda * [ <σ''_z σ_a> P_{z b c z} + <σ'_z σ'_a> P_{z b c a} ]
      4)  (Cw)^2 * <σ'''_z σ'_a σ'_b σ'_c> * H_{z a} * H_{z b} * H_{z c}
      5)  (n_l/n_{l-1})*(Cw)^2 * <σ'_b σ'_c> * <σ'''_z σ'_a> * B_{z z b c} * H_{z a}
      6)  (n_l/n_{l-1})*(Cw)^2 * <σ'_b σ'_c> * [ <σ'''_z σ'_a> P_{z b c z} + <σ''_z σ''_a> P_{z b c a} ] * H_{z a}
      7)  (n_l/n_{l-1})*(Cw)^2 * <σ'_b σ'_c> * <σ'_z σ'_a> * R^{(l)}_{z a b c}
    """
    ratio = n_l / n_lm1
    cw2 = Cw * Cw

    B_zzbc = torch.einsum("zzbc->zbc", B)   # B_{z z b c}
    P_zbcz = torch.einsum("zbcz->zbc", P)   # P_{z b c z}

    term1 = (Lambda * Cw) * torch.einsum("zabc,zb,zc->zabc", E_ddsp_zabc, H, H)

    term2 = (ratio * Cw * Lambda) * torch.einsum("bc,za,zbc->zabc", E_p, E_dd_za, B_zzbc)

    term3a = (ratio * Cw * Lambda) * torch.einsum("bc,za,zbc->zabc", E_p, E_dd_za, P_zbcz)
    term3b = (ratio * Cw * Lambda) * torch.einsum("bc,za,zbca->zabc", E_p, E_p, P)
    term3 = term3a + term3b

    term4 = cw2 * torch.einsum("zabc,za,zb,zc->zabc", E_3pppp_zabc, H, H, H)

    term5 = (ratio * cw2) * torch.einsum("bc,za,zbc,za->zabc", E_p, E_3p_za, B_zzbc, H)

    term6a = (ratio * cw2) * torch.einsum("bc,za,zbc,za->zabc", E_p, E_3p_za, P_zbcz, H)
    term6b = (ratio * cw2) * torch.einsum("bc,za,zbca,za->zabc", E_p, E_dddd_za, P, H)
    term6 = term6a + term6b

    term7 = (ratio * cw2) * torch.einsum("bc,za,zabc->zabc", E_p, E_p, R_l)

    return term1 + term2 + term3 + term4 + term5 + term6 + term7


import torch

def T_update_acdb(
    T_l: torch.Tensor,                 # T^{(l)}_{a c d b}   (i.e. T_{δ1 δ3 δ4 δ2})
    H: torch.Tensor,                   # H^{(l)}_{i j}
    F_l: torch.Tensor,                 # F^{(l)}_{g a h b}   (i.e. F_{δ7 δ1 δ8 δ2})
    Q_l: torch.Tensor,                 # Q^{(l)}_{i j k l}
    Ginv: torch.Tensor,                # (G^{(l)})^{-1}_{i j}

    E_ppss_abcd: torch.Tensor,         # <σ'_a σ'_b σ_c σ_d>_{G^{(l)}}               (a,b,c,d)
    E_z_ps_eac: torch.Tensor,          # <z_e σ'_a σ_c>_{G^{(l)}}                    (e,a,c)
    E_z_ps_fbd: torch.Tensor,          # <z_f σ'_b σ_d>_{G^{(l)}}                    (f,b,d)

    E_pddsp_abcd: torch.Tensor,        # <σ'_a σ''_b σ_c σ'_d>_{G^{(l)}}             (a,b,c,d)
    E_z_ddp_fbd: torch.Tensor,         # <z_f σ''_b σ'_d>_{G^{(l)}}                  (f,b,d)

    E_p: torch.Tensor,                 # <σ'_i σ'_j>_{G^{(l)}}                        (i,j)
    E_dds: torch.Tensor,               # <σ''_i σ_j>_{G^{(l)}}                         (i,j)

    E_ddpsp_badc: torch.Tensor,        # <σ''_b σ'_a σ_d σ'_c>_{G^{(l)}}             (b,a,d,c)
    E_z_ds_ebd: torch.Tensor,          # <z_e σ''_b σ_d>_{G^{(l)}}                    (e,b,d)
    E_z_pp_fac: torch.Tensor,          # <z_f σ'_a σ'_c>_{G^{(l)}}                    (f,a,c)

    E_ddddpp_abcd: torch.Tensor,       # <σ''_a σ''_b σ'_c σ'_d>_{G^{(l)}}           (a,b,c,d)
    E_z_ddp_eac: torch.Tensor,         # <z_e σ''_a σ'_c>_{G^{(l)}}                  (e,a,c)
    E_z_ddp_fbd2: torch.Tensor,        # <z_f σ''_b σ'_d>_{G^{(l)}}                  (f,b,d)

    E_3p_p: torch.Tensor,              # <σ'''_i σ'_j>_{G^{(l)}}                       (i,j)
    E_dd_dd: torch.Tensor,             # <σ''_i σ''_j>_{G^{(l)}}                       (i,j)

    Cw_next: float,                    # C_W^{(l+1)}
    Lambda_next: float,                # λ_W^{(l+1)}
    n_l: int,
    n_lm1: int,
) -> torch.Tensor:
    """
    Computes the layer recursion shown in the figure for:
        T^{(l+1)}_{a c d b}  (i.e. T^{(l+1)}_{δ1 δ3 δ4 δ2})

    All δ-indices range over the same index set of size D.

    Term structure (matching the figure, grouped):
      1)  (Lambda)^2 * <σ'_a σ'_b σ_c σ_d> * H_{a b}
      2)  (n_l/n_{l-1})*(Lambda)^2 * Σ_{e,f,g,h} <z_e σ'_a σ_c><z_f σ'_b σ_d> Ginv_{e g} Ginv_{f h} F^{(l)}_{g a h b}

      3)  Cw*Lambda * <σ'_a σ''_b σ_c σ'_d> * H_{b a} * H_{b d}
      4)  (n_l/n_{l-1})*Cw*Lambda * H_{b d} * Σ_{e,f,g,h} <z_e σ'_a σ_c><z_f σ''_b σ'_d> Ginv_{e g} Ginv_{f h} F^{(l)}_{g a h b}
      5)  (n_l/n_{l-1})*Cw*Lambda * <σ'_b σ'_d> * ( <σ''_a σ_c> Q_{b d a a} + <σ'_a σ'_c> Q_{b d a c} )

      6)  Cw*Lambda * <σ''_b σ'_a σ_d σ'_c> * H_{a b} * H_{a c}
      7)  (n_l/n_{l-1})*Cw*Lambda * H_{a c} * Σ_{e,f,g,h} <z_e σ''_b σ_d><z_f σ'_a σ'_c> Ginv_{e g} Ginv_{f h} F^{(l)}_{g b h a}
      8)  (n_l/n_{l-1})*Cw*Lambda * <σ'_a σ'_c> * ( <σ''_b σ_d> Q_{a c b b} + <σ'_b σ'_d> Q_{a c b d} )

      9)  (n_l/n_{l-1})*(Cw)^2 * <σ'_a σ'_c><σ'_b σ'_d> * T^{(l)}_{a c d b}

     10)  (Cw)^2 * <σ''_a σ''_b σ'_c σ'_d> * H_{a b} H_{a c} H_{b d}
     11)  (n_l/n_{l-1})*(Cw)^2 * H_{a c} H_{b d} * Σ_{e,f,g,h} <z_e σ''_a σ'_c><z_f σ''_b σ'_d> Ginv_{e g} Ginv_{f h} F^{(l)}_{g a h b}

     12)  (n_l/n_{l-1})*(Cw)^2 * H_{b d} <σ'_a σ'_c> * ( <σ'''_b σ'_d> Q_{a c b b} + <σ''_b σ''_d> Q_{a c b d} )
     13)  (n_l/n_{l-1})*(Cw)^2 * H_{a c} <σ'_b σ'_d> * ( <σ'''_a σ'_c> Q_{b d a a} + <σ''_a σ''_c> Q_{b d a c} )
    """
    ratio = n_l / n_lm1
    cw2 = Cw_next * Cw_next
    lam2 = Lambda_next * Lambda_next
    cwlam = Cw_next * Lambda_next

    # diagonals needed: Q_{b d a a} and Q_{a c b b}
    Q_bd_aa = torch.einsum("bdaa->bda", Q_l)  # (b,d,a)
    Q_ac_bb = torch.einsum("acbb->acb", Q_l)  # (a,c,b)

    # F_{g b h a} version
    F_gbha = F_l.permute(0, 3, 2, 1)  # (g,b,h,a)

    # 1) (Lambda)^2 * <σ'_a σ'_b σ_c σ_d> * H_{a b}
    term1 = lam2 * torch.einsum("abcd,ab->acdb", E_ppss_abcd, H)

    # 2) ratio*(Lambda)^2 * Σ <z_e σ'_a σ_c><z_f σ'_b σ_d> Ginv_{e g} Ginv_{f h} F_{g a h b}
    term2 = (ratio * lam2) * torch.einsum(
        "eac,fbd,eg,fh,gahb->acdb",
        E_z_ps_eac, E_z_ps_fbd, Ginv, Ginv, F_l
    )

    # 3) Cw*Lambda * <σ'_a σ''_b σ_c σ'_d> * H_{b a} * H_{b d}
    term3 = cwlam * torch.einsum("abcd,ba,bd->acdb", E_pddsp_abcd, H, H)

    # 4) ratio*Cw*Lambda * H_{b d} * Σ <z_e σ'_a σ_c><z_f σ''_b σ'_d> ... F_{g a h b}
    term4 = (ratio * cwlam) * torch.einsum(
        "bd,eac,fbd,eg,fh,gahb->acdb",
        H, E_z_ps_eac, E_z_ddp_fbd, Ginv, Ginv, F_l
    )

    # 5) ratio*Cw*Lambda * <σ'_b σ'_d> * ( <σ''_a σ_c> Q_{b d a a} + <σ'_a σ'_c> Q_{b d a c} )
    term5a = torch.einsum("bd,ac,bda->acdb", E_p, E_dds, Q_bd_aa)
    term5b = torch.einsum("bd,ac,bdac->acdb", E_p, E_p, Q_l)
    term5 = (ratio * cwlam) * (term5a + term5b)

    # 6) Cw*Lambda * <σ''_b σ'_a σ_d σ'_c> * H_{a b} * H_{a c}
    term6 = cwlam * torch.einsum("badc,ab,ac->acdb", E_ddpsp_badc, H, H)

    # 7) ratio*Cw*Lambda * H_{a c} * Σ <z_e σ''_b σ_d><z_f σ'_a σ'_c> ... F_{g b h a}
    term7 = (ratio * cwlam) * torch.einsum(
        "ac,ebd,fac,eg,fh,gbha->acdb",
        H, E_z_ds_ebd, E_z_pp_fac, Ginv, Ginv, F_gbha
    )

    # 8) ratio*Cw*Lambda * <σ'_a σ'_c> * ( <σ''_b σ_d> Q_{a c b b} + <σ'_b σ'_d> Q_{a c b d} )
    term8a = torch.einsum("ac,bd,acb->acdb", E_p, E_dds, Q_ac_bb)
    term8b = torch.einsum("ac,bd,acbd->acdb", E_p, E_p, Q_l)
    term8 = (ratio * cwlam) * (term8a + term8b)

    # 9) ratio*(Cw)^2 * <σ'_a σ'_c><σ'_b σ'_d> * T^{(l)}_{a c d b}
    term9 = (ratio * cw2) * torch.einsum("ac,bd,acdb->acdb", E_p, E_p, T_l)

    # 10) (Cw)^2 * <σ''_a σ''_b σ'_c σ'_d> * H_{a b} H_{a c} H_{b d}
    term10 = cw2 * torch.einsum("abcd,ab,ac,bd->acdb", E_ddddpp_abcd, H, H, H)

    # 11) ratio*(Cw)^2 * H_{a c} H_{b d} * Σ <z_e σ''_a σ'_c><z_f σ''_b σ'_d> ... F_{g a h b}
    term11 = (ratio * cw2) * torch.einsum(
        "ac,bd,eac,fbd,eg,fh,gahb->acdb",
        H, H, E_z_ddp_eac, E_z_ddp_fbd2, Ginv, Ginv, F_l
    )

    # 12) ratio*(Cw)^2 * H_{b d} <σ'_a σ'_c> * ( <σ'''_b σ'_d> Q_{a c b b} + <σ''_b σ''_d> Q_{a c b d} )
    term12a = torch.einsum("bd,ac,bd,acb->acdb", H, E_p, E_3p_p, Q_ac_bb)
    term12b = torch.einsum("bd,ac,bd,acbd->acdb", H, E_p, E_dd_dd, Q_l)
    term12 = (ratio * cw2) * (term12a + term12b)

    # 13) ratio*(Cw)^2 * H_{a c} <σ'_b σ'_d> * ( <σ'''_a σ'_c> Q_{b d a a} + <σ''_a σ''_c> Q_{b d a c} )
    term13a = torch.einsum("ac,bd,ac,bda->acdb", H, E_p, E_3p_p, Q_bd_aa)
    term13b = torch.einsum("ac,bd,ac,bdac->acdb", H, E_p, E_dd_dd, Q_l)
    term13 = (ratio * cw2) * (term13a + term13b)

    return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10 + term11 + term12 + term13

def P_update_zabc(
    P_l: torch.Tensor,            # P^{(l)}_{z a b c}
    H: torch.Tensor,              # H_{i j}
    B: torch.Tensor,              # B_{i j a b}  (we will take i=j=z inside)
    E_ddp_p_s_zabc: torch.Tensor, # <σ''_z σ'_a σ'_b σ_c>          (z,a,b,c)
    E_p: torch.Tensor,            # <σ'_i σ'_j>                    (i,j) used as (a,b) and (z,c)
    E_dd_zc: torch.Tensor,        # <σ''_z σ_c>                    (z,c)
    Cw: float,                    # C_W^{(l+1)} (constant float)
    n_l: int,
    n_lm1: int,
) -> torch.Tensor:
    """
    Term structure:
      1)  (Cw)^2 * <σ''_z σ'_a σ'_b σ_c> * H_{z a} * H_{z b}
      2)  (n_l/n_{l-1})*(Cw)^2 * <σ'_a σ'_b> * [
            <σ''_z σ_c> * P^{(l)}_{z a b z}
          + <σ'_z σ'_c> * P^{(l)}_{z a b c}
          + <σ''_z σ_c> * B_{z z a b}
          ]
    """
    ratio = n_l / n_lm1
    cw2 = Cw * Cw

    P_zabz = torch.einsum("zabz->zab", P_l)   # P^{(l)}_{z a b z}
    B_zzab = torch.einsum("zzab->zab", B)     # B_{z z a b}

    term1 = cw2 * torch.einsum("zabc,za,zb->zabc", E_ddp_p_s_zabc, H, H)

    inside = (
        torch.einsum("zc,zab->zabc", E_dd_zc, P_zabz)  # <σ''_z σ_c> * P_{zabz}
        + torch.einsum("zc,zabc->zabc", E_p, P_l)      # <σ'_z σ'_c> * P_{zabc}
        + torch.einsum("zc,zab->zabc", E_dd_zc, B_zzab) # <σ''_z σ_c> * B_{zzab}
    )

    term2 = (ratio * cw2) * torch.einsum("ab,zabc->zabc", E_p, inside)

    return term1 + term2

def Q_update_zabc(
    Q_l: torch.Tensor,
    H: torch.Tensor,
    F_lp1: torch.Tensor,
    F_l: torch.Tensor,
    Ginv: torch.Tensor,
    E_ddp_p_s_zabc: torch.Tensor,   # <σ''_z σ'_a σ'_b σ_c>
    E_p: torch.Tensor,              # <σ'_i σ'_j>
    E_dd_bc: torch.Tensor,          # <σ''_b σ_c>
    E_z_ddp_dza: torch.Tensor,      # <z_d σ''_z σ'_a>
    E_z_p_s_ebc: torch.Tensor,      # <z_e σ'_b σ_c>
    Cw: float,                      #this is C_W^{(l+1)}
    LambdaW: float,
    n_l: int,
    n_lm1: int,
) -> torch.Tensor:
    """
    Term structure:
      1)  (Cw)^2 * <σ''_z σ'_a σ'_b σ_c> * H_{z a} * H_{z b}
      2)  (LambdaW/Cw) * F^{(l+1)}_{a z c b}
      3)  (n_l/n_{l-1})*(Cw)^2 * <σ'_z σ'_a> * [ <σ''_b σ_c> * Q^{(l)}_{z a b b} + <σ'_b σ'_c> * Q^{(l)}_{z a b c} ]
      4)  (n_l/n_{l-1})*(Cw)^2 * H_{z a} * Σ_{d,e,f,g} [ <z_d σ''_z σ'_a> <z_e σ'_b σ_c> G^{-1}_{d f} G^{-1}_{e g} F^{(l)}_{f z g b} ]
    """
    ratio = n_l / n_lm1
    cw2 = Cw * Cw

    term1 = cw2 * torch.einsum("zabc,za,zb->zabc", E_ddp_p_s_zabc, H, H)

    term2 = (LambdaW / Cw) * torch.einsum("azcb->zabc", F_lp1)

    Q_zabb = torch.einsum("zabb->zab", Q_l)

    bracket = (
        torch.einsum("bc,zab->zabc", E_dd_bc, Q_zabb)
        + torch.einsum("bc,zabc->zabc", E_p, Q_l)   #E_p= <σ'_b σ'_c>
    )

    term3 = (ratio * cw2) * torch.einsum("za,zabc->zabc", E_p, bracket)  # E_p=<σ'_z σ'_a>

    term4 = (ratio * cw2) * torch.einsum("za,dza,ebc,df,eg,fzgb->zabc", H, E_z_ddp_dza, E_z_p_s_ebc, Ginv, Ginv, F_l)

    return term1 + term2 + term3 + term4

def F_update_abcd(
    F_l: torch.Tensor,
    H: torch.Tensor,
    Ginv: torch.Tensor,
    E_sspp_abcd: torch.Tensor,      # <σ_a σ_b σ'_c σ'_d>
    E_sps_ace: torch.Tensor,        # <σ_a σ'_c z_e>
    E_sps_bdf: torch.Tensor,        # <σ_b σ'_d z_f>
    Cw: float,
    n_l: int,
    n_lm1: int,
) -> torch.Tensor:
    """
    Term structure:
      1)  (Cw)^2 * <σ_a σ_b σ'_c σ'_d> * H_{c d}
      2)  (n_l/n_{l-1})*(Cw)^2 * Σ_{e,f,g,h} <σ_a σ'_c z_e><σ_b σ'_d z_f> G^{-1}_{e g} G^{-1}_{f h} F^{(l)}_{g c h d}
    """
    ratio = n_l / n_lm1
    cw2 = Cw * Cw

    term1 = cw2 * torch.einsum("abcd,cd->abcd", E_sspp_abcd, H)

    term2 = (ratio * cw2) * torch.einsum("ace,bdf,eg,fh,gchd->abcd", E_sps_ace, E_sps_bdf, Ginv, Ginv, F_l)

    return term1 + term2

def B_update_acbd(
    B_l: torch.Tensor,          # B^{(l)}_{a c b d}
    H_l: torch.Tensor,          # H^{(l)}_{i j}
    E_pppp_abcd: torch.Tensor,  # <σ'_a σ'_b σ'_c σ'_d>_{G^{(l)}}   (a,b,c,d)
    E_p_ac: torch.Tensor,       # <σ'_a σ'_c>_{G^{(l)}}             (a,c)
    Cw_next: float,             # C_W^{(l+1)} (constant float)
    n_l: int,
    n_lm1: int,
    E_p_bd: Optional[torch.Tensor] = None,  # <σ'_b σ'_d>_{G^{(l)}}   (b,d); default uses E_p_ac
) -> torch.Tensor:
    """
    B^{(l+1)}_{a c b d}
      = (Cw_next)^2 * [ <σ'_a σ'_b σ'_c σ'_d> * H_{a b} * H_{c d}
                        + (n_l/n_{l-1}) * <σ'_a σ'_c> * <σ'_b σ'_d> * B^{(l)}_{a c b d} ]
    """
    if E_p_bd is None:
        E_p_bd = E_p_ac

    cw2 = Cw_next * Cw_next
    ratio = n_l / n_lm1

    term1 = torch.einsum("abcd,ab,cd->acbd", E_pppp_abcd, H_l, H_l)
    term2 = ratio * torch.einsum("ac,bd,acbd->acbd", E_p_ac, E_p_bd, B_l)

    return cw2 * (term1 + term2)

def D_update_abcd(
    D_l: torch.Tensor,                 # D^{(l)}_{a b c d}
    V_raised: torch.Tensor,            # V^{(l)}_{p q r s}
    H: torch.Tensor,                   # H^{(l)}_{i j}
    G: torch.Tensor,                   # G^{(l)}_{i j}
    Ginv: torch.Tensor,                # (G^{(l)})^{-1}_{i j}

    # base expectations
    E_ssss_abcd: torch.Tensor,         # <σ_a σ_b σ_c σ_d>
    E_sspp_abcd: torch.Tensor,         # <σ_a σ_b σ'_c σ'_d>
    E_ss: torch.Tensor,                # <σ_i σ_j>
    E_pp: torch.Tensor,                # <σ'_i σ'_j>

    # zz expectations
    E_zz_ss_pqab: torch.Tensor,        # <z_p z_q σ_a σ_b>
    E_zz_ss_pqcd: torch.Tensor,        # <z_p z_q σ_c σ_d>
    E_zz_pp_pqcd: torch.Tensor,        # <z_p z_q σ'_c σ'_d>

    Cw_next: float,                    # C_W^{(l+1)}
    Lambda_next: float,                # λ_W^{(l+1)}
    n_l: int,
    n_lm1: int,
) -> torch.Tensor:
    """

    Term structure:
      1)  Cw * ( <σ_a σ_b \Omega_{cd}> - <σ_a σ_b><\Omega_{cd}> )
      2)  (n_l/(4n_{l-1})) * Cw * Σ_{pqrs} V^{pqrs} <(z_p z_q - G_pq) σ_a σ_b> <(z_r z_s - G_rs) \Omega_{cd}>
      3)  (n_l/(2n_{l-1})) * (Cw)^2 * \sum_{pqrs} D^{(l)}_{r s c d} <(z_p z_q - G_pq) σ_a σ_b> Ginv^{p r} Ginv^{q s} <σ'_c σ'_d>
    """
    ratio4 = n_l / (4.0 * n_lm1)
    ratio2 = n_l / (2.0 * n_lm1)
    cw = Cw_next
    cw2 = cw * cw
    lam = Lambda_next

    # \Omega_{cd} mean and mixed moments
    Omega_mean_cd = lam * E_ss + cw * (H * E_pp)                         # (c,d)
    SigmaSigma_Omega = lam * E_ssss_abcd + cw * torch.einsum("abcd,cd->abcd", E_sspp_abcd, H)

    term1 = cw * (SigmaSigma_Omega - torch.einsum("ab,cd->abcd", E_ss, Omega_mean_cd))

    # centered blocks
    X_pqab = E_zz_ss_pqab - torch.einsum("pq,ab->pqab", G, E_ss)

    Zss_pqcd = E_zz_ss_pqcd - torch.einsum("pq,cd->pqcd", G, E_ss)
    Zpp_pqcd = E_zz_pp_pqcd - torch.einsum("pq,cd->pqcd", G, E_pp)
    U_pqcd = lam * Zss_pqcd + cw * torch.einsum("pqcd,cd->pqcd", Zpp_pqcd, H)   # <(zz-G) \Omega_cd>

    term2 = (ratio4 * cw) * torch.einsum("pqrs,pqab,rscd->abcd", V_raised, X_pqab, U_pqcd)

    term3 = (ratio2 * cw2) * torch.einsum(
        "pqab,pr,qs,rscd,cd->abcd",
        X_pqab, Ginv, Ginv, D_l, E_pp
    )

    return term1 + term2 + term3


def A_update_abcd(
    A_l: torch.Tensor,                 # A^{(l)}_{a b c d}
    D_l: torch.Tensor,                 # D^{(l)}_{a b c d}
    V_raised: torch.Tensor,            # V^{(l)}_{p q r s}  (raised indices)
    H: torch.Tensor,                   # H^{(l)}_{i j}
    G: torch.Tensor,                   # G^{(l)}_{i j}
    Ginv: torch.Tensor,                # (G^{(l)})^{-1}_{i j}

    # base expectations
    E_ssss_abcd: torch.Tensor,         # <σ_a σ_b σ_c σ_d>
    E_sspp_abcd: torch.Tensor,         # <σ_a σ_b σ'_c σ'_d>
    E_ppss_abcd: torch.Tensor,         # <σ'_a σ'_b σ_c σ_d>
    E_pppp_abcd: torch.Tensor,         # <σ'_a σ'_b σ'_c σ'_d>
    E_ss: torch.Tensor,                # <σ_i σ_j>
    E_pp: torch.Tensor,                # <σ'_i σ'_j>

    # zz expectations
    E_zz_ss_pqab: torch.Tensor,        # <z_p z_q σ_a σ_b>
    E_zz_pp_pqab: torch.Tensor,        # <z_p z_q σ'_a σ'_b>
    E_zz_ss_pqcd: torch.Tensor,        # <z_p z_q σ_c σ_d>
    E_zz_pp_pqcd: torch.Tensor,        # <z_p z_q σ'_c σ'_d>

    Cw_next: float,                    # C_W^{(l+1)}
    Lambda_next: float,                # λ_W^{(l+1)}
    n_l: int,
    n_lm1: int,
) -> torch.Tensor:
    """
    Implements (8.97) with Ω̂ expanded using (8.74) and with H, G pulled out.

    Term structure:
      1)  <\Omega_ab \Omega_cd> - <\Omega_ab><\Omega_cd>
      2)  (n_l/(4n_{l-1})) * \sum_{pqrs} V^{pqrs} <\Omega_ab(zz-G)_pq> <\Omega_cd(zz-G)_rs>
      3)  (n_l/n_{l-1})*(Cw)^2 <σ'_aσ'_b><σ'_cσ'_d> A^{(l)}_{abcd}
      4)  (n_l/n_{l-1})*(Cw/2) * \sum_{pqrs} [ <\Omega_ab(zz-G)_pq> Ginv^{pr}Ginv^{qs} D^{(l)}_{r s c d} <σ'_cσ'_d>
                                           + <\Omega_cd(zz-G)_pq> Ginv^{pr}Ginv^{qs} D^{(l)}_{r s a b} <σ'_aσ'_b> ]
    """
    ratio4 = n_l / (4.0 * n_lm1)
    ratio  = n_l / n_lm1
    cw = Cw_next
    cw2 = cw * cw
    lam = Lambda_next

    # means
    Omega_mean = lam * E_ss + cw * (H * E_pp)  # (i,j)

    # <\Omega_ab \Omega_cd>
    OmegaOmega = (
        (lam * lam) * E_ssss_abcd
        + (lam * cw) * torch.einsum("abcd,cd->abcd", E_sspp_abcd, H)
        + (lam * cw) * torch.einsum("abcd,ab->abcd", E_ppss_abcd, H)
        + (cw2) * torch.einsum("abcd,ab,cd->abcd", E_pppp_abcd, H, H)
    )

    term1 = OmegaOmega - torch.einsum("ab,cd->abcd", Omega_mean, Omega_mean)

    Zss_pqab = E_zz_ss_pqab - torch.einsum("pq,ab->pqab", G, E_ss)
    Zpp_pqab = E_zz_pp_pqab - torch.einsum("pq,ab->pqab", G, E_pp)
    U_pqab = lam * Zss_pqab + cw * torch.einsum("pqab,ab->pqab", Zpp_pqab, H)  # <\Omega_ab(zz-G)>

    Zss_pqcd = E_zz_ss_pqcd - torch.einsum("pq,cd->pqcd", G, E_ss)
    Zpp_pqcd = E_zz_pp_pqcd - torch.einsum("pq,cd->pqcd", G, E_pp)
    U_pqcd = lam * Zss_pqcd + cw * torch.einsum("pqcd,cd->pqcd", Zpp_pqcd, H)  # <\Omega_cd(zz-G)>

    term2 = ratio4 * torch.einsum("pqrs,pqab,rscd->abcd", V_raised, U_pqab, U_pqcd)

    term3 = (ratio * cw2) * torch.einsum("ab,cd,abcd->abcd", E_pp, E_pp, A_l)

    # mixed D terms
    W_pqcd = torch.einsum("pr,qs,rscd->pqcd", Ginv, Ginv, D_l)

    D_rsab = D_l.permute(2, 3, 0, 1)  # (r,s,a,b)
    W_pqab = torch.einsum("pr,qs,rsab->pqab", Ginv, Ginv, D_rsab)

    sub1 = torch.einsum("pqab,pqcd,cd->abcd", U_pqab, W_pqcd, E_pp)
    sub2 = torch.einsum("pqcd,pqab,ab->abcd", U_pqcd, W_pqab, E_pp)

    term4 = (ratio * (cw / 2.0)) * (sub1 + sub2)

    return term1 + term2 + term3 + term4

def recurse_layer(
    theta_prev: torch.Tensor,
    K_prev: torch.Tensor,
    V_prev: torch.Tensor,   # needs raised indexes
    S_prev: torch.Tensor,
    R_prev: torch.Tensor,
    T_prev: torch.Tensor,
    U_prev: torch.Tensor,
    A_prev: torch.Tensor,
    B_prev: torch.Tensor,
    P_prev: torch.Tensor,
    Q_prev: torch.Tensor,
    D_prev: torch.Tensor,
    F_prev: torch.Tensor,
    C_W: float,
    lambda_W: float,
    n_prev: int,
    n_preprev: int,
):
    # sigma= sin
    # sigma'= cos
    # sigma''= -sin
    # sigma''' = -cos

    # expectations

    # some two indicie expectations
    exp_sin_sin = s2_expval(K_prev)   # E[sin sin]
    exp_cos_cos = c2_expval(K_prev)   # E[cos cos]

    # some 4 incicie expectations
    exp_sin_sin_cos_cos = s2c2_expval(K_prev)  # E[sin sin cos cos] with indices (sin,sin,cos,cos)
    exp_sin_sin_sin_sin = sk_expval(K_prev, 4) # E[sin sin sin sin]
    exp_cos_cos_cos_cos = cr_expval(K_prev, 4) # E[cos cos cos cos]

    # z expectations
    exp_sin_cos_z = s1c1z1_expval(K_prev)  # E[sin cos z] with indices (sin,cos,z)

    # z z expectations
    exp_sin_sin_z_z = s2z2_expval(K_prev)  # E[sin sin z z] with indices (sin,sin,z,z)
    exp_cos_cos_z_z = c2z2_expval(K_prev)  # E[cos cos z z] with indices (cos,cos,z,z)

    # Inverse of K
    K_inv = torch.linalg.inv(K_prev)

    # theta/H and K/G
    theta_new, K_new = compute_new_theta_K(
        theta_prev=theta_prev,
        C_W=C_W,
        C_b=0.0,
        lambda_b=0.0,
        lambda_W=lambda_W,
        K=K_prev,
    )

    # V (raised-indicies)
    # s2z2_expval returns E[sin_a sin_b z_p z_q] with axes (a,b,p,q)
    # must convert to E[z_p z_q sin_a sin_b] with axes (p,q,a,b)
    zz_sin_sin = exp_sin_sin_z_z.permute(2, 3, 0, 1)

    V_new = V_update_raised(
        V_prev=V_prev,
        G=K_prev,
        E_sigma4=exp_sin_sin_sin_sin,
        E_sigma2=exp_sin_sin,
        E_zz_sigma2=zz_sin_sin,
        Cw_next=C_W,
        n_l=n_prev,
        n_lm1=n_preprev,
    )

    # B tensor
    B_new = B_update_acbd(
        B_l=B_prev,
        H_l=theta_prev,
        E_pppp_abcd=exp_cos_cos_cos_cos,  # cos cos cos cos
        E_p_ac=exp_cos_cos,               # cos cos
        Cw_next=C_W,
        n_l=n_prev,
        n_lm1=n_preprev,
        E_p_bd=None,
    )

    # F tensor
    # E[sigma_a sigma_b sigma'_c sigma'_d] = E[sin sin cos cos]
    # E[sigma_a sigma'_c z_e] = E[sin cos z]
    F_new = F_update_abcd(
        F_l=F_prev,
        H=theta_prev,
        Ginv=K_inv,
        E_sspp_abcd=exp_sin_sin_cos_cos,
        E_sps_ace=exp_sin_cos_z,
        E_sps_bdf=exp_sin_cos_z,
        Cw=C_W,
        n_l=n_prev,
        n_lm1=n_preprev,
    )


    # P and Q
    # For sigma = sin:
    # <sigma''_z sigma'_a sigma'_b sigma_c> = <(-sin_z) cos_a cos_b sin_c>
    exp_ddp_p_s_zabc = -exp_sin_sin_cos_cos.permute(0, 2, 3, 1)  # (z,a,b,c)

    #P tensor
    # <sigma''_z sigma_c> = <(-sin_z) sin_c>
    exp_dd_zc = -exp_sin_sin

    P_new = P_update_zabc(
        P_l=P_prev,
        H=theta_prev,
        B=B_prev,
        E_ddp_p_s_zabc=exp_ddp_p_s_zabc,
        E_p=exp_cos_cos,
        E_dd_zc=exp_dd_zc,
        Cw=C_W,
        n_l=n_prev,
        n_lm1=n_preprev,
    )

    # Q tensor
    # <sigma''_b sigma_c> = <(-sin_b) sin_c>
    exp_dd_bc = -exp_sin_sin

    # <z_d sigma''_z sigma'_a> = <z_d (-sin_z) cos_a>
    exp_z_ddp_dza = -exp_sin_cos_z.permute(2, 0, 1)  # (d,z,a)

    # <z_e sigma'_b sigma_c> = <z_e cos_b sin_c>
    exp_z_p_s_ebc = exp_sin_cos_z.permute(2, 1, 0)   # (e,b,c)

    Q_new = Q_update_zabc(
        Q_l=Q_prev,
        H=theta_prev,
        F_lp1=F_new,
        F_l=F_prev,
        Ginv=K_inv,
        E_ddp_p_s_zabc=exp_ddp_p_s_zabc,
        E_p=exp_cos_cos,
        E_dd_bc=exp_dd_bc,
        E_z_ddp_dza=exp_z_ddp_dza,
        E_z_p_s_ebc=exp_z_p_s_ebc,
        Cw=C_W,
        LambdaW=lambda_W,
        n_l=n_prev,
        n_lm1=n_preprev,
    )

    # U tensor
    # <sigma'' sigma'' sigma' sigma'> = <(-sin)(-sin) cos cos> = <sin sin cos cos>
    U_new = U_update_abcd(
        U_l=U_prev,
        H_l=theta_prev,
        E_ddpp_abcd=exp_sin_sin_cos_cos,
        E_p_ad=exp_cos_cos,
        Cw_next=C_W,
        n_l=n_prev,
        n_lm1=n_preprev,
        E_p_bc=None,
    )

    # S tensor
    # <sigma'_a sigma'_b sigma'_c sigma'_d> = cos cos cos cos
    # <sigma''_a sigma''_b> = (-sin)(-sin) = sin sin
    S_new = S_update_abcd(
        S_l=S_prev,
        B=B_prev,
        H=theta_prev,
        E_pppp_abcd=exp_cos_cos_cos_cos,
        E_p_ab=exp_cos_cos,
        E_dd_ab=exp_sin_sin,
        E_ddpp_abcd=exp_sin_sin_cos_cos,
        Cw_next=C_W,
        Lambda_next=lambda_W,
        n_l=n_prev,
        n_lm1=n_preprev,
        E_p_cd=None,
    )

    #R tensor
    # <sigma''_z sigma_a sigma'_b sigma'_c> = <(-sin_z) sin_a cos_b cos_c>
    exp_ddsp_zabc = -exp_sin_sin_cos_cos
    exp_dd_za = -exp_sin_sin
    exp_3pppp_zabc = -exp_cos_cos_cos_cos
    exp_3p_za = -exp_cos_cos
    exp_dddd_za = exp_sin_sin

    R_new = R_update_zabc(
        R_l=R_prev,
        H=theta_prev,
        B=B_prev,
        P=P_prev,
        E_ddsp_zabc=exp_ddsp_zabc,
        E_p=exp_cos_cos,
        E_dd_za=exp_dd_za,
        E_3pppp_zabc=exp_3pppp_zabc,
        E_3p_za=exp_3p_za,
        E_dddd_za=exp_dddd_za,
        Cw=C_W,
        Lambda=lambda_W,
        n_l=n_prev,
        n_lm1=n_preprev,
    )

    # T tensor
    Dsize = K_prev.shape[0]
    zeros_3 = torch.zeros((Dsize, Dsize, Dsize), dtype=K_prev.dtype, device=K_prev.device)

    # <sigma'_a sigma'_b sigma_c sigma_d> = <cos cos sin sin>
    exp_ppss_abcd = exp_sin_sin_cos_cos.permute(2, 3, 0, 1)

    # <z_e sigma'_a sigma_c> = <z_e cos_a sin_c>
    exp_z_ps_eac = exp_sin_cos_z.permute(2, 1, 0)

    # <sigma'_a sigma''_b sigma_c sigma'_d> = <cos_a (-sin_b) sin_c cos_d>
    exp_pddsp_abcd = -exp_sin_sin_cos_cos.permute(2, 0, 1, 3)

    # <z_f sigma''_b sigma'_d> = <z_f (-sin_b) cos_d>
    exp_z_ddp_fbd = -exp_sin_cos_z.permute(2, 0, 1)

    # <sigma''_i sigma_j> = <(-sin_i) sin_j>
    exp_dds = -exp_sin_sin

    # <sigma''_b sigma'_a sigma_d sigma'_c> = <(-sin_b) cos_a sin_d cos_c>
    exp_ddpsp_badc = -exp_sin_sin_cos_cos.permute(0, 2, 1, 3)

    # These are odd so zero
    exp_z_ds_ebd = zeros_3
    exp_z_pp_fac = zeros_3

    exp_ddddpp_abcd = exp_sin_sin_cos_cos
    exp_z_ddp_eac = -exp_sin_cos_z.permute(2, 0, 1)
    exp_3p_p = -exp_cos_cos
    exp_dd_dd = exp_sin_sin

    T_new = T_update_acdb(
        T_l=T_prev,
        H=theta_prev,
        F_l=F_prev,
        Q_l=Q_prev,
        Ginv=K_inv,
        E_ppss_abcd=exp_ppss_abcd,
        E_z_ps_eac=exp_z_ps_eac,
        E_z_ps_fbd=exp_z_ps_eac,
        E_pddsp_abcd=exp_pddsp_abcd,
        E_z_ddp_fbd=exp_z_ddp_fbd,
        E_p=exp_cos_cos,
        E_dds=exp_dds,
        E_ddpsp_badc=exp_ddpsp_badc,
        E_z_ds_ebd=exp_z_ds_ebd,
        E_z_pp_fac=exp_z_pp_fac,
        E_ddddpp_abcd=exp_ddddpp_abcd,
        E_z_ddp_eac=exp_z_ddp_eac,
        E_z_ddp_fbd2=exp_z_ddp_eac,
        E_3p_p=exp_3p_p,
        E_dd_dd=exp_dd_dd,
        Cw_next=C_W,
        Lambda_next=lambda_W,
        n_l=n_prev,
        n_lm1=n_preprev,
    )

    # D and A
    zz_cos_cos = exp_cos_cos_z_z.permute(2, 3, 0, 1)  # (p,q,a,b) for z z cos cos

    D_new = D_update_abcd(
        D_l=D_prev,
        V_raised=V_prev,
        H=theta_prev,
        G=K_prev,
        Ginv=K_inv,
        E_ssss_abcd=exp_sin_sin_sin_sin,
        E_sspp_abcd=exp_sin_sin_cos_cos,
        E_ss=exp_sin_sin,
        E_pp=exp_cos_cos,
        E_zz_ss_pqab=zz_sin_sin,
        E_zz_ss_pqcd=zz_sin_sin,
        E_zz_pp_pqcd=zz_cos_cos,
        Cw_next=C_W,
        Lambda_next=lambda_W,
        n_l=n_prev,
        n_lm1=n_preprev,
    )

    A_new = A_update_abcd(
        A_l=A_prev,
        D_l=D_prev,
        V_raised=V_prev,
        H=theta_prev,
        G=K_prev,
        Ginv=K_inv,
        E_ssss_abcd=exp_sin_sin_sin_sin,
        E_sspp_abcd=exp_sin_sin_cos_cos,
        E_ppss_abcd=exp_ppss_abcd,
        E_pppp_abcd=exp_cos_cos_cos_cos,
        E_ss=exp_sin_sin,
        E_pp=exp_cos_cos,
        E_zz_ss_pqab=zz_sin_sin,
        E_zz_pp_pqab=zz_cos_cos,
        E_zz_ss_pqcd=zz_sin_sin,
        E_zz_pp_pqcd=zz_cos_cos,
        Cw_next=C_W,
        Lambda_next=lambda_W,
        n_l=n_prev,
        n_lm1=n_preprev,
    )

    return [
        theta_new,
        K_new,
        V_new,
        S_new,
        R_new,
        T_new,
        U_new,
        A_new,
        B_new,
        P_new,
        Q_new,
        D_new,
        F_new,
    ]




#computing m tensors
def compute_all_m_tensors(
    n_out: int,                     # number of output neurons (i index range); equals n_L in the paper
    training_size: int,             # number of training indices (alpha range)
    theta_final: torch.Tensor,      # theta_final[beta, alpha] (delta_size x delta_size). Used for H and Htilde_lower.
    K_final: torch.Tensor,          # K_final (delta_size x delta_size). Used in ddNTK-I / ddNTK-II gamma factor.

    # last layer tensors
    A_final: torch.Tensor,          # A[delta, a1, a2, a3] (delta_size^4)
    B_final: torch.Tensor,          # B[delta, a1, a2, a3] (delta_size^4)
    P_final: torch.Tensor,          # P[i0,i1,i2,i3] (delta_size^4)
    Q_final: torch.Tensor,          # Q[i0,i1,i2,i3] (delta_size^4)
    R_final: torch.Tensor,          # R[i0,i1,i2,i3] (delta_size^4)
    S_final: torch.Tensor,          # S[i0,i1,i2,i3] (delta_size^4)
    T_final: torch.Tensor,          # T[i0,i1,i2,i3] (delta_size^4)
    U_final: torch.Tensor,          # U[i0,i1,i2,i3] (delta_size^4)

    # Z-tensors needed by the m-formulas (these are NOT inverses of any kind an the upper indicies are just notation)
    Z_A: torch.Tensor,              # Z_A[a1,a2,a3,a4] (delta_size^4)
    Z_B: torch.Tensor,              # Z_B[a1,a2,a3,a4] (delta_size^4)
    Z_IA: torch.Tensor,             # Z_IA[a1,a2,a3,a4,a5,a6] (delta_size^6)
    Z_IB: torch.Tensor,             # Z_IB[a1,a2,a3,a4,a5,a6] (delta_size^6)
    Z_IIA: torch.Tensor,            # Z_IIA[a1,a2,a3,a4,a5,a6] (delta_size^6)
    Z_IIB: torch.Tensor,            # Z_IIB[a1,a2,a3,a4,a5,a6] (delta_size^6)

    # labels (targets) for training examples only
    y_labels: torch.Tensor,         # y[i, alpha] with shape (n_out, training_size)
) -> Dict[str, torch.Tensor]:
    """
    Computes the five m-tensors at the final layer by calling:
      1) compute_m_NTK
      2) compute_m_delta_NTK
      3) compute_m_dNTK
      4) compute_m_ddNTK_I
      5) compute_m_ddNTK_II

    Conventions:
      - beta/delta indices range over the full delta_size (train+test). We do not slice them.
      - alpha indices range over the training set (0..training_size-1) in the sums.
      - theta_final plays the role of H (and its training block is Htilde_lower).
      - K_final plays the role of K in the ddNTK-I / ddNTK-II gamma factor.
    """
    ts = training_size
    n_L = float(n_out)

    # H_beta_alpha1 is theta_final[beta, alpha1]. We can pass the full theta_final
    # because compute_m_NTK slices its second axis to training_size
    H_beta_alpha1 = theta_final  # (delta_size, delta_size)

    Htilde_lower = theta_final

    m_ntk = compute_m_NTK(
        H_beta_alpha1=H_beta_alpha1,
        Htilde_lower=Htilde_lower,
        y_labels=y_labels,
        training_size=ts,
    )

    m_delta_ntk = compute_m_delta_NTK(
        n_L=n_L,
        A=A_final,
        B=B_final,
        Htilde_lower=Htilde_lower,
        y=y_labels,
        training_size=ts,
    )

    m_dntk = compute_m_dNTK(
        n_L=n_L,
        z_tensor_A=Z_A,
        z_tensor_B=Z_B,
        P_tensor=P_final,
        Q_tensor=Q_final,
        y_labels=y_labels,
        training_size=ts,
    )

    m_ddntk_i = compute_m_ddNTK_I(
        n_L=n_L,
        z_tensor_IA=Z_IA,
        z_tensor_IB=Z_IB,
        R_tensor=R_final,
        K_matrix=K_final,
        labels_y=y_labels,
        training_size=ts,
    )

    m_ddntk_ii = compute_m_ddNTK_II(
        n_L=n_L,
        z_tensor_IIB=Z_IIB,
        z_tensor_IIA=Z_IIA,
        S_tensor=S_final,
        T_tensor=T_final,
        U_tensor=U_final,
        K_matrix=K_final,
        labels_y=y_labels,
        training_size=ts,
    )

    return {
        "m_NTK": m_ntk,
        "m_delta_NTK": m_delta_ntk,
        "m_dNTK": m_dntk,
        "m_ddNTK_I": m_ddntk_i,
        "m_ddNTK_II": m_ddntk_ii,
    }

def compute_mean_output_neurons(
    m_NTK: torch.Tensor,          # m_NTK[i, beta]
    m_delta_NTK: torch.Tensor,    # m_delta_NTK[i, beta]
    m_dNTK: torch.Tensor,         # m_dNTK[i, beta]
    m_ddNTK_I: torch.Tensor,      # m_ddNTK_I[i, beta]
    m_ddNTK_II: torch.Tensor,     # m_ddNTK_II[i, beta]
    H_beta_alpha1: torch.Tensor,  # H[beta, alpha1], shape (delta_size, delta_size)
    Htilde_lower: torch.Tensor,   # Htilde_lower[alpha1, alpha2], shape (delta_size, delta_size) (we invert training block)
    n_Lm1: int,                   # n_{L-1} in the paper
    training_size: int,
) -> torch.Tensor:
    """
    m[i, beta] = E[ y_tilde^{(L)}_{i, beta}(T) ] to order 1/n_{L-1}:

      m[i,beta] =
          m_NTK[i,beta]
        + (1/n_Lm1) * ( m_delta_NTK[i,beta] + m_dNTK[i,beta] + m_ddNTK_I[i,beta] + m_ddNTK_II[i,beta] )
        - (1/n_Lm1) * sum_{alpha1, alpha2}
              H[beta, alpha1] * Htilde_upper[alpha1, alpha2]
              * ( m_delta_NTK[i,alpha2] + m_dNTK[i,alpha2] + m_ddNTK_I[i,alpha2] + m_ddNTK_II[i,alpha2] )

    where Htilde_upper is the inverse of the training block of Htilde_lower,
    and alpha1, alpha2 run only over {0, ..., training_size-1}, while beta runs over train+test.
    """
    ts = training_size
    inv_n = 1.0 / float(n_Lm1)

    # total 1/n correction term
    m_corr = m_delta_NTK + m_dNTK + m_ddNTK_I + m_ddNTK_II  # (i, beta)

    # building the Htilde_upper on training block
    Htilde_upper = torch.linalg.inv(Htilde_lower[:ts, :ts])     # (alpha1, alpha2)
    H_beta_train = H_beta_alpha1[:, :ts]                        # (beta, alpha1)
    m_corr_train = m_corr[:, :ts]                               # (i, alpha2)

    # projection term: sum_{alpha1,alpha2} H[beta,alpha1] Htilde_upper[alpha1,alpha2] m_corr[i,alpha2]
    proj = torch.einsum("ba,ac,ic->ib", H_beta_train, Htilde_upper, m_corr_train)  # (i, beta)

    # final mean prediction
    m_mean = m_NTK + inv_n * (m_corr - proj)
    return m_mean


# if __name__ == "__main__":
#     np.random.seed(44)
#     torch.manual_seed(44)
#     torch.set_num_threads(1)
#
#     # Ensemble code
#     class SinNet(nn.Module):
#         def __init__(self, input_dim, hidden_dim, num_hidden_layers):
#             super(SinNet, self).__init__()
#             layers = []
#
#             # first layer
#             layers.append(nn.Linear(input_dim, hidden_dim, bias=False))
#
#             # rest of hidden layers
#             for _ in range(num_hidden_layers - 1):
#                 layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
#
#             # output layer (scalar output)
#             layers.append(nn.Linear(hidden_dim, 1, bias=False))
#             self.layers = nn.ModuleList(layers)
#
#             # init as in your Test_three
#             with torch.no_grad():
#                 self.layers[0].weight.normal_(0.0, (1 / input_dim) ** 0.5)
#                 for layer in self.layers[1:-1]:
#                     layer.weight.normal_(0.0, (1 / hidden_dim) ** 0.5)
#                 self.layers[-1].weight.normal_(0.0, (1 / hidden_dim) ** 0.5)
#
#         def forward(self, x):
#             for layer in self.layers[:-1]:
#                 x = torch.sin(layer(x))
#             x = self.layers[-1](x)
#             return x
#
#     def train_sin_net_until_predictions_stop(
#         train_vecs,
#         labels,
#         hidden_dim,
#         num_hidden_layers,
#         lr,
#         max_steps,
#         test_vecs,
#         pred_tol=1e-6,
#         patience=200,
#         min_steps=200,
#         device="cpu",
#     ):
#         """
#         Stop condition:
#           max_abs(preds_t - preds_{t-1}) < pred_tol for `patience` consecutive steps,
#           after at least `min_steps` steps.
#
#         Returns:
#           model, final_preds (np.ndarray shape (#all_points,)), steps_used (int), last_delta (float)
#         """
#         # Torch expects N x d
#         if train_vecs.shape[0] < train_vecs.shape[1]:
#             train_vecs = train_vecs.T
#         N, d = train_vecs.shape
#
#         labels = labels.reshape(-1, 1)
#
#         x_train = torch.tensor(train_vecs, dtype=torch.float32, device=device)
#         y_train = torch.tensor(labels, dtype=torch.float32, device=device)
#
#         if test_vecs.shape[0] < test_vecs.shape[1]:
#             test_vecs = test_vecs.T
#         x_test = torch.tensor(test_vecs, dtype=torch.float32, device=device)
#
#         model = SinNet(input_dim=d, hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers).to(device)
#
#         criterion = nn.MSELoss()
#
#         # --- custom learning rates per layer = lr / fan_in ---
#         param_groups = []
#         for layer in model.layers:
#             divisor = layer.in_features
#             layer_lr = lr / divisor
#             param_groups.append({"params": layer.parameters(), "lr": layer_lr})
#         optimizer = torch.optim.SGD(param_groups)
#
#         prev_preds = None
#         stable_count = 0
#         last_delta = float("inf")
#
#         for step in range(max_steps):
#             optimizer.zero_grad()
#             outputs = model(x_train)
#             loss = criterion(outputs, y_train)
#             loss.backward()
#             optimizer.step()
#
#             # compute predictions on all points (train+test)
#             with torch.no_grad():
#                 preds = model(x_test).view(-1)  # (num_points,)
#
#             if prev_preds is not None:
#                 last_delta = (preds - prev_preds).abs().max().item()
#                 if (step + 1) >= min_steps and last_delta < pred_tol:
#                     stable_count += 1
#                 else:
#                     stable_count = 0
#
#                 if stable_count >= patience:
#                     break
#
#             prev_preds = preds
#
#         steps_used = step + 1
#
#         # hard check: did we stop because predictions stabilized?
#         assert stable_count >= patience, (
#             f"Ensemble model did not stabilize within max_steps={max_steps}. "
#             f"Last max_abs_delta={last_delta:.3e}, stable_count={stable_count}."
#         )
#
#         final_preds = preds.detach().cpu().numpy().flatten()
#         return model, final_preds, steps_used, last_delta
#
#     # training plus test on unit circle
#
#
#     N_train = 3
#     N_test = 5
#
#     train_angles = np.linspace(0.5, 2 * np.pi, N_train, endpoint=False)
#     test_angles = np.linspace(0.0, 2 * np.pi, N_test, endpoint=False)
#
#     train_vecs = np.vstack([np.cos(train_angles), np.sin(train_angles)])  # (2, N_train)
#     test_vecs = np.vstack([np.cos(test_angles), np.sin(test_angles)])     # (2, N_test)
#
#     inpt_vecs = np.concatenate([train_vecs, test_vecs], axis=1)           # (2, delta_size)
#     inpt_angles = np.concatenate([train_angles, test_angles], axis=0)     # (delta_size,)
#
#     labels = np.random.uniform(-5, 5, size=N_train)
#
#     # index bookkeeping
#     delta_size = N_train + N_test
#     is_train = np.zeros(delta_size, dtype=bool)
#     is_train[:N_train] = True
#     is_test = ~is_train
#
#     # hyperparams
#
#     hidden_dim = 30
#     depth = 2
#     lr = 0.06
#     eta = lr
#
#     ensemble_size = 12
#     max_steps = 4000
#
#     pred_tol = 1e-5
#     patience = 30
#     min_steps = 50
#
#     # THEORETICAL mean prediction from tensor code
#     device = "cpu"
#     dtype = torch.float64
#
#     X = torch.tensor(inpt_vecs, dtype=dtype, device=device)  # (2, delta_size)
#     n0 = X.shape[0]
#
#     C_W = 1.0
#     C_b = 0.0
#     lambda_W = 1.0
#     lambda_b = 0.0
#
#     # K^(1), theta^(1) from the picture (with C_W=1 and lambda_W=1)
#     K_prev = C_b + (C_W / n0) * (X.T @ X)
#     theta_prev = lambda_b + (lambda_W / n0) * (X.T @ X)
#
#     # all other tensors start at 0
#     zero4 = torch.zeros((delta_size, delta_size, delta_size, delta_size), dtype=dtype, device=device)
#     V_prev = zero4.clone()
#     S_prev = zero4.clone()
#     R_prev = zero4.clone()
#     T_prev = zero4.clone()
#     U_prev = zero4.clone()
#     A_prev = zero4.clone()
#     B_prev = zero4.clone()
#     P_prev = zero4.clone()
#     Q_prev = zero4.clone()
#     D_prev = zero4.clone()
#     F_prev = zero4.clone()
#
#     # recurse depth-1 times
#     for _ in range(depth - 1):
#         theta_prev, K_prev, V_prev, S_prev, R_prev, T_prev, U_prev, A_prev, B_prev, P_prev, Q_prev, D_prev, F_prev = recurse_layer(
#             theta_prev=theta_prev,
#             K_prev=K_prev,
#             V_prev=V_prev,
#             S_prev=S_prev,
#             R_prev=R_prev,
#             T_prev=T_prev,
#             U_prev=U_prev,
#             A_prev=A_prev,
#             B_prev=B_prev,
#             P_prev=P_prev,
#             Q_prev=Q_prev,
#             D_prev=D_prev,
#             F_prev=F_prev,
#             C_W=C_W,
#             lambda_W=lambda_W,
#             n_prev=hidden_dim,  # n_{L-1}
#             n_preprev=n0,       # n_0
#         )
#
#     theta_final = theta_prev
#     K_final = K_prev
#
#     # Z tensors are now computed from training block of Htilde_lower (theta)
#     theta_train = theta_final[:N_train, :N_train] + 1e-9 * torch.eye(N_train, dtype=dtype, device=device)
#     Z_A, Z_B, Z_IA, Z_IB, Z_IIA, Z_IIB = compute_Z_tensors(theta_train, eta)
#
#     y_labels = torch.tensor(labels.reshape(1, -1), dtype=dtype, device=device)  # n_out=1
#
#     m_dict = compute_all_m_tensors(
#         n_out=1,
#         training_size=N_train,
#         theta_final=theta_final,
#         K_final=K_final,
#         A_final=A_prev,
#         B_final=B_prev,
#         P_final=P_prev,
#         Q_final=Q_prev,
#         R_final=R_prev,
#         S_final=S_prev,
#         T_final=T_prev,
#         U_final=U_prev,
#         Z_A=Z_A,
#         Z_B=Z_B,
#         Z_IA=Z_IA,
#         Z_IB=Z_IB,
#         Z_IIA=Z_IIA,
#         Z_IIB=Z_IIB,
#         y_labels=y_labels,
#     )
#
#     mean_pred = compute_mean_output_neurons(
#         m_NTK=m_dict["m_NTK"],
#         m_delta_NTK=m_dict["m_delta_NTK"],
#         m_dNTK=m_dict["m_dNTK"],
#         m_ddNTK_I=m_dict["m_ddNTK_I"],
#         m_ddNTK_II=m_dict["m_ddNTK_II"],
#         H_beta_alpha1=theta_final,
#         Htilde_lower=theta_final,
#         n_Lm1=hidden_dim,
#         training_size=N_train,
#     ).detach().cpu().numpy().reshape(-1)  # (delta_size,)
#
#     # ensemble training until predictions stop changing
#     ensemble_preds = []
#     steps_used_list = []
#     last_delta_list = []
#
#     # train_vecs for the trainer is "columns are vectors"
#     # test_vecs should be all points (train + test) in the same column convention
#     for _ in range(ensemble_size):
#         _, preds, steps_used, last_delta = train_sin_net_until_predictions_stop(
#             train_vecs=train_vecs,
#             labels=labels,
#             hidden_dim=hidden_dim,
#             num_hidden_layers=depth - 1,
#             lr=lr,
#             max_steps=max_steps,
#             test_vecs=inpt_vecs,
#             pred_tol=pred_tol,
#             patience=patience,
#             min_steps=min_steps,
#             device=device,
#         )
#         ensemble_preds.append(preds)
#         steps_used_list.append(steps_used)
#         last_delta_list.append(last_delta)
#
#     ensemble_preds = np.stack(ensemble_preds, axis=0)  # (E, delta_size)
#     ensemble_mean = ensemble_preds.mean(axis=0)
#     ensemble_std = ensemble_preds.std(axis=0)
#
#     print(
#         f"Ensemble early-stop: mean steps={np.mean(steps_used_list):.1f}, "
#         f"max steps={np.max(steps_used_list)}, "
#         f"mean last_delta={np.mean(last_delta_list):.3e}, max last_delta={np.max(last_delta_list):.3e}"
#     )
#
#     # Plotting
#     order = np.argsort(inpt_angles)
#     angles_sorted = inpt_angles[order]
#     is_train_sorted = is_train[order]
#     is_test_sorted = is_test[order]
#
#     theory_sorted = mean_pred[order]
#     ens_mean_sorted = ensemble_mean[order]
#
#     plt.figure(figsize=(9, 5))
#
#     # THEORY mean: train vs test
#     plt.scatter(
#         angles_sorted[is_train_sorted],
#         theory_sorted[is_train_sorted],
#         marker="o",
#         s=90,
#         label="theory mean (train)",
#         alpha=0.4
#     )
#     plt.scatter(
#         angles_sorted[is_test_sorted],
#         theory_sorted[is_test_sorted],
#         marker="o",
#         facecolors="none",
#         edgecolors="black",
#         s=90,
#         label="theory mean (test)",
#         alpha=0.4
#     )
#
#     # ENSEMBLE mean: train vs test
#     plt.scatter(
#         angles_sorted[is_train_sorted],
#         ens_mean_sorted[is_train_sorted],
#         marker="^",
#         s=90,
#         label="ensemble mean (train)",
#         alpha=0.4
#     )
#     plt.scatter(
#         angles_sorted[is_test_sorted],
#         ens_mean_sorted[is_test_sorted],
#         marker="^",
#         facecolors="none",
#         edgecolors="black",
#         s=90,
#         label="ensemble mean (test)",
#         alpha=0.4
#     )
#
#     # Training targets
#     plt.scatter(train_angles, labels, marker="s", s=110, label="training labels", alpha=0.4)
#
#     plt.xlabel("angle (rad)")
#     plt.ylabel("output")
#     plt.title(f"theory mean vs ensemble mean (N_train={N_train}, N_test={N_test}, width={hidden_dim}, lr={lr})")
#     plt.legend(loc="best")
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#
#     np.random.seed(44)
#     torch.manual_seed(44)
#     torch.set_num_threads(1)
#
#     # -----------------------------
#     # Basic problem setup
#     # -----------------------------
#     N_train = 3
#     N_test = 5
#
#     hidden_dim = 30
#     depth = 3
#     lr = 0.06
#     eta = lr
#
#     C_W = 1.0
#     C_b = 0.0
#     lambda_W = 1.0
#     lambda_b = 0.0
#
#     device = "cpu"
#     dtype = torch.float64
#
#     # -----------------------------
#     # Build train and test inputs on the unit circle
#     # -----------------------------
#     train_angles = np.linspace(0.5, 2 * np.pi, N_train, endpoint=False)
#     test_angles = np.linspace(0.0, 2 * np.pi, N_test, endpoint=False)
#
#     train_vecs = np.vstack([np.cos(train_angles), np.sin(train_angles)])   # shape (2, N_train)
#     test_vecs = np.vstack([np.cos(test_angles), np.sin(test_angles)])      # shape (2, N_test)
#
#     inpt_vecs = np.concatenate([train_vecs, test_vecs], axis=1)            # shape (2, delta_size)
#
#     labels = np.random.uniform(-5, 5, size=N_train)
#
#     delta_size = N_train + N_test
#
#     # -----------------------------
#     # Print only basic parameters
#     # -----------------------------
#     print("Running NTK mean prediction computation")
#     print(f"training size   = {N_train}")
#     print(f"test size       = {N_test}")
#     print(f"delta size      = {delta_size}")
#     print(f"input dimension = {inpt_vecs.shape[0]}")
#     print(f"hidden width    = {hidden_dim}")
#     print(f"depth           = {depth}")
#     print(f"learning rate   = {lr}")
#     print(f"eta             = {eta}")
#     print(f"C_W             = {C_W}")
#     print(f"lambda_W        = {lambda_W}")
#     print()
#
#     # -----------------------------
#     # Initial K and theta
#     # K^(1) = (C_W / n0) X^T X
#     # theta^(1) = (lambda_W / n0) X^T X
#     # -----------------------------
#     X = torch.tensor(inpt_vecs, dtype=dtype, device=device)
#     n0 = X.shape[0]
#
#     K_prev = C_b + (C_W / n0) * (X.T @ X)
#     theta_prev = lambda_b + (lambda_W / n0) * (X.T @ X)
#
#     # -----------------------------
#     # All other tensors start at zero
#     # -----------------------------
#     zero4 = torch.zeros(
#         (delta_size, delta_size, delta_size, delta_size),
#         dtype=dtype,
#         device=device
#     )
#
#     V_prev = zero4.clone()
#     S_prev = zero4.clone()
#     R_prev = zero4.clone()
#     T_prev = zero4.clone()
#     U_prev = zero4.clone()
#     A_prev = zero4.clone()
#     B_prev = zero4.clone()
#     P_prev = zero4.clone()
#     Q_prev = zero4.clone()
#     D_prev = zero4.clone()
#     F_prev = zero4.clone()
#
#     # -----------------------------
#     # Recurse through layers
#     # -----------------------------
#     for _ in range(depth - 1):
#         (
#             theta_prev,
#             K_prev,
#             V_prev,
#             S_prev,
#             R_prev,
#             T_prev,
#             U_prev,
#             A_prev,
#             B_prev,
#             P_prev,
#             Q_prev,
#             D_prev,
#             F_prev,
#         ) = recurse_layer(
#             theta_prev=theta_prev,
#             K_prev=K_prev,
#             V_prev=V_prev,
#             S_prev=S_prev,
#             R_prev=R_prev,
#             T_prev=T_prev,
#             U_prev=U_prev,
#             A_prev=A_prev,
#             B_prev=B_prev,
#             P_prev=P_prev,
#             Q_prev=Q_prev,
#             D_prev=D_prev,
#             F_prev=F_prev,
#             C_W=C_W,
#             lambda_W=lambda_W,
#             n_prev=hidden_dim,
#             n_preprev=n0,
#         )
#
#     theta_final = theta_prev
#     K_final = K_prev
#
#     # -----------------------------
#     # Compute Z tensors from the training block of theta
#     # -----------------------------
#     theta_train = theta_final[:N_train, :N_train] + 1e-9 * torch.eye(
#         N_train, dtype=dtype, device=device
#     )
#
#     Z_A, Z_B, Z_IA, Z_IB, Z_IIA, Z_IIB = compute_Z_tensors(theta_train, eta)
#
#     # -----------------------------
#     # Labels for the m-recursions
#     # n_out = 1 because scalar output network
#     # -----------------------------
#     y_labels = torch.tensor(labels.reshape(1, -1), dtype=dtype, device=device)
#
#     # -----------------------------
#     # Compute m tensors only transiently
#     # They are not printed or saved
#     # -----------------------------
#     m_dict = compute_all_m_tensors(
#         n_out=1,
#         training_size=N_train,
#         theta_final=theta_final,
#         K_final=K_final,
#         A_final=A_prev,
#         B_final=B_prev,
#         P_final=P_prev,
#         Q_final=Q_prev,
#         R_final=R_prev,
#         S_final=S_prev,
#         T_final=T_prev,
#         U_final=U_prev,
#         Z_A=Z_A,
#         Z_B=Z_B,
#         Z_IA=Z_IA,
#         Z_IB=Z_IB,
#         Z_IIA=Z_IIA,
#         Z_IIB=Z_IIB,
#         y_labels=y_labels,
#     )
#
#     mean_pred = compute_mean_output_neurons(
#         m_NTK=m_dict["m_NTK"],
#         m_delta_NTK=m_dict["m_delta_NTK"],
#         m_dNTK=m_dict["m_dNTK"],
#         m_ddNTK_I=m_dict["m_ddNTK_I"],
#         m_ddNTK_II=m_dict["m_ddNTK_II"],
#         H_beta_alpha1=theta_final,
#         Htilde_lower=theta_final,
#         n_Lm1=hidden_dim,
#         training_size=N_train,
#     ).detach().cpu().numpy().reshape(-1)
#
#     # Optional: free the intermediate dictionary immediately
#     del m_dict
#
#     # -----------------------------
#     # Split predictions into train/test parts
#     # -----------------------------
#     train_pred = mean_pred[:N_train]
#     test_pred = mean_pred[N_train:]
#
#     # -----------------------------
#     # Plot only:
#     #   - NTK mean prediction on train points
#     #   - NTK mean prediction on test points
#     #   - training labels
#     # No connecting lines
#     # -----------------------------
#     plt.figure(figsize=(8, 5))
#
#     # training labels
#     plt.scatter(
#         train_angles,
#         labels,
#         marker="s",
#         s=110,
#         alpha=0.6,
#         label="training labels"
#     )
#
#     # NTK mean prediction on training points
#     plt.scatter(
#         train_angles,
#         train_pred,
#         marker="o",
#         s=90,
#         alpha=0.6,
#         label="NTK mean prediction (train)"
#     )
#
#     # NTK mean prediction on test points
#     plt.scatter(
#         test_angles,
#         test_pred,
#         marker="o",
#         s=90,
#         alpha=0.6,
#         facecolors="none",
#         edgecolors="black",
#         label="NTK mean prediction (test)"
#     )
#
#     plt.xlabel("angle (rad)")
#     plt.ylabel("output")
#     plt.title(
#         f"NTK mean prediction\n"
#         f"N_train={N_train}, N_test={N_test}, width={hidden_dim}, depth={depth}, lr={lr}"
#     )
#     plt.legend(loc="best")
#     plt.tight_layout()
#     output_path = f"ntk_prediction_Ntrain{N_train}_Ntest{N_test}_width{hidden_dim}_depth{depth}_lr{lr}.png"
#     plt.savefig(output_path, dpi=300, bbox_inches="tight")
#     plt.close()
#
#     print(f"Saved plot to {output_path}")


if __name__ == "__main__":


    # -----------------------------
    # Reproducibility / Killarney setup
    # -----------------------------
    BASE_SEED = 44
    np.random.seed(BASE_SEED)
    torch.manual_seed(BASE_SEED)

    # Use SLURM CPU allocation if available; otherwise use 1 thread
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))

    device = "cpu"
    dtype = torch.float64

    # -----------------------------
    # Fixed constants
    # -----------------------------
    C_W = 1.0
    C_b = 0.0
    lambda_W = 1.0
    lambda_b = 0.0

    lr = 0.06
    eta = lr

    # -----------------------------
    # Choose the configurations you want to run
    # -----------------------------
    CONFIGS = [
        {"N_train": 10, "N_test": 10,  "hidden_dim": 100,  "depth": 2},
        {"N_train": 20, "N_test": 10, "hidden_dim": 100, "depth": 2},
        {"N_train": 30, "N_test": 10, "hidden_dim": 100, "depth": 2},
        {"N_train": 50, "N_test": 10, "hidden_dim": 100, "depth": 2},
    ]

    output_dir = "C:\\Users\\alexm\\Downloads"
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------
    # Helper function for one run
    # -----------------------------
    def run_one_config(config, config_id):
        N_train = config["N_train"]
        N_test = config["N_test"]
        hidden_dim = config["hidden_dim"]
        depth = config["depth"]

        delta_size = N_train + N_test

        # Use a different but deterministic seed for each config
        rng = np.random.default_rng(BASE_SEED + config_id)

        # -----------------------------
        # Build train and test inputs on the unit circle
        # -----------------------------
        train_angles = np.linspace(0.5, 2 * np.pi, N_train, endpoint=False)
        test_angles = np.linspace(0.0, 2 * np.pi, N_test, endpoint=False)

        train_vecs = np.vstack([np.cos(train_angles), np.sin(train_angles)])
        test_vecs = np.vstack([np.cos(test_angles), np.sin(test_angles)])

        inpt_vecs = np.concatenate([train_vecs, test_vecs], axis=1)

        labels = rng.uniform(-5, 5, size=N_train)

        # -----------------------------
        # Print only basic parameters
        # -----------------------------
        print("=" * 70)
        print(f"Running config {config_id}")
        print(f"training size   = {N_train}")
        print(f"test size       = {N_test}")
        print(f"delta size      = {delta_size}")
        print(f"input dimension = {inpt_vecs.shape[0]}")
        print(f"hidden width    = {hidden_dim}")
        print(f"depth           = {depth}")
        print(f"learning rate   = {lr}")
        print(f"eta             = {eta}")
        print(f"C_W             = {C_W}")
        print(f"lambda_W        = {lambda_W}")
        print("=" * 70, flush=True)

        # -----------------------------
        # Initial K and theta
        # -----------------------------
        X = torch.tensor(inpt_vecs, dtype=dtype, device=device)
        n0 = X.shape[0]

        K_prev = C_b + (C_W / n0) * (X.T @ X)
        theta_prev = lambda_b + (lambda_W / n0) * (X.T @ X)

        # -----------------------------
        # All other tensors start at zero
        # -----------------------------
        zero4 = torch.zeros(
            (delta_size, delta_size, delta_size, delta_size),
            dtype=dtype,
            device=device,
        )

        V_prev = zero4.clone()
        S_prev = zero4.clone()
        R_prev = zero4.clone()
        T_prev = zero4.clone()
        U_prev = zero4.clone()
        A_prev = zero4.clone()
        B_prev = zero4.clone()
        P_prev = zero4.clone()
        Q_prev = zero4.clone()
        D_prev = zero4.clone()
        F_prev = zero4.clone()

        # -----------------------------
        # Recurse through layers
        # -----------------------------
        for layer_idx in range(depth - 1):
            if layer_idx == 0:
                n_preprev = n0
            else:
                n_preprev = hidden_dim

            (
                theta_prev,
                K_prev,
                V_prev,
                S_prev,
                R_prev,
                T_prev,
                U_prev,
                A_prev,
                B_prev,
                P_prev,
                Q_prev,
                D_prev,
                F_prev,
            ) = recurse_layer(
                theta_prev=theta_prev,
                K_prev=K_prev,
                V_prev=V_prev,
                S_prev=S_prev,
                R_prev=R_prev,
                T_prev=T_prev,
                U_prev=U_prev,
                A_prev=A_prev,
                B_prev=B_prev,
                P_prev=P_prev,
                Q_prev=Q_prev,
                D_prev=D_prev,
                F_prev=F_prev,
                C_W=C_W,
                lambda_W=lambda_W,
                n_prev=hidden_dim,
                n_preprev=n_preprev,
            )

        theta_final = theta_prev
        K_final = K_prev

        # -----------------------------
        # Compute Z tensors from training block of theta
        # -----------------------------
        theta_train = theta_final[:N_train, :N_train] + 1e-9 * torch.eye(
            N_train, dtype=dtype, device=device
        )

        Z_A, Z_B, Z_IA, Z_IB, Z_IIA, Z_IIB = compute_Z_tensors(theta_train, eta)

        y_labels = torch.tensor(labels.reshape(1, -1), dtype=dtype, device=device)

        # -----------------------------
        # Compute m tensors transiently only
        # Not printed and not saved
        # -----------------------------
        m_dict = compute_all_m_tensors(
            n_out=1,
            training_size=N_train,
            theta_final=theta_final,
            K_final=K_final,
            A_final=A_prev,
            B_final=B_prev,
            P_final=P_prev,
            Q_final=Q_prev,
            R_final=R_prev,
            S_final=S_prev,
            T_final=T_prev,
            U_final=U_prev,
            Z_A=Z_A,
            Z_B=Z_B,
            Z_IA=Z_IA,
            Z_IB=Z_IB,
            Z_IIA=Z_IIA,
            Z_IIB=Z_IIB,
            y_labels=y_labels,
        )

        mean_pred = compute_mean_output_neurons(
            m_NTK=m_dict["m_NTK"],
            m_delta_NTK=m_dict["m_delta_NTK"],
            m_dNTK=m_dict["m_dNTK"],
            m_ddNTK_I=m_dict["m_ddNTK_I"],
            m_ddNTK_II=m_dict["m_ddNTK_II"],
            H_beta_alpha1=theta_final,
            Htilde_lower=theta_final,
            n_Lm1=hidden_dim,
            training_size=N_train,
        ).detach().cpu().numpy().reshape(-1)

        del m_dict

        # -----------------------------
        # Split predictions into train/test parts
        # -----------------------------
        train_pred = mean_pred[:N_train]
        test_pred = mean_pred[N_train:]

        # -----------------------------
        # Plot only:
        #   - training labels
        #   - NTK mean prediction on train points
        #   - NTK mean prediction on test points
        # -----------------------------
        plt.figure(figsize=(8, 5))

        plt.scatter(
            train_angles,
            labels,
            marker="s",
            s=110,
            alpha=0.6,
            label="training labels",
        )

        plt.scatter(
            train_angles,
            train_pred,
            marker="o",
            s=90,
            alpha=0.6,
            label="NTK mean prediction (train)",
        )

        plt.scatter(
            test_angles,
            test_pred,
            marker="o",
            s=90,
            alpha=0.6,
            facecolors="none",
            edgecolors="black",
            label="NTK mean prediction (test)",
        )

        plt.xlabel("angle (rad)")
        plt.ylabel("output")
        plt.title(
            f"NTK mean prediction\n"
            f"N_train={N_train}, N_test={N_test}, "
            f"width={hidden_dim}, depth={depth}, lr={lr}"
        )
        plt.legend(loc="best")
        plt.tight_layout()

        output_path = os.path.join(
            output_dir,
            f"ntk_prediction_cfg{config_id}_"
            f"Ntrain{N_train}_Ntest{N_test}_"
            f"width{hidden_dim}_depth{depth}_lr{lr}.png",
        )

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved plot to {output_path}", flush=True)

        # Delete large tensors before moving to the next config
        del (
            X,
            K_prev,
            theta_prev,
            K_final,
            theta_final,
            V_prev,
            S_prev,
            R_prev,
            T_prev,
            U_prev,
            A_prev,
            B_prev,
            P_prev,
            Q_prev,
            D_prev,
            F_prev,
            Z_A,
            Z_B,
            Z_IA,
            Z_IB,
            Z_IIA,
            Z_IIB,
        )

    # -----------------------------
    # Run all configs
    # -----------------------------
    for config_id, config in enumerate(CONFIGS):
        run_one_config(config, config_id)

    print("All NTK prediction plots finished.", flush=True)