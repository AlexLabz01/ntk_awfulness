import numpy as np
import torch
from typing import Optional
import itertools
from typing import Dict


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



def chi_parallel_update(
    Cw: float,
    K_l: float,
    E_sp_s_z: float,           # <σ' σ z>_{K^{(l)}}
) -> float:
    """
    Chi_parallel^{(l+1)} = (C_W /  K^{(l)}) * <σ' σ z>_{K^{(l)}}
    """
    return (Cw / K_l) * E_sp_s_z


def chi_perp_update(
    Cw: float,
    E_sp_sp: float,            # <σ' σ'>_{K^{(l)}}
) -> float:
    """
    Chi_perp^{(l)} = C_W * <σ' σ'>_{K^{(l)}}
    """
    return Cw * E_sp_sp


def R_update_scalar(
    R_l: float,
    H_l: float,
    B_l: float,
    P_l: float,
    chi_perp_l: float,
    Cw: float,
    lambda_next: float,
    E_dd_spsps: float,         # <σ'' σ' σ' σ>_{K^{(l)}}
    E_3p_spspsp: float,        # <σ''' σ' σ' σ'>_{K^{(l)}}
    E_dd_s: float,             # <σ'' σ>_{K^{(l)}}
    E_3p_s: float,             # <σ''' σ>_{K^{(l)}}
    E_sp_sp: float,            # <σ' σ'>_{K^{(l)}}
    E_dd_dd: float,            # <σ'' σ''>_{K^{(l)}}
) -> float:
    """
    Term structure:
      1)  (Chi_perp^{(l)})^2 * R^{(l)}
      2)  lambda_W^{(l+1)}*C_W * <σ'' σ' σ' σ> * (H^{(l)})^2
      3)  (C_W)^2 * <σ''' σ' σ' σ'> * (H^{(l)})^3
      4)  Chi_perp^{(l)} * [ lambda_W^{(l+1)}<σ'' σ> + C_W*H^{(l)}<σ''' σ> ] * (B^{(l)} + P^{(l)})
      5)  Chi_perp^{(l)} * [ lambda_W^{(l+1)}<σ' σ'> + C_W*H^{(l)}<σ'' σ''> ] * P^{(l)}
    """
    cw2 = Cw * Cw

    term1 = (chi_perp_l * chi_perp_l) * R_l
    term2 = (lambda_next * Cw) * E_dd_spsps * (H_l * H_l)
    term3 = cw2 * E_3p_spspsp * (H_l * H_l * H_l)

    bracket1 = (lambda_next * E_dd_s) + (Cw * H_l * E_3p_s)
    term4 = chi_perp_l * bracket1 * (B_l + P_l)

    bracket2 = (lambda_next * E_sp_sp) + (Cw * H_l * E_dd_dd)
    term5 = chi_perp_l * bracket2 * P_l

    return term1 + term2 + term3 + term4 + term5


def S_update_scalar(
    S_l: float,
    H_l: float,
    B_l: float,
    chi_perp_l: float,
    Cw: float,
    lambda_next: float,
    E_spspspsp: float,         # <σ' σ' σ' σ'>_{K^{(l)}}
    E_dddd_spsp: float,        # <σ'' σ'' σ' σ'>_{K^{(l)}}
    E_sp_sp: float,            # <σ' σ'>_{K^{(l)}}
    E_dd_dd: float,            # <σ'' σ''>_{K^{(l)}}
) -> float:
    """
    Term structure:
      1)  (Chi_perp^{(l)})^2 * S^{(l)}
      2)  C_W*lambda_W^{(l+1)} * <σ' σ' σ' σ'> * (H^{(l)})^2
      3)  (C_W)^2 * <σ'' σ'' σ' σ'> * (H^{(l)})^3
      4)  Chi_perp^{(l)} * [ lambda_W^{(l+1)}<σ' σ'> + C_W*H^{(l)}<σ'' σ''> ] * B^{(l)}
    """
    cw2 = Cw * Cw

    term1 = (chi_perp_l * chi_perp_l) * S_l
    term2 = (Cw * lambda_next) * E_spspspsp * (H_l * H_l)
    term3 = cw2 * E_dddd_spsp * (H_l * H_l * H_l)

    bracket = (lambda_next * E_sp_sp) + (Cw * H_l * E_dd_dd)
    term4 = chi_perp_l * bracket * B_l

    return term1 + term2 + term3 + term4


def U_update_scalar(
    U_l: float,
    H_l: float,
    chi_perp_l: float,
    Cw: float,
    E_dddd_spsp: float,        # <σ'' σ'' σ' σ'>_{K^{(l)}}
) -> float:
    """
    Term structure:
      1)  (Chi_perp^{(l)})^2 * U^{(l)}
      2)  (C_W)^2 * <σ'' σ'' σ' σ'> * (H^{(l)})^3
    """
    cw2 = Cw * Cw

    term1 = (chi_perp_l * chi_perp_l) * U_l
    term2 = cw2 * E_dddd_spsp * (H_l * H_l * H_l)

    return term1 + term2

def T_update_scalar(
    T_l: float,
    H_l: float,
    K_l: float,
    F_l: float,
    Q_l: float,
    chi_perp_l: float,
    Cw: float,
    lambda_next: float,
    E_dd_spsps: float,         # <σ'' σ' σ' σ>_{K^{(l)}} (scalar)
    E_dddd_spsp: float,        # <σ'' σ'' σ' σ'>_{K^{(l)}} (scalar)
    E_spsp_ss: float,          # <σ' σ' σ σ>_{K^{(l)}} (scalar)
    E_z_sp_s: float,           # <z σ' σ>_{K^{(l)}} (scalar)
    E_z_dd_sp: float,          # <z σ'' σ'>_{K^{(l)}} (scalar)
    E_dd_s: float,             # <σ'' σ>_{K^{(l)}} (scalar)
    E_sp_sp: float,            # <σ' σ'>_{K^{(l)}} (scalar)
    E_3p_sp: float,            # <σ''' σ'>_{K^{(l)}} (scalar)
    E_dd_dd: float,            # <σ'' σ''>_{K^{(l)}} (scalar)
) -> float:
    """
    Term structure:
      1)  (chi_perp^{(l)})^2 * T^{(l)}
      2)  2*Cw*lambda_W^{(l+1)} * <σ'' σ' σ' σ> * (H^{(l)})^2
      3)  (Cw)^2 * <σ'' σ'' σ' σ'> * (H^{(l)})^3
      4)  (lambda_W^{(l+1)})^2 * <σ' σ' σ σ> * H^{(l)}
      5)  [ lambda_W^{(l+1)}<z σ' σ> + Cw*H^{(l)}<z σ'' σ'> ]^2 * F^{(l)} / (K^{(l)})^2
      6)  2*chi_perp^{(l)} * [ lambda_W^{(l+1)}(<σ'' σ> + <σ' σ'>)
                              + Cw*H^{(l)}(<σ''' σ'> + <σ'' σ''>) ] * Q^{(l)}
    """
    cw2 = Cw * Cw
    ratioF = F_l / (K_l * K_l)

    term1 = (chi_perp_l * chi_perp_l) * T_l
    term2 = (2.0 * Cw * lambda_next) * E_dd_spsps * (H_l * H_l)
    term3 = cw2 * E_dddd_spsp * (H_l * H_l * H_l)
    term4 = (lambda_next * lambda_next) * E_spsp_ss * H_l

    bracketF = (lambda_next * E_z_sp_s) + (Cw * H_l * E_z_dd_sp)
    term5 = (bracketF * bracketF) * ratioF

    bracketQ = (lambda_next * (E_dd_s + E_sp_sp)) + (Cw * H_l * (E_3p_sp + E_dd_dd))
    term6 = 2.0 * chi_perp_l * bracketQ * Q_l

    return term1 + term2 + term3 + term4 + term5 + term6

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


#recurse through the layers
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
    # Convert to E[z_p z_q sin_a sin_b] with axes (p,q,a,b)
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

    # final layer tensors (Model A: full delta_size index set on every axis unless noted)
    A_final: torch.Tensor,          # A[delta, a1, a2, a3] (delta_size^4)
    B_final: torch.Tensor,          # B[delta, a1, a2, a3] (delta_size^4)
    P_final: torch.Tensor,          # P[i0,i1,i2,i3] (delta_size^4)
    Q_final: torch.Tensor,          # Q[i0,i1,i2,i3] (delta_size^4)
    R_final: torch.Tensor,          # R[i0,i1,i2,i3] (delta_size^4)
    S_final: torch.Tensor,          # S[i0,i1,i2,i3] (delta_size^4)
    T_final: torch.Tensor,          # T[i0,i1,i2,i3] (delta_size^4)
    U_final: torch.Tensor,          # U[i0,i1,i2,i3] (delta_size^4)

    # Z-tensors needed by the m-formulas (these are NOT inverses; "upper indices" is just notation)
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
    # because compute_m_NTK slices its second axis to training_size anyway.
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

    # build Htilde_upper on training block
    Htilde_upper = torch.linalg.inv(Htilde_lower[:ts, :ts])     # (alpha1, alpha2)
    H_beta_train = H_beta_alpha1[:, :ts]                        # (beta, alpha1)
    m_corr_train = m_corr[:, :ts]                               # (i, alpha2)

    # projection term: sum_{alpha1,alpha2} H[beta,alpha1] Htilde_upper[alpha1,alpha2] m_corr[i,alpha2]
    proj = torch.einsum("ba,ac,ic->ib", H_beta_train, Htilde_upper, m_corr_train)  # (i, beta)

    # final mean prediction
    m_mean = m_NTK + inv_n * (m_corr - proj)
    return m_mean

# -----------------------------
# Helpers for "scalar as tensor"
# -----------------------------
def t1(val, dtype=torch.float64, device="cpu"):
    return torch.full((1, 1), float(val), dtype=dtype, device=device)

def t3(val, dtype=torch.float64, device="cpu"):
    return torch.full((1, 1, 1), float(val), dtype=dtype, device=device)

def t4(val, dtype=torch.float64, device="cpu"):
    return torch.full((1, 1, 1, 1), float(val), dtype=dtype, device=device)

def close(a, b, tol=1e-10):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


if __name__ == "__main__":
    device = "cpu"
    dtype = torch.float64
    tol = 1e-10
    torch.manual_seed(0)


    Cw = float(torch.rand(()).item() + 0.5)
    Lam = float(torch.rand(()).item() + 0.5)

    #hidden layer case lets say
    n_l = 128
    n_lm1 = 128

    H_l = float(torch.rand(()).item() + 0.5)
    K_l = float(torch.rand(()).item() + 0.5)

    U_l = float(torch.randn(()).item())
    S_l = float(torch.randn(()).item())
    R_l = float(torch.randn(()).item())
    T_l = float(torch.randn(()).item())

    B_l = float(torch.randn(()).item())
    P_l = float(torch.randn(()).item())
    F_l = float(torch.randn(()).item())
    Q_l = float(torch.randn(()).item())

    # Expectations (scalars)
    E_sp_sp      = float(torch.rand(()).item() + 0.1)   # <σ' σ'>
    E_dd_s       = float(torch.randn(()).item())        # <σ'' σ>
    E_3p_s       = float(torch.randn(()).item())        # <σ''' σ>
    E_dd_dd      = float(torch.rand(()).item() + 0.1)   # <σ'' σ''>

    E_dd_spsps   = float(torch.randn(()).item())        # <σ'' σ' σ' σ>
    E_3p_spspsp  = float(torch.randn(()).item())        # <σ''' σ' σ' σ'>
    E_spspspsp   = float(torch.randn(()).item())        # <σ' σ' σ' σ'>
    E_dddd_spsp  = float(torch.randn(()).item())        # <σ'' σ'' σ' σ'>

    E_spsp_ss    = float(torch.randn(()).item())        # <σ' σ' σ σ>
    E_z_sp_s     = float(torch.randn(()).item())        # <z σ' σ>
    E_z_dd_sp    = float(torch.randn(()).item())        # <z σ'' σ'>
    E_3p_sp      = float(torch.randn(()).item())        # <σ''' σ'>

    #the chi that we actually use
    chi_perp = chi_perp_update(Cw=Cw, E_sp_sp=E_sp_sp)

    # U tensors/scalars

    U_next_scalar = U_update_scalar(
        U_l=U_l, H_l=H_l, chi_perp_l=chi_perp, Cw=Cw, E_dddd_spsp=E_dddd_spsp
    )

    U_next_tensor = U_update_abcd(
        U_l=t4(U_l, dtype=dtype, device=device),
        H_l=t1(H_l, dtype=dtype, device=device),
        E_ddpp_abcd=t4(E_dddd_spsp, dtype=dtype, device=device),
        E_p_ad=t1(E_sp_sp, dtype=dtype, device=device),
        Cw_next=Cw,
        n_l=n_l,
        n_lm1=n_lm1,
        E_p_bc=None,
    ).item()

    # S tensors/scalars
    S_next_scalar = S_update_scalar(
        S_l=S_l, H_l=H_l, B_l=B_l, chi_perp_l=chi_perp,
        Cw=Cw, lambda_next=Lam,
        E_spspspsp=E_spspspsp,
        E_dddd_spsp=E_dddd_spsp,
        E_sp_sp=E_sp_sp,
        E_dd_dd=E_dd_dd
    )

    S_next_tensor = S_update_abcd(
        S_l=t4(S_l, dtype=dtype, device=device),
        B=t4(B_l, dtype=dtype, device=device),
        H=t1(H_l, dtype=dtype, device=device),
        E_pppp_abcd=t4(E_spspspsp, dtype=dtype, device=device),
        E_p_ab=t1(E_sp_sp, dtype=dtype, device=device),
        E_dd_ab=t1(E_dd_dd, dtype=dtype, device=device),
        E_ddpp_abcd=t4(E_dddd_spsp, dtype=dtype, device=device),
        Cw_next=Cw,
        Lambda_next=Lam,
        n_l=n_l,
        n_lm1=n_lm1,
        E_p_cd=None,
    ).item()


    # R tensors/scalars
    R_next_scalar = R_update_scalar(
        R_l=R_l, H_l=H_l, B_l=B_l, P_l=P_l,
        chi_perp_l=chi_perp, Cw=Cw, lambda_next=Lam,
        E_dd_spsps=E_dd_spsps,
        E_3p_spspsp=E_3p_spspsp,
        E_dd_s=E_dd_s,
        E_3p_s=E_3p_s,
        E_sp_sp=E_sp_sp,
        E_dd_dd=E_dd_dd
    )

    R_next_tensor = R_update_zabc(
        R_l=t4(R_l, dtype=dtype, device=device),
        H=t1(H_l, dtype=dtype, device=device),
        B=t4(B_l, dtype=dtype, device=device),
        P=t4(P_l, dtype=dtype, device=device),

        E_ddsp_zabc=t4(E_dd_spsps, dtype=dtype, device=device),
        E_p=t1(E_sp_sp, dtype=dtype, device=device),
        E_dd_za=t1(E_dd_s, dtype=dtype, device=device),
        E_3pppp_zabc=t4(E_3p_spspsp, dtype=dtype, device=device),
        E_3p_za=t1(E_3p_s, dtype=dtype, device=device),
        E_dddd_za=t1(E_dd_dd, dtype=dtype, device=device),

        Cw=Cw, Lambda=Lam, n_l=n_l, n_lm1=n_lm1
    ).item()

    # T tensors/scalars
    T_next_scalar = T_update_scalar(
        T_l=T_l, H_l=H_l, K_l=K_l, F_l=F_l, Q_l=Q_l,
        chi_perp_l=chi_perp, Cw=Cw, lambda_next=Lam,
        E_dd_spsps=E_dd_spsps,
        E_dddd_spsp=E_dddd_spsp,
        E_spsp_ss=E_spsp_ss,
        E_z_sp_s=E_z_sp_s,
        E_z_dd_sp=E_z_dd_sp,
        E_dd_s=E_dd_s,
        E_sp_sp=E_sp_sp,
        E_3p_sp=E_3p_sp,
        E_dd_dd=E_dd_dd
    )

    # For the single input case inverses are obviously just reciprocols
    Ginv_1 = t1(1.0 / K_l, dtype=dtype, device=device)

    #putting in all the ungodly amount of inputs
    T_next_tensor = T_update_acdb(
        T_l=t4(T_l, dtype=dtype, device=device),
        H=t1(H_l, dtype=dtype, device=device),
        F_l=t4(F_l, dtype=dtype, device=device),
        Q_l=t4(Q_l, dtype=dtype, device=device),
        Ginv=Ginv_1,

        E_ppss_abcd=t4(E_spsp_ss, dtype=dtype, device=device),
        E_z_ps_eac=t3(E_z_sp_s, dtype=dtype, device=device),
        E_z_ps_fbd=t3(E_z_sp_s, dtype=dtype, device=device),

        E_pddsp_abcd=t4(E_dd_spsps, dtype=dtype, device=device),
        E_z_ddp_fbd=t3(E_z_dd_sp, dtype=dtype, device=device),

        E_p=t1(E_sp_sp, dtype=dtype, device=device),
        E_dds=t1(E_dd_s, dtype=dtype, device=device),

        E_ddpsp_badc=t4(E_dd_spsps, dtype=dtype, device=device),
        E_z_ds_ebd=t3(E_z_sp_s, dtype=dtype, device=device),
        E_z_pp_fac=t3(E_z_dd_sp, dtype=dtype, device=device),

        E_ddddpp_abcd=t4(E_dddd_spsp, dtype=dtype, device=device),
        E_z_ddp_eac=t3(E_z_dd_sp, dtype=dtype, device=device),
        E_z_ddp_fbd2=t3(E_z_dd_sp, dtype=dtype, device=device),

        E_3p_p=t1(E_3p_sp, dtype=dtype, device=device),
        E_dd_dd=t1(E_dd_dd, dtype=dtype, device=device),

        Cw_next=Cw, Lambda_next=Lam, n_l=n_l, n_lm1=n_lm1
    ).item()

    # Moment of truth
    results = {
        "U": (U_next_scalar, U_next_tensor),
        "S": (S_next_scalar, S_next_tensor),
        "R": (R_next_scalar, R_next_tensor),
        "T": (T_next_scalar, T_next_tensor),
    }

    all_ok = True
    for name, (sval, tval) in results.items():
        ok = close(float(sval), float(tval), tol=tol)
        all_ok = all_ok and ok
        print(f"{name}: scalar={sval:.12g} tensor={tval:.12g}  abs_err={abs(sval-tval):.3e}  ALL OK={ok}")

    assert all_ok, "Uh oh, somethings wrong"