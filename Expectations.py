import torch
import itertools


def partition_pairs_singles(L):
    """
    Produces all the partitions of L into disjoint sets of pairs and singletons

    Return list of dicts
    """
    # Base case
    if len(L) == 0:
        return [{"pairs": [], "singles": []}]
    if len(L) == 1:
        return [{"pairs": [], "singles": [L[0]]}]

    new_elem = L[0]
    partitions = []

    # Inductive step: append new_elem as a singleton, and then append as a pair
    for rest in partition_pairs_singles(L[1:]):
        partitions.append({"pairs": rest["pairs"], "singles": [new_elem] + rest["singles"]})
    for i in range(1, len(L)):
        new_elem_2 = L[i]
        remaining = L[1:i] + L[i + 1:]
        for rest in partition_pairs_singles(remaining):
            partitions.append({"pairs": [[new_elem, new_elem_2]] + rest["pairs"], "singles": rest["singles"]})

    return partitions


def exp_val(K, I, J, L):
    k = len(I)
    r = len(J)

    if k == r == 0:
        normalize = torch.tensor(1.0)
    else:
        normalize = 2 ** (k + r - 1)

    sgn_fip_list = [-1, 1]

    if k == 0:
        eps_sin_full_list = [torch.tensor([0.0], dtype=K.dtype, device=K.device)]
        sgn_fip_list = [1]
    else:
        sin_sign_combinations = list(itertools.product([-1, 1], repeat=k - 1))
        eps_sin_full_list = []
        for combo in sin_sign_combinations:
            eps_sin_full_list.append(torch.tensor([1.0, *combo], dtype=K.dtype, device=K.device))
    if r == 0:
        eps_cos_full_list = [torch.tensor([0.0], dtype=K.dtype, device=K.device)]
        sgn_fip_list = [1]
    else:
        cos_sign_combinations = list(itertools.product([-1, 1], repeat=r - 1))
        eps_cos_full_list = []
        for combo in cos_sign_combinations:
            eps_cos_full_list.append(torch.tensor([1.0, *combo], dtype=K.dtype, device=K.device))

    partitions = partition_pairs_singles(L)

    vals = []
    for sgn_fip_val in sgn_fip_list:
        for eps_signs_sin in eps_sin_full_list:
            for eps_signs_cos in eps_cos_full_list:
                sgn_fip = torch.tensor(sgn_fip_val, dtype=K.dtype, device=K.device)
                epsilon = torch.zeros(K.shape[0], dtype=K.dtype)

                epsilon.scatter_add_(0, torch.tensor(I), eps_signs_sin)
                epsilon.scatter_add_(0, torch.tensor(J), sgn_fip * eps_signs_cos)

                epsilon = epsilon.unsqueeze(1)

                eps_sin_sum = int(eps_signs_sin.sum().item())
                running_sum = 0
                for pi in partitions:
                    pairs = pi['pairs']
                    sings = pi['singles']
                    running_p_prod = torch.tensor(1.0, dtype=K.dtype)
                    running_s_prod = torch.tensor(1.0, dtype=K.dtype)
                    if len(sings) % 2 == k % 2:
                        for p in pairs:
                            running_p_prod *= K[p[0], p[1]]
                        for s in sings:
                            running_s_prod *= (K @ epsilon)[s, 0]
                        running_sum += ((-1.0) ** ((len(sings)-eps_sin_sum)/2)) * running_p_prod * running_s_prod

                vals.append(torch.exp(-0.5 * (epsilon.T @ K @ epsilon)) * running_sum)

    return sum(vals) / normalize


def s2_expval(M):
    """
    M should be a square matrix
    \E[\sin \sin]
    """
    diag = torch.diagonal(M)
    K = torch.exp(-0.5 * (diag.unsqueeze(0) + diag.unsqueeze(1))) * torch.sinh(M)
    return K


def c2_expval(M):
    """
    M should be a square matrix
    \E[\cos \cos]
    """
    diag = torch.diagonal(M)
    K = torch.exp(-0.5 * (diag.unsqueeze(0) + diag.unsqueeze(1))) * torch.cosh(M)
    return K


def s2c1_expval(M):
    """
    M should be a square matrix
    \E[\sin \sin \cos]
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
    E[ \sin \cos z ]
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
    E[ \cos z z ]
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
    E[ \sin \sin z z ]
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
    E[ \cos \cos z z ]
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
    E[ \sin \sin \cos \cos ]
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
    \E[\Prod_{i=1}^k \sin(z_i)]
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
    \E[\Prod_{i=1}^r \cos(z_i)]
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


