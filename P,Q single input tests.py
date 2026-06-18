import torch

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







def P_update_scalar_single_input(
    P_l: float,                # P^{(l)} (scalar)
    H_l: float,                # H^{(l)} (scalar)
    B_l: float,                # B^{(l)} (scalar)
    Cw: float,                 # C_W (constant float)
    E_sp_sp: float,            # <σ' σ'>_{K^{(l)}} (scalar)  (used to build chi_perp)
    E_dd_s: float,             # <σ'' σ>_{K^{(l)}} (scalar)
    E_dd_spsps: float,         # <σ'' σ' σ' σ>_{K^{(l)}} (scalar)
) -> float:
    """
    Single-input recursion:
      P^{(l+1)} =
          C_W^2 <σ'' σ' σ' σ> (H^{(l)})^2
        + C_W χ_perp^{(l)} <σ'' σ> B^{(l)}
        + [ C_W χ_perp^{(l)} <σ'' σ> + (χ_perp^{(l)})^2 ] P^{(l)}

    Shorthand (defined inside):
      χ_perp^{(l)} = C_W <σ' σ'>_{K^{(l)}}
    """
    chi_perp = Cw * E_sp_sp

    cw2 = Cw * Cw
    term1 = cw2 * E_dd_spsps * (H_l * H_l)
    term2 = (Cw * chi_perp * E_dd_s) * B_l
    term3 = (Cw * chi_perp * E_dd_s + chi_perp * chi_perp) * P_l

    return term1 + term2 + term3


def Q_update_scalar_single_input(
    Q_l: float,                # Q^{(l)} (scalar)
    H_l: float,                # H^{(l)} (scalar)
    K_l: float,                # K^{(l)} (scalar)
    F_l: float,                # F^{(l)} (scalar)
    F_lp1: float,              # F^{(l+1)} (scalar)
    Cw: float,                 # C_W (constant float)
    lambda_next: float,        # λ_W^{(l+1)} (constant float)
    E_sp_sp: float,            # <σ' σ'>_{K^{(l)}} (scalar) (used to build chi_perp)
    E_sp_s_z: float,           # <σ' σ z>_{K^{(l)}} (scalar) (used to build chi_parallel)
    E_dd_sp_z: float,          # <σ'' σ' z>_{K^{(l)}} (scalar) (used to build h)
    E_dd_s: float,             # <σ'' σ>_{K^{(l)}} (scalar)
    E_dd_spsps: float,         # <σ'' σ' σ' σ>_{K^{(l)}} (scalar)
) -> float:
    """
    Single-input recursion:
      Q^{(l+1)} =
          C_W^2 <σ'' σ' σ' σ> (H^{(l)})^2
        + (λ_W^{(l+1)}/C_W) F^{(l+1)}
        + 2 h^{(l)} χ_parallel^{(l)} H^{(l)} F^{(l)}
        + [ C_W χ_perp^{(l)} <σ'' σ> + (χ_perp^{(l)})^2 ] Q^{(l)}

    Shorthands (defined inside, with expectations MULTIPLYING the fractions):
      χ_parallel^{(l)} = (C_W / K^{(l)}) <σ' σ z>_{K^{(l)}}
      χ_perp^{(l)}     = C_W <σ' σ'>_{K^{(l)}}
      h^{(l)}          = (C_W / (2 K^{(l)})) <σ'' σ' z>_{K^{(l)}}
    """
    chi_perp = Cw * E_sp_sp
    chi_parallel = (Cw / K_l) * E_sp_s_z
    h = (Cw / (2.0 * K_l)) * E_dd_sp_z

    cw2 = Cw * Cw
    term1 = cw2 * E_dd_spsps * (H_l * H_l)
    term2 = (lambda_next / Cw) * F_lp1
    term3 = 2.0 * h * chi_parallel * H_l * F_l
    term4 = (Cw * chi_perp * E_dd_s + chi_perp * chi_perp) * Q_l

    return term1 + term2 + term3 + term4


# -----------------------------
# Helpers for "scalar as tensor" (D=1 trivial tensors)
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
    lam = float(torch.rand(()).item() + 0.5)

    # hidden layer step where n_in is n_out
    n_l = 12
    n_lm1 = 12

    H_l = float(torch.rand(()).item() + 0.5)
    K_l = float(torch.rand(()).item() + 0.5)

    P_l = float(torch.randn(()).item())
    Q_l = float(torch.randn(()).item())
    B_l = float(torch.randn(()).item())
    F_l = float(torch.randn(()).item())
    F_lp1 = float(torch.randn(()).item())

    # expectations
    E_sp_sp    = float(torch.rand(()).item() + 0.1)   # <σ' σ'>
    E_dd_s     = float(torch.randn(()).item())        # <σ'' σ>
    E_dd_spsps = float(torch.randn(()).item())        # <σ'' σ' σ' σ>

    E_sp_s_z   = float(torch.randn(()).item())        # <σ' σ z>
    E_dd_sp_z  = float(torch.randn(()).item())        # <σ'' σ' z>

    P_next_scalar = P_update_scalar_single_input(
        P_l=P_l, H_l=H_l, B_l=B_l,
        Cw=Cw,
        E_sp_sp=E_sp_sp,
        E_dd_s=E_dd_s,
        E_dd_spsps=E_dd_spsps
    )

    Q_next_scalar = Q_update_scalar_single_input(
        Q_l=Q_l, H_l=H_l, K_l=K_l,
        F_l=F_l, F_lp1=F_lp1,
        Cw=Cw, lambda_next=lam,
        E_sp_sp=E_sp_sp,
        E_sp_s_z=E_sp_s_z,
        E_dd_sp_z=E_dd_sp_z,
        E_dd_s=E_dd_s,
        E_dd_spsps=E_dd_spsps
    )
    Ginv_1 = t1(1.0 / K_l, dtype=dtype, device=device)

    # P_update_zabc:
    #   P_l[z,a,b,c], H[z,a], B[i,j,a,b] (uses B[z,z,a,b]),
    #   E_ddp_p_s_zabc[z,a,b,c], E_p[i,j], E_dd_zc[z,c]
    P_next_tensor = P_update_zabc(
        P_l=t4(P_l, dtype=dtype, device=device),
        H=t1(H_l, dtype=dtype, device=device),
        B=t4(B_l, dtype=dtype, device=device),
        E_ddp_p_s_zabc=t4(E_dd_spsps, dtype=dtype, device=device),
        E_p=t1(E_sp_sp, dtype=dtype, device=device),
        E_dd_zc=t1(E_dd_s, dtype=dtype, device=device),
        Cw=Cw,
        n_l=n_l,
        n_lm1=n_lm1,
    ).item()

    # Q_update_zabc:
    #   Q_l[z,a,b,c], H[z,a], F_lp1[a,z,c,b], F_l[f,z,g,b], Ginv[d,f]
    #   E_ddp_p_s_zabc[z,a,b,c], E_p[i,j], E_dd_bc[b,c],
    #   E_z_ddp_dza[d,z,a], E_z_p_s_ebc[e,b,c]
    Q_next_tensor = Q_update_zabc(
        Q_l=t4(Q_l, dtype=dtype, device=device),
        H=t1(H_l, dtype=dtype, device=device),
        F_lp1=t4(F_lp1, dtype=dtype, device=device),
        F_l=t4(F_l, dtype=dtype, device=device),
        Ginv=Ginv_1,

        E_ddp_p_s_zabc=t4(E_dd_spsps, dtype=dtype, device=device),
        E_p=t1(E_sp_sp, dtype=dtype, device=device),
        E_dd_bc=t1(E_dd_s, dtype=dtype, device=device),
        E_z_ddp_dza=t3(E_dd_sp_z, dtype=dtype, device=device),
        E_z_p_s_ebc=t3(E_sp_s_z, dtype=dtype, device=device),

        Cw=Cw,
        LambdaW=lam,
        n_l=n_l,
        n_lm1=n_lm1,
    ).item()

    results = {
        "P": (P_next_scalar, P_next_tensor),
        "Q": (Q_next_scalar, Q_next_tensor),
    }

    all_ok = True
    for name, (sval, tval) in results.items():
        ok = close(float(sval), float(tval), tol=tol)
        all_ok = all_ok and ok
        print(f"{name}: single input vals={sval:.12g} multi-input vals={tval:.12g}  absolute error={abs(sval-tval):.3e}  thresh_reached={ok}", flush=True)

    assert all_ok, "Oopsies"