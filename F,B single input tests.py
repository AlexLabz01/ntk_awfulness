import torch


def B_update_acbd(
    B_l: torch.Tensor,          # B^{(l)}_{a c b d}
    H_l: torch.Tensor,          # H^{(l)}_{i j}
    E_pppp_abcd: torch.Tensor,  # <σ'_a σ'_b σ'_c σ'_d>_{G^{(l)}}   (a,b,c,d)
    E_p_ac: torch.Tensor,       # <σ'_a σ'_c>_{G^{(l)}}             (a,c)
    Cw_next: float,             # C_W^{(l+1)} (constant float)
    n_l: int,
    n_lm1: int,
    E_p_bd: torch.Tensor,  # <σ'_b σ'_d>_{G^{(l)}}
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



def D_update_scalar_single_input(
    D_l: float,
    V_l: float,
    H_l: float,
    K_l: float,
    Cw: float,
    lambda_next: float,
    E_sigma_sigma_z: float,        # <σ σ z>   (if you need it later; included for completeness)
    E_sp_s_z: float,               # <σ' σ z>
    E_dd_sp_z: float,              # <σ'' σ' z>
    E_ssss: float,                 # <σ σ σ σ>
    E_ss: float,                   # <σ σ>
    E_sspp: float,                 # <σ σ σ' σ'>
) -> float:
    """
    Single-input recursion (no multi-input comparison yet, as requested):

    D^{(l+1)} =
        χ_perp^{(l)} χ_parallel^{(l)} D^{(l)}
      + (λ_W/C_W) * [ C_W^2 <σ σ σ σ> - (C_W <σ σ>)^2 + (χ_parallel^{(l)})^2 V^{(l)} ]
      + H^{(l)} * [ C_W^2 <σ σ σ' σ'> - C_W <σ σ> χ_perp^{(l)} + 2 h^{(l)} χ_parallel^{(l)} V^{(l)} ]

    Shorthands (expectations MULTIPLY the fractions):
      χ_parallel^{(l)} = (C_W / K^{(l)}) <σ' σ z>
      χ_perp^{(l)}     = C_W <σ' σ'>
      h^{(l)}          = (C_W / (2 K^{(l)})) <σ'' σ' z>
    """
    chi_parallel = (Cw / K_l) * E_sp_s_z
    chi_perp = Cw * 0.0
    h = (Cw / (2.0 * K_l)) * E_dd_sp_z

    term1 = (chi_perp * chi_parallel) * D_l

    bracket_lambda = (Cw * Cw) * E_ssss - (Cw * E_ss) ** 2 + (chi_parallel ** 2) * V_l
    term2 = (lambda_next / Cw) * bracket_lambda

    bracket_H = (Cw * Cw) * E_sspp - (Cw * E_ss) * chi_perp + 2.0 * h * chi_parallel * V_l
    term3 = H_l * bracket_H

    return term1 + term2 + term3


def F_update_scalar_single_input(
    F_l: float,                  # F^{(l)}
    H_l: float,                  # H^{(l)}
    K_l: float,                  # K^{(l)}
    Cw: float,                   # C_W
    E_s_sprime_z: float,         # <σ σ' z>_{K^{(l)}}
    E_sspp: float,               # <σ σ σ' σ'>_{K^{(l)}}
) -> float:
    """
    Single-input recursion:
      F^{(l+1)} = (χ_parallel^{(l)})^2 F^{(l)} + C_W^2 <σ σ σ' σ'> H^{(l)}

    Shorthand:
      χ_parallel^{(l)} = (C_W / K^{(l)}) <σ σ' z>_{K^{(l)}}
    """
    chi_parallel = (Cw / K_l) * E_s_sprime_z
    return (chi_parallel * chi_parallel) * F_l + (Cw * Cw) * E_sspp * H_l


def B_update_scalar_single_input(
    B_l: float,                  # B^{(l)}
    H_l: float,                  # H^{(l)}
    Cw: float,                   # C_W
    E_sp_sp: float,              # <σ' σ'>_{K^{(l)}}   (for chi_perp)
    E_pppp: float,               # <σ' σ' σ' σ'>_{K^{(l)}}
) -> float:
    """
    Single-input recursion:
      B^{(l+1)} = (χ_perp^{(l)})^2 B^{(l)} + C_W^2 <σ' σ' σ' σ'> (H^{(l)})^2

    Shorthand:
      χ_perp^{(l)} = C_W <σ' σ'>_{K^{(l)}}
    """
    chi_perp = Cw * E_sp_sp
    return (chi_perp * chi_perp) * B_l + (Cw * Cw) * E_pppp * (H_l * H_l)



def t1(val, dtype=torch.float64, device="cpu"):
    return torch.full((1, 1), float(val), dtype=dtype, device=device)

def t3(val, dtype=torch.float64, device="cpu"):
    return torch.full((1, 1, 1), float(val), dtype=dtype, device=device)

def t4(val, dtype=torch.float64, device="cpu"):
    return torch.full((1, 1, 1, 1), float(val), dtype=dtype, device=device)

def close(a, b, tol=1e-10):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))



if __name__ == "__main__":
    """
    Algebraic match in D=1 with G^{-1} = 1/K:

    Multi-input F_update_abcd term2 in D=1 becomes:
      C_W^2 * (<σ σ' z>)^2 * (1/K)^2 * F
    which equals (χ_parallel)^2 F when χ_parallel = (C_W/K)<σ σ' z>.

    Multi-input B_update_acbd in D=1 becomes:
      C_W^2 * <σ'σ'σ'σ'> * H^2 + C_W^2 * (<σ'σ'>)^2 * B
    which equals C_W^2<E_pppp>H^2 + (χ_perp)^2 B when χ_perp = C_W<σ'σ'>.

    We set n_l == n_lm1 so (n_l/n_lm1)=1 for the comparison.
    """

    device = "cpu"
    dtype = torch.float64
    tol = 1e-10
    torch.manual_seed(0)

    # constants
    Cw = float(torch.rand(()).item() + 0.5)
    K_l = float(torch.rand(()).item() + 0.5)
    H_l = float(torch.rand(()).item() + 0.5)

    # set ratio = 1
    n_l = 64
    n_lm1 = 64

    # base scalars
    F_l = float(torch.randn(()).item())
    B_l = float(torch.randn(()).item())

    # expectations for single-input
    E_s_sprime_z = float(torch.randn(()).item())       # <σ σ' z>
    E_sspp = float(torch.randn(()).item())             # <σ σ σ' σ'>
    E_sp_sp = float(torch.rand(()).item() + 0.1)       # <σ' σ'>
    E_pppp = float(torch.randn(()).item())             # <σ' σ' σ' σ'>

    F_next_single = F_update_scalar_single_input(
        F_l=F_l, H_l=H_l, K_l=K_l, Cw=Cw,
        E_s_sprime_z=E_s_sprime_z,
        E_sspp=E_sspp
    )

    B_next_single = B_update_scalar_single_input(
        B_l=B_l, H_l=H_l, Cw=Cw,
        E_sp_sp=E_sp_sp,
        E_pppp=E_pppp
    )
    Ginv_1 = t1(1.0 / K_l, dtype=dtype, device=device)

    # F_update_abcd
    F_next_multi = F_update_abcd(
        F_l=t4(F_l, dtype=dtype, device=device),
        H=t1(H_l, dtype=dtype, device=device),
        Ginv=Ginv_1,
        E_sspp_abcd=t4(E_sspp, dtype=dtype, device=device),
        E_sps_ace=t3(E_s_sprime_z, dtype=dtype, device=device),
        E_sps_bdf=t3(E_s_sprime_z, dtype=dtype, device=device),
        Cw=Cw,
        n_l=n_l,
        n_lm1=n_lm1
    ).item()

    # B_update_acbd
    #   B_l[a,c,b,d] (D=1 -> 1x1x1x1)
    #   H[a,b] and H[c,d] (both from H_l 1x1)
    #   E_pppp[a,b,c,d]
    #   E_p_ac and E_p_bd are <σ' σ'>
    B_next_multi = B_update_acbd(
        B_l=t4(B_l, dtype=dtype, device=device),
        H_l=t1(H_l, dtype=dtype, device=device),
        E_pppp_abcd=t4(E_pppp, dtype=dtype, device=device),
        E_p_ac=t1(E_sp_sp, dtype=dtype, device=device),
        Cw_next=Cw,
        n_l=n_l,
        n_lm1=n_lm1,
        E_p_bd=None
    ).item()

    results = {
        "F": (F_next_single, F_next_multi),
        "B": (B_next_single, B_next_multi),
    }

    all_ok = True
    for name, (sval, tval) in results.items():
        ok = close(float(sval), float(tval), tol=tol)
        all_ok = all_ok and ok
        print(f"{name}: sing_inpt={sval:.12g} multi={tval:.12g}  absolute_error={abs(sval-tval):.3e}  All_good={ok}", flush=True)

    assert all_ok, "Oopsies"