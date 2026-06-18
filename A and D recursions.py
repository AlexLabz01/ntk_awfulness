import numpy as np
import torch

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

    # Ω̂_{cd} mean and mixed moments
    Omega_mean_cd = lam * E_ss + cw * (H * E_pp)                         # (c,d)
    SigmaSigma_Omega = lam * E_ssss_abcd + cw * torch.einsum("abcd,cd->abcd", E_sspp_abcd, H)

    term1 = cw * (SigmaSigma_Omega - torch.einsum("ab,cd->abcd", E_ss, Omega_mean_cd))

    # centered blocks
    X_pqab = E_zz_ss_pqab - torch.einsum("pq,ab->pqab", G, E_ss)

    Zss_pqcd = E_zz_ss_pqcd - torch.einsum("pq,cd->pqcd", G, E_ss)
    Zpp_pqcd = E_zz_pp_pqcd - torch.einsum("pq,cd->pqcd", G, E_pp)
    U_pqcd = lam * Zss_pqcd + cw * torch.einsum("pqcd,cd->pqcd", Zpp_pqcd, H)   # <(zz-G) Ω̂_cd>

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


# single input recursions

def single_compact_step_A_D(
    A_l: float,
    D_l: float,
    V_l: float,
    G_l: float,
    H_l: float,
    Cw: float,
    lam: float,
    # expectations
    E_ss: float,         # g=<σ^2>
    E_ssss: float,       # <σ^4>
    E_pp: float,         # <(σ')^2>
    E_pppp: float,       # <(σ')^4>
    E_sspp: float,       # <σ^2 (σ')^2>
    E_z_sp_s: float,     # <z σ' σ>
    E_pp_z2_minus_G: float,  # <(σ')^2 (z^2 - G)>
):
    #shorcuts
    chi_perp = Cw * E_pp
    chi_par  = (Cw / G_l) * E_z_sp_s
    h        = (Cw / (4.0 * G_l * G_l)) * E_pp_z2_minus_G
    g        = E_ss

    # V recursion
    V_lp1 = (chi_par * chi_par) * V_l + (Cw * Cw) * (E_ssss - g * g)

    # D recursion
    D_lp1 = (chi_perp * chi_par) * D_l + (lam / Cw) * V_lp1 \
            + H_l * ((Cw * Cw) * E_sspp - Cw * g * chi_perp + 2.0 * h * chi_par * V_l)

    # A recursion
    A_lp1 = (chi_perp * chi_perp) * A_l \
            - (lam / Cw) ** 2 * V_lp1 \
            + 2.0 * (lam / Cw) * D_lp1 \
            + 4.0 * h * chi_perp * H_l * D_l \
            + (H_l * H_l) * ((Cw * Cw) * E_pppp - (chi_perp * chi_perp) + (2.0 * h) ** 2 * V_l)

    return A_lp1, D_lp1, V_lp1


# expectations for tan

def gh_expectations_tanh(G: float, n: int = 200):
    xs, ws = np.polynomial.hermite.hermgauss(n)
    z = np.sqrt(2.0 * G) * xs
    w = ws / np.sqrt(np.pi)

    s = np.tanh(z)
    sp = 1.0 - s**2

    E_ss    = float(np.sum(w * (s*s)))
    E_ssss  = float(np.sum(w * (s**4)))
    E_pp    = float(np.sum(w * (sp*sp)))
    E_pppp  = float(np.sum(w * ((sp*sp)**2)))
    E_sspp  = float(np.sum(w * ((s*s) * (sp*sp))))

    E_z_sp_s = float(np.sum(w * (z * sp * s)))
    E_pp_z2_minus_G = float(np.sum(w * ((sp*sp) * (z*z - G))))

    E_z2_ss = float(np.sum(w * ((z*z) * (s*s))))
    E_z2_pp = float(np.sum(w * ((z*z) * (sp*sp))))

    return dict(
        E_ss=E_ss, E_ssss=E_ssss, E_pp=E_pp, E_pppp=E_pppp, E_sspp=E_sspp,
        E_z_sp_s=E_z_sp_s, E_pp_z2_minus_G=E_pp_z2_minus_G,
        E_z2_ss=E_z2_ss, E_z2_pp=E_z2_pp
    )



def t1(x, dtype=torch.float64, device="cpu"):
    return torch.full((1, 1), float(x), dtype=dtype, device=device)

def t4(x, dtype=torch.float64, device="cpu"):
    return torch.full((1, 1, 1, 1), float(x), dtype=dtype, device=device)

def close(a, b, tol=1e-7):
    return abs(a-b) <= tol * max(1.0, abs(a), abs(b))


if __name__ == "__main__":
    device = "cpu"
    dtype = torch.float64
    tol = 1e-6

    torch.manual_seed(0)

    # parameters
    Cw = 1.2
    lam = 0.7
    H_l = 0.9
    G_l = 1.1

    n_l = 32
    n_lm1 = 32

    A_l = 0.20
    D_l = -0.05
    V_l = 0.15

    # taking gpt's suggestion for consistent expectations under N(0,G_l)
    ex = gh_expectations_tanh(G_l, n=250)

    # single inpt
    A_lp1_s, D_lp1_s, V_lp1_s = single_compact_step_A_D(
        A_l=A_l, D_l=D_l, V_l=V_l,
        G_l=G_l, H_l=H_l, Cw=Cw, lam=lam,
        E_ss=ex["E_ss"], E_ssss=ex["E_ssss"],
        E_pp=ex["E_pp"], E_pppp=ex["E_pppp"], E_sspp=ex["E_sspp"],
        E_z_sp_s=ex["E_z_sp_s"], E_pp_z2_minus_G=ex["E_pp_z2_minus_G"]
    )

    H = t1(H_l, dtype=dtype, device=device)
    G = t1(G_l, dtype=dtype, device=device)
    Ginv = t1(1.0 / G_l, dtype=dtype, device=device)

    A = t4(A_l, dtype=dtype, device=device)
    D = t4(D_l, dtype=dtype, device=device)

    E_ss = t1(ex["E_ss"], dtype=dtype, device=device)
    E_pp = t1(ex["E_pp"], dtype=dtype, device=device)

    E_ssss_abcd = t4(ex["E_ssss"], dtype=dtype, device=device)
    E_pppp_abcd = t4(ex["E_pppp"], dtype=dtype, device=device)
    E_sspp_abcd = t4(ex["E_sspp"], dtype=dtype, device=device)
    E_ppss_abcd = t4(ex["E_sspp"], dtype=dtype, device=device)
    E_zz_ss = t4(ex["E_z2_ss"], dtype=dtype, device=device)
    E_zz_pp = t4(ex["E_z2_pp"], dtype=dtype, device=device)

    # V_scalar = (n_l/n_{l-1}) * G^4 * V^{0000} so V^{0000} = V_scalar / ((n_l/n_{l-1})*G^4)
    V0000 = V_l / ((n_l / n_lm1) * (G_l ** 4))
    V_raised = t4(V0000, dtype=dtype, device=device)

    # multi inpt
    D_lp1_t = D_update_abcd(
        D_l=D,
        V_raised=V_raised,
        H=H, G=G, Ginv=Ginv,
        E_ssss_abcd=E_ssss_abcd,
        E_sspp_abcd=E_sspp_abcd,
        E_ss=E_ss,
        E_pp=E_pp,
        E_zz_ss_pqab=E_zz_ss,
        E_zz_ss_pqcd=E_zz_ss,
        E_zz_pp_pqcd=E_zz_pp,
        Cw_next=Cw, Lambda_next=lam,
        n_l=n_l, n_lm1=n_lm1
    ).item()

    A_lp1_t = A_update_abcd(
        A_l=A, D_l=D,
        V_raised=V_raised,
        H=H, G=G, Ginv=Ginv,
        E_ssss_abcd=E_ssss_abcd,
        E_sspp_abcd=E_sspp_abcd,
        E_ppss_abcd=E_ppss_abcd,
        E_pppp_abcd=E_pppp_abcd,
        E_ss=E_ss, E_pp=E_pp,
        E_zz_ss_pqab=E_zz_ss,
        E_zz_pp_pqab=E_zz_pp,
        E_zz_ss_pqcd=E_zz_ss,
        E_zz_pp_pqcd=E_zz_pp,
        Cw_next=Cw, Lambda_next=lam,
        n_l=n_l, n_lm1=n_lm1
    ).item()

    print(f"Single inpt:  D_l={D_lp1_s:.12g}   A_l={A_lp1_s:.12g}   V_l={V_lp1_s:.12g}")
    print(f"Multi inpt:     D_l={D_lp1_t:.12g}   err={abs(D_lp1_s-D_lp1_t):.3e}")
    print(f"Multi inpt:     A_l={A_lp1_t:.12g}   err={abs(A_lp1_s-A_lp1_t):.3e}")

    assert close(D_lp1_s, D_lp1_t, tol=tol), "D is no bueno"
    assert close(A_lp1_s, A_lp1_t, tol=tol), "A is no bueno"