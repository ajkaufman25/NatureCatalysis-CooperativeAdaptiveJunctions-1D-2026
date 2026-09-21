#!/usr/bin/env python3
"""Validate the buried/buried (BB) operating point and export spatial profiles.

This script is intentionally downstream of BB_schottky_loadline_solver.py.

Workflow
--------
1. Recompute the illuminated BB photovoltaic open-circuit voltage.
2. Recompute the HOR/ORR load-line intersection.
3. Gauge-shift the load-line BVP to the *actual* catalyst potentials U_H and U_O.
4. Re-solve the full Poisson + electron/hole drift-diffusion + SRH BVP at those
   absolute potentials to verify that the operating point is unchanged.
5. Export band, quasi-Fermi-level, carrier-density, current, generation, and
   recombination profiles and publication-quality diagnostic plots.

Required files in the same directory
------------------------------------
- cooperative_adaptive_junction_simulator.py
- BB_schottky_loadline_solver.py
"""

from pathlib import Path
import importlib.util
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location(
    "bbfocus_validate", ROOT / "BB_schottky_loadline_solver.py"
)
bb = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bb
spec.loader.exec_module(bb)


def shifted_seed(model, sol, common_shift_V):
    """Apply the exact fixed-barrier gauge transformation to a BVP solution."""
    y = sol.y.copy()
    y[0] += common_shift_V / model.VT
    return bb.base._Seed(sol.x, y)


def profile_dataframe(model, sol):
    """Return physical spatial profiles on a dense uniform grid."""
    s,n,p,Ucb,Uvb,Ufn,Ufp,Jn,Jp,N,P,d = model.profiles(sol, npts=2401)
    R = model.srh_over_Gref(N, P) * model.Gref
    G = np.full_like(s, model.Gref)
    return pd.DataFrame({
        "x_um": s * model.p.L_s * 1e6,
        "n_cm3": n / 1e6,
        "p_cm3": p / 1e6,
        "U_CB_V_vs_RHE": Ucb,
        "U_VB_V_vs_RHE": Uvb,
        "U_Fn_V_vs_RHE": Ufn,
        "U_Fp_V_vs_RHE": Ufp,
        # Conventional semiconductor-energy convention: electron energy = -U.
        "E_C_eV": -Ucb,
        "E_V_eV": -Uvb,
        "E_Fn_eV": -Ufn,
        "E_Fp_eV": -Ufp,
        "Jn_mA_cm2": Jn / 10.0,
        "Jp_mA_cm2": Jp / 10.0,
        "Jtotal_mA_cm2": (Jn + Jp) / 10.0,
        "G_cm3_s": G / 1e6,
        "R_SRH_cm3_s": R / 1e6,
    })


def main():
    out = ROOT / "bb_profile_validation_output"
    out.mkdir(exist_ok=True)

    m = bb.BBSchottkyDevice()
    s = bb.BBLoadLineSolver(m)

    # Independent recomputation of the two-terminal BB photovoltaic element.
    voc, vocsol_gauge, _ = s.solve_voc()

    # Independent recomputation of the electrochemical load-line intersection.
    result, opsol_gauge = s.solve_operating_point(voc=voc)
    UH, UO = float(result["U_H_V"]), float(result["U_O_V"])

    # Critical validation: re-solve at the actual absolute catalyst potentials.
    opseed = shifted_seed(m, opsol_gauge, UH)
    opsol = m.solve_state(UH, UO, 1.0, previous=opseed, tol=1e-8, nmesh=1800)
    if opsol.status != 0:
        raise RuntimeError(opsol.message)
    opdg = m.diagnostics(opsol, UH, UO, 1.0)

    # Open circuit in the same absolute gauge for a clean profile comparison.
    UH_voc = UH
    UO_voc = UH + voc
    vocseed = shifted_seed(m, vocsol_gauge, UH_voc)
    vocsol = m.solve_state(UH_voc, UO_voc, 1.0, previous=vocseed,
                           tol=1e-8, nmesh=1800)
    if vocsol.status != 0:
        raise RuntimeError(vocsol.message)
    vocdg = m.diagnostics(vocsol, UH_voc, UO_voc, 1.0)

    opdf = profile_dataframe(m, opsol)
    vocdf = profile_dataframe(m, vocsol)

    # Exact gauge-invariance comparison against the load-line gauge U_H=0.
    _,ng,pg,Ucbg,_,_,_,Jng,Jpg,*_ = m.profiles(opsol_gauge, npts=2401)
    _,na,pa,Ucba,_,_,_,Jna,Jpa,*_ = m.profiles(opsol, npts=2401)

    Irev_mA_cm2 = -opdg["Jsem_mA_cm2"]
    summary = {
        "Voc_V": voc,
        "operating_current_uA_cm2": opdg["Jsem_A_m2"]/1e4*1e6,
        "reverse_current_magnitude_uA_cm2": -opdg["Jsem_A_m2"]/1e4*1e6,
        "U_H_V": UH,
        "U_O_V": UO,
        "U_O_minus_U_H_V": UO-UH,
        "V_minus_Voc_mV": 1000*((UO-UH)-voc),
        "j_H_uA_cm2": opdg["jH_A_m2"]/1e4*1e6,
        "j_O_uA_cm2": opdg["jO_A_m2"]/1e4*1e6,
        "current_balance_H_A_m2": opdg["Jsem_A_m2"]+opdg["jH_A_m2"],
        "current_balance_O_A_m2": -opdg["Jsem_A_m2"]+opdg["jO_A_m2"],
        "U_CB_H_V": opdg["Ucb_H_V"],
        "U_CB_O_V": opdg["Ucb_O_V"],
        "surface_UCB_difference_mV": 1000*(opdg["Ucb_O_V"]-opdg["Ucb_H_V"]),
        "band_bending_H_V": opdg["band_bending_H_V"],
        "band_bending_O_V": opdg["band_bending_O_V"],
        "U_CB_bulk_V": opdg["Ucb_bulk_V"],
        "QFL_center_split_V": opdg["QFL_center_split_V"],
        "Jn_H_mA_cm2": opdg["Jn_H_mA_cm2"],
        "Jp_H_mA_cm2": opdg["Jp_H_mA_cm2"],
        "Jn_O_mA_cm2": opdg["Jn_O_mA_cm2"],
        "Jp_O_mA_cm2": opdg["Jp_O_mA_cm2"],
        "Jgen_mA_cm2": opdg["Jgen_mA_cm2"],
        "SRH_recombination_mA_cm2": opdg["Jrec_mA_cm2"],
        "interface_counterflow_mA_cm2": opdg["counterflow_mA_cm2"],
        "reverse_current_budget_mA_cm2": Irev_mA_cm2,
        "budget_residual_mA_cm2":
            opdg["Jgen_mA_cm2"] -
            (opdg["Jrec_mA_cm2"] + opdg["counterflow_mA_cm2"] + Irev_mA_cm2),
        "max_total_current_span_A_m2": float(np.ptp(Jna+Jpa)),
        "BVP_max_rms_residual": opdg["max_BVP_rms_residual"],
        "gauge_invariance_max_rel_n":
            float(np.max(np.abs(na-ng)/np.maximum(ng,1e-300))),
        "gauge_invariance_max_rel_p":
            float(np.max(np.abs(pa-pg)/np.maximum(pg,1e-300))),
        "gauge_invariance_max_abs_Jn_A_m2": float(np.max(np.abs(Jna-Jng))),
        "gauge_invariance_max_abs_Jp_A_m2": float(np.max(np.abs(Jpa-Jpg))),
        "gauge_invariance_max_UCB_shift_error_V":
            float(np.max(np.abs((Ucba-Ucbg)-UH))),
        "Voc_recheck_current_uA_cm2": vocdg["Jsem_A_m2"]/1e4*1e6,
        "Voc_surface_UCB_difference_uV":
            1e6*(vocdg["Ucb_O_V"]-vocdg["Ucb_H_V"]),
    }

    pd.DataFrame([summary]).to_csv(out/"BB_validation_summary.csv", index=False)
    opdf.to_csv(out/"BB_operating_profiles.csv", index=False)
    vocdf.to_csv(out/"BB_Voc_profiles.csv", index=False)

    # Plot 1: conventional band diagram at physical BB operating point.
    fig, ax = plt.subplots(figsize=(7.4,5.0))
    ax.plot(opdf.x_um, opdf.E_C_eV, label=r"$E_C$")
    ax.plot(opdf.x_um, opdf.E_V_eV, label=r"$E_V$")
    ax.plot(opdf.x_um, opdf.E_Fn_eV, linestyle="--", label=r"$E_{Fn}$")
    ax.plot(opdf.x_um, opdf.E_Fp_eV, linestyle="--", label=r"$E_{Fp}$")
    ax.scatter([0.0,1.0], [-UH,-UO], marker="s", s=50, label="Metal Fermi levels")
    ax.set_xlabel(r"Position in SrTiO$_3$ ($\mu$m)")
    ax.set_ylabel("Electron energy (eV; higher upward)")
    ax.set_title("BB operating-point band diagram")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out/"BB_band_diagram_operating_point.pdf", bbox_inches="tight")
    fig.savefig(out/"BB_band_diagram_operating_point.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: conduction band at Voc versus loaded state in the same gauge.
    fig, ax = plt.subplots(figsize=(7.4,4.8))
    ax.plot(vocdf.x_um, vocdf.E_C_eV, label=r"$E_C$ at $V_{\mathrm{OC}}$")
    ax.plot(opdf.x_um, opdf.E_C_eV, linestyle="--",
            label=r"$E_C$ at loaded BB state")
    ax.set_xlabel(r"Position in SrTiO$_3$ ($\mu$m)")
    ax.set_ylabel(r"$E_C$ (eV)")
    ax.set_title("BB conduction-band profile: open circuit vs reverse-loaded state")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out/"BB_band_profile_Voc_vs_operating.pdf", bbox_inches="tight")
    fig.savefig(out/"BB_band_profile_Voc_vs_operating.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Plot 3: individual and total current.
    fig, ax = plt.subplots(figsize=(7.4,4.8))
    ax.plot(opdf.x_um, opdf.Jn_mA_cm2, label=r"$J_n$")
    ax.plot(opdf.x_um, opdf.Jp_mA_cm2, label=r"$J_p$")
    ax.plot(opdf.x_um, opdf.Jtotal_mA_cm2, linestyle="--", label=r"$J_n+J_p$")
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel(r"Position in SrTiO$_3$ ($\mu$m)")
    ax.set_ylabel(r"Current density (mA cm$^{-2}$)")
    ax.set_title("BB operating-point carrier currents")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out/"BB_current_profiles.pdf", bbox_inches="tight")
    fig.savefig(out/"BB_current_profiles.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Plot 4: carrier concentrations.
    fig, ax = plt.subplots(figsize=(7.4,4.8))
    ax.semilogy(opdf.x_um, opdf.n_cm3, label=r"$n$")
    ax.semilogy(opdf.x_um, opdf.p_cm3, label=r"$p$")
    ax.set_xlabel(r"Position in SrTiO$_3$ ($\mu$m)")
    ax.set_ylabel(r"Carrier concentration (cm$^{-3}$)")
    ax.set_title("BB operating-point carrier concentrations")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out/"BB_carrier_profiles.pdf", bbox_inches="tight")
    fig.savefig(out/"BB_carrier_profiles.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Plot 5: bulk generation and SRH recombination.
    fig, ax = plt.subplots(figsize=(7.4,4.8))
    ax.plot(opdf.x_um, opdf.G_cm3_s, label="Generation")
    ax.plot(opdf.x_um, opdf.R_SRH_cm3_s, label="SRH recombination")
    ax.set_xlabel(r"Position in SrTiO$_3$ ($\mu$m)")
    ax.set_ylabel(r"Volumetric rate (cm$^{-3}$ s$^{-1}$)")
    ax.set_title("BB generation and bulk recombination")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out/"BB_generation_recombination.pdf", bbox_inches="tight")
    fig.savefig(out/"BB_generation_recombination.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(pd.DataFrame([summary]).T.to_string(header=False))


if __name__ == "__main__":
    main()
