"""SI-facing data schema: energies in eV, voltages in V and explicit signs.

No physical equation or fitted parameter is changed by this exporter. Legacy
U/MIEC/counterflow columns are confined to data/reference and provenance.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'data/reference'


def state_row(model: Any, sol: Any, V_HER: float, V_OER: float,
              architecture: str, *, light_fraction: float = 1.0,
              faradaic_active: bool = True) -> dict:
    """One scalar summary from a converged state using exact SI definitions."""
    s,n,p,Uc,Uv,Un,Up,Jn,Jp,N,P,d=model.profiles(sol,2401)
    w=model.facet.central_bulk_window
    Ebulk=float(np.mean(-Uc[(s>=.5-w)&(s<=.5+w)]))
    Jsem=float((Jn+Jp)[0]); Jcontact=float(-Jp[0]-Jn[-1])
    Jsrh=float(np.trapezoid(model.srh_over_Gref(N,P),s)*model.Jscale)
    Jgen=light_fraction*model.Jscale
    jH,jO,etaH,etaO=model.faradaic_currents_A_m2(V_HER,V_OER)
    if not faradaic_active:jH=jO=0.0
    eH,eO=-float(V_HER),-float(V_OER)
    return dict(
        architecture=architecture,
        state=('finite_current' if faradaic_active else
               ('dark_equilibrium' if light_fraction==0 else 'illuminated_catalyst_off')),
        faradaic_active=faradaic_active,
        V_cat_HER_V_vs_RHE=float(V_HER), V_cat_OER_V_vs_RHE=float(V_OER),
        E_cat_HER_eV=eH, E_cat_OER_eV=eO,
        Delta_V_cat_V=float(V_OER-V_HER), Delta_E_cat_eV=eH-eO,
        eta_HER_V=float(etaH) if faradaic_active else np.nan,
        eta_OER_V=float(etaO) if faradaic_active else np.nan,
        J_sem_mA_cm2=Jsem/10,
        J_OWS_mA_cm2=Jsem/10 if faradaic_active and Jsem>0 else np.nan,
        J_gen_mA_cm2=Jgen/10,
        J_contact_mA_cm2=Jcontact/10 if Jsem>=-1e-10 else np.nan,
        signed_opposite_carrier_term_mA_cm2=Jcontact/10,
        J_SRH_mA_cm2=Jsrh/10,
        E_CB_s_HER_eV=float(-Uc[0]), E_CB_s_OER_eV=float(-Uc[-1]),
        E_CB_bulk_eV=Ebulk,
        V_bb_HER_V=float(-Uc[0]-Ebulk), V_bb_OER_V=float(-Uc[-1]-Ebulk),
        Delta_E_CB_surface_eV=float(Uc[0]-Uc[-1]),
        Phi_B_HER_eV=float(-Uc[0]-eH), Phi_B_OER_eV=float(-Uc[-1]-eO),
        Delta_E_F_center_eV=float(Up[len(s)//2]-Un[len(s)//2]),
        j_HER_anodic_mA_cm2=float(jH/10), j_OER_anodic_mA_cm2=float(jO/10),
        catalyst_balance_HER_A_m2=float(Jsem+jH),
        catalyst_balance_OER_A_m2=float(-Jsem+jO),
        integrated_current_balance_A_m2=float(Jgen-Jsem-Jcontact-Jsrh),
        BVP_rms_max=float(np.max(sol.rms_residuals)),
        adaptive_mesh_nodes=len(sol.x))


def convert_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert a stored voltage-coordinate profile, without interpolation."""
    out=pd.DataFrame({'x_um':frame.x_um})
    for new,old in [('E_CB_eV','U_CB_V_vs_RHE'),('E_VB_eV','U_VB_V_vs_RHE'),
                    ('E_Fn_eV','U_Fn_V_vs_RHE'),('E_Fp_eV','U_Fp_V_vs_RHE')]:
        out[new]=-frame[old]
    for c in ['n_cm3','p_cm3','Jn_mA_cm2','Jp_mA_cm2','Jtotal_mA_cm2']:
        out[c]=frame[c]
    return out


def convert_sweep(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert source table names and energy-difference numerical values.

    A source QFL splitting in V is (E_Fn-E_Fp)/q. Its numeric eV value
    is the same, with NO sign reversal. Catalyst separation is OER-HER
    in voltage, which equals HER-OER in electron energy divided by q.
    """
    mapping={
        'mean_bending_V':'mean_V_bb_0_V', 'asymmetry_V':'Delta_V_bb_0_V',
        'k':'k_m_s', 'MIEC_separation_V':'Delta_V_cat_V',
        'contact_separation_V':'Delta_V_cat_V',
        'QFL_split_V':'Delta_E_F_center_eV',
        'QFL_center_split_V':'Delta_E_F_center_eV',
        'Jgen_mA_cm2':'J_gen_mA_cm2', 'counterflow_mA_cm2':'J_contact_mA_cm2',
        'SRH_mA_cm2':'J_SRH_mA_cm2',
        'band_bending_H_V':'V_bb_HER_V', 'band_bending_O_V':'V_bb_OER_V',
        'U_H_V_vs_RHE':'V_cat_HER_V_vs_RHE','U_O_V_vs_RHE':'V_cat_OER_V_vs_RHE',
        'U_H_V':'V_cat_HER_V_vs_RHE','U_O_V':'V_cat_OER_V_vs_RHE',
        'eta_H_V':'eta_HER_V', 'eta_O_V':'eta_OER_V',
        'current_budget_residual_A_m2':'integrated_current_balance_A_m2',
        'BVP_residual':'BVP_rms_max', 'final_BVP_residual':'BVP_rms_max',
        'band_bending_H_input_V':'V_bb_HER_0_V',
        'band_bending_O_input_V':'V_bb_OER_0_V',
    }
    out=frame.copy()
    # Some upstream tables carry both a canonical name and its older alias.
    for old,new in mapping.items():
        if old not in out or old==new:continue
        if new in out:
            if not np.allclose(out[old],out[new],equal_nan=True):
                raise ValueError(f'Conflicting source aliases: {old}, {new}')
            out=out.drop(columns=old)
        else:out=out.rename(columns={old:new})
    if 'Delta_V_cat_V' in out:out['Delta_E_cat_eV']=out.Delta_V_cat_V
    for side in ['HER','OER']:
        col=f'V_cat_{side}_V_vs_RHE'
        if col in out:out[f'E_cat_{side}_eV']=-out[col]
    if 'current_budget_residual_mA_cm2' in out:
        out['integrated_current_balance_A_m2']=10*out['current_budget_residual_mA_cm2']
    if 'eta_H_abs_V' in out and 'eta_HER_V' not in out:
        out['eta_HER_V']=-out['eta_H_abs_V']
    # Frozen-redox initial guesses are numerical diagnostics, not the SI dark
    # equilibrium. Retain them only in the unchanged reference CSVs.
    out=out.drop(columns=[c for c in out if c.startswith('initial_') or
                         c in ('eta_H_abs_V','current_budget_residual_mA_cm2')])
    return out


def selected_catalyst_off_values(si_file: Path | None=None) -> pd.DataFrame:
    """Read the four displayed selections directly from SI Table S8.

    The SI provides their separations, but no common-mode selection rule
    for AA/AB/BA. These display values are never input to an OWS solver.
    They must not be called freshly recomputed or unique open-circuit states.
    """
    import re
    si_file=si_file or ROOT/'docs/NatureCatalysis_SI_energy_voltage_BA_clarified.tex'
    text=si_file.read_text()
    line=next(l for l in text.splitlines() if l.startswith('Selected catalyst-off'))
    values=[float(x.strip().replace('\\\\','')) for x in line.split('&')[1:]]
    assert len(values)==4
    return pd.DataFrame([
        dict(architecture=mode,selected_Delta_V_cat_V=value,
             provenance='SI Table S8; prescribed comparison value' if mode!='BB' else 'SI Table S8; BB barrier-difference voltage',
             selection_rule_specified=(mode=='BB'),
             numerical_state_recomputed=False)
        for mode,value in zip(['AA','AB','BA','BB'],values)])


def export_reference_tables(outdir: Path, reference_dir: Path=REFERENCE) -> None:
    """Create SI-named views of unchanged stored tables (no BVP solves)."""
    outdir.mkdir(parents=True,exist_ok=True)
    for name in ['native_band_bending_sweep.csv','common_transfer_rate_sweep.csv','intensity_sweep.csv']:
        convert_sweep(pd.read_csv(reference_dir/name)).to_csv(outdir/name,index=False)
    t=convert_sweep(pd.read_csv(reference_dir/'thickness/AA_thickness_sweep_combined_dense.csv'))
    t.to_csv(outdir/'AA_thickness_sweep.csv',index=False)
    # The current SI figure uses the 31-point global grid only.
    t[t.source=='full'].to_csv(outdir/'AA_thickness_plot_data.csv',index=False)
    selected_catalyst_off_values().to_csv(outdir/'selected_catalyst_off_states.csv',index=False)
