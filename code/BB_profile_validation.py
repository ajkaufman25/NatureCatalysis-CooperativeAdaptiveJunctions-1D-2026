#!/usr/bin/env python3
"""Validate BB profiles, absolute energy reference and semiconductor gauge invariance.

The focused outer load-line solver shares the publication semiconductor BVP.
This script shifts its V_cat,HER=0 representation to the actual catalyst
potentials, then independently re-solves the BVP at those potentials. At
photovoltaic open circuit Faradaic exchange is disabled. That state and the
reverse HOR/ORR loaded state are separate from finite-current forward OWS.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import BB_schottky_loadline_solver as bb
from publication_data import state_row
from validation_checks import audit_state
ROOT=Path(__file__).resolve().parents[1]


def profile_dataframe(model,sol):
    """Energies in eV with E_RHE=0; densities and rates use labelled cm units."""
    frame=model.energy_profile_dataframe(sol,npts=2401)
    s,n,p,*_=model.profiles(sol,2401)
    frame['G_cm3_s']=model.Gref/1e6
    frame['R_SRH_cm3_s']=model.srh_over_Gref(n/model.nscale,p/model.nscale)*model.Gref/1e6
    return frame


def validate(outdir:Path,plots:bool=True):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    m=bb.BBSchottkyDevice();solver=bb.BBLoadLineSolver(m)
    voc,voc_gauge,_=solver.solve_voc()
    result,op_gauge=solver.solve_operating_point(voc)
    h,o=result['U_H_V'],result['U_O_V']
    # Common shifts preserve the fixed-barrier semiconductor problem. The
    # Faradaic laws retain their absolute reference and are not gauge shifted.
    seed=bb.base._gauge_shift_seed(m,op_gauge,h)
    op=m.solve_state(h,o,1.,previous=seed,tol=1e-8,nmesh=1800)
    check=audit_state(m,op,h,o,bvp_limit=1.1e-8)
    seed_voc=bb.base._gauge_shift_seed(m,voc_gauge,h)
    oc=m.solve_state(h,h+voc,1.,previous=seed_voc,tol=1e-8,nmesh=1800)
    check_oc=audit_state(m,oc,h,h+voc,faradaic_active=False,bvp_limit=1.1e-8)
    actual=profile_dataframe(m,op);gauge=profile_dataframe(m,op_gauge)
    open_circuit=profile_dataframe(m,oc)
    comparisons={}
    for name in ['E_CB_eV','E_VB_eV','E_Fn_eV','E_Fp_eV']:
        # A +h shift in volts is a -h shift in one-electron energies in eV.
        error=float(np.max(np.abs(actual[name]-gauge[name]+h)))
        comparisons[name+'_shift_error']=error
        if error>1e-6:raise RuntimeError('BB energy gauge-invariance check failed: '+name)
    for name in ['n_cm3','p_cm3']:
        error=float(np.max(np.abs(actual[name]/gauge[name]-1)))
        comparisons[name+'_relative_error']=error
        if error>1e-5:raise RuntimeError('BB density gauge-invariance check failed: '+name)
    for name in ['Jn_mA_cm2','Jp_mA_cm2','Jtotal_mA_cm2']:
        error=float(np.max(np.abs(actual[name]-gauge[name])))
        comparisons[name+'_absolute_error']=error
        if error>1e-6:raise RuntimeError('BB current gauge-invariance check failed: '+name)
    rows=[]
    for sol,hh,oo,active,label in [(op,h,o,True,'reverse_HOR_ORR_load_diagnostic'),
                                   (oc,h,h+voc,False,'illuminated_catalyst_off')]:
        row=state_row(m,sol,hh,oo,'BB',faradaic_active=active)
        row.update(state=label,V_OC_V=voc);rows.append(row)
    pd.DataFrame(rows).to_csv(outdir/'BB_validation_states.csv',index=False)
    actual.to_csv(outdir/'BB_reverse_load_profile.csv',index=False)
    open_circuit.to_csv(outdir/'BB_open_circuit_profile.csv',index=False)
    report={'loaded_endpoint':check,'open_circuit_endpoint':check_oc,
            'gauge_invariance':comparisons,'passed':True}
    (outdir/'endpoint_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    if plots:make_plots(actual,open_circuit,rows[0],outdir)
    print(f'BB gauge/profile checks passed; V_OC={voc:.9g} V',flush=True)
    return report


def make_plots(profile,open_circuit,row,outdir):
    """Standalone diagnostic figures; no retired SI numbering is reused."""
    specs=[('BB_band_diagram','Electron energy (eV)',
            [('E_CB_eV',r'$E_{\rm CB}$'),('E_VB_eV',r'$E_{\rm VB}$'),
             ('E_Fn_eV',r'$E_{F,n}$'),('E_Fp_eV',r'$E_{F,p}$')]),
           ('BB_current_profiles',r'Current density (mA cm$^{-2}$)',
            [('Jn_mA_cm2',r'$J_n$'),('Jp_mA_cm2',r'$J_p$'),('Jtotal_mA_cm2',r'$J_n+J_p$')]),
           ('BB_carrier_profiles',r'Carrier density (cm$^{-3}$)',
            [('n_cm3',r'$n$'),('p_cm3',r'$p$')]),
           ('BB_generation_recombination',r'Volumetric rate (cm$^{-3}$ s$^{-1}$)',
            [('G_cm3_s','Generation'),('R_SRH_cm3_s','SRH recombination')])]
    for name,ylabel,series in specs:
        fig,ax=plt.subplots(figsize=(7.4,4.8))
        for col,label in series:
            ax.plot(profile.x_um,profile[col],label=label,
                    linestyle='--' if col.startswith('E_F') else '-')
        if name=='BB_carrier_profiles':ax.set_yscale('log')
        if name=='BB_band_diagram':
            ax.scatter([0,profile.x_um.iloc[-1]],[row['E_cat_HER_eV'],row['E_cat_OER_eV']],
                       marker='s',label=r'$E_{\rm cat}$')
        ax.set(xlabel=r'Position ($\mu$m)',ylabel=ylabel)
        ax.legend(frameon=False,ncol=2);ax.grid(alpha=.15)
        fig.tight_layout();fig.savefig(outdir/(name+'.pdf'));plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.4,4.8))
    ax.plot(open_circuit.x_um,open_circuit.E_CB_eV,label=r'$E_{\rm CB}$ at $V_{\rm OC}$')
    ax.plot(profile.x_um,profile.E_CB_eV,'--',label=r'$E_{\rm CB}$ under reverse load')
    ax.set(xlabel=r'Position ($\mu$m)',ylabel=r'$E_{\rm CB}$ (eV)')
    ax.legend(frameon=False);ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(outdir/'BB_open_circuit_vs_loaded.pdf');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,default=ROOT/'reproduced_output/BB_profiles')
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args();validate(args.outdir,not args.no_plots)

if __name__=='__main__':main()
