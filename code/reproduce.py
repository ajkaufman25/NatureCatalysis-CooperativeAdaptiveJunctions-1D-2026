#!/usr/bin/env python3
"""Reproduce SI-aligned model outputs with explicit source/solve provenance.

Default: fresh AA/BA operating states and common-Fermi dark equilibria; load
stored sensitivity and thickness tables. --full-sweeps recomputes all three
sensitivity sweeps and all 62 thickness points. --diagnostics additionally
solves the BB reverse load. Historical AB/BA zero-current families remain
reference-only diagnostics and never replace the selected states in SI Table S8.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict, replace
import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import scipy
from cooperative_adaptive_junction_simulator import (
    ModelParams, FacetConfig, FaradaicConfig, CooperativeAdaptiveJunction,
    solve_adaptive_case, solve_ba_finite_current, solve_bb_control,
    sweep_ab_catalyst_off, sweep_ba_catalyst_off,
    _continue_intensity_from_state, _map_solution_seed_between_intensities,
    facet_from_mean_delta)
from scipy.optimize import brentq
from publication_data import (ROOT,REFERENCE,state_row,convert_sweep,
                              convert_profile,export_reference_tables,
                              selected_catalyst_off_values)
from validation_checks import audit_state
from AA_thickness_sweep import run_thickness


def write_json(path:Path,obj):
    """Write machine-readable metadata; Python/NumPy scalars are normalized."""
    def convert(x):
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,Path):return str(x)
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(obj,indent=2,default=convert)+'\n')


def compare_sweep(new:pd.DataFrame,reference:pd.DataFrame,keys:list[str],group:str):
    """Compare endpoint values, ignoring provenance/status text and seed data."""
    fields=['J_OWS_mA_cm2','Delta_V_cat_V','J_gen_mA_cm2','J_contact_mA_cm2',
            'J_SRH_mA_cm2','Delta_E_F_center_eV']
    comparisons=[]
    for _,r in new.iterrows():
        mask=np.ones(len(reference),dtype=bool)
        for key in keys:mask &= np.isclose(reference[key],r[key],rtol=1e-10,atol=1e-12)
        if mask.sum()!=1:
            raise RuntimeError(f'{group}: nonunique/missing reference point {[r[k] for k in keys]}')
        old=reference[mask].iloc[0]
        for field in fields:
            if field not in old or field not in r:continue
            if not (pd.notna(old[field]) and pd.notna(r[field])):continue
            difference=abs(float(old[field])-float(r[field]))
            comparisons.append(dict(group=group,coordinates=';'.join(f'{k}={r[k]:.8g}' for k in keys),
                                    quantity=field,reference=old[field],recomputed=r[field],
                                    absolute_difference=difference,passed=difference<1e-6))
    return comparisons


def _continue_aa_step(previous, *, mean=.925, delta=.15, k=10.0):
    """Solve a new AA parameter point using only a freshly solved prior guess.

    Parameter continuation changes the Newton initial guess, never the endpoint
    equations. It avoids an unnecessary cold-start high-injection solve, which
    can encounter a singular Jacobian at some of the supplied sweep points.
    Every target is re-solved and current-balanced at its own parameters.
    """
    old,oldsol,oldJ,oldh,oldo=previous
    params=replace(old.p,k_n=float(k),k_p=float(k))
    model=CooperativeAdaptiveJunction(params,facet_from_mean_delta(mean,delta))
    seed=_map_solution_seed_between_intensities(old,oldsol,model)
    # New surface potentials enter through the boundary residual; this is not
    # a common-mode shift of the physical solution or reuse of saved densities.
    initial=model.solve_state(oldh,oldo,1.0,previous=seed,tol=8e-7,nmesh=1100)
    if initial.status!=0:
        raise RuntimeError('AA parameter-continuation seed failed: '+initial.message)
    cache=[]
    def residual(current):
        h=model.invert_H_for_forward_current(current)
        o=model.invert_O_for_forward_current(current)
        guess=initial if not cache else min(cache,key=lambda z:abs(z[0]-current))[1]
        sol=model.solve_state(h,o,1.0,previous=guess,tol=3e-7,nmesh=1100)
        if sol.status!=0:raise RuntimeError('AA parameter-continuation BVP failed: '+sol.message)
        diag=model.diagnostics(sol,h,o,1.0)
        cache.append((current,sol,h,o))
        return diag['Jsem_A_m2']-current
    estimate=max(1e-8,oldJ*model.Jscale/old.Jscale)
    residual(estimate)
    lo,hi=.65*estimate,1.35*estimate
    flo,fhi=residual(lo),residual(hi)
    for _ in range(12):
        if flo*fhi<=0:break
        lo*=.6;hi*=1.5;flo,fhi=residual(lo),residual(hi)
    if flo*fhi>0:raise RuntimeError('AA parameter-continuation current root not bracketed')
    J=brentq(residual,lo,hi,xtol=2e-9,rtol=2e-10)
    residual(J)
    _,sol,h,o=cache[-1]
    return model,sol,J,h,o


def continue_aa(previous, *, mean=.925, delta=.15, k=10.0):
    """Limit changes of initial-guess parameters; solve every intermediate state.

    The 25 mV and half-decade steps are numerical continuation increments,
    not extra plotted sweep points or altered model parameters. Endpoints
    alone enter the reference comparisons and publication tables.
    """
    old=previous[0]
    oldmean=(old.facet.band_bending_H_V+old.facet.band_bending_O_V)/2
    olddelta=old.facet.band_bending_O_V-old.facet.band_bending_H_V
    oldlog=np.log10(old.p.k_n);newlog=np.log10(k)
    steps=max(1,int(np.ceil(abs(mean-oldmean)/.025)),
              int(np.ceil(abs(delta-olddelta)/.025)),
              int(np.ceil(abs(newlog-oldlog)/.5)))
    state=previous
    for a in np.linspace(0,1,steps+1)[1:]:
        state=_continue_aa_step(state,mean=oldmean+a*(mean-oldmean),
                               delta=olddelta+a*(delta-olddelta),
                               k=10**(oldlog+a*(newlog-oldlog)))
    return state


def checked_aa(previous=None,**kwargs):
    """Solve AA and retain the complete state for independent endpoint checks."""
    if previous is None:
        m,ds,dd,il,idi,J,h,o,sol,dg=solve_adaptive_case(**kwargs)
    else:
        m,sol,J,h,o=continue_aa(previous,**kwargs)
    audit=audit_state(m,sol,h,o,bvp_limit=3.1e-7)
    row=state_row(m,sol,h,o,'AA')
    row.update(status='ok',calculation_origin='recomputed_BVP',
               initialization='fresh_parameter_continuation' if previous else 'dark_and_light_continuation',
               utilization_percent=100*row['J_OWS_mA_cm2']/row['J_gen_mA_cm2'])
    return row,audit,(m,sol,J,h,o)


def recompute_sensitivities(data:Path,checks:Path,reference:Path,aa_state):
    """Preserve the upstream grids; continue from fresh states, never CSVs."""
    all_audits=[];comparisons=[];rows=[]
    previous=aa_state; native_states=[aa_state]
    for mean in [.70,.85,.925,1.00,1.15]:
        for delta in [.00,.05,.10,.15,.20,.30]:
            if delta==0:
                # The upstream symmetric row is assigned analytically. Do not
                # misreport it as a converged finite-current catalytic state.
                row=dict(J_OWS_mA_cm2=0.0,status='analytic_symmetry',
                         calculation_origin='symmetry_assignment_no_BVP')
            else:
                previous=min(native_states,key=lambda state:
                    abs((state[0].facet.band_bending_H_V+state[0].facet.band_bending_O_V)/2-mean)
                    +abs(state[0].facet.band_bending_O_V-state[0].facet.band_bending_H_V-delta))
                row,audit,previous=checked_aa(previous=previous,mean=mean,delta=delta)
                native_states.append(previous)
                all_audits.append(dict(group='native',mean=mean,delta=delta,**audit))
            row.update(mean_V_bb_0_V=mean,Delta_V_bb_0_V=delta)
            rows.append(row)
            print(f'native mean={mean:g} V delta={delta:g} V: J={row["J_OWS_mA_cm2"]:.9g}',flush=True)
            pd.DataFrame(rows).to_csv(data/'native_band_bending_sweep.csv',index=False)
    gdf=pd.DataFrame(rows)
    comparisons+=compare_sweep(gdf,convert_sweep(pd.read_csv(reference/'native_band_bending_sweep.csv')),
                               ['mean_V_bb_0_V','Delta_V_bb_0_V'],'native')
    rows=[]
    previous=aa_state
    for k in [.1,1.,10.,100.,1000.]:
        row,audit,previous=checked_aa(previous=previous,k=k);row['k_m_s']=k;rows.append(row)
        all_audits.append(dict(group='transfer',k_m_s=k,**audit))
        print(f'transfer k={k:g} m/s: J={row["J_OWS_mA_cm2"]:.9g}',flush=True)
    kdf=pd.DataFrame(rows);kdf.to_csv(data/'common_transfer_rate_sweep.csv',index=False)
    comparisons+=compare_sweep(kdf,convert_sweep(pd.read_csv(reference/'common_transfer_rate_sweep.csv')),
                               ['k_m_s'],'transfer')
    # Continue outward from 10 mW/cm^2 exactly as in the upstream solver.
    m,sol,J,h,o=aa_state
    states={10.0:aa_state}
    for points in ([20.,30.],[5.,3.,1.]):
        previous=aa_state
        for intensity in points:
            mm,ss,jj,hh,oo,dg=_continue_intensity_from_state(*previous,intensity)
            states[intensity]=(mm,ss,jj,hh,oo);previous=states[intensity]
    rows=[]
    for intensity in sorted(states):
        mm,ss,jj,hh,oo=states[intensity]
        audit=audit_state(mm,ss,hh,oo,bvp_limit=1.3e-6)
        row=state_row(mm,ss,hh,oo,'AA')
        row.update(intensity_mW_cm2=intensity,status='ok',calculation_origin='recomputed_BVP',
                   utilization_percent=100*row['J_OWS_mA_cm2']/row['J_gen_mA_cm2'])
        rows.append(row);all_audits.append(dict(group='intensity',intensity_mW_cm2=intensity,**audit))
        print(f'intensity={intensity:g} mW/cm^2: J={row["J_OWS_mA_cm2"]:.9g}',flush=True)
    idf=pd.DataFrame(rows);idf.to_csv(data/'intensity_sweep.csv',index=False)
    comparisons+=compare_sweep(idf,convert_sweep(pd.read_csv(reference/'intensity_sweep.csv')),
                               ['intensity_mW_cm2'],'intensity')
    comp=pd.DataFrame(comparisons);comp.to_csv(checks/'sensitivity_reference_comparison.csv',index=False)
    write_json(checks/'sensitivity_endpoint_checks.json',all_audits)
    if not comp.passed.all():
        raise RuntimeError('Sensitivity regression failed; see sensitivity_reference_comparison.csv')
    return dict(native_BVP_states=25,native_symmetry_assignments=5,
                transfer_BVP_states=5,intensity_states=6,
                maximum_reference_difference=float(comp.absolute_difference.max()))


def run_diagnostics(outdir:Path,reference:Path):
    """Recompute optional controls separately from SI forward-OWS results."""
    outdir.mkdir(parents=True,exist_ok=True)
    m,solver,jv,voc,vocsol,vocdg,op,sol,dg=solve_bb_control()
    audit=audit_state(m,sol,op['U_H_V'],op['U_O_V'],bvp_limit=1.1e-8)
    row=state_row(m,sol,op['U_H_V'],op['U_O_V'],'BB')
    row['state']='reverse_HOR_ORR_load_diagnostic';row['V_OC_V']=voc
    pd.DataFrame([row]).to_csv(outdir/'BB_reverse_load.csv',index=False)
    m.energy_profile_dataframe(sol).to_csv(outdir/'BB_reverse_load_profile.csv',index=False)
    # The curve was solved in the BB gauge V_cat,HER=0. These surface
    # energies are explicitly gauge-labelled, unlike the physical profile.
    pd.DataFrame(dict(Delta_V_cat_V=jv.V_terminal_V,
                      J_sem_mA_cm2=jv.Jsem_mA_cm2,
                      E_CB_s_HER_gauge_eV=-jv.U_CB_H_V,
                      E_CB_s_OER_gauge_eV=-jv.U_CB_O_V,
                      BVP_rms_max=jv.BVP_residual)).to_csv(outdir/'BB_semiconductor_JV.csv',index=False)
    ref=float(pd.read_csv(reference/'BB_operating_point.csv').iloc[0]['Jsem_uA_cm2'])/1000
    if abs(row['J_sem_mA_cm2']-ref)>1e-7:
        raise RuntimeError('BB reverse-load reference comparison failed')
    details={'BB':dict(V_OC_V=voc,**audit)}
    details['mixed_contact_families']={
        'source':'unchanged data/reference/AB_catalyst_off_Voc_sweep.csv and BA_catalyst_off_Voc_sweep.csv',
        'freshly_recomputed':False,
        'used_in_current_SI_figures':False,
        'selection_constraints_for_SI_states_specified':False,
        'note':'Historical asymptotic diagnostic sweeps are not the selected SI Table S8 states.'}
    write_json(outdir/'diagnostic_summary.json',details)
    return details


def reproduce(outdir:Path,full_sweeps:bool=False,reference_dir:Path|None=None,
              diagnostics:bool=False,plots:bool=True,allow_existing:bool=False):
    """Run the requested calculations and emit auditable provenance metadata."""
    outdir=Path(outdir).resolve()
    if outdir.exists() and any(outdir.iterdir()) and not allow_existing:
        raise FileExistsError(f'Output directory is not empty: {outdir}. Choose a new directory or --overwrite.')
    outdir.mkdir(parents=True,exist_ok=True)
    reference=Path(reference_dir).resolve() if reference_dir else REFERENCE
    if not (reference/'AA_operating_profile.csv').exists() and (reference/'reference').is_dir():
        reference=reference/'reference'
    data=outdir/'data';checks=outdir/'validation'
    data.mkdir(exist_ok=True);checks.mkdir(exist_ok=True)
    start=time.monotonic()
    metadata=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,
                  scipy=scipy.__version__,pandas=pd.__version__,full_sweeps=full_sweeps,
                  reference_directory=(str(reference.relative_to(ROOT)) if reference.is_relative_to(ROOT) else str(reference)),
                  started_at_UTC=datetime.now(timezone.utc).isoformat(),
                  physical_inputs={'semiconductor':asdict(ModelParams()),'facets':asdict(FacetConfig()),
                                   'Faradaic':asdict(FaradaicConfig())},groups={})
    write_json(outdir/'run_metadata.json',metadata)
    print('Solving AA and BA finite-current endpoints',flush=True)
    aa_row,aa_audit,aa_state=checked_aa()
    aa,aasol,Jaa,haa,oaa=aa_state
    ba,Jba,hba,oba,basol,badg,bachk=solve_ba_finite_current()
    ba_audit=audit_state(ba,basol,hba,oba,bvp_limit=1.1e-8,balance_limit_A_m2=1e-8)
    ba_row=state_row(ba,basol,hba,oba,'BA')
    ba_row.update(status='ok',calculation_origin='recomputed_BVP')
    rows=[aa_row,ba_row]
    pd.DataFrame(rows).to_csv(data/'operating_states.csv',index=False)
    for mode,m,sol in [('AA',aa,aasol),('BA',ba,basol)]:
        m.energy_profile_dataframe(sol).to_csv(data/f'{mode}_operating_profile.csv',index=False)
    metadata['groups']['operating']='fresh AA and BA BVP/load-line solves'
    checks_baseline={'AA':aa_audit,'BA':ba_audit,
                     'BA_two_potential_root_success':bool(bachk.success),
                     'BA_two_potential_root_residual_A_m2':bachk.fun.tolist()}
    # A common 0-V catalyst potential is an explicit state definition in the SI.
    dark=[]
    for mode in ['AA','AB','BA','BB']:
        m=CooperativeAdaptiveJunction(ModelParams(),FacetConfig(),
                                      lambda_H=float(mode[0]=='B'),lambda_O=float(mode[1]=='B'))
        sol=m.solve_dark_equilibrium()
        audit=audit_state(m,sol,0.0,0.0,0.0,faradaic_active=False,bvp_limit=1.1e-8)
        row=state_row(m,sol,0.0,0.0,mode,light_fraction=0,faradaic_active=False)
        row['calculation_origin']='recomputed_common_Fermi_dark_BVP';dark.append(row)
        m.energy_profile_dataframe(sol).to_csv(data/f'{mode}_dark_equilibrium_profile.csv',index=False)
        checks_baseline[f'{mode}_dark']=audit
    pd.DataFrame(dark).to_csv(data/'dark_equilibrium_states.csv',index=False)
    # Explicit regression of the BA distinction: fixed barrier, reduced
    # local bending, and almost eliminated left-right surface difference.
    bd=next(r for r in dark if r['architecture']=='BA')
    ba_physics=dict(Phi_B_HER_eV=ba_row['Phi_B_HER_eV'],
                    dark_V_bb_HER_V=bd['V_bb_HER_V'],
                    operating_V_bb_HER_V=ba_row['V_bb_HER_V'],
                    surface_band_shift_eV=ba_row['E_CB_s_HER_eV']-bd['E_CB_s_HER_eV'],
                    bulk_band_shift_eV=ba_row['E_CB_bulk_eV']-bd['E_CB_bulk_eV'],
                    surface_energy_difference_eV=ba_row['Delta_E_CB_surface_eV'])
    assert abs(ba_physics['Phi_B_HER_eV']-.9552120108495786)<1e-10
    assert abs(ba_physics['operating_V_bb_HER_V']-.308893959609)<1e-7
    assert ba_physics['operating_V_bb_HER_V']<ba_physics['dark_V_bb_HER_V']
    assert abs(ba_physics['surface_energy_difference_eV']-.0016669487594)<1e-8
    checks_baseline['BA_physical_definitions']=ba_physics
    comparisons=[]
    for mode in ['AA','BA']:
        expected=convert_profile(pd.read_csv(reference/f'{mode}_operating_profile.csv'))
        current=pd.read_csv(data/f'{mode}_operating_profile.csv')
        for field in expected:
            # Densities span many orders of magnitude, so their comparison
            # is relative. Energies/currents use absolute SI-reported units.
            delta=np.abs(current[field].to_numpy()-expected[field].to_numpy())
            if field in ('n_cm3','p_cm3'):
                metric=float(np.max(delta/np.maximum(np.abs(expected[field]),1e-300)))
                limit=1e-5;kind='relative'
            else:metric=float(np.max(delta));limit=1e-6;kind='absolute'
            comparisons.append(dict(architecture=mode,quantity=field,comparison=kind,
                                    maximum_difference=metric,tolerance=limit,passed=metric<=limit))
    comp=pd.DataFrame(comparisons);comp.to_csv(checks/'baseline_reference_comparison.csv',index=False)
    if not comp.passed.all():raise RuntimeError('Baseline profile regression failed')
    write_json(checks/'baseline_endpoint_checks.json',checks_baseline)
    metadata['groups']['dark']='four fresh common-Fermi, Faradaic-disabled equilibria'
    # These are supplied selections, not hidden inputs or invented constraints.
    selected_catalyst_off_values().to_csv(data/'selected_catalyst_off_states.csv',index=False)
    metadata['groups']['selected_catalyst_off']='SI Table S8 displayed selections; AA/AB/BA not numerically reconstructed (selection constraints unspecified)'
    if full_sweeps:
        summary=recompute_sensitivities(data,checks,reference,aa_state)
        metadata['groups']['sensitivities']=dict(origin='fresh BVP solves',**summary)
        thickness,summary=run_thickness(outdir/'thickness',reference_dir=reference)
        thickness.to_csv(data/'AA_thickness_sweep.csv',index=False)
        thickness[thickness.source=='full'].to_csv(data/'AA_thickness_plot_data.csv',index=False)
        metadata['groups']['thickness']=dict(origin='fresh BVP solves',**summary)
    else:
        export_reference_tables(data,reference)
        metadata['groups']['sensitivities']='imported stored tables; no sensitivity BVP rerun'
        metadata['groups']['thickness']='imported stored 62-point table; no thickness BVP rerun'
    if diagnostics:
        run_diagnostics(outdir/'diagnostics',reference)
        metadata['groups']['additional_diagnostics']='fresh BB reverse load; historical mixed-contact families retained as reference only'
    if plots:
        from plot_publication import make_figures
        make_figures(data,outdir/'figures')
        metadata['groups']['figures']='S16-S19 rendered from this run; S15 is SI TikZ source'
    metadata['elapsed_seconds']=time.monotonic()-start
    metadata['passed']=True
    write_json(outdir/'run_metadata.json',metadata)
    print(f'Completed in {metadata["elapsed_seconds"]:.1f} s: {outdir}',flush=True)
    return metadata


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--outdir',type=Path,default=ROOT/'reproduced_output/full')
    ap.add_argument('--full-sweeps',action='store_true',help='Recompute every sensitivity and thickness state; never use cached sweep outputs')
    ap.add_argument('--precomputed-dir',type=Path,default=None,help='Reference CSV directory; used only for comparisons with --full-sweeps')
    ap.add_argument('--diagnostics',action='store_true',help='Also recompute the BB reverse-load diagnostic (historical mixed families remain reference-only)')
    ap.add_argument('--no-plots',action='store_true')
    ap.add_argument('--overwrite',action='store_true',help='Allow writing into the specified existing output directory; does not delete it')
    args=ap.parse_args()
    reproduce(args.outdir,args.full_sweeps,args.precomputed_dir,args.diagnostics,
              not args.no_plots,args.overwrite)


if __name__=='__main__':main()
