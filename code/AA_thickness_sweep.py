#!/usr/bin/env python3
"""Recompute AA thickness states; continuation supplies starting guesses only.

The default grid retains all 31 global and 31 refined upstream points. No
stored solution profile is loaded. The current SI's Figure S19 uses only the
31 global points; the refined grid is retained as an additional diagnostic.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import replace
import argparse
import json
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from cooperative_adaptive_junction_simulator import (
    ModelParams, CooperativeAdaptiveJunction, facet_from_mean_delta,
    _map_solution_seed_between_intensities)
from publication_data import state_row, convert_sweep, REFERENCE
from validation_checks import audit_state

ROOT=Path(__file__).resolve().parents[1]


def solve_point(length_um: float, previous=None):
    """Solve one thickness using the original scalar-current continuation.

    G and Jscale change through Beer-Lambert total absorption. Rescale the
    previous n,p,J guesses to the new scales, then solve every new endpoint.
    The previous current times Jscale_new/Jscale_old is a bracket estimate,
    not an imposed quantum yield. Failed endpoints raise an exception.
    """
    if not np.isfinite(length_um) or length_um<=0:
        raise ValueError('Thickness must be finite and positive (um)')
    p=replace(ModelParams(),L_s=float(length_um)*1e-6)
    model=CooperativeAdaptiveJunction(p,facet_from_mean_delta(.925,.15))
    if previous is None:
        J,UH,UO,sol,d,_=model.solve_ows()
    else:
        old,oldsol,oldJ=previous
        seed=_map_solution_seed_between_intensities(old,oldsol,model)
        cache=[]
        def residual(current):
            uh=model.invert_H_for_forward_current(current)
            uo=model.invert_O_for_forward_current(current)
            guess=seed if not cache else min(cache,key=lambda t:abs(t[0]-current))[1]
            state=model.solve_state(uh,uo,1.0,previous=guess,tol=3e-7,nmesh=1100)
            if state.status:
                raise RuntimeError(f'L={length_um:g} um, J={current:g} A/m^2: {state.message}')
            diag=model.diagnostics(state,uh,uo,1.0)
            cache.append((current,state,diag))
            return diag['Jsem_A_m2']-current
        estimate=oldJ*model.Jscale/old.Jscale
        residual(estimate)
        lo=max(1e-10,.65*estimate);hi=1.35*estimate
        flo=residual(lo);fhi=residual(hi)
        for _ in range(10):
            if flo*fhi<=0:break
            lo*=.6;hi*=1.5;flo=residual(lo);fhi=residual(hi)
        if flo*fhi>0:
            raise RuntimeError(f'No current bracket at L={length_um:g} um')
        J=brentq(residual,lo,hi,xtol=2e-9,rtol=2e-10)
        residual(J);_,sol,d=cache[-1]
        UH=model.invert_H_for_forward_current(J)
        UO=model.invert_O_for_forward_current(J)
    audit=audit_state(model,sol,UH,UO,bvp_limit=3.1e-7)
    row=state_row(model,sol,UH,UO,'AA')
    row.update(thickness_nm=length_um*1000,thickness_um=length_um,
               alpha_L=p.alpha_abs*p.L_s,
               absorbed_fraction=float(-np.expm1(-p.alpha_abs*p.L_s)),
               utilization_percent=100*row['J_OWS_mA_cm2']/row['J_gen_mA_cm2'],
               status='ok',calculation_origin='recomputed_BVP')
    return row,(model,sol,J),audit


def run_thickness(outdir: Path, lengths_um=None, reference_dir: Path=REFERENCE):
    """Recompute, checkpoint and compare all requested thickness states."""
    outdir.mkdir(parents=True,exist_ok=True)
    points=([(float(x),'verification') for x in lengths_um] if lengths_um else
            [(float(x),'full') for x in np.logspace(-2,1,31)]+
            [(float(x),'refined') for x in np.geomspace(.05,.5,31)])
    points.sort(key=lambda item:item[0])
    rows=[];audits=[];previous=None
    for length,source in points:
        row,previous,audit=solve_point(length,previous)
        row['source']=source;rows.append(row)
        audits.append(dict(thickness_um=length,source=source,**audit))
        pd.DataFrame(rows).to_csv(outdir/'AA_thickness_sweep.csv',index=False)
        (outdir/'endpoint_checks.json').write_text(json.dumps(audits,indent=2)+'\n')
        print(f'thickness {len(rows):2d}/{len(points)}: L={length:.6g} um, J={row["J_OWS_mA_cm2"]:.9g} mA/cm^2',flush=True)
    df=pd.DataFrame(rows).sort_values('thickness_um')
    ref=convert_sweep(pd.read_csv(reference_dir/'thickness/AA_thickness_sweep_combined_dense.csv'))
    fields=['J_OWS_mA_cm2','Delta_V_cat_V','J_gen_mA_cm2','J_SRH_mA_cm2',
            'J_contact_mA_cm2','V_bb_HER_V','V_bb_OER_V','Delta_E_F_center_eV']
    comparisons=[]
    for _,row in df.iterrows():
        k=(ref.thickness_um-row.thickness_um).abs().idxmin();r=ref.loc[k]
        if abs(r.thickness_um-row.thickness_um)>1e-8:continue
        for field in fields:
            delta=abs(float(row[field])-float(r[field]))
            comparisons.append(dict(thickness_um=row.thickness_um,quantity=field,
                                    reference=r[field],recomputed=row[field],
                                    absolute_difference=delta,passed=delta<1e-7))
    comp=pd.DataFrame(comparisons)
    comp.to_csv(outdir/'reference_comparison.csv',index=False)
    if len(comp) and not comp.passed.all():
        raise RuntimeError('Thickness reference comparison failed; see reference_comparison.csv')
    summary=dict(points=len(df),reference_comparisons=len(comp),
                 reference_tolerance=1e-7,
                 maximum_absolute_differences=(comp.groupby('quantity').absolute_difference.max().to_dict() if len(comp) else {}),
                 maximum_BVP_rms=float(df.BVP_rms_max.max()),
                 maximum_integrated_balance_A_m2=float(df.integrated_current_balance_A_m2.abs().max()),
                 all_endpoint_checks_passed=True)
    (outdir/'reproduction_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return df,summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--outdir',type=Path,default=ROOT/'reproduced_output/thickness')
    ap.add_argument('--lengths-um',type=float,nargs='+')
    args=ap.parse_args()
    _,summary=run_thickness(args.outdir,args.lengths_um)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
