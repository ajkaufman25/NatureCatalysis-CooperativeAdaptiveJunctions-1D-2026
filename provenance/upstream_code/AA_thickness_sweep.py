#!/usr/bin/env python3
"""Reproduce the AA thickness sweep using the unchanged publication simulator.

Default: 31 logarithmic global points plus 31 refined shoulder points. Thickness continuation supplies initial guesses; this script does not load
precomputed states. Each endpoint satisfies the unchanged steady equations.
The reported operating current is the final semiconductor current diagnostic.
"""
from pathlib import Path
from dataclasses import replace
import argparse
import json
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from cooperative_adaptive_junction_simulator import ModelParams, CooperativeAdaptiveJunction, facet_from_mean_delta, _map_solution_seed_between_intensities

ROOT = Path(__file__).resolve().parents[1]

def solve_point(length_um, previous=None):
    p = replace(ModelParams(), L_s=float(length_um)*1e-6)
    model = CooperativeAdaptiveJunction(p, facet_from_mean_delta(.925, .15))
    if previous is None:
        J, UH, UO, sol, d, _ = model.solve_ows()
    else:
        old, oldsol, oldJ = previous
        seed = _map_solution_seed_between_intensities(old, oldsol, model)
        cache = []
        def residual(current):
            uh=model.invert_H_for_forward_current(current)
            uo=model.invert_O_for_forward_current(current)
            guess=seed if not cache else min(cache,key=lambda item:abs(item[0]-current))[1]
            state=model.solve_state(uh,uo,1.0,previous=guess,tol=3e-7,nmesh=1100)
            if state.status: raise RuntimeError(f'L={length_um}, J={current}: {state.message}')
            diag=model.diagnostics(state,uh,uo,1.0)
            cache.append((current,state,diag))
            return diag['Jsem_A_m2']-current
        estimate=oldJ*model.Jscale/old.Jscale
        # Evaluate near the previous solution before bracketing the load line.
        residual(estimate)
        lo=max(1e-10,.65*estimate);hi=1.35*estimate
        flo=residual(lo);fhi=residual(hi)
        for _ in range(10):
            if flo*fhi<=0: break
            lo*=.6;hi*=1.5;flo=residual(lo);fhi=residual(hi)
        J=brentq(residual,lo,hi,xtol=2e-9,rtol=2e-10)
        residual(J);_,sol,d=cache[-1]
        UH=model.invert_H_for_forward_current(J);UO=model.invert_O_for_forward_current(J)
    if sol.status != 0:
        raise RuntimeError(sol.message)
    row = dict(thickness_nm=length_um*1000, thickness_um=length_um,
        alpha_L=p.alpha_abs*p.L_s, absorbed_fraction=-np.expm1(-p.alpha_abs*p.L_s),
        J_OWS_mA_cm2=d['Jsem_mA_cm2'], U_H_V_vs_RHE=UH, U_O_V_vs_RHE=UO,
        contact_separation_V=UO-UH, eta_H_V=d['eta_H_V'], eta_O_V=d['eta_O_V'],
        Jgen_mA_cm2=d['Jgen_mA_cm2'], utilization_percent=100*d['Jsem_mA_cm2']/d['Jgen_mA_cm2'],
        SRH_mA_cm2=d['Jrec_mA_cm2'], counterflow_mA_cm2=d['counterflow_mA_cm2'],
        band_bending_H_V=d['band_bending_H_V'], band_bending_O_V=d['band_bending_O_V'],
        QFL_center_split_V=d['QFL_center_split_V'], BVP_residual=d['max_BVP_rms_residual'],
        current_budget_residual_A_m2=d['current_budget_residual_A_m2'],
        catalyst_balance_H_A_m2=d['MIEC_balance_H_A_m2'],
        catalyst_balance_O_A_m2=d['MIEC_balance_O_A_m2'], status='ok')
    return row,(model,sol,J)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--outdir',type=Path,default=ROOT/'reproduced_output'/'thickness')
    ap.add_argument('--lengths-um',type=float,nargs='+',help='Optional subset for independent spot verification')
    args=ap.parse_args(); args.outdir.mkdir(parents=True,exist_ok=True)
    points=[(x,'verification') for x in args.lengths_um] if args.lengths_um else ([(x,'full') for x in np.logspace(-2,1,31)]+[(x,'refined') for x in np.geomspace(.05,.5,31)])
    points.sort(key=lambda item:item[0])
    rows=[];previous=None
    for length, source in points:
        row,previous=solve_point(length,previous);row['source']=source;rows.append(row)
        pd.DataFrame(rows).sort_values('thickness_um').to_csv(args.outdir/'AA_thickness_sweep_reproduced.csv',index=False)
        print(f'{len(rows)}/{len(points)} L={length:.8g} um J={row["J_OWS_mA_cm2"]:.10g} mA/cm2',flush=True)
    df=pd.DataFrame(rows).sort_values('thickness_um')
    reference=pd.read_csv(ROOT/'data'/'thickness'/'AA_thickness_sweep_combined_dense.csv')
    comparisons=[]
    fields=['J_OWS_mA_cm2','contact_separation_V','Jgen_mA_cm2','SRH_mA_cm2','counterflow_mA_cm2','band_bending_H_V','band_bending_O_V']
    for _,r in df.iterrows():
        k=(reference.thickness_um-r.thickness_um).abs().idxmin(); ref=reference.loc[k]
        if abs(ref.thickness_um-r.thickness_um)>1e-8: continue
        for field in fields:
            comparisons.append(dict(thickness_um=r.thickness_um,quantity=field,reference=ref[field],reproduced=r[field],absolute_difference=abs(r[field]-ref[field])))
    comp=pd.DataFrame(comparisons);comp.to_csv(args.outdir/'reference_comparison.csv',index=False)
    summary=dict(points=len(df),max_abs_difference_by_quantity=comp.groupby('quantity').absolute_difference.max().to_dict(),max_bvp_residual=float(df.BVP_residual.max()),max_current_budget_residual_A_m2=float(df.current_budget_residual_A_m2.abs().max()),max_catalyst_balance_A_m2=float(df[['catalyst_balance_H_A_m2','catalyst_balance_O_A_m2']].abs().max().max()))
    (args.outdir/'reproduction_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
