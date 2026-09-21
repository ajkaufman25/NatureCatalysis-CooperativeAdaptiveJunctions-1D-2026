#!/usr/bin/env python3
"""Refine fresh AA/BA endpoints at fixed operating catalyst potentials.

This tests sensitivity of physical outputs to tighter collocation settings.
It is a fixed-boundary refinement check, not a second catalytic current root
or an independent physical/discretization model. The baseline is solved
freshly; no saved profile supplies the initial solution.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from cooperative_adaptive_junction_simulator import solve_adaptive_case,solve_ba_finite_current
from validation_checks import audit_state
from publication_data import state_row
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--outdir',type=Path,default=ROOT/'reproduced_output/convergence')
    args=ap.parse_args();args.outdir.mkdir(parents=True,exist_ok=True)
    aa=solve_adaptive_case();ba=solve_ba_finite_current()
    m,ds,dd,il,idi,j,h,o,sol,dg=aa
    states=[('AA',m,sol,h,o,3e-8)]
    m,j,h,o,sol,dg,chk=ba;states.append(('BA',m,sol,h,o,1e-9))
    rows=[];audits={}
    for mode,m,sol,h,o,tol in states:
        refined=m.solve_state(h,o,1.,previous=sol,tol=tol,nmesh=3600)
        audits[mode]=audit_state(m,refined,h,o,bvp_limit=1.01*tol)
        before=state_row(m,sol,h,o,mode);after=state_row(m,refined,h,o,mode)
        for field in ['J_OWS_mA_cm2','J_contact_mA_cm2','J_SRH_mA_cm2',
                      'V_bb_HER_V','V_bb_OER_V','Delta_E_F_center_eV']:
            difference=abs(after[field]-before[field])
            rows.append(dict(architecture=mode,quantity=field,baseline=before[field],
                             refined=after[field],absolute_difference=difference,
                             tolerance=1e-6,passed=difference<1e-6))
        # More finely sampled quadrature checks the integrated SRH diagnostic.
        values=[]
        for points in [2401,9601]:
            s,n,p,uc,uv,un,up,jn,jp,N,P,d=m.profiles(refined,points)
            values.append(float(np.trapezoid(m.srh_over_Gref(N,P),s)*m.Jscale/10))
        rows.append(dict(architecture=mode,quantity='SRH_quadrature_mA_cm2',baseline=values[0],
                         refined=values[1],absolute_difference=abs(values[1]-values[0]),
                         tolerance=1e-6,passed=abs(values[1]-values[0])<1e-6))
    frame=pd.DataFrame(rows);frame.to_csv(args.outdir/'refinement_comparison.csv',index=False)
    (args.outdir/'endpoint_checks.json').write_text(json.dumps(audits,indent=2)+'\n')
    if not frame.passed.all():raise RuntimeError('Refinement comparison failed')
    print('AA/BA fixed-boundary refinement and quadrature checks passed',flush=True)

if __name__=='__main__':main()
