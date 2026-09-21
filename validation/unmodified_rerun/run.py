"""Rerun the byte-preserved upstream model for the baseline comparison."""
from pathlib import Path
import sys,json,time,importlib.util
root=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('caj_reference',root/'provenance/upstream_code/cooperative_adaptive_junction_simulator.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
out=root/'validation/unmodified_rerun';out.mkdir(parents=True,exist_ok=True)
all_results={}
for mode in ['AA','BA','BB']:
    start=time.monotonic();print('START',mode,flush=True)
    if mode=='AA':
        model,ds,dd,il,idi,J,UH,UO,sol,dg=m.solve_adaptive_case()
    elif mode=='BA':
        model,J,UH,UO,sol,dg,chk=m.solve_ba_finite_current()
        dg['two_potential_check_success']=bool(chk.success)
        dg['two_potential_check_residual_A_m2']=chk.fun.tolist()
    else:
        model,s,jv,voc,vocsol,vocdg,op,sol,dg=m.solve_bb_control()
        UH,UO=op['U_H_V'],op['U_O_V'];dg['Voc_V']=float(voc)
    model.profile_dataframe(sol).to_csv(out/f'{mode}_operating_profile.csv',index=False)
    eq=model.solve_state(0.0,0.0,0.0,previous=model.equilibrium_seed(0.0),tol=1e-8,nmesh=1800)
    assert eq.status==0,eq.message
    dark=model.diagnostics(eq,0,0,0)
    all_results[mode]=dict(operating=dg,dark_zero_common_potential=dark,elapsed_seconds=time.monotonic()-start)
    (out/'summary.json').write_text(json.dumps(all_results,indent=2)+'\n')
    print(mode,dg['Jsem_mA_cm2'],dg['band_bending_H_V'],dg['max_BVP_rms_residual'],'s=',time.monotonic()-start,flush=True)
