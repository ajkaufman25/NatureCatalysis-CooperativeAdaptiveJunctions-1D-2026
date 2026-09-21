"""Endpoint checks independent of the SI-facing export names.

This module does not solve a BVP. It checks a supplied converged solution,
reconstructs SI energy-form statistics/currents, and raises on failed criteria.
The criteria distinguish collocation, physical current conservation and
reference regression. See docs/NUMERICAL_METHODS.md.
"""
from __future__ import annotations
from typing import Any
import numpy as np

Q_C = 1.602176634e-19
KB_J_K = 1.380649e-23


def audit_state(model: Any, sol: Any, V_HER: float, V_OER: float,
                light_fraction: float = 1.0, *, faradaic_active: bool = True,
                bvp_limit: float = 1.3e-6,
                balance_limit_A_m2: float = 1e-6) -> dict[str, float | bool | int]:
    """Validate a true BVP endpoint, not a cached CSV or an initial guess.

    The integrated balance uses the same 2401-point trapezoidal diagnostic
    as the original solver. Its 1e-6 A/m^2 criterion includes quadrature error.
    The local transfer scalars are converted to +x conventional currents
    with the four signs given explicitly in SI eq:transfer-signs.
    """
    if sol.status != 0:
        raise RuntimeError('Unconverged BVP: '+str(sol.message))
    s,n,p,Uc,Uv,Un,Up,Jn,Jp,N,P,d = model.profiles(sol, 2401)
    E_C_J, E_V_J, E_Fn_J, E_Fp_J = [-Q_C*u for u in (Uc,Uv,Un,Up)]
    E_H_J, E_O_J = -Q_C*V_HER, -Q_C*V_OER
    KT = KB_J_K*model.p.T
    n_check = model.Nc*np.exp(-(E_C_J-E_Fn_J)/KT)
    p_check = model.Nv*np.exp(-(E_Fp_J-E_V_J)/KT)
    neq = model.Nc*np.exp(-(E_C_J[[0,-1]]-np.array([E_H_J,E_O_J]))/KT)
    peq = model.Nv*np.exp(-(np.array([E_H_J,E_O_J])-E_V_J[[0,-1]])/KT)
    jn_out = Q_C*model.p.k_n*(n[[0,-1]]-neq)
    jp_out = Q_C*model.p.k_p*(p[[0,-1]]-peq)
    boundary_expected = np.array([jn_out[0],-jp_out[0],-jn_out[1],jp_out[1]])
    boundary_actual = np.array([Jn[0],Jp[0],Jn[-1],Jp[-1]])
    Jsem = float((Jn+Jp)[0])
    Jgen = float(light_fraction*model.Jscale)
    Jsrh = float(np.trapezoid(model.srh_over_Gref(N,P),s)*model.Jscale)
    # Exact signed identity; no max(...,0) clipping is used in this check.
    Jcontact = float(-Jp[0]-Jn[-1])
    if faradaic_active:
        jH,jO,*_ = model.faradaic_currents_A_m2(V_HER,V_OER)
    else:
        jH=jO=0.0
    y=sol.sol(s); yp=sol.sol(s,1)
    rhs=model._ode(light_fraction)(s,y)
    sampled_defect=float(np.max(np.abs((yp-rhs)/(1+np.abs(rhs)))))
    # Reconstruct the SI's energy-gradient currents using spline derivatives.
    grad_En=(KT/model.p.L_s)*(yp[2]-yp[0])
    grad_Ep=-(KT/model.p.L_s)*(yp[0]+yp[3])
    energy_current_defect=max(float(np.max(np.abs(model.p.mu_n*n*grad_En-Jn))),
                              float(np.max(np.abs(model.p.mu_p*p*grad_Ep-Jp))))
    bc=model._bc(V_HER,V_OER)(sol.y[:,0],sol.y[:,-1])
    # Include all carrier-density and interface detailed-balance exponents.
    rH=(V_HER-model.U0)/model.VT; rO=(V_OER-model.U0)/model.VT
    exponents=np.r_[y[2],y[3],y[0,0]-rH,rH-y[0,0],y[0,-1]-rO,rO-y[0,-1]]
    clipping_active=bool(np.any(exponents<=-220) or np.any(exponents>=120))
    metrics={
        'bvp_status':int(sol.status), 'adaptive_mesh_nodes':int(len(sol.x)),
        'bvp_iterations':int(sol.niter),
        'BVP_rms_max':float(np.max(sol.rms_residuals)),
        'boundary_residual_dimensionless_max':float(np.max(np.abs(bc))),
        'sampled_ode_defect_max':sampled_defect,
        'energy_current_defect_A_m2':energy_current_defect,
        'n_energy_statistics_relative_error':float(np.max(np.abs(n_check/n-1))),
        'p_energy_statistics_relative_error':float(np.max(np.abs(p_check/p-1))),
        'band_gap_error_eV':float(np.max(np.abs((-Uc)-(-Uv)-model.p.Eg))),
        'transfer_boundary_error_A_m2':float(np.max(np.abs(boundary_expected-boundary_actual))),
        'total_current_span_A_m2':float(np.ptp(Jn+Jp)),
        'catalyst_balance_HER_A_m2':float(Jsem+jH),
        'catalyst_balance_OER_A_m2':float(-Jsem+jO),
        'integrated_current_balance_A_m2':float(Jgen-Jsem-Jcontact-Jsrh),
        'final_exponential_clipping_active':clipping_active,
        'minimum_log_or_balance_exponent':float(exponents.min()),
        'maximum_log_or_balance_exponent':float(exponents.max()),
    }
    criteria={
        'BVP_rms_max':bvp_limit,
        'boundary_residual_dimensionless_max':max(bvp_limit,1e-8),
        'n_energy_statistics_relative_error':1e-10,
        'p_energy_statistics_relative_error':1e-10,
        'band_gap_error_eV':1e-12,
        'transfer_boundary_error_A_m2':1e-6,
        'total_current_span_A_m2':1e-8,
        'catalyst_balance_HER_A_m2':balance_limit_A_m2,
        'catalyst_balance_OER_A_m2':balance_limit_A_m2,
        'integrated_current_balance_A_m2':1e-6,
    }
    for name,limit in criteria.items():
        if not np.isfinite(metrics[name]) or abs(metrics[name])>limit:
            raise RuntimeError(f'Endpoint check failed: {name}={metrics[name]:.6g}; limit={limit:.6g}')
    if clipping_active:
        raise RuntimeError('A final endpoint reaches a numerical exponential clip')
    if faradaic_active and Jsem>0 and (Jp[0]>1e-8 or Jn[-1]>1e-8):
        raise RuntimeError('The SI forward contact-recombination sign assumptions are not satisfied')
    metrics['passed']=True
    return metrics
