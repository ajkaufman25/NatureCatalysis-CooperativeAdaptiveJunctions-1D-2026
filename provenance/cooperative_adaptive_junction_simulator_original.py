#!/usr/bin/env python3
"""
Publication simulator for cooperative adaptive junctions in faceted n-SrTiO3.

This file is intentionally self-contained.  It implements the publication model
used in the Supporting Information of "Cooperative Adaptive Junctions Govern
Overall Photoelectrochemical Water Splitting" and generates the numerical
controls and figures discussed there.

The model is deliberately separated into three physical blocks:

(1) Semiconductor: 1D Poisson + electron/hole drift-diffusion + full mid-gap
    Shockley-Read-Hall (SRH) recombination under uniform photogeneration.

(2) Native facet asymmetry: the aqueous STO(100) (HER) and STO(110) (OER)
    surface band energies are represented by a pinned-surface limiting boundary
    condition.  The relaxed-dark band-bending targets are 0.85 and 1.00 V,
    respectively.  Their 0.15-V difference is the static symmetry-breaking
    input.  The absolute common-mode bending is not the source of directionality.

(3) Catalyst/MIEC: each catalyst is represented as a perfectly screened,
    redox-capable electron reservoir.  In the adaptive limit the STO surface
    band edge remains pinned while the catalyst electron electrochemical
    potential moves, changing the effective semiconductor/catalyst barrier.
    In the non-adaptive (buried/Schottky) control the STO surface band edge moves
    one-for-one with catalyst potential, so the dark barrier height is fixed.

The semiconductor/catalyst carrier-transfer boundary law is the reversible
constant-DOS form of Mills, Lin and Boettcher (PRL 2014).  Catalyst/electrolyte
HER/HOR and OER/ORR currents are reversible around 0 and 1.229 V vs RHE,
respectively, with one-sided transport limits only for the back reactions HOR
and ORR.  No empirical 'wrong carrier' selectivity multiplier is used.

Energy plots are shown in the conventional semiconductor convention: electron
energy increases upward.  The numerical solver internally uses electrochemical
potentials U in V vs RHE; plotted one-electron energies are E=-U (eV, apart
from an arbitrary additive constant).

Run examples
------------
  python cooperative_adaptive_junction_simulator.py --outdir output
  python cooperative_adaptive_junction_simulator.py --outdir output --full-sweeps

The default run generates the adaptive baseline, the final BB/AB/BA controls,
and publication figures.  --full-sweeps additionally recomputes the native
band-bending grid, common Mills-rate sweep, and illumination-intensity sweep.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import argparse
import math
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.optimize import brentq, root, curve_fit

# ----------------------------- physical constants -----------------------------
q = 1.602176634e-19          # C
EPS0 = 8.8541878128e-12      # F m-1
KB = 1.380649e-23            # J K-1
H = 6.62607015e-34           # J s
M0 = 9.1093837015e-31        # kg
C_LIGHT = 299792458.0        # m s-1


@dataclass(frozen=True)
class ModelParams:
    """Bulk STO, illumination, interface-transfer, and MIEC parameters."""
    T: float = 298.15
    L_s: float = 1.0e-6
    eps_r_s: float = 300.0
    Eg: float = 3.20
    m_n_rel: float = 1.8
    m_p_rel: float = 3.0
    # Representative first-order STO mobilities used in the publication model.
    mu_n: float = 5.0e-4      # m2 V-1 s-1 = 5 cm2 V-1 s-1
    mu_p: float = 1.0e-5      # m2 V-1 s-1 = 0.1 cm2 V-1 s-1
    tau_n: float = 1.0e-4     # 100 us, deliberately idealized bulk lifetime
    tau_p: float = 1.0e-4
    # Reversible semiconductor/catalyst transfer coefficient.  The same value
    # is applied to electrons and holes at both contacts in the baseline.
    k_n: float = 10.0         # m s-1
    k_p: float = 10.0
    wavelength: float = 365e-9
    intensity_mW_cm2: float = 10.0
    alpha_abs: float = 1.0e6  # m-1; used only to convert incident light to G


@dataclass(frozen=True)
class FacetConfig:
    """Static facet-specific dark electrostatic asymmetry."""
    ND_cm3: float = 1.0e18
    band_bending_H_V: float = 0.85   # STO(100), HER/electron side
    band_bending_O_V: float = 1.00   # STO(110), OER/hole side
    central_bulk_window: float = 0.08


@dataclass(frozen=True)
class FaradaicConfig:
    """Absolute-RHE reversible catalyst/electrolyte kinetics."""
    U_HER_eq_V: float = 0.0
    U_OER_eq_V: float = 1.229
    j0_H_A_cm2: float = 1.0e-6
    j0_O_A_cm2: float = 1.0e-12
    b_H_V_dec: float = 0.100
    b_O_V_dec: float = 0.040
    jlim_HOR_A_cm2: float = 5.0e-5
    jlim_ORR_A_cm2: float = 5.0e-5


@dataclass(frozen=True)
class Numerics:
    """Numerical tolerances for the collocation boundary-value solver."""
    bvp_tol_dark: float = 2.0e-7
    bvp_tol_light: float = 6.0e-7
    bvp_nodes: int = 800
    bvp_max_nodes: int = 180000


class _Seed:
    """Lightweight interpolating initial guess accepted by solve_bvp."""
    status = 0
    def __init__(self, x, y):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
    def sol(self, xnew):
        xnew = np.asarray(xnew)
        return np.vstack([np.interp(xnew, self.x, row) for row in self.y])


class CooperativeAdaptiveJunction:
    """1D STO + two screened catalyst reservoirs.

    lambda_H=lambda_O=0 is the fully adaptive Mills limit.
    lambda_i=1 is the fixed-barrier buried/Schottky limit at contact i.
    Intermediate lambda values are reserved for numerical continuation only and
    are not assigned separate physical meaning.
    """
    def __init__(self, p: ModelParams, facet: FacetConfig = FacetConfig(),
                 far: FaradaicConfig = FaradaicConfig(), num: Numerics = Numerics(),
                 lambda_H: float = 0.0, lambda_O: float = 0.0):
        self.p, self.facet, self.far, self.num = p, facet, far, num
        self.lambda_H, self.lambda_O = float(lambda_H), float(lambda_O)
        self.VT = KB*p.T/q
        self.eps_s = EPS0*p.eps_r_s
        self.Nc = 2*(2*np.pi*p.m_n_rel*M0*KB*p.T/H**2)**1.5
        self.Nv = 2*(2*np.pi*p.m_p_rel*M0*KB*p.T/H**2)**1.5
        self.ni = math.sqrt(self.Nc*self.Nv)*math.exp(-p.Eg/(2*self.VT))
        # Absolute electrochemical-potential reference used throughout.
        self.Ucb0 = -0.400  # V vs RHE, flat-band conduction-band reference
        self.U0 = self.Ucb0 + self.VT*math.log(self.Nc/self.ni)

        photon_energy = H*C_LIGHT/p.wavelength
        incident_W_m2 = 10.0*p.intensity_mW_cm2
        absorbed_fraction = 1.0-math.exp(-p.alpha_abs*p.L_s)
        # The spatial model uses uniform G.  The Beer-Lambert expression here
        # is used only to convert the stated incident 365-nm intensity into the
        # same total number of absorbed photons in the 1-um slab.
        self.Gref = incident_W_m2*absorbed_fraction/photon_energy/p.L_s
        self.Jscale = q*self.Gref*p.L_s

        # Dimensionless scaling.  The quasi-Fermi formulation is algebraically
        # equivalent to ordinary drift + diffusion but is better conditioned
        # across strongly depleted regions.
        self.mu_ref = min(p.mu_n,p.mu_p)
        self.nscale = self.Gref*p.L_s**2/(self.mu_ref*self.VT)
        self.nib = self.ni/self.nscale
        self.mob_n = p.mu_n/self.mu_ref
        self.mob_p = p.mu_p/self.mu_ref
        self.Apois = p.L_s**2*q*self.nscale/(self.eps_s*self.VT)
        self.kappa_n = p.k_n*p.L_s/(self.mu_ref*self.VT)
        self.kappa_p = p.k_p*p.L_s/(self.mu_ref*self.VT)

        self.ND_cm3 = float(facet.ND_cm3)
        self.ND_m3 = self.ND_cm3*1e6
        self.ND_scaled = self.ND_m3/self.nscale
        self.n0 = 0.5*(self.ND_m3+math.sqrt(self.ND_m3**2+4*self.ni**2))
        self.p0 = self.ni**2/self.n0
        self.U_mid = 0.5*(far.U_HER_eq_V+far.U_OER_eq_V)

        # Calibrate the absolute surface band energies so the fully relaxed dark
        # reference has the requested depletion bendings.  Only their 0.15-V
        # difference is the directional symmetry-breaking parameter.
        self.Ucb_bulk_dark_target = far.U_HER_eq_V-self.VT*math.log(self.Nc/self.n0)
        self.Ucb_H_pin = self.Ucb_bulk_dark_target-facet.band_bending_H_V
        self.Ucb_O_pin = self.Ucb_bulk_dark_target-facet.band_bending_O_V

        # Physical fixed Schottky barriers for the matched buried controls.
        # These are defined directly from the neutral-bulk Ec-Ef separation
        # plus the native depletion bending and are independent of the HER/OER
        # solution redox potentials.
        self.delta_Ec_bulk_eV = -self.Ucb_bulk_dark_target
        self.Phi_H_Sch_eV = self.delta_Ec_bulk_eV + facet.band_bending_H_V
        self.Phi_O_Sch_eV = self.delta_Ec_bulk_eV + facet.band_bending_O_V
        self.Phi_difference_Sch_eV = self.Phi_O_Sch_eV-self.Phi_H_Sch_eV

        # Adaptive barriers at the HER/OER equilibrium potentials are useful
        # diagnostics only; they are not the fixed barriers of the buried controls.
        self.adaptive_barrier_H_at_eq_eV = far.U_HER_eq_V-self.Ucb_H_pin
        self.adaptive_barrier_O_at_eq_eV = far.U_OER_eq_V-self.Ucb_O_pin
        self.dark_barrier_H_eV = self.Phi_H_Sch_eV
        self.dark_barrier_O_eV = self.Phi_O_Sch_eV
        self.dark_barrier_difference_eV = self.Phi_difference_Sch_eV
        self.dark_sol = None
        self.dark_diag = None

    # ----------------------- semiconductor constitutive laws ------------------
    def srh_over_Gref(self, N, P):
        n, p = N*self.nscale, P*self.nscale
        R = (n*p-self.ni**2)/(self.p.tau_p*(n+self.ni)+self.p.tau_n*(p+self.ni))
        return R/self.Gref

    def _initial_mesh(self, nmesh):
        z=np.linspace(0.0,1.0,nmesh)
        return 0.5*(1.0-np.cos(np.pi*z))

    def _ode(self, light_fraction):
        def ode(s,y,pars=None):
            v,d,lnN,lnP,jn,jp=y
            N=np.exp(np.clip(lnN,-220,120)); P=np.exp(np.clip(lnP,-220,120))
            rr=self.srh_over_Gref(N,P)
            out=np.empty_like(y)
            # s=x/L, v=(U_CB-U_CB^0)/V_T, d=-dv/ds.
            out[0]=-d
            out[1]=self.Apois*(P-N+self.ND_scaled)
            # Drift-diffusion expressed through electron/hole quasi-Fermi levels.
            out[2]=jn/(self.mob_n*np.maximum(N,1e-300))-d
            out[3]=-jp/(self.mob_p*np.maximum(P,1e-300))+d
            out[4]=rr-light_fraction
            out[5]=light_fraction-rr
            return out
        return ode

    # ------------------------- interface energy models -------------------------
    def surface_band_potentials(self,U_H,U_O):
        """Return absolute STO surface U_CB for chosen adaptive/buried limits.

        Adaptive (lambda=0): U_CB,s is fixed by native facet pinning, so moving
        U_M changes Phi_B=U_M-U_CB,s.

        Buried (lambda=1): U_CB,s moves one-for-one with U_M, so Phi_B remains
        equal to its dark value.  This is the fixed Schottky-barrier control.
        """
        UcbH_ad=self.Ucb_H_pin
        UcbO_ad=self.Ucb_O_pin
        UcbH_bur=float(U_H)-self.Phi_H_Sch_eV
        UcbO_bur=float(U_O)-self.Phi_O_Sch_eV
        UcbH=(1.0-self.lambda_H)*UcbH_ad+self.lambda_H*UcbH_bur
        UcbO=(1.0-self.lambda_O)*UcbO_ad+self.lambda_O*UcbO_bur
        return UcbH,UcbO

    def _bc(self,U_H,U_O):
        UcbH,UcbO=self.surface_band_potentials(U_H,U_O)
        vHp=(UcbH-self.Ucb0)/self.VT; vOp=(UcbO-self.Ucb0)/self.VT
        rH=(U_H-self.U0)/self.VT; rO=(U_O-self.U0)/self.VT
        def bc(ya,yb):
            vL,dL,lnNL,lnPL,jnL,jpL=ya
            vR,dR,lnNR,lnPR,jnR,jpR=yb
            NL=np.exp(np.clip(lnNL,-220,120)); PL=np.exp(np.clip(lnPL,-220,120))
            NR=np.exp(np.clip(lnNR,-220,120)); PR=np.exp(np.clip(lnPR,-220,120))
            # Constant-DOS Mills detailed-balance carrier populations.
            NeqL=self.nib*np.exp(np.clip(vL-rH,-220,120))
            PeqL=self.nib*np.exp(np.clip(rH-vL,-220,120))
            NeqR=self.nib*np.exp(np.clip(vR-rO,-220,120))
            PeqR=self.nib*np.exp(np.clip(rO-vR,-220,120))
            return np.array([
                vL-vHp, vR-vOp,
                jnL-self.kappa_n*(NL-NeqL),
                jpL+self.kappa_p*(PL-PeqL),
                jnR+self.kappa_n*(NR-NeqR),
                jpR-self.kappa_p*(PR-PeqR),
            ])
        return bc

    # ------------------------------ BVP solution ------------------------------
    def equilibrium_seed(self,U_common,nmesh=None):
        if nmesh is None:nmesh=self.num.bvp_nodes
        s=self._initial_mesh(nmesh)
        UcbH,UcbO=self.surface_band_potentials(U_common,U_common)
        vH=(UcbH-self.Ucb0)/self.VT; vO=(UcbO-self.Ucb0)/self.VT
        def ode(s,y):
            v,d=y; Ucb=self.Ucb0+self.VT*v
            n=self.Nc*np.exp(np.clip((Ucb-U_common)/self.VT,-220,120))
            p=self.Nv*np.exp(np.clip((U_common-(Ucb+self.p.Eg))/self.VT,-220,120))
            out=np.empty_like(y); out[0]=-d
            out[1]=self.Apois*(p/self.nscale-n/self.nscale+self.ND_scaled)
            return out
        def bc(ya,yb):return np.array([ya[0]-vH,yb[0]-vO])
        y0=np.zeros((2,s.size));y0[0]=np.linspace(vH,vO,s.size)
        solp=solve_bvp(ode,bc,s,y0,tol=2e-7,max_nodes=120000,verbose=0)
        if solp.status!=0:raise RuntimeError('equilibrium Poisson seed failed: '+solp.message)
        v,d=solp.sol(s);Ucb=self.Ucb0+self.VT*v
        n=self.Nc*np.exp(np.clip((Ucb-U_common)/self.VT,-220,120))
        p=self.Nv*np.exp(np.clip((U_common-(Ucb+self.p.Eg))/self.VT,-220,120))
        y=np.zeros((6,s.size));y[0]=v;y[1]=d
        y[2]=np.log(np.maximum(n/self.nscale,1e-300));y[3]=np.log(np.maximum(p/self.nscale,1e-300))
        return _Seed(s,y)

    def _high_injection_seed(self,previous,light_fraction,nmesh=None):
        if nmesh is None:nmesh=self.num.bvp_nodes
        s=self._initial_mesh(nmesh);y=previous.sol(s).copy()
        n=np.exp(np.clip(y[2],-220,120))*self.nscale
        p=np.exp(np.clip(y[3],-220,120))*self.nscale
        dn=2.0*light_fraction*self.Gref*min(self.p.tau_n,self.p.tau_p)
        y[2]=np.log(np.maximum((n+dn)/self.nscale,1e-300))
        y[3]=np.log(np.maximum((p+dn)/self.nscale,1e-300))
        y[4]=0.5*light_fraction*(1-2*s);y[5]=-y[4]
        return _Seed(s,y)

    def solve_state(self,U_H,U_O,light_fraction,previous=None,tol=None,nmesh=None,high_injection=False):
        if tol is None:tol=self.num.bvp_tol_light if light_fraction>0 else self.num.bvp_tol_dark
        if nmesh is None:nmesh=self.num.bvp_nodes
        s=self._initial_mesh(nmesh);bc=self._bc(float(U_H),float(U_O))
        if previous is None:
            Umean=0.5*(U_H+U_O);prev=self.equilibrium_seed(Umean,nmesh)
            # Homotopy in catalyst-potential separation keeps Newton on the
            # physical branch through strongly depleted exponential regimes.
            nstep=max(2,int(abs(U_O-U_H)/0.06)+2)
            for a in np.linspace(0,1,nstep)[1:]:
                ua=Umean+a*(U_H-Umean);ub=Umean+a*(U_O-Umean)
                sol=solve_bvp(self._ode(0.0),self._bc(ua,ub),s,prev.sol(s),
                              tol=max(tol,7e-7),max_nodes=self.num.bvp_max_nodes)
                if sol.status!=0:return sol
                prev=sol
            previous=prev
        if light_fraction<=0:
            return solve_bvp(self._ode(0.0),bc,s,previous.sol(s),tol=tol,max_nodes=self.num.bvp_max_nodes)
        seed=self._high_injection_seed(previous,light_fraction,nmesh) if high_injection else previous
        sol=solve_bvp(self._ode(light_fraction),bc,s,seed.sol(s),tol=tol,max_nodes=self.num.bvp_max_nodes)
        if sol.status==0:return sol
        # Logarithmic light continuation is a numerical stabilization only;
        # endpoint equations are unchanged.
        prev=previous
        for lf in np.geomspace(1e-4,light_fraction,11):
            guess=self._high_injection_seed(prev,float(lf),nmesh)
            sol=solve_bvp(self._ode(float(lf)),bc,s,guess.sol(s),tol=max(tol,1.2e-6),max_nodes=self.num.bvp_max_nodes)
            if sol.status!=0:return sol
            prev=sol
        return prev

    # ----------------------- reversible Faradaic kinetics ----------------------
    def faradaic_currents_A_m2(self,U_H,U_O):
        etaH=U_H-self.far.U_HER_eq_V;etaO=U_O-self.far.U_OER_eq_V
        fHp=10.0**np.clip(etaH/self.far.b_H_V_dec,-120,120)
        fHm=10.0**np.clip(-etaH/self.far.b_H_V_dec,-120,120)
        fOp=10.0**np.clip(etaO/self.far.b_O_V_dec,-120,120)
        fOm=10.0**np.clip(-etaO/self.far.b_O_V_dec,-120,120)
        jH_cm2=self.far.j0_H_A_cm2*(fHp-fHm)/(1+(self.far.j0_H_A_cm2/self.far.jlim_HOR_A_cm2)*fHp)
        jO_cm2=self.far.j0_O_A_cm2*(fOp-fOm)/(1+(self.far.j0_O_A_cm2/self.far.jlim_ORR_A_cm2)*fOm)
        return 1e4*jH_cm2,1e4*jO_cm2,float(etaH),float(etaO)

    def invert_H_for_forward_current(self,J_A_m2):
        return brentq(lambda u:self.faradaic_currents_A_m2(u,self.far.U_OER_eq_V)[0]+J_A_m2,
                      -1.2,-1e-14,xtol=1e-13,rtol=1e-12)
    def invert_O_for_forward_current(self,J_A_m2):
        return brentq(lambda u:self.faradaic_currents_A_m2(self.far.U_HER_eq_V,u)[1]-J_A_m2,
                      self.far.U_OER_eq_V+1e-14,2.4,xtol=1e-13,rtol=1e-12)

    # ------------------------------ diagnostics --------------------------------
    def profiles(self,sol,npts=2001):
        s=np.linspace(0,1,npts);v,d,lnN,lnP,jn,jp=sol.sol(s)
        N=np.exp(np.clip(lnN,-220,120));P=np.exp(np.clip(lnP,-220,120))
        n=N*self.nscale;p=P*self.nscale
        Ucb=self.Ucb0+self.VT*v;Uvb=Ucb+self.p.Eg
        Ufn=Ucb-self.VT*np.log(np.maximum(n/self.Nc,1e-300))
        Ufp=Uvb+self.VT*np.log(np.maximum(p/self.Nv,1e-300))
        return s,n,p,Ucb,Uvb,Ufn,Ufp,jn*self.Jscale,jp*self.Jscale,N,P,d

    def band_bending(self,sol):
        s,n,p,Ucb,Uvb,Ufn,Ufp,Jn,Jp,N,P,d=self.profiles(sol,2401)
        w=self.facet.central_bulk_window;mask=(s>=0.5-w)&(s<=0.5+w)
        Ubulk=float(np.mean(Ucb[mask]))
        return Ubulk-float(Ucb[0]),Ubulk-float(Ucb[-1]),Ubulk

    def diagnostics(self,sol,U_H,U_O,light_fraction):
        s,n,p,Ucb,Uvb,Ufn,Ufp,Jn,Jp,N,P,d=self.profiles(sol,2401)
        Jsem=float((Jn+Jp)[0]);jH,jO,etaH,etaO=self.faradaic_currents_A_m2(U_H,U_O)
        rr=self.srh_over_Gref(N,P);Jrec=float(np.trapezoid(rr,s)*self.Jscale);Jgen=float(light_fraction*self.Jscale)
        BH,BO,Ubulk=self.band_bending(sol);mid=len(s)//2
        wrong_H=max(0.0,-float(Jp[0])) if Jsem>=0 else max(0.0,float(Jn[0]))
        wrong_O=max(0.0,-float(Jn[-1])) if Jsem>=0 else max(0.0,float(Jp[-1]))
        counter=wrong_H+wrong_O;budget=Jgen-Jrec-counter-Jsem if Jsem>=0 else np.nan
        return dict(U_H_V=float(U_H),U_O_V=float(U_O),MIEC_separation_V=float(U_O-U_H),
                    eta_H_V=etaH,eta_O_V=etaO,Jsem_A_m2=Jsem,Jsem_mA_cm2=Jsem/10,
                    jH_A_m2=jH,jH_mA_cm2=jH/10,jO_A_m2=jO,jO_mA_cm2=jO/10,
                    MIEC_balance_H_A_m2=Jsem+jH,MIEC_balance_O_A_m2=-Jsem+jO,
                    Jgen_A_m2=Jgen,Jgen_mA_cm2=Jgen/10,Jrec_A_m2=Jrec,Jrec_mA_cm2=Jrec/10,
                    counterflow_A_m2=counter,counterflow_mA_cm2=counter/10,
                    current_budget_residual_A_m2=budget,
                    band_bending_H_V=BH,band_bending_O_V=BO,Ucb_bulk_V=Ubulk,
                    Ucb_H_V=float(Ucb[0]),Ucb_O_V=float(Ucb[-1]),
                    Ufn_center_V=float(Ufn[mid]),Ufp_center_V=float(Ufp[mid]),
                    QFL_center_split_V=float(Ufp[mid]-Ufn[mid]),
                    Jn_H_A_m2=float(Jn[0]),Jp_H_A_m2=float(Jp[0]),Jn_O_A_m2=float(Jn[-1]),Jp_O_A_m2=float(Jp[-1]),
                    Jn_H_mA_cm2=float(Jn[0]/10),Jp_H_mA_cm2=float(Jp[0]/10),
                    Jn_O_mA_cm2=float(Jn[-1]/10),Jp_O_mA_cm2=float(Jp[-1]/10),
                    n_H_cm3=float(n[0]/1e6),p_H_cm3=float(p[0]/1e6),n_O_cm3=float(n[-1]/1e6),p_O_cm3=float(p[-1]/1e6),
                    max_BVP_rms_residual=float(np.max(sol.rms_residuals)) if hasattr(sol,'rms_residuals') else np.nan)

    def solve_relaxed_dark(self):
        sol=self.solve_state(self.far.U_HER_eq_V,self.far.U_OER_eq_V,0.0,previous=None,tol=self.num.bvp_tol_dark)
        if sol.status!=0:raise RuntimeError('relaxed dark BVP failed: '+sol.message)
        self.dark_sol=sol;self.dark_diag=self.diagnostics(sol,self.far.U_HER_eq_V,self.far.U_OER_eq_V,0.0)
        return sol,self.dark_diag

    def solve_ows(self,light_fraction=1.0,initial_sol=None,verbose=False):
        if self.dark_sol is None:self.solve_relaxed_dark()
        if initial_sol is None:
            initial_sol=self.solve_state(self.far.U_HER_eq_V,self.far.U_OER_eq_V,light_fraction,
                                         previous=self.dark_sol,high_injection=True)
            if initial_sol.status!=0:raise RuntimeError('initial light BVP failed: '+initial_sol.message)
        cache=[]
        def residual(J):
            UH=self.invert_H_for_forward_current(J);UO=self.invert_O_for_forward_current(J)
            prev=initial_sol if not cache else min(cache,key=lambda z:abs(z[0]-J))[1]
            sol=self.solve_state(UH,UO,light_fraction,previous=prev,tol=max(self.num.bvp_tol_light,9e-7))
            if sol.status!=0:sol=self.solve_state(UH,UO,light_fraction,previous=self.dark_sol,high_injection=True,tol=1.3e-6)
            if sol.status!=0:raise RuntimeError(f'OWS trial BVP failed at J={J}: {sol.message}')
            dg=self.diagnostics(sol,UH,UO,light_fraction);cache.append((J,sol,dg,UH,UO))
            if verbose:print('trial',J/10,'mA/cm2 ->',dg['Jsem_mA_cm2'])
            return dg['Jsem_A_m2']-J
        Jseed=max(1e-8,self.diagnostics(initial_sol,self.far.U_HER_eq_V,self.far.U_OER_eq_V,light_fraction)['Jsem_A_m2'])
        lo=1e-10;hi=max(2*Jseed,0.2);flo=residual(lo);fhi=residual(hi)
        for _ in range(12):
            if flo*fhi<=0:break
            hi*=1.7;fhi=residual(hi)
        if flo*fhi>0:raise RuntimeError('No forward OWS current root bracketed')
        J=brentq(lambda x:residual(x),lo,hi,xtol=2e-9,rtol=2e-10,maxiter=80)
        near=min(cache,key=lambda z:abs(z[0]-J));UH=self.invert_H_for_forward_current(J);UO=self.invert_O_for_forward_current(J)
        sol=self.solve_state(UH,UO,light_fraction,previous=near[1],tol=3e-7,nmesh=1100)
        if sol.status!=0:sol=near[1]
        return J,UH,UO,sol,self.diagnostics(sol,UH,UO,light_fraction),initial_sol

    def profile_dataframe(self,sol,npts=1801):
        s,n,p,Ucb,Uvb,Ufn,Ufp,Jn,Jp,N,P,d=self.profiles(sol,npts)
        return pd.DataFrame(dict(x_um=s*self.p.L_s*1e6,U_CB_V_vs_RHE=Ucb,U_VB_V_vs_RHE=Uvb,
                                 U_Fn_V_vs_RHE=Ufn,U_Fp_V_vs_RHE=Ufp,n_cm3=n/1e6,p_cm3=p/1e6,
                                 Jn_mA_cm2=Jn/10,Jp_mA_cm2=Jp/10,Jtotal_mA_cm2=(Jn+Jp)/10))



# ---------------------- publication control calculations ----------------------
def make_params(intensity=10.0, k=10.0):
    return ModelParams(intensity_mW_cm2=float(intensity), k_n=float(k), k_p=float(k))


def facet_from_mean_delta(mean, delta):
    return FacetConfig(1e18, float(mean-delta/2), float(mean+delta/2))


def solve_adaptive_case(mean=.925, delta=.15, k=10.0, intensity=10.0):
    """Fully adaptive AA OWS solution."""
    facet=facet_from_mean_delta(mean,delta)
    p=make_params(intensity,k)
    m=CooperativeAdaptiveJunction(p,facet,lambda_H=0.0,lambda_O=0.0)
    ds,dd=m.solve_relaxed_dark()
    il=m.solve_state(m.far.U_HER_eq_V,m.far.U_OER_eq_V,1.0,
                     previous=ds,high_injection=True,tol=1.2e-6)
    if il.status!=0: raise RuntimeError(il.message)
    J,UH,UO,fs,fd,_=m.solve_ows(1.0,il)
    return m,ds,dd,il,m.diagnostics(il,m.far.U_HER_eq_V,m.far.U_OER_eq_V,1.0),J,UH,UO,fs,fd


class BBLoadLineSolver:
    """Buried/buried control as a photovoltaic J-V curve plus HOR/ORR load.

    Both contacts use fixed physical Schottky barriers.  The semiconductor
    depends only on the terminal metal-potential difference V=U_O-U_H.  The
    H2/O2 chemistry is then added as an external electrochemical load curve.
    """
    def __init__(self, device: CooperativeAdaptiveJunction):
        if not (np.isclose(device.lambda_H,1.0) and np.isclose(device.lambda_O,1.0)):
            raise ValueError('BBLoadLineSolver requires both contacts buried')
        self.m=device
        self._cache={}

    def _nearest_seed(self,V):
        if not self._cache:return None
        return self._cache[min(self._cache,key=lambda x:abs(x-V))][0]

    def solve_semiconductor_at_V(self,V,tol=3e-7,nmesh=900):
        V=float(V)
        key=round(V,14)
        if key in self._cache:return self._cache[key]
        UH,UO=0.0,V
        prev=self._nearest_seed(V)
        if prev is None:
            dark=self.m.solve_state(UH,UO,0.0,previous=None,tol=self.m.num.bvp_tol_dark,nmesh=nmesh)
            if dark.status!=0:raise RuntimeError(dark.message)
            sol=self.m.solve_state(UH,UO,1.0,previous=dark,high_injection=True,tol=tol,nmesh=nmesh)
        else:
            sol=self.m.solve_state(UH,UO,1.0,previous=prev,tol=tol,nmesh=nmesh)
        if sol.status!=0:raise RuntimeError(sol.message)
        dg=self.m.diagnostics(sol,UH,UO,1.0)
        self._cache[key]=(sol,dg)
        return sol,dg

    def build_jv(self,vmin=0.0,vmax=0.22,dv=0.005):
        rows=[]
        for V in np.arange(vmin,vmax+0.5*dv,dv):
            sol,dg=self.solve_semiconductor_at_V(float(V))
            rows.append(dict(V_terminal_V=V,Jsem_mA_cm2=dg['Jsem_mA_cm2'],
                             U_CB_H_V=dg['Ucb_H_V'],U_CB_O_V=dg['Ucb_O_V'],
                             BVP_residual=dg['max_BVP_rms_residual']))
        return pd.DataFrame(rows)

    def solve_voc(self):
        center=self.m.Phi_difference_Sch_eV
        V=brentq(lambda x:self.solve_semiconductor_at_V(x)[1]['Jsem_A_m2'],
                 center-.03,center+.03,xtol=2e-12,rtol=2e-12,maxiter=80)
        sol,dg=self.solve_semiconductor_at_V(V,tol=2e-7,nmesh=1200)
        return V,sol,dg

    def U_H_for_reverse_I(self,I):
        lim=self.m.far.jlim_HOR_A_cm2*1e4
        if not 0<I<lim:raise ValueError('I must be below HOR limiting current')
        return brentq(lambda U:self.m.faradaic_currents_A_m2(U,self.m.far.U_OER_eq_V)[0]-I,
                      self.m.far.U_HER_eq_V,1.5,xtol=2e-14,rtol=2e-14)

    def U_O_for_reverse_I(self,I):
        lim=self.m.far.jlim_ORR_A_cm2*1e4
        if not 0<I<lim:raise ValueError('I must be below ORR limiting current')
        return brentq(lambda U:self.m.faradaic_currents_A_m2(self.m.far.U_HER_eq_V,U)[1]+I,
                      -0.5,self.m.far.U_OER_eq_V,xtol=2e-14,rtol=2e-14)

    def load_at_I(self,I):
        UH=self.U_H_for_reverse_I(I);UO=self.U_O_for_reverse_I(I)
        return UH,UO,UO-UH

    def solve_operating_point(self,voc=None):
        if voc is None:voc,_,_=self.solve_voc()
        lim=min(self.m.far.jlim_HOR_A_cm2,self.m.far.jlim_ORR_A_cm2)*1e4
        # z=-log10(delta), delta=1-I/Ilim keeps precision close to transport limit.
        def R(z):
            I=lim*(1.0-10.0**(-float(z)))
            UH,UO,V=self.load_at_I(I)
            return self.solve_semiconductor_at_V(V)[1]['Jsem_A_m2']+I
        zr=brentq(R,2.0,6.5,xtol=2e-11,rtol=2e-11,maxiter=100)
        delta=10.0**(-zr);I=lim*(1.0-delta);UH,UO,V=self.load_at_I(I)
        sol,dg=self.solve_semiconductor_at_V(V,tol=2e-7,nmesh=1200)
        return dict(Voc_V=voc,z_root=zr,delta_fraction=delta,I_reverse_A_m2=I,
                    I_reverse_uA_cm2=I/1e4*1e6,Jsem_A_m2=dg['Jsem_A_m2'],
                    Jsem_uA_cm2=dg['Jsem_A_m2']/1e4*1e6,U_H_V=UH,U_O_V=UO,
                    V_terminal_V=V,jH_A_m2=dg['jH_A_m2'],jO_A_m2=dg['jO_A_m2'],
                    U_CB_H_V=dg['Ucb_H_V'],U_CB_O_V=dg['Ucb_O_V'],
                    surface_band_difference_V=dg['Ucb_O_V']-dg['Ucb_H_V'],
                    BVP_residual=dg['max_BVP_rms_residual']),sol,dg


def _gauge_shift_seed(model,sol,common_shift_V):
    y=sol.y.copy();y[0]+=common_shift_V/model.VT
    return _Seed(sol.x,y)


def solve_bb_control(p=None,facet=None):
    p=make_params() if p is None else p
    facet=FacetConfig() if facet is None else facet
    m=CooperativeAdaptiveJunction(p,facet,lambda_H=1.0,lambda_O=1.0)
    s=BBLoadLineSolver(m)
    jv=s.build_jv();voc,vocsol,vocdg=s.solve_voc();op,gauge_sol,gauge_dg=s.solve_operating_point(voc)
    # Re-solve at the actual absolute catalyst potentials to verify gauge invariance.
    seed=_gauge_shift_seed(m,gauge_sol,op['U_H_V'])
    physical=m.solve_state(op['U_H_V'],op['U_O_V'],1.0,previous=seed,tol=1e-8,nmesh=1800)
    if physical.status!=0:raise RuntimeError(physical.message)
    pdg=m.diagnostics(physical,op['U_H_V'],op['U_O_V'],1.0)
    op.update(dict(Jsem_A_m2=pdg['Jsem_A_m2'],Jsem_uA_cm2=pdg['Jsem_A_m2']/1e4*1e6,
                   jH_A_m2=pdg['jH_A_m2'],jO_A_m2=pdg['jO_A_m2'],
                   U_CB_H_V=pdg['Ucb_H_V'],U_CB_O_V=pdg['Ucb_O_V'],
                   surface_band_difference_V=pdg['Ucb_O_V']-pdg['Ucb_H_V'],
                   H_balance_A_m2=pdg['MIEC_balance_H_A_m2'],O_balance_A_m2=pdg['MIEC_balance_O_A_m2'],
                   BVP_residual=pdg['max_BVP_rms_residual']))
    return m,s,jv,voc,vocsol,vocdg,op,physical,pdg


class MixedZeroCurrentSolver:
    """Catalyst-off zero-current sweep for one adaptive and one buried contact."""
    def __init__(self,mode,p=None,facet=None):
        if mode not in ('AB','BA'):raise ValueError('mode must be AB or BA')
        self.mode=mode
        self.p=make_params() if p is None else p
        self.facet=FacetConfig() if facet is None else facet
        self.m=CooperativeAdaptiveJunction(self.p,self.facet,
            lambda_H=0.0 if mode[0]=='A' else 1.0,
            lambda_O=0.0 if mode[1]=='A' else 1.0)
        self.previous=None
        self.last_float=None

    def solve_point(self,U_buried,U_float_guess=None,tol=2.5e-6,nmesh=650):
        # AB: fixed U_O, solve floating U_H. BA: fixed U_H, solve floating U_O.
        if U_float_guess is None:
            U_float_guess=(-0.69 if self.mode=='AB' else 2.19) if self.last_float is None else self.last_float
        cache=[]
        def eval_float(Uf):
            UH,UO=(Uf,U_buried) if self.mode=='AB' else (U_buried,Uf)
            seed=self.previous if not cache else min(cache,key=lambda z:abs(z[0]-Uf))[1]
            if seed is None:
                sol=self.m.solve_state(UH,UO,1.0,previous=None,tol=tol,nmesh=nmesh)
            else:
                sol=self.m.solve_state(UH,UO,1.0,previous=seed,tol=tol,nmesh=nmesh)
            if sol.status!=0:raise RuntimeError(sol.message)
            dg=self.m.diagnostics(sol,UH,UO,1.0);cache.append((Uf,sol,dg))
            return dg['Jsem_A_m2']
        # Secant is efficient on a continuation branch; a broad bracket is fallback.
        try:
            r=root(lambda x:np.array([eval_float(float(x[0]))]),np.array([U_float_guess]),tol=2e-9)
            if not r.success:raise RuntimeError(r.message)
            Uf=float(r.x[0])
        except Exception:
            center=float(U_float_guess);xs=np.linspace(center-0.5,center+0.5,31);last=None;br=None
            for x in xs:
                try:f=eval_float(float(x))
                except Exception:continue
                if last is not None and f*last[1]<=0:br=(last[0],float(x));break
                last=(float(x),float(f))
            if br is None:raise RuntimeError('Could not bracket mixed-contact zero-current root')
            Uf=brentq(lambda x:eval_float(float(x)),*br,xtol=2e-9,rtol=2e-9)
        near=min(cache,key=lambda z:abs(z[0]-Uf));self.previous=near[1];self.last_float=Uf
        UH,UO=(Uf,U_buried) if self.mode=='AB' else (U_buried,Uf)
        dg=self.m.diagnostics(near[1],UH,UO,1.0)
        return UH,UO,near[1],dg


def estimate_saturation_asymptote(df,buried_col,voc_col,direction='positive',fit_min=None,fit_max=None):
    d=df.copy();x=d[buried_col].to_numpy(float);y=d[voc_col].to_numpy(float)
    xx=x if direction=='positive' else -x
    mask=np.isfinite(xx)&np.isfinite(y)
    if fit_min is not None:mask &= xx>=fit_min
    if fit_max is not None:mask &= xx<=fit_max
    xx=xx[mask];y=y[mask]
    if len(xx)<4:return float(np.nanmax(y))
    def fn(z,vinf,A,k):return vinf-A*np.exp(-k*z)
    p0=[float(np.nanmax(y)),max(1e-6,float(np.nanmax(y)-np.nanmin(y))),20.0]
    try:
        popt,_=curve_fit(fn,xx,y,p0=p0,maxfev=20000)
        return float(popt[0])
    except Exception:
        return float(y[-1])


def sweep_ab_catalyst_off(precomputed=None):
    """Upper-bound catalyst-off Voc for adaptive HER / buried OER.

    The ideal buried O contact is swept into reverse bias.  The adaptive H
    potential is floated at J_sem=0.  The voltage approaches an asymptotic
    ceiling as the buried contact becomes perfectly hole selective.
    """
    if precomputed is not None and Path(precomputed).exists():
        df=pd.read_csv(precomputed).sort_values('U_O_V').reset_index(drop=True)
        vmax=float(df['Voc_V'].max())
        # Deep reverse-bias points are already on the plateau; average the last
        # several highest-reverse-bias values for a stable quoted ceiling.
        tail=df.nsmallest(min(8,len(df)),'U_O_V')
        vmax=float(tail['Voc_V'].mean())
        return df,vmax,None,None
    s=MixedZeroCurrentSolver('AB')
    # Direct continuation is reliable through U_O=-0.40 V, where the voltage
    # is already within ~2e-5 V of the asymptote.
    points=np.r_[0.0,-np.arange(.02,.42,.02)]
    rows=[];last=-0.690984;rep=None
    for UO in points:
        UH,UO2,sol,dg=s.solve_point(float(UO),last)
        last=UH
        rows.append(dict(U_H_V=UH,U_O_V=UO2,Voc_V=UO2-UH,Jsem_mA_cm2=dg['Jsem_mA_cm2'],
                         surface_band_diff_V=dg['Ucb_O_V']-dg['Ucb_H_V'],
                         Jn_H_mA_cm2=dg['Jn_H_mA_cm2'],Jp_H_mA_cm2=dg['Jp_H_mA_cm2'],
                         QFL_center_split_V=dg['QFL_center_split_V'],BVP_residual=dg['max_BVP_rms_residual']))
        if abs(UO+0.40)<1e-9:rep=(s.m,sol,UH,UO2,dg)
    df=pd.DataFrame(rows).sort_values('U_O_V').reset_index(drop=True)
    vmax=estimate_saturation_asymptote(df,'U_O_V','Voc_V','negative',fit_min=.30,fit_max=.40)
    return df,vmax,rep,s


def sweep_ba_catalyst_off(precomputed=None):
    """Upper-bound catalyst-off Voc for buried HER / adaptive OER."""
    if precomputed is not None and Path(precomputed).exists():
        df=pd.read_csv(precomputed).sort_values('U_H_V').reset_index(drop=True)
        vmax=estimate_saturation_asymptote(df,'U_H_V','Voc_V','positive',fit_min=.025,fit_max=.15)
        return df,vmax,None,None
    s=MixedZeroCurrentSolver('BA')
    points=[-0.10,-0.05,0.0,0.025,0.05,0.075,0.10,0.125,0.15]
    rows=[];last=2.19;rep=None
    for UH in points:
        UH2,UO,sol,dg=s.solve_point(float(UH),last)
        last=UO
        rows.append(dict(U_H_V=UH2,U_O_V=UO,Voc_V=UO-UH2,Jsem_mA_cm2=dg['Jsem_mA_cm2'],
                         surface_band_diff_V=dg['Ucb_O_V']-dg['Ucb_H_V'],
                         Jn_O_mA_cm2=dg['Jn_O_mA_cm2'],Jp_O_mA_cm2=dg['Jp_O_mA_cm2'],
                         QFL_center_split_V=dg['QFL_center_split_V'],BVP_residual=dg['max_BVP_rms_residual']))
        if abs(UH-.10)<1e-9:rep=(s.m,sol,UH2,UO,dg)
    df=pd.DataFrame(rows).sort_values('U_H_V').reset_index(drop=True)
    vmax=estimate_saturation_asymptote(df,'U_H_V','Voc_V','positive',fit_min=.025,fit_max=.15)
    return df,vmax,rep,s


def solve_ba_finite_current(p=None,facet=None):
    """Finite-current BA OWS state from Faradaic load-line intersection."""
    p=make_params() if p is None else p;facet=FacetConfig() if facet is None else facet
    m=CooperativeAdaptiveJunction(p,facet,lambda_H=1.0,lambda_O=0.0)
    # Physical illumination seed at the zero-overpotential catalyst potentials.
    seed=m.solve_state(0.0,1.229,1.0,previous=None,high_injection=True,tol=1.2e-6,nmesh=800)
    if seed.status!=0:raise RuntimeError(seed.message)
    cache=[]
    def F(J):
        UH=m.invert_H_for_forward_current(J);UO=m.invert_O_for_forward_current(J)
        prev=seed if not cache else min(cache,key=lambda z:abs(z[0]-J))[1]
        sol=m.solve_state(UH,UO,1.0,previous=prev,tol=8e-7,nmesh=850)
        if sol.status!=0:raise RuntimeError(sol.message)
        dg=m.diagnostics(sol,UH,UO,1.0);cache.append((J,sol,dg,UH,UO))
        return dg['Jsem_A_m2']-J
    # Baseline root lies near 0.30 A/m2 = 0.030 mA/cm2.
    J=brentq(F,0.10,0.60,xtol=2e-11,rtol=2e-11,maxiter=100)
    near=min(cache,key=lambda z:abs(z[0]-J));UH=m.invert_H_for_forward_current(J);UO=m.invert_O_for_forward_current(J)
    sol=m.solve_state(UH,UO,1.0,previous=near[1],tol=1e-8,nmesh=1800)
    if sol.status!=0:raise RuntimeError(sol.message)
    dg=m.diagnostics(sol,UH,UO,1.0)
    # Independent two-potential check.
    state=[sol]
    def R(x):
        h,o=map(float,x)
        ss=m.solve_state(h,o,1.0,previous=state[-1],tol=5e-7,nmesh=900)
        if ss.status!=0:return np.array([1e3,1e3])
        state.append(ss);dd=m.diagnostics(ss,h,o,1.0)
        return np.array([dd['Jsem_A_m2']+dd['jH_A_m2'],-dd['Jsem_A_m2']+dd['jO_A_m2']])
    chk=root(R,np.array([UH,UO]),tol=1e-10)
    return m,J,UH,UO,sol,dg,chk


# ------------------------------- plotting ------------------------------------
def save_figure(fig,path):
    fig.tight_layout();fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'),dpi=300,bbox_inches='tight');plt.close(fig)


def conventional_energy(ax,model,sol,UH,UO,title,redox=True):
    df=model.profile_dataframe(sol);x=df.x_um.to_numpy()
    ax.plot(x,-df.U_CB_V_vs_RHE,label=r'$E_C$',lw=1.9)
    ax.plot(x,-df.U_VB_V_vs_RHE,label=r'$E_V$',lw=1.9)
    ax.plot(x,-df.U_Fn_V_vs_RHE,label=r'$E_{F,n}$',ls='--',lw=1.3)
    ax.plot(x,-df.U_Fp_V_vs_RHE,label=r'$E_{F,p}$',ls='--',lw=1.3)
    ax.plot([-0.08,0],[-UH,-UH],lw=3,label='catalyst electron level')
    ax.plot([model.p.L_s*1e6,model.p.L_s*1e6+0.08],[-UO,-UO],lw=3)
    if redox:
        ax.axhline(0,ls=':',lw=1.0,label='solution redox level')
        ax.axhline(-1.229,ls=':',lw=1.0)
    ax.axvline(0,lw=.6);ax.axvline(model.p.L_s*1e6,lw=.6)
    ax.set_title(title,fontsize=9.5);ax.set_xlabel(r'Position ($\mu$m)')
    ax.set_ylabel('Electron energy (eV; higher upward)');ax.grid(alpha=.12)


def figure_model_framework(outdir,model):
    fig,ax=plt.subplots(figsize=(8.6,4.5));ax.axis('off')
    ax.add_patch(plt.Rectangle((.29,.27),.42,.44,fill=False,lw=1.5))
    ax.add_patch(plt.Rectangle((.06,.31),.18,.36,fill=False,lw=1.5))
    ax.add_patch(plt.Rectangle((.76,.31),.18,.36,fill=False,lw=1.5))
    ax.text(.50,.62,'n-SrTiO$_3$\nPoisson + drift-diffusion + SRH',ha='center',va='center')
    ax.text(.15,.49,'HER-side\ncontact',ha='center',va='center');ax.text(.85,.49,'OER-side\ncontact',ha='center',va='center')
    ax.annotate('',xy=(.30,.49),xytext=(.23,.49),arrowprops=dict(arrowstyle='<->'))
    ax.annotate('',xy=(.77,.49),xytext=(.70,.49),arrowprops=dict(arrowstyle='<->'))
    ax.text(.265,.55,'Mills',ha='center',fontsize=8);ax.text(.735,.55,'Mills',ha='center',fontsize=8)
    ax.text(.50,.18,r'Native facet asymmetry: $U_{SC,H}^0=0.85$ V, $U_{SC,O}^0=1.00$ V',ha='center')
    ax.text(.50,.08,r'Adaptive (A): $U_{CB,s}=U_{CB,s}^0$;  Buried (B): $U_{CB,s}=U_{cat}-\Phi_B^0$',ha='center',fontsize=9)
    ax.text(.15,.77,'HER/HOR\n0 V vs RHE',ha='center');ax.text(.85,.77,'OER/ORR\n1.229 V vs RHE',ha='center')
    save_figure(fig,outdir/'Figure_S14_model_framework')


def figure_band_diagrams(outdir,states):
    fig,axs=plt.subplots(2,2,figsize=(11.2,8.4),sharex=True)
    for ax,(model,sol,UH,UO,title) in zip(axs.ravel(),states):
        conventional_energy(ax,model,sol,UH,UO,title)
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=6,frameon=False,fontsize=8)
    fig.subplots_adjust(bottom=.12,hspace=.30,wspace=.25)
    fig.savefig(outdir/'Figure_S15_conventional_band_diagrams.pdf',bbox_inches='tight')
    fig.savefig(outdir/'Figure_S15_conventional_band_diagrams.png',dpi=300,bbox_inches='tight');plt.close(fig)


def figure_controls(outdir,bb_jv,bb_op,abdf,ab_vmax,badf,ba_vmax,ba_sweep,ba_dg):
    fig,axs=plt.subplots(2,2,figsize=(11.2,8.3))
    # BB photovoltaic J-V and electrochemical load.
    ax=axs[0,0];ax.plot(bb_jv.V_terminal_V,bb_jv.Jsem_mA_cm2,label='BB semiconductor J-V')
    lim=0.05;zs=np.linspace(2.2,5.2,80);Is=lim*(1-10**(-zs))
    # Direct analytic inversion through the same model is not available here; use
    # the validated operating point and a visual guide from the Faradaic law.
    # Construct via formulas using baseline parameters.
    f=FaradaicConfig();vals=[]
    for I_mA in Is:
        I=I_mA*10.0
        # temporary adaptive model only supplies Faradaic inversion
        tm=CooperativeAdaptiveJunction(ModelParams(),FacetConfig())
        UH=brentq(lambda u:tm.faradaic_currents_A_m2(u,1.229)[0]-I,0,1.5)
        UO=brentq(lambda u:tm.faradaic_currents_A_m2(0,u)[1]+I,-.5,1.229)
        vals.append((UO-UH,-I_mA))
    vals=np.array(vals);ax.plot(vals[:,0],vals[:,1],label='HOR/ORR load')
    ax.scatter([bb_op['V_terminal_V']],[bb_op['Jsem_uA_cm2']/1000],s=50,zorder=4,label='operating point')
    ax.axvline(bb_op['Voc_V'],ls=':',lw=1,label=r'$V_{OC}$')
    ax.set_xlim(.13,.18);ax.set_ylim(-.25,.25);ax.set_xlabel('Terminal separation (V)');ax.set_ylabel(r'Current (mA cm$^{-2}$)');ax.set_title('(a) BB load line');ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=8)
    # AB upper bound.
    ax=axs[0,1];x=-abdf.U_O_V.to_numpy();ax.plot(x,abdf.Voc_V,marker='o',ms=2.5)
    ax.axhline(1.229,ls=':',label='1.229 V OWS minimum');ax.axhline(ab_vmax,ls='--',label=f'ceiling {ab_vmax:.3f} V')
    ax.set_xlabel('Buried O-side reverse bias coordinate, $-U_O$ (V)');ax.set_ylabel(r'$V_{OC}$ (V)');ax.set_title('(b) AB catalyst-off voltage ceiling');ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=8)
    # BA upper bound.
    ax=axs[1,0];ax.plot(badf.U_H_V,badf.Voc_V,marker='o',ms=3)
    ax.axhline(1.229,ls=':',label='1.229 V OWS minimum');ax.axhline(ba_vmax,ls='--',label=f'ceiling {ba_vmax:.3f} V')
    ax.set_xlabel('Buried H-side bias coordinate, $U_H$ (V)');ax.set_ylabel(r'$V_{OC}$ (V)');ax.set_title('(c) BA catalyst-off voltage ceiling');ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=8)
    # BA finite-current load line.
    ax=axs[1,1];s=ba_sweep.sort_values('trial_J_mA_cm2');mask=s.trial_J_mA_cm2<=.12
    ax.plot(s.loc[mask,'trial_J_mA_cm2'],s.loc[mask,'Jsem_mA_cm2'],marker='o',ms=3,label=r'$J_{sem}$')
    ax.plot(s.loc[mask,'trial_J_mA_cm2'],s.loc[mask,'trial_J_mA_cm2'],ls='--',label=r'$J_{Faradaic}=J$')
    ax.scatter([ba_dg['Jsem_mA_cm2']],[ba_dg['Jsem_mA_cm2']],s=50,zorder=4,label='BA OWS point')
    ax.set_xlabel(r'Trial OWS current (mA cm$^{-2}$)');ax.set_ylabel(r'Current (mA cm$^{-2}$)');ax.set_title('(d) BA finite-current intersection');ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=8)
    save_figure(fig,outdir/'Figure_S16_adaptive_buried_controls')


# --------------------------- AA sensitivity sweeps ----------------------------
def run_k_sweep():
    rows=[]
    for k in [0.1,1,10,100,1000]:
        m,ds,dd,il,idi,J,UH,UO,fs,fd=solve_adaptive_case(k=k)
        rows.append(dict(k_m_s=k,J_OWS_mA_cm2=fd['Jsem_mA_cm2'],counterflow_mA_cm2=fd['counterflow_mA_cm2'],
                         SRH_mA_cm2=fd['Jrec_mA_cm2'],MIEC_separation_V=fd['MIEC_separation_V'],QFL_split_V=fd['QFL_center_split_V']))
    return pd.DataFrame(rows)


def _map_solution_seed_between_intensities(src_model,src_sol,dst_model,nmesh=900):
    x=dst_model._initial_mesh(nmesh);y=src_sol.sol(x).copy()
    y[2]+=np.log(src_model.nscale/dst_model.nscale);y[3]+=np.log(src_model.nscale/dst_model.nscale)
    y[4]*=src_model.Jscale/dst_model.Jscale;y[5]*=src_model.Jscale/dst_model.Jscale
    return _Seed(x,y)


def _continue_intensity_from_state(src_model,src_sol,Jsrc,UHsrc,UOsrc,intensity):
    m=CooperativeAdaptiveJunction(make_params(intensity=intensity),FacetConfig())
    m.solve_relaxed_dark();seed=_map_solution_seed_between_intensities(src_model,src_sol,m)
    ratio=float(intensity)/float(src_model.p.intensity_mW_cm2);J0=max(1e-8,float(Jsrc)*ratio)
    UH0=m.invert_H_for_forward_current(J0);UO0=m.invert_O_for_forward_current(J0)
    sol=m.solve_state(UH0,UO0,1.0,previous=seed,tol=1.2e-6,nmesh=900)
    if sol.status!=0:sol=m.solve_state(UHsrc,UOsrc,1.0,previous=seed,tol=2e-6,nmesh=900)
    if sol.status!=0:raise RuntimeError(sol.message)
    cache=[]
    def F(J):
        uh=m.invert_H_for_forward_current(J);uo=m.invert_O_for_forward_current(J)
        prev=sol if not cache else min(cache,key=lambda z:abs(z[0]-J))[1]
        ss=m.solve_state(uh,uo,1.0,previous=prev,tol=1.2e-6,nmesh=900)
        if ss.status!=0:raise RuntimeError(ss.message)
        dg=m.diagnostics(ss,uh,uo,1.0);cache.append((J,ss,dg,uh,uo));return dg['Jsem_A_m2']-J
    vals=J0*np.geomspace(.5,1.5,18);last=None;bracket=None
    for J in vals:
        try:f=F(float(J))
        except Exception:continue
        if last is not None and f*last[1]<=0:bracket=(last[0],float(J));break
        last=(float(J),float(f))
    if bracket is None:raise RuntimeError('No OWS bracket in intensity continuation')
    rr=brentq(F,*bracket,xtol=2e-8,rtol=1e-9,maxiter=60);near=min(cache,key=lambda z:abs(z[0]-rr))
    _,ss,dg,uh,uo=near;return m,ss,rr,uh,uo,dg


def run_intensity_sweep():
    m,ds,dd,il,idi,J,UH,UO,fs,fd=solve_adaptive_case(intensity=10.0);results={10.0:(m,fs,J,UH,UO,fd)}
    src=(m,fs,J,UH,UO)
    for I in [20.,30.]:
        mm,ss,jj,hh,oo,dg=_continue_intensity_from_state(*src,I);results[I]=(mm,ss,jj,hh,oo,dg);src=(mm,ss,jj,hh,oo)
    src=(m,fs,J,UH,UO)
    for I in [5.,3.,1.]:
        mm,ss,jj,hh,oo,dg=_continue_intensity_from_state(*src,I);results[I]=(mm,ss,jj,hh,oo,dg);src=(mm,ss,jj,hh,oo)
    rows=[]
    for I in sorted(results):
        mm,ss,jj,hh,oo,dg=results[I]
        rows.append(dict(intensity_mW_cm2=I,J_OWS_mA_cm2=dg['Jsem_mA_cm2'],Jgen_mA_cm2=dg['Jgen_mA_cm2'],
                         utilization_percent=100*dg['Jsem_mA_cm2']/dg['Jgen_mA_cm2'],counterflow_mA_cm2=dg['counterflow_mA_cm2'],
                         SRH_mA_cm2=dg['Jrec_mA_cm2'],MIEC_separation_V=dg['MIEC_separation_V'],QFL_split_V=dg['QFL_center_split_V'],
                         BVP_residual=dg['max_BVP_rms_residual'],status='ok'))
    return pd.DataFrame(rows)


def run_native_grid():
    rows=[]
    for mean in [0.70,0.85,0.925,1.00,1.15]:
        for delta in [0.00,0.05,0.10,0.15,0.20,0.30]:
            if delta==0:
                rows.append(dict(mean_bending_V=mean,asymmetry_V=delta,J_OWS_mA_cm2=0.0,status='symmetric'));continue
            try:
                m,ds,dd,il,idi,J,UH,UO,fs,fd=solve_adaptive_case(mean,delta)
                rows.append(dict(mean_bending_V=mean,asymmetry_V=delta,J_OWS_mA_cm2=fd['Jsem_mA_cm2'],
                                 counterflow_mA_cm2=fd['counterflow_mA_cm2'],utilization_percent=100*fd['Jsem_mA_cm2']/fd['Jgen_mA_cm2'],status='ok'))
            except Exception:
                rows.append(dict(mean_bending_V=mean,asymmetry_V=delta,J_OWS_mA_cm2=np.nan,status='failed'))
    return pd.DataFrame(rows)


def figure_native_sensitivity(outdir,gdf):
    fig,axs=plt.subplots(1,2,figsize=(10.8,4.2));piv=gdf.pivot(index='mean_bending_V',columns='asymmetry_V',values='J_OWS_mA_cm2').sort_index()
    im=axs[0].imshow(piv.values,origin='lower',aspect='auto',extent=[piv.columns.min(),piv.columns.max(),piv.index.min(),piv.index.max()])
    axs[0].set_xlabel(r'Facet asymmetry $\Delta U_{SC}$ (V)');axs[0].set_ylabel('Mean dark band bending (V)');axs[0].set_title('OWS current');fig.colorbar(im,ax=axs[0],label=r'mA cm$^{-2}$')
    for mean in sorted(gdf.mean_bending_V.unique()):
        sl=gdf[np.isclose(gdf.mean_bending_V,mean)].sort_values('asymmetry_V');axs[1].plot(sl.asymmetry_V,sl.J_OWS_mA_cm2,marker='o',label=f'{mean:g} V mean')
    axs[1].set_xlabel(r'Facet asymmetry $\Delta U_{SC}$ (V)');axs[1].set_ylabel(r'$J_{OWS}$ (mA cm$^{-2}$)');axs[1].set_title('Differential facet asymmetry controls collection');axs[1].legend(frameon=False,fontsize=7);axs[1].grid(alpha=.15)
    save_figure(fig,outdir/'Figure_S17_native_asymmetry_sensitivity')


def figure_rate_intensity(outdir,kdf,idf):
    fig,axs=plt.subplots(2,2,figsize=(10.8,8.0))
    axs[0,0].semilogx(kdf.k_m_s,kdf.J_OWS_mA_cm2,marker='o',label='OWS');axs[0,0].semilogx(kdf.k_m_s,kdf.counterflow_mA_cm2,marker='s',label='counterflow');axs[0,0].set_xlabel(r'common $k_n=k_p$ (m s$^{-1}$)');axs[0,0].set_ylabel(r'current (mA cm$^{-2}$)');axs[0,0].set_title('Common transfer-rate sweep');axs[0,0].legend(frameon=False);axs[0,0].grid(alpha=.15)
    axs[0,1].semilogx(kdf.k_m_s,kdf.MIEC_separation_V,marker='o',label='catalyst separation');axs[0,1].semilogx(kdf.k_m_s,kdf.QFL_split_V,marker='s',label='center QFL split');axs[0,1].set_xlabel(r'common $k$ (m s$^{-1}$)');axs[0,1].set_ylabel('V');axs[0,1].set_title('Voltage weakly dependent on common k');axs[0,1].legend(frameon=False);axs[0,1].grid(alpha=.15)
    axs[1,0].plot(idf.intensity_mW_cm2,idf.J_OWS_mA_cm2,marker='o',label='OWS');axs[1,0].plot(idf.intensity_mW_cm2,idf.Jgen_mA_cm2,marker='s',label='absorbed photon-equivalent');axs[1,0].set_xlabel(r'365-nm intensity (mW cm$^{-2}$)');axs[1,0].set_ylabel(r'current (mA cm$^{-2}$)');axs[1,0].set_title('Photocurrent follows excitation');axs[1,0].legend(frameon=False);axs[1,0].grid(alpha=.15)
    axs[1,1].plot(idf.intensity_mW_cm2,idf.MIEC_separation_V,marker='o',label='catalyst separation');axs[1,1].plot(idf.intensity_mW_cm2,idf.QFL_split_V,marker='s',label='center QFL split');axs[1,1].set_xlabel(r'365-nm intensity (mW cm$^{-2}$)');axs[1,1].set_ylabel('V');axs[1,1].set_title('Adaptive voltage increases with excitation');axs[1,1].legend(frameon=False);axs[1,1].grid(alpha=.15)
    save_figure(fig,outdir/'Figure_S18_rate_and_intensity_sensitivity')


def _ba_loadline_sweep(m,seed):
    rows=[];cache=[]
    for J_mA in np.unique(np.r_[np.linspace(.0001,.12,28),.03037917]):
        J=float(J_mA*10.0);UH=m.invert_H_for_forward_current(J);UO=m.invert_O_for_forward_current(J)
        prev=seed if not cache else min(cache,key=lambda z:abs(z[0]-J))[1]
        try:sol=m.solve_state(UH,UO,1.0,previous=prev,tol=1.2e-6,nmesh=650)
        except Exception:continue
        if sol.status!=0:continue
        dg=m.diagnostics(sol,UH,UO,1.0);cache.append((J,sol,dg))
        rows.append(dict(trial_J_mA_cm2=J_mA,Jsem_mA_cm2=dg['Jsem_mA_cm2'],residual_mA_cm2=dg['Jsem_mA_cm2']-J_mA,
                         U_H_V=UH,U_O_V=UO,U_O_minus_U_H_V=UO-UH))
    return pd.DataFrame(rows)


def run(outdir:Path,full_sweeps=False,precomputed_dir:Optional[Path]=None):
    outdir.mkdir(parents=True,exist_ok=True);figdir=outdir/'figures';resdir=outdir/'results';figdir.mkdir(exist_ok=True);resdir.mkdir(exist_ok=True)
    p=make_params();facet=FacetConfig()

    # AA fully adaptive baseline.
    aa,ds,dd,il,idi,Jaa,UHaa,UOaa,solaa,dgaa=solve_adaptive_case()

    # BB two-terminal photovoltaic control and reverse H2/O2 load state.
    bb,bbsolver,bbjv,bbvoc,bbvocsol,bbvocdg,bbop,bbsol,bbdg=solve_bb_control(p,facet)

    # AB/BA catalyst-off voltage ceilings.  Validated extended AB data may be
    # supplied to avoid the numerically stiff deep-reverse-bias continuation.
    ab_pre=None;ba_pre=None
    if precomputed_dir is not None:
        cand=precomputed_dir/'AB_reverse_bias_extended_Voc_sweep.csv';ab_pre=cand if cand.exists() else None
        cand=precomputed_dir/'BA_catalyst_off_open_circuit_sweep.csv';ba_pre=cand if cand.exists() else None
    abdf,abmax,abrep,absolver=sweep_ab_catalyst_off(ab_pre)
    badf,bamax,barep,basolver=sweep_ba_catalyst_off(ba_pre)

    # BA finite-current OWS state; AB is screened out because its upper-bound
    # catalyst-off voltage is below 1.229 V.
    bam,Jba,UHba,UOba,solba,dgba,bachk=solve_ba_finite_current(p,facet)
    ba_pre=None
    if precomputed_dir is not None:
        for name in ('BA_finite_current_loadline.csv','BA_finite_current_loadline_sweep.csv'):
            cand=precomputed_dir/name
            if cand.exists():
                ba_pre=cand;break
    ba_load=pd.read_csv(ba_pre) if ba_pre is not None else _ba_loadline_sweep(bam,solba)

    # Architecture summary.
    controls=pd.DataFrame([
        dict(mode='AA',HER='adaptive',OER='adaptive',catalyst_off_Voc_max_V=np.nan,OWS_thermodynamically_accessible=True,
             steady_state='OWS',J_mA_cm2=dgaa['Jsem_mA_cm2'],U_H_V=UHaa,U_O_V=UOaa,separation_V=UOaa-UHaa),
        dict(mode='BA',HER='buried',OER='adaptive',catalyst_off_Voc_max_V=bamax,OWS_thermodynamically_accessible=bamax>1.229,
             steady_state='OWS',J_mA_cm2=dgba['Jsem_mA_cm2'],U_H_V=UHba,U_O_V=UOba,separation_V=UOba-UHba),
        dict(mode='AB',HER='adaptive',OER='buried',catalyst_off_Voc_max_V=abmax,OWS_thermodynamically_accessible=abmax>1.229,
             steady_state='reverse expected; OWS screened out',J_mA_cm2=np.nan,U_H_V=np.nan,U_O_V=np.nan,separation_V=np.nan),
        dict(mode='BB',HER='buried',OER='buried',catalyst_off_Voc_max_V=bbvoc,OWS_thermodynamically_accessible=False,
             steady_state='HOR+ORR reverse load',J_mA_cm2=bbdg['Jsem_mA_cm2'],U_H_V=bbop['U_H_V'],U_O_V=bbop['U_O_V'],separation_V=bbop['V_terminal_V']),
    ])
    controls.to_csv(resdir/'architecture_controls_summary.csv',index=False)
    bbjv.to_csv(resdir/'BB_semiconductor_JV.csv',index=False);pd.DataFrame([bbop]).to_csv(resdir/'BB_operating_point.csv',index=False)
    abdf.to_csv(resdir/'AB_catalyst_off_Voc_sweep.csv',index=False);badf.to_csv(resdir/'BA_catalyst_off_Voc_sweep.csv',index=False)
    ba_load.to_csv(resdir/'BA_finite_current_loadline.csv',index=False)
    aa.profile_dataframe(solaa).to_csv(resdir/'AA_operating_profile.csv',index=False)
    bb.profile_dataframe(bbsol).to_csv(resdir/'BB_operating_profile.csv',index=False)
    bam.profile_dataframe(solba).to_csv(resdir/'BA_operating_profile.csv',index=False)

    # Figures S14-S16 use only the final architecture logic.
    figure_model_framework(figdir,aa)
    states=[(aa,ds,0.0,0.0,'(a) Native STO facet junction'),
            (aa,solaa,UHaa,UOaa,'(b) AA: adaptive/adaptive OWS'),
            (bb,bbsol,bbop['U_H_V'],bbop['U_O_V'],'(c) BB: buried/buried reverse load'),
            (bam,solba,UHba,UOba,'(d) BA: buried HER / adaptive OER OWS')]
    figure_band_diagrams(figdir,states)
    figure_controls(figdir,bbjv,bbop,abdf,abmax,badf,bamax,ba_load,dgba)

    # Existing AA sensitivity figures S17-S18.
    if precomputed_dir is not None and (precomputed_dir/'native_band_bending_sweep.csv').exists():
        gdf=pd.read_csv(precomputed_dir/'native_band_bending_sweep.csv')
        kfile=precomputed_dir/'common_transfer_rate_sweep.csv';kdf=pd.read_csv(kfile) if kfile.exists() else run_k_sweep()
        if 'k_m_s' not in kdf.columns and 'k' in kdf.columns:kdf=kdf.rename(columns={'k':'k_m_s'})
        if 'QFL_split_V' not in kdf.columns and 'QFL_center_split_V' in kdf.columns:kdf=kdf.rename(columns={'QFL_center_split_V':'QFL_split_V'})
        ifile=precomputed_dir/'intensity_sweep.csv';idf=pd.read_csv(ifile) if ifile.exists() else run_intensity_sweep()
    else:
        gdf=run_native_grid();kdf=run_k_sweep();idf=run_intensity_sweep()
    gdf.to_csv(resdir/'native_band_bending_sweep.csv',index=False);kdf.to_csv(resdir/'common_transfer_rate_sweep.csv',index=False);idf.to_csv(resdir/'intensity_sweep.csv',index=False)
    figure_native_sensitivity(figdir,gdf);figure_rate_intensity(figdir,kdf,idf)

    summary=pd.DataFrame([dict(Jgen_mA_cm2=dgaa['Jgen_mA_cm2'],AA_J_OWS_mA_cm2=dgaa['Jsem_mA_cm2'],AA_U_H_V=UHaa,AA_U_O_V=UOaa,
                                   AA_separation_V=UOaa-UHaa,BB_Voc_V=bbvoc,AB_Voc_ceiling_V=abmax,BA_Voc_ceiling_V=bamax,
                                   BA_J_OWS_mA_cm2=dgba['Jsem_mA_cm2'],BA_separation_V=UOba-UHba)])
    summary.to_csv(resdir/'baseline_summary.csv',index=False)
    return summary,controls,gdf,kdf,idf


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--outdir',default='caj_publication_output')
    ap.add_argument('--full-sweeps',action='store_true')
    ap.add_argument('--precomputed-dir',default=None,help='Optional directory containing validated sweep CSVs')
    args=ap.parse_args();pre=Path(args.precomputed_dir) if args.precomputed_dir else None
    summary,controls,gdf,kdf,idf=run(Path(args.outdir),args.full_sweeps,pre)
    print('\nBASELINE\n',summary.to_string(index=False));print('\nARCHITECTURES\n',controls.to_string(index=False))


if __name__=='__main__':main()
