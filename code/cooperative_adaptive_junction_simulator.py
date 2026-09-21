#!/usr/bin/env python3
"""One-dimensional cooperative adaptive-junction model used in the SI.

Physical equations and nondimensionalization are documented, equation by
equation, in docs/NUMERICAL_METHODS.md. Publication symbols/units are listed
in docs/NOTATION.md. The model parameters and steady equations are unchanged.

PUBLICATION CONVENTION
E_CB, E_VB, E_Fn, E_Fp and E_cat are electron energies, reported in eV.
V_cat and V_bb are voltages in V. E_RHE = 0 and E_cat = -q*V_cat.
The internal U arrays are numerical voltage coordinates: U = -E/q.
They are not additional physical potentials. In numerical-value notation,
E_eV = -U_V. Energy differences divided by q have the same numerical values
in V as the corresponding one-electron energies in eV; helpers below mark
these conversions explicitly. Never multiply an eV-valued number by q twice.

GEOMETRY AND CURRENT
x=0 is HER, x=L is OER. Jn and Jp are conventional currents along +x.
Outward carrier-transfer scalars use positive q for both carrier species:
Jn(0)=jn,H, Jp(0)=-jp,H, Jn(L)=-jn,O, Jp(L)=jp,O.
The first architecture letter refers to HER and the second to OER.

ENTRY POINTS
  python code/reproduce.py --outdir reproduced_output/quick
  python code/reproduce.py --full-sweeps --diagnostics --outdir reproduced_output/full

The first command solves AA/BA operating states and true dark equilibria,
then uses the stored sensitivity/thickness tables. The second recomputes all
sensitivity/thickness states and the optional BB reverse-load diagnostic.
Every output group records whether it was solved, imported or analytically
assigned. The current SI's selected AA/AB/BA catalyst-off separations are
explicitly identified as SI-supplied comparison values, not unique Voc values.
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


def energy_eV_from_voltage_V(voltage_V):
    """Electron energy in eV from a V-valued coordinate, E_RHE=0.

    This implements E[J] = -q[C]*V[V], followed by J -> eV. Therefore
    only a sign change appears in numerical values. Scalars stay scalars.
    """
    return -voltage_V


def voltage_V_from_energy_eV(energy_eV):
    """Inverse electron energy/potential conversion; numerical V = -eV."""
    return -energy_eV


def barrier_voltage_V(energy_difference_eV):
    """Return Delta E/q in V for an energy DIFFERENCE supplied in eV.

    This conversion has no sign reversal. A positive 0.955-eV barrier
    corresponds to Phi_B/q = +0.955 V. The returned numerical value is
    unchanged because 1 eV is the energy q*1 V.
    """
    return energy_difference_eV


def energy_difference_eV_from_voltage_V(voltage_difference_V):
    """Return q*Delta V in eV (positive voltage drop -> positive energy).

    This converts a voltage DROP, not an electron level versus RHE.
    Consequently the numerical value is unchanged and no minus sign occurs.
    """
    return voltage_difference_V


@dataclass(frozen=True)
class ModelParams:
    """Bulk STO, illumination and carrier-transfer inputs (SI Table S7).

    All lengths are m; mobilities m^2/(V s); lifetimes s; Eg is eV;
    DOS masses are relative to m_e; intensity is mW/cm^2; k is m/s.
    See docs/NOTATION.md for the complete input-to-SI mapping.
    """
    T: float = 298.15
    L_s: float = 1.0e-6
    eps_r_s: float = 300.0
    Eg: float = 3.20          # eV; convert Eg/q to V for internal U equations
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
    """Native dark depletion-bending inputs in volts, relative to common EF=0.

    ND_cm3 is cm^-3. central_bulk_window is a dimensionless HALF width:
    diagnostics average U_CB over 0.42 <= x/L <= 0.58 at its default 0.08.
    For thin slabs this is a center reference, not a guaranteed neutral bulk.
    """
    ND_cm3: float = 1.0e18
    band_bending_H_V: float = 0.85   # STO(100), HER/electron side
    band_bending_O_V: float = 1.00   # STO(110), OER/hole side
    central_bulk_window: float = 0.08


@dataclass(frozen=True)
class FaradaicConfig:
    """Reversible catalyst/electrolyte kinetics; voltage reference is RHE.

    Legacy U_HER_eq_V and U_OER_eq_V are the SI's V_HER^eq and V_OER^eq.
    j0 and limiting currents are A/cm^2; b is V/decade. The Faradaic routine
    returns conventional anodic-positive currents in A/m^2.
    """
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
    """Defaults for dimensionless residual-controlled collocation.

    Specific continuation/end-point routines override these defaults as
    listed in docs/NUMERICAL_METHODS.md. bvp_nodes is the INITIAL mesh size;
    bvp_max_nodes bounds the adaptive mesh. tol is an ODE residual tolerance,
    not a relative error bound on a voltage, concentration or current.
    """
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
        self.ni = math.sqrt(self.Nc*self.Nv)*math.exp(-barrier_voltage_V(p.Eg)/(2*self.VT))
        # Internal band-coordinate reference: SI E_CB^ref=+0.400 eV.
        # Ucb0=-E_CB^ref/q, expressed numerically in V.
        self.Ucb0 = voltage_V_from_energy_eV(0.400)  # V-valued coordinate
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

        # Calibrate pinned band coordinates using the SI common-EF=0 dark
        # reference and neutral-bulk statistics (n0). Finite-slab dark
        # Center-referenced finite-slab bendings can differ slightly from these
        # targets. Their 0.15-V difference sets the directional asymmetry.
        self.Ucb_bulk_dark_target = far.U_HER_eq_V-self.VT*math.log(self.Nc/self.n0)
        self.Ucb_H_pin = self.Ucb_bulk_dark_target-facet.band_bending_H_V
        self.Ucb_O_pin = self.Ucb_bulk_dark_target-facet.band_bending_O_V

        # Physical fixed Schottky barriers for the matched buried controls.
        # These are defined directly from the neutral-bulk Ec-Ef separation
        # plus the native depletion bending and are independent of the HER/OER
        # solution redox potentials.
        self.delta_Ec_bulk_eV = energy_eV_from_voltage_V(self.Ucb_bulk_dark_target)
        self.Phi_H_Sch_eV = self.delta_Ec_bulk_eV + energy_difference_eV_from_voltage_V(facet.band_bending_H_V)
        self.Phi_O_Sch_eV = self.delta_Ec_bulk_eV + energy_difference_eV_from_voltage_V(facet.band_bending_O_V)
        self.Phi_difference_Sch_eV = self.Phi_O_Sch_eV-self.Phi_H_Sch_eV

        # Adaptive barriers at the HER/OER equilibrium potentials are useful
        # diagnostics only; they are not the fixed barriers of the buried controls.
        self.adaptive_barrier_H_at_eq_eV = energy_difference_eV_from_voltage_V(far.U_HER_eq_V-self.Ucb_H_pin)
        self.adaptive_barrier_O_at_eq_eV = energy_difference_eV_from_voltage_V(far.U_OER_eq_V-self.Ucb_O_pin)
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
        """Cosine mesh on [0,1], clustered at the two depleted contacts.

        This is only the starting grid; solve_bvp subsequently refines it.
        """
        z=np.linspace(0.0,1.0,nmesh)
        return 0.5*(1.0-np.cos(np.pi*z))

    def _ode(self, light_fraction):
        """Six dimensionless first-order equations (SI transport equations).

        y=(v,d,lnN,lnP,jn,jp), s=x/L;
        v=(U_CB-Ucb0)/VT; d=-dv/ds; N=n/nscale; P=p/nscale;
        jn=Jn/Jscale; jp=Jp/Jscale. All six state entries are dimensionless.
        Apois=q*nscale*L^2/(epsilon*VT), mob=mu/mu_ref,
        nscale=Gref*L^2/(mu_ref*VT), Jscale=q*Gref*L.
        light_fraction=G/Gref. rr=R_SRH/Gref.

        The six derivatives are, in order: band coordinate, its slope,
        electron and hole logarithmic densities, electron and hole currents.
        The last two sum to zero, enforcing constant total current.
        Exponential clipping protects intermediate Newton guesses; the
        validation suite verifies that it is inactive at accepted endpoints.
        """
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
        """Return V-valued STO surface coordinates for adaptive/buried limits.

        Adaptive (lambda=0): U_CB,s is fixed by native facet pinning, so moving
        U_M changes Phi_B[J]=q*(U_M-U_CB,s).

        Buried (lambda=1): U_CB,s moves one-for-one with U_M, so Phi_B remains
        equal to its dark value. In energy notation E_CB,s=E_cat+Phi_B.
        In U coordinates U_CB,s=V_cat-Phi_B/q. The helper makes the
        conversion of an eV-valued barrier into Phi_B/q in V explicit.
        """
        UcbH_ad=self.Ucb_H_pin
        UcbO_ad=self.Ucb_O_pin
        UcbH_bur=float(U_H)-barrier_voltage_V(self.Phi_H_Sch_eV)
        UcbO_bur=float(U_O)-barrier_voltage_V(self.Phi_O_Sch_eV)
        UcbH=(1.0-self.lambda_H)*UcbH_ad+self.lambda_H*UcbH_bur
        UcbO=(1.0-self.lambda_O)*UcbO_ad+self.lambda_O*UcbO_bur
        return UcbH,UcbO

    def _bc(self,U_H,U_O):
        """Six boundary residuals: two bands plus four carrier-transfer laws.

        U_H=V_cat,HER and U_O=V_cat,OER (V vs RHE).
        Local outgoing carrier-transfer scalars use +q for each species.
        Conversion to conventional +x currents is exactly SI eq:transfer-signs:
          jn(0)=+kappa_n*(N-Neq); jp(0)=-kappa_p*(P-Peq);
          jn(1)=-kappa_n*(N-Neq); jp(1)=+kappa_p*(P-Peq).
        kappa=k*L/(mu_ref*VT), so these are dimensionless currents.
        Neq/Peq are local detailed-balance populations, not a prescribed
        illuminated equilibrium. They depend on E_cat and the surface bands.
        """
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
        """Poisson-only equilibrium guess at one common Fermi voltage.

        Boltzmann statistics close Poisson with EF,n=EF,p=E_cat.
        Both current entries are initially zero. This seed is re-solved by
        the full BVP before it is reported as a physical dark reference.
        """
        if nmesh is None:nmesh=self.num.bvp_nodes
        s=self._initial_mesh(nmesh)
        UcbH,UcbO=self.surface_band_potentials(U_common,U_common)
        vH=(UcbH-self.Ucb0)/self.VT; vO=(UcbO-self.Ucb0)/self.VT
        def ode(s,y):
            v,d=y; Ucb=self.Ucb0+self.VT*v
            n=self.Nc*np.exp(np.clip((Ucb-U_common)/self.VT,-220,120))
            p=self.Nv*np.exp(np.clip((U_common-(Ucb+barrier_voltage_V(self.p.Eg)))/self.VT,-220,120))
            out=np.empty_like(y); out[0]=-d
            out[1]=self.Apois*(p/self.nscale-n/self.nscale+self.ND_scaled)
            return out
        def bc(ya,yb):return np.array([ya[0]-vH,yb[0]-vO])
        y0=np.zeros((2,s.size));y0[0]=np.linspace(vH,vO,s.size)
        solp=solve_bvp(ode,bc,s,y0,tol=2e-7,max_nodes=120000,verbose=0)
        if solp.status!=0:raise RuntimeError('equilibrium Poisson seed failed: '+solp.message)
        v,d=solp.sol(s);Ucb=self.Ucb0+self.VT*v
        n=self.Nc*np.exp(np.clip((Ucb-U_common)/self.VT,-220,120))
        p=self.Nv*np.exp(np.clip((U_common-(Ucb+barrier_voltage_V(self.p.Eg)))/self.VT,-220,120))
        y=np.zeros((6,s.size));y[0]=v;y[1]=d
        y[2]=np.log(np.maximum(n/self.nscale,1e-300));y[3]=np.log(np.maximum(p/self.nscale,1e-300))
        return _Seed(s,y)

    def _high_injection_seed(self,previous,light_fraction,nmesh=None):
        """Construct a positive illuminated initial guess, not a constraint.

        Adding 2*G*tau to both populations helps Newton leave the depleted
        dark state. Every carrier density and current is free in the final BVP.
        """
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
        """Solve the semiconductor for prescribed catalyst voltages and light.

        No catalyst/solution charge-balance equation is imposed here. The
        outer operating-current solver supplies those two balances.
        Without a previous solution, start with common-potential Poisson and
        continue the dark boundary separation in roughly 60-mV steps. Try
        the requested illumination directly; on failure use 11 logarithmic
        illumination steps from 1e-4 to the requested fraction. Tolerances
        may be relaxed for these seed-generating solves; callers must check
        final status and achieved residual. This function returns the SciPy
        result, including status/message, adaptive mesh and RMS residuals.
        """
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
        """Internal diagnostics tuple: s,n,p,Ucb,Uvb,Ufn,Ufp,Jn,Jp,N,P,d.

        n,p are m^-3; the four U arrays are V-valued coordinates; J are
        A/m^2. Use energy_profile_dataframe() for publication energies.
        Interpolation uses the converged collocation spline, not a new solve.
        """
        s=np.linspace(0,1,npts);v,d,lnN,lnP,jn,jp=sol.sol(s)
        N=np.exp(np.clip(lnN,-220,120));P=np.exp(np.clip(lnP,-220,120))
        n=N*self.nscale;p=P*self.nscale
        Ucb=self.Ucb0+self.VT*v;Uvb=Ucb+barrier_voltage_V(self.p.Eg)
        Ufn=Ucb-self.VT*np.log(np.maximum(n/self.Nc,1e-300))
        Ufp=Uvb+self.VT*np.log(np.maximum(p/self.Nv,1e-300))
        return s,n,p,Ucb,Uvb,Ufn,Ufp,jn*self.Jscale,jp*self.Jscale,N,P,d

    def band_bending(self,sol):
        """Return V_bb,HER, V_bb,OER and the mean central U_CB (all V).

        q*V_bb,i=E_CB,s,i-E_CB,bulk; hence V_bb=U_CB,bulk-U_CB,s.
        This measures local surface-to-center bending, distinct from Phi_B
        and from the OER-minus-HER surface energy difference. The central
        average is over s in [0.5-w,0.5+w], with w=0.08 by default.
        """
        s,n,p,Ucb,Uvb,Ufn,Ufp,Jn,Jp,N,P,d=self.profiles(sol,2401)
        w=self.facet.central_bulk_window;mask=(s>=0.5-w)&(s<=0.5+w)
        Ubulk=float(np.mean(Ucb[mask]))
        return Ubulk-float(Ucb[0]),Ubulk-float(Ucb[-1]),Ubulk

    def diagnostics(self,sol,U_H,U_O,light_fraction):
        """Compatibility diagnostics using the original U/MIEC field names.

        U-level values are voltage coordinates. The historical clipped
        counterflow quantity is used only by the preserved internal API.
        SI-facing files use publication_data.state_row(), which exports
        energies and evaluates the exact signed current identity separately.
        Faradaic values here are hypothetical unless that exchange is active.
        """
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
        """Legacy numerical dark seed with each catalyst at its redox voltage.

        Retained unchanged for continuation reproducibility. U_H=0 and
        U_O=1.229 V are different Fermi voltages; this is NOT the SI common-
        Fermi dark equilibrium. Use solve_dark_equilibrium() for that state.
        The name is retained only for compatibility with existing drivers.
        """
        sol=self.solve_state(self.far.U_HER_eq_V,self.far.U_OER_eq_V,0.0,previous=None,tol=self.num.bvp_tol_dark)
        if sol.status!=0:raise RuntimeError('relaxed dark BVP failed: '+sol.message)
        self.dark_sol=sol;self.dark_diag=self.diagnostics(sol,self.far.U_HER_eq_V,self.far.U_OER_eq_V,0.0)
        return sol,self.dark_diag

    def solve_dark_equilibrium(self, V_common=0.0, tol=1e-8, nmesh=1800):
        """SI dark reference: G=0, both catalyst Fermi voltages equal.

        No Faradaic current is included in this semiconductor-only solve.
        The publication exporter explicitly disables Faradaic diagnostics.
        This does not replace the legacy seed used in solve_ows().
        """
        seed = self.equilibrium_seed(float(V_common), nmesh)
        sol = self.solve_state(V_common, V_common, 0.0, previous=seed,
                               tol=tol, nmesh=nmesh)
        if sol.status != 0:
            raise RuntimeError('Common-Fermi dark equilibrium failed: '+sol.message)
        return sol

    def energy_profile_dataframe(self, sol, npts=1801):
        """SI-facing energies (eV), currents (mA/cm^2) and densities (cm^-3).

        E_RHE=0. U coordinates are kept in profiles()/profile_dataframe()
        for compatibility; no U column is exported here.
        """
        s,n,p,Ucb,Uvb,Ufn,Ufp,Jn,Jp,N,P,d = self.profiles(sol,npts)
        return pd.DataFrame(dict(
            x_um=s*self.p.L_s*1e6,
            E_CB_eV=energy_eV_from_voltage_V(Ucb),
            E_VB_eV=energy_eV_from_voltage_V(Uvb),
            E_Fn_eV=energy_eV_from_voltage_V(Ufn),
            E_Fp_eV=energy_eV_from_voltage_V(Ufp),
            n_cm3=n/1e6, p_cm3=p/1e6,
            Jn_mA_cm2=Jn/10, Jp_mA_cm2=Jp/10,
            Jtotal_mA_cm2=(Jn+Jp)/10))

    def solve_ows(self,light_fraction=1.0,initial_sol=None,verbose=False):
        """Forward operating point using a scalar current load-line root.

        For trial J>0, invert j_H(V_H)=-J and j_O(V_O)=+J, solve the
        semiconductor BVP, then bracket/solve F(J)=J_sem-J with brentq.
        Cached solutions are starting guesses only. Re-solve at the accepted
        root with tol=3e-7 and 1100 initial nodes before returning it.
        Root tolerances are in A/m^2; BVP tolerances are dimensionless.
        """
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
        if sol.status!=0:raise RuntimeError('Final OWS endpoint failed: '+sol.message)
        return J,UH,UO,sol,self.diagnostics(sol,UH,UO,light_fraction),initial_sol

    def profile_dataframe(self,sol,npts=1801):
        """Legacy internal-voltage export for regression/source compatibility.

        Publication code uses energy_profile_dataframe() instead.
        """
        s,n,p,Ucb,Uvb,Ufn,Ufp,Jn,Jp,N,P,d=self.profiles(sol,npts)
        return pd.DataFrame(dict(x_um=s*self.p.L_s*1e6,U_CB_V_vs_RHE=Ucb,U_VB_V_vs_RHE=Uvb,
                                 U_Fn_V_vs_RHE=Ufn,U_Fp_V_vs_RHE=Ufp,n_cm3=n/1e6,p_cm3=p/1e6,
                                 Jn_mA_cm2=Jn/10,Jp_mA_cm2=Jp/10,Jtotal_mA_cm2=(Jn+Jp)/10))



# ---------------------- publication control calculations ----------------------
def make_params(intensity=10.0, k=10.0):
    """Baseline material inputs with intensity in mW/cm^2 and common k in m/s."""
    return ModelParams(intensity_mW_cm2=float(intensity), k_n=float(k), k_p=float(k))


def facet_from_mean_delta(mean, delta):
    """Native bending inputs in V: H=mean-delta/2; O=mean+delta/2."""
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
        self._cache_settings={}

    def _nearest_seed(self,V):
        if not self._cache:return None
        return self._cache[min(self._cache,key=lambda x:abs(x-V))][0]

    def solve_semiconductor_at_V(self,V,tol=3e-7,nmesh=900):
        V=float(V)
        key=round(V,14)
        if key in self._cache:
            used_tol, used_nmesh = self._cache_settings[key]
            if used_tol <= tol and used_nmesh >= nmesh:
                return self._cache[key]
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
        self._cache_settings[key]=(tol,nmesh)
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
        center=barrier_voltage_V(self.m.Phi_difference_Sch_eV)
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
    """Diagnostic family of mixed-contact zero-current states.

    One buried-contact voltage is selected and the other voltage is solved
    from J_sem=0. This family has an unselected common-mode freedom; a fitted
    plateau is not the SI Table S8 selected value or a unique device Voc.
    """
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
        # SciPy root uses the Powell hybrid method by default (not secant).
        # A broad scalar bracket with brentq is the fallback.
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
        near=min(cache,key=lambda z:abs(z[0]-Uf))
        UH,UO=(Uf,U_buried) if self.mode=='AB' else (U_buried,Uf)
        endpoint=self.m.solve_state(UH,UO,1.0,previous=near[1],tol=tol,nmesh=nmesh)
        if endpoint.status!=0:
            raise RuntimeError('Final mixed-contact endpoint failed: '+endpoint.message)
        dg=self.m.diagnostics(endpoint,UH,UO,1.0)
        if abs(dg['Jsem_A_m2'])>1e-6:
            raise RuntimeError('Mixed-contact zero-current residual exceeds 1e-6 A/m^2')
        self.previous=endpoint;self.last_float=Uf
        return UH,UO,endpoint,dg


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
    """Diagnostic zero-current separation sweep: adaptive HER / buried OER.

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
    # Historical diagnostic range. Deep-bias points can fail the absolute
    # transfer-boundary tolerance because large one-way currents nearly cancel.
    # These optional points are not part of the current SI reproduction.
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
    """Diagnostic zero-current separation sweep: buried HER / adaptive OER."""
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
    if not chk.success or np.max(np.abs(chk.fun))>1e-8:
        raise RuntimeError('BA two-potential balance check failed: '+chk.message)
    return m,J,UH,UO,sol,dg,chk


# ------------------------------- plotting ------------------------------------










# --------------------------- AA sensitivity sweeps ----------------------------
def run_k_sweep():
    rows=[]
    for k in [0.1,1,10,100,1000]:
        m,ds,dd,il,idi,J,UH,UO,fs,fd=solve_adaptive_case(k=k)
        rows.append(dict(k_m_s=k,J_OWS_mA_cm2=fd['Jsem_mA_cm2'],counterflow_mA_cm2=fd['counterflow_mA_cm2'],
                         SRH_mA_cm2=fd['Jrec_mA_cm2'],MIEC_separation_V=fd['MIEC_separation_V'],QFL_split_V=fd['QFL_center_split_V']))
    return pd.DataFrame(rows)


def _map_solution_seed_between_intensities(src_model,src_sol,dst_model,nmesh=900):
    """Rescale a previous guess when intensity or thickness changes.

    lnN/lnP shift by log(old_nscale/new_nscale); jn/jp multiply by
    old_Jscale/new_Jscale. Physical concentrations/current guesses are
    thereby preserved at a common scaled coordinate. A thickness change
    still requires a complete new BVP solve; this is not a physical mapping
    between converged solutions of different geometries.
    """
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
    """Compatibility entry point for the SI-aligned reproduction workflow.

    full_sweeps=True always recomputes sensitivity and thickness states,
    irrespective of precomputed_dir. See reproduce.py for per-group logs.
    """
    from reproduce import reproduce
    return reproduce(Path(outdir), full_sweeps=full_sweeps,
                     reference_dir=precomputed_dir)


def main():
    from reproduce import main as reproduce_main
    reproduce_main()


if __name__ == '__main__':
    main()
