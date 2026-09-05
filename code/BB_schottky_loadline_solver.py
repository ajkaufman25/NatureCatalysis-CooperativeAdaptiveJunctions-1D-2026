#!/usr/bin/env python3
"""Focused BB solver: fixed-barrier Schottky photovoltaic element + HOR/ORR load.

This diagnostic deliberately isolates the buried/buried (BB) case.
It does NOT solve the catalyst potentials as a two-dimensional mixed-potential root.
Instead it constructs two independently well-conditioned one-dimensional objects:

  (1) the illuminated BB semiconductor J-V curve, using the full 1D
      Poisson + electron/hole drift-diffusion + SRH model with two fixed
      Schottky barriers; and
  (2) the reversible HOR/ORR electrochemical load curve V_load(I)=U_O(I)-U_H(I).

The physical BB operating point is their scalar load-line intersection

      J_BB(V_load(I)) + I = 0,

where I>0 denotes the magnitude of the reverse HOR+ORR current.

Near the common 50 uA cm^-2 reverse limiting current, the root is solved using
z=-log10(1-I/I_lim), which gives high precision without placing the root at an
ill-conditioned current asymptote.  For the expected BB state z is modest
(~4), not an extreme value.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import importlib.util
import math
import numpy as np
import pandas as pd
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "cooperative_adaptive_junction_simulator.py"

spec = importlib.util.spec_from_file_location("caj_base", BASE_PATH)
base = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = base
spec.loader.exec_module(base)


class BBSchottkyDevice(base.CooperativeAdaptiveJunction):
    """Two ordinary fixed-barrier Schottky contacts.

    The Schottky barriers are specified directly and are independent of the
    HER/OER redox potentials.  The terminal metal potentials U_H and U_O are
    free to shift.  At each buried contact

        U_CB,s = U_metal - Phi_B^0,

    so U_metal-U_CB,s remains exactly constant.
    """

    def __init__(self, p=None, facet=None, far=None, num=None):
        if p is None:
            p = base.ModelParams()
        if facet is None:
            facet = base.FacetConfig()
        if far is None:
            far = base.FaradaicConfig()
        if num is None:
            num = base.Numerics()
        super().__init__(p, facet=facet, far=far, num=num, lambda_H=1.0, lambda_O=1.0)

        # Neutral-bulk conduction-band offset relative to a common electron
        # electrochemical potential.  With the present n-STO parameters it is
        # 0.105212... eV.  The fixed Schottky barriers are this offset plus the
        # two native depletion bendings (0.85 and 1.00 V).
        self.delta_Ec_bulk_eV = -self.Ucb_bulk_dark_target
        self.Phi_H_eV = self.delta_Ec_bulk_eV + self.facet.band_bending_H_V
        self.Phi_O_eV = self.delta_Ec_bulk_eV + self.facet.band_bending_O_V

        # Override metadata so diagnostics and tables report the actual BB
        # barriers used by this diagnostic rather than a redox-referenced value.
        self.dark_barrier_H_eV = self.Phi_H_eV
        self.dark_barrier_O_eV = self.Phi_O_eV
        self.dark_barrier_difference_eV = self.Phi_O_eV - self.Phi_H_eV

    def surface_band_potentials(self, U_H, U_O):
        return float(U_H) - self.Phi_H_eV, float(U_O) - self.Phi_O_eV


class BBLoadLineSolver:
    def __init__(self, device: BBSchottkyDevice):
        self.m = device
        self._cache = {}  # voltage -> (solution, diagnostics)
        self._last_V = None
        self._last_sol = None

    def _nearest_seed(self, V):
        if not self._cache:
            return None
        key = min(self._cache, key=lambda x: abs(x - V))
        return self._cache[key][0]

    def solve_semiconductor_at_V(self, V, tol=3e-7, nmesh=1000):
        """Solve illuminated BB semiconductor at terminal voltage V=U_O-U_H.

        Gauge choice: U_H=0, U_O=V.  Because both barriers are fixed, a common
        shift of both metal potentials is a gauge transformation and the BB
        semiconductor J-V depends only on V.
        """
        V = float(V)
        if V in self._cache:
            return self._cache[V]
        UH, UO = 0.0, V
        prev = self._nearest_seed(V)
        if prev is None:
            # Physical initial guess: solve the zero-terminal-voltage dark
            # equilibrium first, then turn on the light.  At V=0 the two
            # contacts retain the native 0.15-eV Schottky asymmetry.
            dark = self.m.solve_state(UH, UO, 0.0, previous=None,
                                      tol=self.m.num.bvp_tol_dark, nmesh=nmesh)
            if dark.status != 0:
                raise RuntimeError(f"BB dark seed failed at V={V}: {dark.message}")
            sol = self.m.solve_state(UH, UO, 1.0, previous=dark,
                                     high_injection=True, tol=tol, nmesh=nmesh)
        else:
            sol = self.m.solve_state(UH, UO, 1.0, previous=prev,
                                     tol=tol, nmesh=nmesh)
        if sol.status != 0:
            raise RuntimeError(f"BB illuminated BVP failed at V={V}: {sol.message}")
        dg = self.m.diagnostics(sol, UH, UO, 1.0)
        self._cache[V] = (sol, dg)
        return sol, dg

    def build_jv(self, vmin=0.0, vmax=0.22, dv=0.005):
        """Continuation sweep from short-circuit side through open circuit."""
        volts = np.arange(vmin, vmax + 0.5*dv, dv)
        rows = []
        for V in volts:
            sol, dg = self.solve_semiconductor_at_V(V)
            rows.append({
                "V_terminal_V": V,
                "Jsem_mA_cm2": dg["Jsem_mA_cm2"],
                "Ucb_H_V": dg["Ucb_H_V"],
                "Ucb_O_V": dg["Ucb_O_V"],
                "BVP_residual": dg["max_BVP_rms_residual"],
            })
        return pd.DataFrame(rows)

    def semiconductor_current_A_m2(self, V):
        return self.solve_semiconductor_at_V(V)[1]["Jsem_A_m2"]

    def solve_voc(self):
        # First use a modest bracket around the known native barrier difference.
        center = self.m.Phi_O_eV - self.m.Phi_H_eV
        lo, hi = center - 0.03, center + 0.03
        flo = self.semiconductor_current_A_m2(lo)
        fhi = self.semiconductor_current_A_m2(hi)
        if flo*fhi > 0:
            raise RuntimeError("Could not bracket BB open-circuit voltage")
        V = brentq(lambda x: self.semiconductor_current_A_m2(x), lo, hi,
                   xtol=2e-12, rtol=2e-12, maxiter=80)
        sol, dg = self.solve_semiconductor_at_V(V, tol=2e-7, nmesh=1200)
        return V, sol, dg

    # ---------------- electrochemical load curve ----------------
    def U_H_for_reverse_I(self, I_A_m2):
        """HOR potential for reverse-current magnitude I>0: j_H(U_H)=+I."""
        if not (0 < I_A_m2 < self.m.far.jlim_HOR_A_cm2*1e4):
            raise ValueError("I must lie strictly between 0 and HOR limiting current")
        f = lambda U: self.m.faradaic_currents_A_m2(U, self.m.far.U_OER_eq_V)[0] - I_A_m2
        return brentq(f, self.m.far.U_HER_eq_V, 1.5, xtol=2e-14, rtol=2e-14, maxiter=120)

    def U_O_for_reverse_I(self, I_A_m2):
        """ORR potential for reverse-current magnitude I>0: j_O(U_O)=-I."""
        if not (0 < I_A_m2 < self.m.far.jlim_ORR_A_cm2*1e4):
            raise ValueError("I must lie strictly between 0 and ORR limiting current")
        f = lambda U: self.m.faradaic_currents_A_m2(self.m.far.U_HER_eq_V, U)[1] + I_A_m2
        return brentq(f, -0.5, self.m.far.U_OER_eq_V, xtol=2e-14, rtol=2e-14, maxiter=120)

    def load_at_I(self, I_A_m2):
        UH = self.U_H_for_reverse_I(I_A_m2)
        UO = self.U_O_for_reverse_I(I_A_m2)
        return UH, UO, UO-UH

    def solve_operating_point(self, voc=None):
        """High-precision BB load-line intersection.

        Parameterize the near-limiting reverse current by
          z = -log10(delta), delta = 1-I/Ilim.
        For the expected physical solution z is O(4), safely away from the
        pathological z>>10 asymptote encountered in earlier 2-D solvers.
        """
        jlim_H = self.m.far.jlim_HOR_A_cm2*1e4
        jlim_O = self.m.far.jlim_ORR_A_cm2*1e4
        if abs(jlim_H-jlim_O) > 1e-12*max(jlim_H,jlim_O):
            raise NotImplementedError("Focused BB solver presently assumes equal HOR/ORR limits")
        jlim = jlim_H

        if voc is None:
            voc = self.solve_voc()[0]

        # Initial estimate from the local illuminated semiconductor conductance.
        h = 1e-3
        jm = self.semiconductor_current_A_m2(voc-h)
        jp = self.semiconductor_current_A_m2(voc+h)
        dJdV = (jp-jm)/(2*h)
        V_est = voc - jlim/dJdV  # dJ/dV<0 -> V_est > Voc

        # Electrochemical-only estimate of z from V_load(z)=V_est.
        def I_of_z(z):
            return jlim*(1.0-10.0**(-float(z)))
        def Vload_of_z(z):
            return self.load_at_I(I_of_z(z))[2]

        # z=1 corresponds 90% of limit; z=8 is far closer than physically
        # expected but remains finite.  V_load decreases monotonically here.
        z_lo, z_hi = 1.0, 8.0
        z0 = brentq(lambda z: Vload_of_z(z)-V_est, z_lo, z_hi,
                    xtol=1e-12, rtol=1e-12, maxiter=100)

        # Full load-line residual: semiconductor current plus reverse current.
        # At the physical point J_sem=-I.
        def Rz(z):
            I = I_of_z(z)
            UH, UO, V = self.load_at_I(I)
            Jsem = self.semiconductor_current_A_m2(V)
            return Jsem + I

        # Bracket around the physics-based z0 rather than globally searching
        # the asymptote.  Expand in 0.5-decade increments if needed.
        zl, zh = max(0.5, z0-1.0), min(9.0, z0+1.0)
        fl, fh = Rz(zl), Rz(zh)
        for _ in range(10):
            if fl*fh <= 0:
                break
            zl = max(0.25, zl-0.5)
            zh = min(10.0, zh+0.5)
            fl, fh = Rz(zl), Rz(zh)
        if fl*fh > 0:
            raise RuntimeError("Could not bracket physical BB load-line intersection near expected regime")

        z = brentq(Rz, zl, zh, xtol=3e-12, rtol=3e-12, maxiter=100)
        I = I_of_z(z)
        UH, UO, V = self.load_at_I(I)
        sol, dg = self.solve_semiconductor_at_V(V, tol=2e-7, nmesh=1300)
        jH, jO, _, _ = self.m.faradaic_currents_A_m2(UH, UO)

        return {
            "Voc_V": voc,
            "dJdV_A_m2_V": dJdV,
            "V_initial_estimate_V": V_est,
            "z_initial_estimate": z0,
            "z_root": z,
            "delta_fraction": 10.0**(-z),
            "I_reverse_A_m2": I,
            "I_reverse_uA_cm2": I/1e4*1e6,
            "Jsem_A_m2": dg["Jsem_A_m2"],
            "Jsem_uA_cm2": dg["Jsem_A_m2"]/1e4*1e6,
            "U_H_V": UH,
            "U_O_V": UO,
            "V_terminal_V": V,
            "jH_A_m2": jH,
            "jO_A_m2": jO,
            "current_balance_A_m2": dg["Jsem_A_m2"] + I,
            "H_balance_A_m2": dg["Jsem_A_m2"] + jH,
            "O_balance_A_m2": -dg["Jsem_A_m2"] + jO,
            "Ucb_H_V": dg["Ucb_H_V"],
            "Ucb_O_V": dg["Ucb_O_V"],
            "surface_band_difference_V": dg["Ucb_O_V"]-dg["Ucb_H_V"],
            "BVP_residual": dg["max_BVP_rms_residual"],
            "Phi_H_eV": self.m.Phi_H_eV,
            "Phi_O_eV": self.m.Phi_O_eV,
        }, sol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="bb_solver_output")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    m = BBSchottkyDevice()
    s = BBLoadLineSolver(m)
    jv = s.build_jv(0.0, 0.22, 0.005)
    voc, vocsol, vocdg = s.solve_voc()
    result, opsol = s.solve_operating_point(voc=voc)

    pd.DataFrame([result]).to_csv(out/"bb_operating_point.csv", index=False)
    jv.to_csv(out/"bb_semiconductor_JV.csv", index=False)

    # High-resolution electrochemical load curve around the physical root.
    jlim = m.far.jlim_HOR_A_cm2*1e4
    zs = np.linspace(2.5, 6.0, 141)
    rows=[]
    for z in zs:
        I=jlim*(1-10**(-z)); UH,UO,V=s.load_at_I(I)
        rows.append({"z":z,"delta_fraction":10**(-z),"I_uA_cm2":I/1e4*1e6,
                     "U_H_V":UH,"U_O_V":UO,"V_load_V":V,
                     "Jsem_uA_cm2":s.semiconductor_current_A_m2(V)/1e4*1e6,
                     "loadline_residual_uA_cm2":(s.semiconductor_current_A_m2(V)+I)/1e4*1e6})
    pd.DataFrame(rows).to_csv(out/"bb_loadline_scan.csv", index=False)

    print("Fixed BB Schottky barriers:")
    print(f"  Phi_H = {m.Phi_H_eV:.12f} eV")
    print(f"  Phi_O = {m.Phi_O_eV:.12f} eV")
    print(f"  DeltaPhi = {m.Phi_O_eV-m.Phi_H_eV:.12f} eV")
    print(f"BB Voc = {voc:.12f} V")
    print("Operating point:")
    for k,v in result.items():
        if isinstance(v,float): print(f"  {k}: {v:.12g}")
        else: print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
