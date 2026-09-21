"""Equation/notation regression and freshly solved AA/BA endpoint checks.

Run from the repository root: python -m unittest discover -s tests -v
The upstream module is immutable provenance, not a second physical model.
"""
from pathlib import Path
from dataclasses import asdict
import importlib.util
import sys
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import cooperative_adaptive_junction_simulator as model
from publication_data import state_row,selected_catalyst_off_values
from validation_checks import audit_state
spec=importlib.util.spec_from_file_location('caj_upstream_test',ROOT/'provenance/upstream_code/cooperative_adaptive_junction_simulator.py')
upstream=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=upstream
spec.loader.exec_module(upstream)


class PhysicalEquationsUnchanged(unittest.TestCase):
    def test_parameter_defaults_are_unchanged(self):
        for name in ['ModelParams','FacetConfig','FaradaicConfig','Numerics']:
            self.assertEqual(asdict(getattr(model,name)()),asdict(getattr(upstream,name)()))

    def test_scaled_ode_and_boundaries_match_upstream(self):
        rng=np.random.default_rng(20260920)
        y=np.vstack([rng.uniform(-45,-5,11),rng.uniform(-20,20,11),
                     rng.uniform(-20,5,11),rng.uniform(-35,-1,11),
                     rng.uniform(-2,2,11),rng.uniform(-2,2,11)])
        for h,o in [(0,0),(0,1),(1,0),(1,1)]:
            new=model.CooperativeAdaptiveJunction(model.ModelParams(),lambda_H=h,lambda_O=o)
            old=upstream.CooperativeAdaptiveJunction(upstream.ModelParams(),lambda_H=h,lambda_O=o)
            for lf in [0,.1,1.]:
                np.testing.assert_array_equal(new._ode(lf)(np.linspace(0,1,11),y),
                                              old._ode(lf)(np.linspace(0,1,11),y))
            for vh,vo in [(-.3,1.59),(0,0),(.6,.75)]:
                np.testing.assert_array_equal(new._bc(vh,vo)(y[:,0],y[:,-1]),
                                              old._bc(vh,vo)(y[:,0],y[:,-1]))
                np.testing.assert_array_equal(new.surface_band_potentials(vh,vo),
                                              old.surface_band_potentials(vh,vo))

    def test_srh_and_faradaic_laws_match_upstream(self):
        new=model.CooperativeAdaptiveJunction(model.ModelParams())
        old=upstream.CooperativeAdaptiveJunction(upstream.ModelParams())
        n=np.geomspace(1e-12,1e6,31);p=n[::-1]
        np.testing.assert_array_equal(new.srh_over_Gref(n,p),old.srh_over_Gref(n,p))
        for vh in [-.35,0,.60]:
            for vo in [.75,1.229,1.65]:
                self.assertEqual(new.faradaic_currents_A_m2(vh,vo),old.faradaic_currents_A_m2(vh,vo))


class NotationAndUnits(unittest.TestCase):
    def test_energy_voltage_map(self):
        voltage=np.array([-.400,0.,1.229])
        np.testing.assert_array_equal(model.energy_eV_from_voltage_V(voltage),[.400,0.,-1.229])
        np.testing.assert_array_equal(model.voltage_V_from_energy_eV(model.energy_eV_from_voltage_V(voltage)),voltage)
        self.assertEqual(model.barrier_voltage_V(.955),.955)

    def test_adaptive_and_buried_barrier_slopes(self):
        for buried in [False,True]:
            m=model.CooperativeAdaptiveJunction(model.ModelParams(),lambda_H=float(buried))
            phis=[];energies=[]
            for v in [-.2,-.199]:
                uc,_=m.surface_band_potentials(v,1.5)
                energies.append(-v);phis.append(-uc+v)
            sb=-(phis[1]-phis[0])/(energies[1]-energies[0])
            self.assertAlmostEqual(sb,0 if buried else 1,places=11)

    def test_reference_band_edges_and_barriers(self):
        m=model.CooperativeAdaptiveJunction(model.ModelParams())
        self.assertAlmostEqual(-m.Ucb_H_pin,.9552120108495786,places=12)
        self.assertAlmostEqual(-m.Ucb_O_pin,1.1052120108495786,places=12)
        self.assertAlmostEqual(m.Phi_O_Sch_eV-m.Phi_H_Sch_eV,.15,places=12)
        self.assertAlmostEqual(m.Jscale/10,1.860914591440716,places=12)
        self.assertAlmostEqual(m.ND_m3,1e24)

    def test_four_geometrical_current_signs(self):
        m=model.CooperativeAdaptiveJunction(model.ModelParams())
        h,o=-.2,1.5;ucH,ucO=m.surface_band_potentials(h,o)
        ys=[]
        for uc,vcat,is_right in [(ucH,h,False),(ucO,o,True)]:
            v=(uc-m.Ucb0)/m.VT;r=(vcat-m.U0)/m.VT
            neq=m.nib*np.exp(v-r);peq=m.nib*np.exp(r-v)
            n,p=neq+1.0,peq+1.0
            jn=m.kappa_n*(n-neq);jp=m.kappa_p*(p-peq)
            ys.append(np.array([v,0,np.log(n),np.log(p),-jn if is_right else jn,
                                jp if is_right else -jp]))
        bc=m._bc(h,o)(*ys)
        np.testing.assert_allclose(bc,0,atol=1e-8,rtol=0)

    def test_catalyst_off_values_are_supplied_selections(self):
        frame=selected_catalyst_off_values()
        np.testing.assert_allclose(frame.selected_Delta_V_cat_V,[2.72,.696,2.17,.15])
        self.assertFalse(frame.numerical_state_recomputed.any())
        self.assertEqual(frame.selection_rule_specified.tolist(),[False,False,False,True])

    def test_BB_cache_honors_refinement_request(self):
        m=model.CooperativeAdaptiveJunction(model.ModelParams(),lambda_H=1,lambda_O=1)
        s=model.BBLoadLineSolver(m)
        class Solution:status=0
        with patch.object(m,'solve_state',return_value=Solution()) as solve, \
             patch.object(m,'diagnostics',return_value={'Jsem_A_m2':1.0}):
            s.solve_semiconductor_at_V(.15,tol=1e-6,nmesh=100)
            before=solve.call_count
            s.solve_semiconductor_at_V(.15,tol=1e-6,nmesh=100)
            self.assertEqual(solve.call_count,before)
            s.solve_semiconductor_at_V(.15,tol=1e-8,nmesh=200)
            self.assertGreater(solve.call_count,before)


class FreshEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.aa=model.solve_adaptive_case()
        cls.ba=model.solve_ba_finite_current()

    def test_fresh_AA_and_BA_charge_balance(self):
        m,ds,dd,il,idi,j,h,o,sol,dg=self.aa
        self.assertTrue(audit_state(m,sol,h,o,bvp_limit=3.1e-7)['passed'])
        self.assertAlmostEqual(dg['Jsem_mA_cm2'],.9453855415407645,places=7)
        m,j,h,o,sol,dg,check=self.ba
        self.assertTrue(audit_state(m,sol,h,o,bvp_limit=1.1e-8)['passed'])
        self.assertTrue(check.success)
        self.assertAlmostEqual(dg['Jsem_mA_cm2'],.03037917334853866,places=7)

    def test_BA_fixed_barrier_reduced_bending(self):
        m,j,h,o,sol,dg,check=self.ba
        dark=m.solve_dark_equilibrium()
        d=state_row(m,dark,0,0,'BA',light_fraction=0,faradaic_active=False)
        op=state_row(m,sol,h,o,'BA')
        self.assertAlmostEqual(op['Phi_B_HER_eV'],d['Phi_B_HER_eV'],places=10)
        self.assertLess(op['V_bb_HER_V'],d['V_bb_HER_V'])
        self.assertAlmostEqual(op['V_bb_HER_V'],.308893959609,places=7)
        self.assertAlmostEqual(op['Delta_E_CB_surface_eV'],.0016669487594,places=8)
        self.assertGreater(op['E_CB_bulk_eV']-d['E_CB_bulk_eV'],
                           op['E_CB_s_HER_eV']-d['E_CB_s_HER_eV'])
        audit_state(m,dark,0,0,0,faradaic_active=False,bvp_limit=1.1e-8)
        profile=m.energy_profile_dataframe(dark)
        np.testing.assert_allclose(profile.E_Fn_eV,0,atol=1e-8)
        np.testing.assert_allclose(profile.E_Fp_eV,0,atol=1e-8)

    def test_publication_schema_and_energy_statistics(self):
        m,ds,dd,il,idi,j,h,o,sol,dg=self.aa
        frame=m.energy_profile_dataframe(sol)
        self.assertFalse(any(c.startswith('U_') for c in frame))
        np.testing.assert_allclose(frame.E_CB_eV-frame.E_VB_eV,m.p.Eg,atol=1e-12)
        r=state_row(m,sol,h,o,'AA')
        self.assertAlmostEqual(r['Delta_E_cat_eV'],r['Delta_V_cat_V'],places=12)
        self.assertAlmostEqual(r['E_cat_HER_eV'],-r['V_cat_HER_V_vs_RHE'],places=12)
        self.assertAlmostEqual(r['J_gen_mA_cm2'],r['J_OWS_mA_cm2']+r['J_contact_mA_cm2']+r['J_SRH_mA_cm2'],places=7)

if __name__=='__main__':unittest.main()
