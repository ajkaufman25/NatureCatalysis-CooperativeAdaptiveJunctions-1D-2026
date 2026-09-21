# Numerical methods and equation-to-code guide

This guide connects the BA-clarified SI to the actual implementation. The physical steady equations and baseline inputs are unchanged. SI labels such as `eq:poisson` below identify equations in the supplied LaTeX source, independently of rendered equation numbers.

## 1. Physical problem and boundary data

The semiconductor occupies 0≤x≤L. It contains uniformly ionized donors, nondegenerate carriers and spatially uniform generation. The SI equations are

$$n=N_Ce^{-(E_{\rm CB}-E_{F,n})/(k_BT)},\qquad
p=N_Ve^{-(E_{F,p}-E_{\rm VB})/(k_BT)},$$

$$E_{\rm CB}''=\frac{q^2}{\epsilon}(p-n+N_D),\quad
J_n=\mu_n nE_{F,n}',\quad J_p=\mu_p pE_{F,p}',$$

$$J_n'=q(R-G),\qquad J_p'=q(G-R),$$

$$R=\frac{np-n_i^2}{\tau_p(n+n_i)+\tau_n(p+n_i)}.$$

Here E is in joules in the dimensional equations, epsilon=epsilon_0 epsilon_r, and primes denote x derivatives. See SI `eq:poisson`, `eq:dd`, `eq:continuity`, `eq:srh`. The code uses the algebraically equivalent coordinate U=−E/q. The band energy and the quasi-Fermi energy are distinct state information; they are not equated under illumination.

Generation is G=I_0(1−exp(−alpha L))/[(hc/wavelength)L]. The Beer–Lambert factor determines the total number of generated pairs. It does not introduce a spatial generation gradient. For thickness or intensity changes, the normalization is recalculated for the new parameters.

At an A contact E_CB,s is pinned. At a B contact E_CB,s=E_cat+Phi_B^0. `surface_band_potentials()` returns those conditions on the internal U coordinate. `lambda_i` selects the two limiting boundary conditions.

## 2. Nondimensionalization and the six state variables

Define

$$s=x/L,\quad V_T=k_BT/q,\quad \mu_* = \min(\mu_n,\mu_p),$$

$$n_* = \frac{G_{\rm ref}L^2}{\mu_* V_T},\qquad J_* = qG_{\rm ref}L,$$

$$N=n/n_*,\quad P=p/n_*,\quad j_n=J_n/J_*,\quad j_p=J_p/J_*.$$

`Gref` is the fully illuminated uniform generation rate for the current parameter set. The continuation parameter `light_fraction` multiplies that rate, leaving these scales fixed during an individual illumination continuation. The arrays `jn`, `jp` below are dimensionless global conventional currents, not the SI's dimensional local outward-transfer scalars.

The six components in `y` are

$$y=(v,d,\ell_n,\ell_p,j_n,j_p),\quad
v=(U_{\rm CB}-U_{\rm CB}^{\rm ref})/V_T,\quad
d=-dv/ds,\quad \ell_n=\ln N,\quad\ell_p=\ln P.$$

Log densities maintain positive carrier populations. Define

$$A=\frac{q n_*L^2}{\epsilon V_T},\quad m_n=\mu_n/\mu_*,\quad m_p=\mu_p/\mu_*,\quad
D=N_D/n_*,\quad r=R/G_{\rm ref}.$$

`_ode(light_fraction)` evaluates, in this order,

$$\frac{dv}{ds}=-d,\qquad \frac{dd}{ds}=A(P-N+D),$$

$$\frac{d\ell_n}{ds}=\frac{j_n}{m_nN}-d,\qquad
\frac{d\ell_p}{ds}=-\frac{j_p}{m_pP}+d,$$

$$\frac{dj_n}{ds}=r-f,\qquad \frac{dj_p}{ds}=f-r,$$

where f=`light_fraction`. The last two equations give spatial constancy of total current. The variables called `mob_n`, `mob_p`, `Apois`, `ND_scaled` and `nscale` store these coefficients.

## 3. The six boundary residuals

Two residuals set v at the two surfaces. Four implement reversible carrier transfer. With kappa_n=k_n L/(mu_* V_T) and similarly for holes,

$$j_n(0)=\kappa_n(N_H-N_{{\rm eq},H}),\qquad
j_p(0)=-\kappa_p(P_H-P_{{\rm eq},H}),$$

$$j_n(1)=-\kappa_n(N_O-N_{{\rm eq},O}),\qquad
j_p(1)=\kappa_p(P_O-P_{{\rm eq},O}).$$

These are exactly the SI's four geometrical signs in `eq:transfer-signs`, divided by J_*. They appear in `_bc()` as a residual equal to left side minus right side.

The equilibrium populations are set by each catalyst energy:

$$n_{{\rm eq},i}=N_Ce^{-(E_{{\rm CB},s,i}-E_{{\rm cat},i})/(k_BT)},\quad
p_{{\rm eq},i}=N_Ve^{-(E_{{\rm cat},i}-E_{{\rm VB},s,i})/(k_BT)}.$$

For conditioning, the code writes N_eq=(n_i/n_*) exp(v−r_i) and P_eq=(n_i/n_*) exp(r_i−v), where r_i=(V_cat,i−U_0)/V_T and U_0=U_CB^ref+V_T ln(N_C/n_i). The n_i prefactor and U_0 offset cancel to give the physical Boltzmann factors above. No extra selectivity factor is present.

## 4. Collocation, initial mesh and residual tolerances

`scipy.integrate.solve_bvp` uses fourth-order collocation, an adaptive mesh and a damped Newton method. No user Jacobian is supplied; SciPy uses forward finite differences. The initial mesh is cosine-clustered,

$$s_j=\tfrac12[1-\cos(\pi j/(M-1))],$$

to resolve both depleted contacts. M is an initial mesh size, not the final number of spatial points. Adaptive refinement is allowed up to 180,000 nodes in the main six-variable problem.

SciPy controls an interval RMS residual of the normalized ODE defect (y′−f)/(1+|f|). The `tol` value is dimensionless. It is neither the voltage error nor a guaranteed relative concentration/current error. With no separate `bc_tol`, the componentwise absolute boundary tolerance defaults to `tol`. The status, maximum interval RMS residual, final mesh count and iteration count are written to the validation reports.

Typical settings retained from the solver are:

| Calculation | Initial nodes | Requested ODE tolerance |
|---|---:|---:|
| General dark / illuminated defaults | 800 | 2e-7 / 6e-7 |
| Final AA operating point | 1100 | 3e-7 |
| Final BA operating point | 1800 | 1e-8 |
| Reported common-Fermi dark references | 1800 | 1e-8 |
| Thickness continuation endpoints | 1100 | 3e-7 |
| Intensity continuation endpoints | 900 | 1.2e-6 |
| BB absolute-reference validation | 1800 | 1e-8 |

Other seed/root trial calls use the explicit tolerances beside their `solve_state()` calls. The collocation interpolant is sampled at 1801 points for publication profiles and 2401 points for most integrated diagnostics. Those uniform sampling grids are distinct from the adaptive solve mesh.

## 5. Initial guesses and continuation

`equilibrium_seed()` first solves the two-variable Poisson equilibrium problem at a common catalyst potential and constructs equilibrium carrier densities. When distinct catalyst potentials are required, `solve_state()` changes the contact separation in increments of approximately 0.06 V before turning on illumination. If a direct illuminated solve fails, it attempts eleven logarithmically spaced illumination steps from 1e-4 to the requested light fraction.

`_high_injection_seed()` adds a carrier-density estimate proportional to 2G min(tau_n,tau_p) to an initial guess. It does not impose that carrier density on the converged solution. `_Seed` uses interpolation only to transfer an initial guess onto a new mesh.

Sensitivity sweeps continue from freshly solved states. To avoid cold-start singular Jacobians at some native-asymmetry points, the sweep driver limits changes in mean/differential bending to 0.025 V and common transfer coefficient to half a decade per continuation step. Each intermediate state is current-balanced at its own parameters; only requested grid endpoints are exported. This changes the numerical path, not the physical endpoint equations or parameter grids.

Intensity continuation starts from the freshly solved 10 mW cm⁻² baseline and proceeds to 20,30 and separately to 5,3,1 mW cm⁻². Thickness continuation uses the previous thickness as a guess. Because n_* and J_* change, the log-density and current coordinates are rescaled before reuse. Each endpoint is solved at its new L and G(L); stored CSV profiles are never used as solution seeds in a full run.

The protected exponent ranges, inherited from the solver, are [−220,120] for natural-exponential density factors and [−120,120] for the base-10 Faradaic powers. They prevent overflow on Newton trial iterates. Accepted publication states are checked to remain inside the density/interface clipping bounds. Clipping is a numerical guard and is not an additional physical constitutive law.

## 6. Forward OWS: current as the outer unknown

For a positive trial current J, invert the two reversible catalyst laws with `brentq`:

$$j_H(V_H)=-J,\qquad j_O(V_O)=J.$$

Those potentials determine the boundary data for the semiconductor BVP. Compute its total current J_sem and solve the scalar residual

$$F(J)=J_{\rm sem}(V_H(J),V_O(J))-J=0.$$

This automatically enforces both catalyst charge balances. The usual AA current-root tolerances are xtol=2e-9 A/m² and rtol=2e-10; BA uses xtol=rtol=2e-11 in its bracketing solve. The inverse Faradaic potential roots have their separate voltage tolerances. The accepted endpoint is re-solved at the final settings. A failed refinement raises an error instead of silently returning an earlier cached trial. Cache reuse for BB also respects requests for tighter tolerance or larger initial meshes.

The BA routine additionally solves both catalyst potential balances together with SciPy `root` and checks its success and residual. That is an independent outer-variable formulation using the same spatial model, not an independent discretization.

The phenomenological Faradaic expressions, exchange currents, slopes and one-sided reverse limits are exactly SI `eq:faradaic`. Their A/cm² inputs are converted to A/m² for matching the semiconductor currents. No zero-current selection from Table S8 enters these forward-current roots.

## 7. Dark, catalyst-off and reverse-load states

`solve_dark_equilibrium()` sets both catalyst potentials to 0 V, turns off generation and reports Faradaic exchange as disabled. J_n=J_p=0 and E_F=0 are then the SI dark reference. The older `solve_relaxed_dark()` API supplies a different fixed-redox-voltage numerical seed and is labelled accordingly.

For an illuminated catalyst-off state only J_sem=0 is required; individual electron and hole currents may cancel. With ideal adaptive contacts the two catalyst exchange balances do not select a unique common mode. The SI gives selected AA/AB/BA voltage separations but no additional charge/common-mode selection constraint. Those values are explicitly supplied comparison data, not reconstructed solutions.

The BB semiconductor depends only on terminal potential difference. Its photovoltaic open circuit is found from J_sem(Delta V)=0. The optional reverse HOR/ORR load has magnitude I>0, with j_H=I and j_O=−I. The intersection is J_BB(Delta V_load(I))+I=0. Near the gas-consuming limit the code solves in z=−log10(1−I/I_lim), which avoids using the nearly saturated current itself as a poorly conditioned outer coordinate.

A common voltage shift of the two B contacts is a symmetry of the semiconductor block. Energies shift by the opposite signed amount, while densities and currents stay unchanged. The actual Faradaic reactions keep their RHE reference and do not share that invariance. The validator therefore restores the absolute catalyst potentials and re-solves before exporting physical RHE-referenced energies. Gauge-specific JV energies are labelled `*_gauge_eV`.

## 8. Acceptance checks and numerical interpretation

`validation_checks.audit_state()` reconstructs energy-form carrier statistics, the band gap and all four transfer boundaries from the solution. It checks total-current constancy, both catalyst balances and the exact integrated identity

$$J_{\rm gen}=J_{\rm sem}+[-J_p(0)-J_n(L)]+q\int_0^L R\,dx.$$

For forward OWS the bracketed term is J_contact≥0. The integral uses a 2401-point trapezoidal diagnostic, so its error contains sampling/quadrature error in addition to collocation error. A maximum integrated discrepancy of 1e-6 A/m² is accepted. Boundary-transfer agreement is checked at 1e-6 A/m², total-current spatial variation at 1e-8 A/m², and the two catalytic balances normally at 1e-6 A/m², with a stricter 1e-8 check for the baseline BA state. Each audit records the actual values.

The sampled normalized spline ODE defect and the dimensional current reconstructed from differentiated quasi-Fermi splines are also reported. They are diagnostics, not replacements for the collocation tolerance. Large carrier densities amplify small derivative defects and cancellation when energy gradients are reconstructed; a dimensional reconstructed-current discrepancy is not identical to a charge-balance residual. Reference comparisons and optional tighter-mesh endpoint checks address errors in physical outputs separately.

Baseline profiles are compared with immutable upstream profiles, using relative differences for densities and absolute differences for energies/currents. Sensitivity values are compared at identical parameter coordinates with 1e-6 absolute tolerance in their reported units. Thickness comparisons use 1e-7. No numerical result is rounded before comparison. A failed comparison stops reproduction.

## References

The physical definitions and equations are in the supplied SI, particularly `eq:energy-potential-map`, `eq:poisson`, `eq:dd`, `eq:carriertransfer`, `eq:transfer-signs`, `eq:adaptivebc`, `eq:buriedbc`, `eq:faradaic` and `eq:current-balance`.

T. J. Mills, F. Lin and S. W. Boettcher, Physical Review Letters **112**, 148304 (2014), DOI: 10.1103/PhysRevLett.112.148304, supplies the reversible constant-DOS transfer framework.

SciPy 1.17 `solve_bvp` documentation describes the implemented collocation algorithm and residual definitions: https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.integrate.solve_bvp.html . See also J. Kierzenka and L. F. Shampine, ACM Transactions on Mathematical Software **27**, 299–316 (2001), DOI: 10.1145/502800.502801.

NumPy documents `trapezoid` as introduced in 2.0: https://numpy.org/doc/stable/reference/generated/numpy.trapezoid.html . The requirements reflect the function actually used here.
