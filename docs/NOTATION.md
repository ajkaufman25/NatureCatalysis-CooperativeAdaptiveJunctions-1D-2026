# SI notation, dimensions and signs

The reference is the unchanged BA-clarified LaTeX SI in this directory. H/HER denotes x=0 and O/OER denotes x=L. The first architecture letter identifies HER. SI dimensional energy equations use joules and q>0 in coulombs; reported one-electron energies use eV.

## Energy and voltage coordinates

$$E_{\rm cat,i}=-qV_{\rm cat,i},\qquad E_{\rm RHE}=0.$$

For an electron level, the numerical mapping is `E_eV = -U_V`. For an **energy difference divided by q**, the mapping is `Delta_E_eV -> Delta_E_over_q_V` with no sign change. In particular, a positive barrier of 0.955 eV corresponds to Phi_B/q=+0.955 V. The code's explicit identity-valued `barrier_voltage_V()` marks this unit conversion; multiplying its eV argument by q and then treating the result as volts would be incorrect.

| SI quantity | Internal representation | Public output |
|---|---|---|
| E_CB, E_VB | `Ucb`, `Uvb` in V, E=-qU | `E_CB_eV`, `E_VB_eV` |
| E_F,n, E_F,p | `Ufn`, `Ufp` in V, E=-qU | `E_Fn_eV`, `E_Fp_eV` |
| V_cat,HER; V_cat,OER | `U_H`, `U_O` in V vs RHE | `V_cat_HER_V_vs_RHE`, `V_cat_OER_V_vs_RHE` |
| E_cat,HER; E_cat,OER | negative of the corresponding numeric V | `E_cat_HER_eV`, `E_cat_OER_eV` |
| Delta V_cat | `U_O-U_H` | `Delta_V_cat_V` |
| E_cat,HER − E_cat,OER | q Delta V_cat | `Delta_E_cat_eV` |
| E_F,n − E_F,p at center | `Ufp-Ufn` | `Delta_E_F_center_eV` |
| V_bb,i | `Ucb_bulk-Ucb_surface` | `V_bb_HER_V`, `V_bb_OER_V` |
| Phi_B,i | E_CB,s,i − E_cat,i | `Phi_B_HER_eV`, `Phi_B_OER_eV` |
| E_CB,s,O − E_CB,s,H | `Ucb_H-Ucb_O` | `Delta_E_CB_surface_eV` |
| J_sem | `Jn+Jp` along +x | `J_sem_mA_cm2` |
| J_OWS | positive operating J_sem | `J_OWS_mA_cm2` |
| J_contact | `-Jp[0]-Jn[-1]` in forward OWS | `J_contact_mA_cm2` |
| J_SRH | q integral of R over thickness | `J_SRH_mA_cm2` |

The band relation is E_VB=E_CB−E_g. The positive catalyst separation has opposite contact order in the two representations:

$$\Delta V_{\rm cat}=V_{\rm cat,OER}-V_{\rm cat,HER},\qquad
q\Delta V_{\rm cat}=E_{\rm cat,HER}-E_{\rm cat,OER}.$$

## Baseline input map (SI Table S7)

| SI input | Python field | Value and stored units |
|---|---|---|
| T | `ModelParams.T` | 298.15 K |
| L | `L_s` | 1e-6 m |
| N_D | `FacetConfig.ND_cm3` | 1e18 cm⁻³; converted to 1e24 m⁻³ |
| epsilon_r | `eps_r_s` | 300 |
| E_g | `Eg` | 3.20 eV |
| m_n*/m_e, m_p*/m_e | `m_n_rel`, `m_p_rel` | 1.8, 3.0 |
| mu_n, mu_p | `mu_n`, `mu_p` | 5e-4, 1e-5 m² V⁻¹ s⁻¹ |
| tau_n, tau_p | `tau_n`, `tau_p` | 1e-4 s each |
| wavelength | `wavelength` | 365e-9 m |
| I_0 | `intensity_mW_cm2` | 10 mW cm⁻² |
| alpha | `alpha_abs` | 1e6 m⁻¹ |
| V_bb,H^0, V_bb,O^0 | `band_bending_H_V`, `band_bending_O_V` | 0.85, 1.00 V |
| k_n, k_p | `k_n`, `k_p` | 10 m s⁻¹ each |
| V_HER^eq, V_OER^eq | `FaradaicConfig.U_HER_eq_V`, `U_OER_eq_V` | 0, 1.229 V vs RHE |
| j_0,H, j_0,O | `j0_H_A_cm2`, `j0_O_A_cm2` | 1e-6, 1e-12 A cm⁻² |
| b_H, b_O | `b_H_V_dec`, `b_O_V_dec` | 0.100, 0.040 V decade⁻¹ |
| reverse limiting currents | `jlim_HOR_A_cm2`, `jlim_ORR_A_cm2` | 5e-5 A cm⁻² each |

The code retains `lambda_H`/`lambda_O` as contact switches: 0 is A and 1 is B. These are distinct from the optical wavelength. Intermediate switch values are not additional physical architectures in the SI.

E_CB^ref=+0.400 eV corresponds to the internal reference `Ucb0=-0.400` V. The surface pins are separately calibrated from the chosen E_F=0 dark reference and the native bendings. They are approximately +0.955 and +1.105 eV. Changing a coordinate reference must not silently change those prescribed physical pins.

## Currents and local transfer signs

Both J_n and J_p are conventional currents along increasing x. Local outward carrier-transfer scalars use positive q for both species:

$$j_{n,i}=qk_n(n_{s,i}-n_{{\rm eq},i}),\qquad
j_{p,i}=qk_p(p_{s,i}-p_{{\rm eq},i}).$$

The geometry and charge signs give

$$J_n(0)=j_{n,H},\quad J_p(0)=-j_{p,H},\quad
J_n(L)=-j_{n,O},\quad J_p(L)=j_{p,O}.$$

Faradaic current is anodic-positive. Thus HER has j_H<0, OER has j_O>0, and the floating catalyst balances are J_sem+j_H=0 and −J_sem+j_O=0. For forward operation, J_OWS=J_sem>0.

J_contact is the sum of the two opposite-carrier contact fluxes. It is evaluated as the exact signed combination −J_p(0)−J_n(L), without clipping in the publication exporter. In the reverse BB diagnostic, that signed combination is retained in `signed_opposite_carrier_term_mA_cm2`; `J_contact_mA_cm2` and `J_OWS_mA_cm2` are left undefined where the SI forward-OWS interpretation does not apply.

## Local bending, fixed barriers and BA

The central band reference is the mean on 0.42≤x/L≤0.58. It need not be a neutral bulk in a thin slab. The quantities

$$qV_{\rm bb,H}=E_{\rm CB,s,H}-E_{\rm CB,bulk},\qquad
\Phi_{B,H}=E_{\rm CB,s,H}-E_{\rm cat,H}$$

have different reference levels. In BA, the second stays fixed at about 0.955 eV while the first decreases from approximately 0.850 eV in the common-Fermi dark reference to 0.309 eV during operation. The floating bulk shifts upward more strongly than the HER surface. Simultaneously, E_CB,s,O−E_CB,s,H falls from 0.150 eV to 1.67 meV. Neither the catalyst potential alone nor the fixed barrier determines the local depletion bending.

## Storage precision

A/m² to mA/cm² is division by 10. Densities in m⁻³ to cm⁻³ are division by 1e6. Input intensity in mW/cm² to W/m² is multiplication by 10. CSVs retain numerical precision for regression; figure annotations use approximately three significant figures.

The original U/MIEC/counterflow/current_budget names are preserved only in immutable reference material and the compatibility-level internal API. New publication files consistently use the names above. No automatic global string replacement is used to convert energies or reverse a difference's sign.
