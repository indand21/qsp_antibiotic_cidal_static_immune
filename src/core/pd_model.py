"""
Pharmacodynamic model: bacterial dynamics, immune response, cytokine production
"""

import numpy as np
from typing import Dict, Tuple

class BacterialPopulationODE:
    """
    Multi-population bacterial ODE system:
      - B_rep: replicating cells
      - B_pers: persister cells
      - B_SCV: small-colony variants (heteroresistant)
      - N_eff: effective immune effectors (neutrophils/macrophages)
      - Damage: accumulated cidal damage (unitless)
      - IL6, TNF: inflammatory cytokines
      - PAMP: pathogen-associated molecular patterns released during killing
    """

    def __init__(self, params: Dict):
        """
        Parameters:
            params: dict with keys 'bacteria', 'immune', 'cytokine'
        """
        self.p_bact = params['bacteria']
        self.p_imm = params['immune']
        self.p_cyto = params['cytokine']

    def h_static(self, C: float, EC50: float = 1.0, hill: float = 1.0) -> float:
        """
        Hill-type inhibition for bacteriostatic drugs
        H = 1 - C^hill / (EC50^hill + C^hill)
        Lower growth multiplier as concentration increases

        Units: C and EC50 are in mg/L (consistent with MIC)
        EC50=1.0 mg/L corresponds to typical MIC for susceptible strains
        """
        return 1.0 - (C**hill) / (EC50**hill + C**hill)

    def f_cidal_mechanism(self, C: float, Damage: float) -> float:
        """
        Cidal killing as function of accumulated damage
        f_cidal = Damage^n / (Damage50^n + Damage^n)
        Higher damage -> higher killing rate

        Units: C in mg/L, Damage dimensionless
        Damage50 calibrated for mg/L concentration scale
        """
        n = 2.0
        Damage50 = 3.0  # calibrated for mg/L scale
        return (Damage**n) / (Damage50**n + Damage**n)

    def rhs(self, t: float, y: np.ndarray, C_effect: float, 
            drug_class: str = 'cidal', is_static: bool = False) -> np.ndarray:
        """
        Right-hand side of the bacterial + immune ODE

        State vector y:
            y[0] = B_rep (replicating cells, CFU/mL)
            y[1] = B_pers (persisters, CFU/mL)
            y[2] = B_SCV (small-colony variants, CFU/mL)
            y[3] = N_eff (neutrophil/macrophage equivalents)
            y[4] = Damage (accumulated cidal damage, unitless)
            y[5] = IL6 (pg/mL)
            y[6] = TNF (pg/mL)

        Parameters:
            t: time (hours)
            y: state vector
            C_effect: drug concentration at effect site (mg/L)
            drug_class: 'cidal' or 'static'
            is_static: legacy parameter, True if drug is bacteriostatic
        """
        B_rep = max(y[0], 1e-6)
        B_pers = max(y[1], 0)
        B_SCV = max(y[2], 0)
        N_eff = max(y[3], 0)
        Damage = max(y[4], 0)
        IL6 = max(y[5], 0)
        TNF = max(y[6], 0)

        # PAMP state (index 7) — new
        PAMP = max(y[7], 0) if len(y) > 7 else 0.0

        B_total = B_rep + B_pers + B_SCV

        dydt = np.zeros(8)  # 7 PD states + PAMP

        # --- Replicating population ---
        # Logistic growth with carrying capacity
        growth_term = self.p_bact.k_growth * (1.0 - B_total / self.p_bact.B_max) * B_rep

        # Bacteriostatic effect: reduces growth proportionally
        if is_static or drug_class == 'static':
            H_static = self.h_static(C_effect, EC50=0.1, hill=1.2)  # EC50=0.1 mg/L for clinical potency
            growth_term *= H_static

        # Immune-mediated killing
        immune_kill = self.p_imm.k_kill_base * N_eff * B_rep

        # Cidal drug killing: via damage accumulation + direct concentration-dependent kill
        # The direct term provides fast initial kill; damage term provides sustained kill
        # NOTE: Kill rates calibrated for C_effect in mg/L (peak ~7.7 mg/L for meropenem 500mg)
        if drug_class == 'cidal' and not is_static:
            f_cidal = self.f_cidal_mechanism(C_effect, Damage)
            # Direct concentration-dependent kill (fast, saturable)
            # At C=5 mg/L: rate = 8.0 * 5/6 = 6.7/h → combined with growth (0.5/h): net ~6.2/h
            # 3-log kill in ~8-12 h with q8h dosing
            direct_kill = 8.0 * C_effect / (C_effect + 1.0) * B_rep
            # Damage-dependent kill (sustained, requires accumulation)
            damage_kill = 3.0 * f_cidal * B_rep
            cidal_kill = direct_kill + damage_kill
        else:
            cidal_kill = 0.0

        # Transition to persisters
        to_pers = self.p_bact.k_pers * B_rep

        dydt[0] = growth_term - immune_kill - cidal_kill - to_pers

        # --- Persister population ---
        # Persisters are relatively protected from drugs
        from_rep = self.p_bact.k_pers * B_rep
        immune_kill_pers = 0.1 * self.p_imm.k_kill_base * N_eff * B_pers  # slower immune kill
        exit_pers = 0.05 * B_pers  # slow reactivation

        dydt[1] = from_rep - immune_kill_pers - exit_pers

        # --- Small-colony variant population ---
        # Emerge under prolonged static pressure
        # Compute H_static for mutation logic
        if is_static or drug_class == 'static':
            H_static_check = self.h_static(C_effect, EC50=0.1, hill=1.2)
        else:
            H_static_check = 1.0

        if H_static_check < 0.3:  # significant static inhibition
            mutation_rate = self.p_bact.mu_mut * B_rep
        else:
            mutation_rate = 0

        immune_kill_scv = 0.05 * self.p_imm.k_kill_base * N_eff * B_SCV

        dydt[2] = mutation_rate - immune_kill_scv

        # --- Immune effectors (neutrophils/macrophages) ---
        # Recruitment proportional to bacterial burden
        recruit = self.p_imm.k_prod * (B_total / (self.p_imm.EC50_immune + B_total))
        degrade = self.p_imm.k_deg_immune * N_eff

        dydt[3] = recruit - degrade

        # --- Cidal damage accumulation ---
        # Units: C_effect in mg/L
        # At C=5 mg/L: k_dmg = 12.0 * 5 = 60/h → Damage_eq = 200 → f_cidal ≈ 1.0
        if drug_class == 'cidal' and not is_static:
            k_dmg = 12.0 * C_effect  # damage accumulation rate (mg/L scale)
            k_repair = self.p_bact.k_repair
            dydt[4] = k_dmg - k_repair * Damage
        else:
            dydt[4] = -self.p_bact.k_repair * Damage  # passive repair

        # --- IL-6 production ---
        # Two sources: (1) burden-dependent baseline, (2) PAMP-mediated burst during killing
        # Cidal drugs trigger more IL-6 via TLR9 (DNA release) — captured in PAMP burst
        if drug_class == 'cidal' and not is_static:
            alpha_cyto = self.p_cyto.alpha_cidal
        else:
            alpha_cyto = self.p_cyto.alpha_static

        # Baseline IL-6 from bacterial burden
        IL6_prod_baseline = alpha_cyto * self.p_cyto.k_IL6_prod * (B_rep + 0.5*B_pers) / 1e6
        # PAMP-mediated IL-6 burst (much stronger production during active killing)
        IL6_prod_pamp = self.p_cyto.k_IL6_prod * 5000.0 * PAMP / (PAMP + 1e6)
        IL6_prod = IL6_prod_baseline + IL6_prod_pamp
        IL6_clear = self.p_cyto.k_IL6_clear * IL6

        dydt[5] = IL6_prod - IL6_clear

        # --- TNF production (linked to positive IL-6 production, not net IL-6 change) ---
        TNF_prod = self.p_cyto.TNF_IL6_ratio * IL6_prod
        TNF_clear = 0.3 * TNF

        dydt[6] = TNF_prod - TNF_clear

        # --- PAMP dynamics ---
        # PAMPs are released during cidal bacterial killing (cell lysis releases DNA/LPS)
        # This drives the IL-6 burst observed during active bacterial clearance
        if drug_class == 'cidal' and not is_static:
            # Recompute kill rate for PAMP release
            f_cidal_pamp = self.f_cidal_mechanism(C_effect, Damage)
            direct_kill_pamp = 8.0 * C_effect / (C_effect + 1.0) * B_rep
            damage_kill_pamp = 3.0 * f_cidal_pamp * B_rep
            pamp_release = 1e7 * (direct_kill_pamp + damage_kill_pamp)
        else:
            pamp_release = 0.0

        pamp_clear = 2.0 * PAMP  # PAMPs cleared rapidly (t1/2 ≈ 20 min)
        dydt[7] = pamp_release - pamp_clear

        return dydt


def create_ode_system(params: Dict) -> BacterialPopulationODE:
    """Factory to create ODE system"""
    return BacterialPopulationODE(params)
