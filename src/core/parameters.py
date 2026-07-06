"""
Parameter definitions and default values for QSP model
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict

@dataclass
class BacterialParameters:
    """Bacterial growth and population dynamics parameters"""
    k_growth: float = 0.5  # per hour, intrinsic growth rate
    B_max: float = 1e9  # CFU/mL, carrying capacity (log-scale typically 10^8-10^10)
    k_pers: float = 0.01  # per hour, transition to persisters
    mu_mut: float = 1e-6  # mutation rate per cell per generation (SCV emergence under static) - increased from 1e-7 for clinically relevant heteroresistance
    k_repair: float = 0.3  # per hour, damage repair rate (t1/2 ≈ 2.3h, persists between doses)
    MIC_baseline: float = 1.0  # mg/L, baseline MIC for susceptible strain

@dataclass
class ImmuneParameters:
    """Immune effector dynamics"""
    N_eff_0: float = 1e7  # baseline neutrophils/macrophages per mL
    k_prod: float = 0.5  # per hour, immune recruitment rate
    EC50_immune: float = 1e5  # CFU/mL, half-max for immune recruitment
    k_deg_immune: float = 0.05  # per hour, immune cell degradation
    k_kill_base: float = 1e-8  # per mL per hour, baseline immune killing capacity

@dataclass
class CytokineParameters:
    """Inflammatory mediators"""
    # FIXED: Reduced from 10.0 to 4.0 to match literature IL-6 levels
    # At B=10^8 CFU/mL: produces ~54,000 pg/mL (literature severe pneumonia: 10,000-100,000 pg/mL)
    k_IL6_prod: float = 4.0  # pg/mL/hour per 10^6 CFU, baseline production
    alpha_static: float = 1.0  # relative cytokine production by static drugs
    alpha_cidal: float = 3.0  # relative cytokine production by cidal drugs (TLR9-mediated)
    k_IL6_clear: float = 0.2  # per hour, IL-6 degradation
    TNF_IL6_ratio: float = 0.3  # TNF produced as fraction of IL-6

@dataclass
class PKParameters:
    """Pharmacokinetic parameters (to be populated per drug)

    Note: CL and Q are total population values in L/h (NOT per-kg).
    Vc and Vp are per-kg (L/kg) and are scaled by patient weight in the simulation.
    This convention follows the analytical PK computation which uses:
        k_elim = CL / (Vc * weight_kg)  # per hour
    """
    CL: float  # L/h, total systemic clearance (population mean for 70 kg adult)
    Vc: float  # L/kg, central volume of distribution (scaled by weight in simulation)
    Vp: float  # L/kg, peripheral volume of distribution (scaled by weight in simulation)
    Q: float   # L/h, inter-compartmental clearance (population mean for 70 kg adult)
    Ka: float  # per hour, absorption rate constant (if oral)
    Kp: float  # tissue penetration coefficient (effect-site/plasma AUC ratio)

def get_default_parameters() -> Dict:
    """
    Returns complete default parameter set
    """
    return {
        'bacteria': BacterialParameters(),
        'immune': ImmuneParameters(),
        'cytokine': CytokineParameters(),
        # Drug-specific PK will be loaded separately
    }

def get_drug_pk_parameters(drug_name: str) -> PKParameters:
    """
    Return population mean PK parameters for selected drugs.

    CL and Q are total values in L/h (population mean for 70 kg adult).
    Vc and Vp are per-kg in L/kg (scaled by patient weight in simulation).

    References:
        Meropenem: Drusano 1995 (CL 9-15 L/h, Vd ~0.3 L/kg, t1/2 ~1 h)
        Doxycycline: Agwuh & MacGowan 2006 (CL 3-8 L/h, Vd 0.5-1.0 L/kg)
        Linezolid: Blevins 2003 (CL 3-6 L/h, Vd 0.6-0.8 L/kg)
        Ciprofloxacin: Lettieri 1992 (CL 15-30 L/h, Vd 2.0 L/kg)
    """
    if drug_name.lower() == 'doxycycline':
        # Oral doxycycline: 100-200 mg q12-24h
        # Literature (Agwuh & MacGowan 2006): 200 mg gives Cmax ~3-5 mg/L at
        #   tmax ~2-3 h, t1/2 ~16-18 h, AUC ~90-113 mg.h/L. The cited AUC fixes the
        #   apparent oral clearance: CL/F = Dose/AUC = 200/100 ~= 2.0 L/h (an earlier
        #   code comment quoting "3-8 L/h" is inconsistent with that AUC and the
        #   18 h half-life). CL set to 1.8 L/h, which places t1/2 (16.2 h at 70 kg,
        #   = 0.693 * Vc*weight / CL), AUC0-inf (~111 mg.h/L) and Cmax (~4.3 mg/L)
        #   all within the published ranges. The prior CL=5.0 gave only ~5.8 h,
        #   which curve-level external validation exposed as under-prediction of the
        #   24 h tail. Ka raised 0.5 -> 1.3 /h to reproduce the observed ~2.7 h tmax.
        return PKParameters(
            CL=1.8,        # L/h apparent CL/F (consistent with AUC ~90-113 and t1/2 ~16-18 h)
            Vc=0.6,        # L/kg (literature: 0.5-1.0 L/kg)
            Vp=1.2,        # L/kg
            Q=5.0,         # L/h (estimated)
            Ka=1.3,        # per hour (oral absorption; tmax ~2.7 h)
            Kp=0.7         # lung penetration ratio
        )
    elif drug_name.lower() == 'meropenem':
        # IV meropenem: 500-2000 mg q6-8h
        # Literature: CL 9-15 L/h, Vd ~0.3 L/kg, t1/2 ~1 h, lung penetration moderate
        return PKParameters(
            CL=15.0,       # L/h (total; literature: 9-15 L/h for normal renal function)
            Vc=0.25,       # L/kg (literature: 0.2-0.3 L/kg)
            Vp=0.15,       # L/kg
            Q=8.0,         # L/h (estimated)
            Ka=0.0,        # IV only
            Kp=0.4         # lung penetration ratio (literature: 0.3-0.5)
        )
    elif drug_name.lower() == 'linezolid':
        # IV/PO linezolid: 600 mg q12h
        # Literature: half-life 4-5h, Vd ~0.6-0.8 L/kg, lung penetration good
        return PKParameters(
            CL=5.0,        # L/h (total; literature: 3-6 L/h)
            Vc=0.65,       # L/kg (literature: 0.6-0.8 L/kg)
            Vp=0.35,       # L/kg
            Q=4.0,         # L/h (estimated)
            Ka=0.4,        # per hour (if oral)
            Kp=0.75        # lung penetration ratio (literature: 0.6-0.8)
        )
    elif drug_name.lower() == 'ciprofloxacin':
        # Fluoroquinolone: 400 mg IV or 500-750 mg PO q12h
        # Literature: CL 15-30 L/h, Vd ~2.0 L/kg
        return PKParameters(
            CL=22.0,       # L/h (total; literature: 15-30 L/h)
            Vc=2.0,        # L/kg (literature: 1.5-2.5 L/kg)
            Vp=1.0,        # L/kg
            Q=10.0,        # L/h (estimated)
            Ka=0.8,        # per hour (oral)
            Kp=0.6         # moderate lung penetration
        )
    else:
        raise ValueError(f"Unknown drug: {drug_name}")

def normalize_pk_parameters(params: PKParameters, weight_kg: float) -> Dict:
    """
    Scale per-kg PK parameters (Vc, Vp) by patient weight.
    CL and Q are already total values (L/h) and are not scaled.

    Returns:
        Dict with keys 'CL' (L/h), 'Vc' (L), 'Vp' (L), 'Q' (L/h), 'Ka', 'Kp'
    """
    return {
        'CL': params.CL,          # L/h (already total, not scaled)
        'Vc': params.Vc * weight_kg,  # L (scaled from L/kg)
        'Vp': params.Vp * weight_kg,  # L (scaled from L/kg)
        'Q': params.Q,            # L/h (already total, not scaled)
        'Ka': params.Ka,
        'Kp': params.Kp,
    }
