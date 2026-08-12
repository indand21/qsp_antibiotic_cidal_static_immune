"""Static-vs-cidal strategy-preference margin engine (regime study).

At a point in (immune capacity, persister rate, inflammation susceptibility)
space we run the model twice at IDENTICAL exposure, toggling only the drug
mechanism (cidal killing vs static growth-inhibition), and compare peak host
damage. Lower peak D_host is better, so the preference margin is
    delta = peak_D_host(static) - peak_D_host(cidal)
delta > 0 => cidal preferred; delta < 0 => static preferred.
"""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import trapezoid

from src.core.parameters import (
    get_default_parameters,
    get_drug_pk_parameters,
    normalize_pk_parameters,
)
from src.core.pd_model import create_ode_system
from src.core.pk_models import TwoCompartmentPKModel, DosingRegimen
from src.core.simulation import run_simulation

# Standard host initial conditions (immunocompetent defaults); N_eff is overridden
# per call to set immune capacity.
DEFAULT_INIT = {
    "B_rep": 1e6, "B_pers": 1e3, "B_SCV": 0, "N_eff": 1e7,
    "Damage": 0, "IL6": 10, "TNF": 5, "PAMP": 0, "D_host": 0,
}

# Representative exposure: meropenem-like PK, 1000 mg q8h x6, 30-min infusion.
_BASE_DOSE_MG = 1000.0
_WEIGHT_KG = 70.0


def _make_pk():
    pk = normalize_pk_parameters(get_drug_pk_parameters("meropenem"), _WEIGHT_KG)
    return TwoCompartmentPKModel(**pk, effect_site_model=True)


def run_one(n_eff, k_pers, k_infl, drug_class, exposure_scale=1.0, t_span=(0, 96),
            scv_midpoint=None, init_overrides=None):
    """Run one simulation at a grid point for the given drug mechanism.

    init_overrides, if given, updates the initial conditions (e.g. elevated
    baseline cytokines for a hyperinflammatory phenotype).
    """
    params = get_default_parameters()
    params["bacteria"].k_pers = k_pers
    params["damage"].k_infl = k_infl
    if scv_midpoint is not None:
        params["bacteria"].scv_switch_midpoint = scv_midpoint
    pd_model = create_ode_system(params)

    regimen = DosingRegimen(
        dose_mg=_BASE_DOSE_MG * exposure_scale,
        interval_hours=8, start_time=0, n_doses=6, infusion_duration_min=30,
    )
    ic = dict(DEFAULT_INIT)
    ic["N_eff"] = n_eff
    if init_overrides:
        ic.update(init_overrides)
    return run_simulation(
        pk_model=_make_pk(), regimen=regimen, pd_model=pd_model,
        initial_conditions=ic, t_span=t_span, drug_class=drug_class,
        weight_kg=_WEIGHT_KG,
    )


def decompose_damage(result, damage_params):
    """Cumulative pathogen-driven and inflammation-driven injury over the run.

    Recomputes the two source terms from the trajectory and integrates each
    over time (before healing), giving their total contribution to host damage.
    """
    t = result.t
    B_total = result.y[:, 4] + result.y[:, 5] + result.y[:, 6]  # B_rep+B_pers+B_SCV
    il6 = result.y[:, 9]
    tnf = result.y[:, 10]
    infl_il6 = np.clip(il6 / damage_params.IL6_ref - 1.0, 0.0, None)
    infl_tnf = np.clip(tnf / damage_params.TNF_ref - 1.0, 0.0, None)
    infl_intensity = infl_il6 + damage_params.w_TNF * infl_tnf

    pathogen_rate = damage_params.k_path * B_total / (B_total + damage_params.B50)
    infl_rate = (damage_params.k_infl * infl_intensity
                 / (infl_intensity + damage_params.I50))
    return float(trapezoid(pathogen_rate, t)), float(trapezoid(infl_rate, t))


def zero_crossing(x_values, deltas):
    """Interpolated x at the first sign change of `deltas` (nan if none)."""
    x = np.asarray(x_values, dtype=float)
    d = np.asarray(deltas, dtype=float)
    for i in range(len(d) - 1):
        if d[i] == 0.0:
            return float(x[i])
        if d[i] * d[i + 1] < 0.0:
            frac = d[i] / (d[i] - d[i + 1])
            return float(x[i] + frac * (x[i + 1] - x[i]))
    if len(d) and d[-1] == 0.0:
        return float(x[-1])
    return float("nan")


@dataclass
class MarginResult:
    delta: float
    peak_static: float
    peak_cidal: float
    path_static: float
    infl_static: float
    path_cidal: float
    infl_cidal: float
    terminal_static: float = 0.0
    terminal_cidal: float = 0.0
    delta_terminal: float = 0.0


def margin_at_point(n_eff, k_pers, k_infl, exposure_scale=1.0, t_span=(0, 96),
                    scv_midpoint=None):
    """Compare static vs cidal at identical exposure; return the margin + decomposition."""
    r_static = run_one(n_eff, k_pers, k_infl, "static", exposure_scale, t_span, scv_midpoint)
    r_cidal = run_one(n_eff, k_pers, k_infl, "cidal", exposure_scale, t_span, scv_midpoint)
    dmg = get_default_parameters()["damage"]
    dmg.k_infl = k_infl
    path_s, infl_s = decompose_damage(r_static, dmg)
    path_c, infl_c = decompose_damage(r_cidal, dmg)
    peak_s = r_static.peak_host_damage()
    peak_c = r_cidal.peak_host_damage()
    term_s = r_static.terminal_host_damage()
    term_c = r_cidal.terminal_host_damage()
    return MarginResult(
        delta=peak_s - peak_c,
        peak_static=peak_s, peak_cidal=peak_c,
        path_static=path_s, infl_static=infl_s,
        path_cidal=path_c, infl_cidal=infl_c,
        terminal_static=term_s, terminal_cidal=term_c, delta_terminal=term_s - term_c,
    )


def margin_grid(n_eff_values, k_pers_values, k_infl=0.03, exposure_scale=1.0):
    """2-D grid of delta over (n_eff_values x k_pers_values)."""
    grid = np.zeros((len(n_eff_values), len(k_pers_values)))
    for i, n_eff in enumerate(n_eff_values):
        for j, k_pers in enumerate(k_pers_values):
            grid[i, j] = margin_at_point(n_eff, k_pers, k_infl, exposure_scale).delta
    return grid
