"""Regime-map computations for the JTB cidal/static study.

Builds the threshold surface (immune capacity x persister rate, across exposure),
the hyperinflammatory trade-off map (immune capacity x inflammation susceptibility),
and robustness sweeps, all from the Phase 2a strategy-margin engine. Functions
return plain dicts of numpy arrays for JSON serialisation by the driver script.
"""
import numpy as np

from src.analysis.strategy_margin import margin_at_point, margin_grid, zero_crossing


def compute_threshold_surface(n_eff_values, k_pers_values, exposures):
    """Delta over (n_eff x k_pers) at each exposure, plus the Delta=0 immune boundary."""
    n_eff_values = np.asarray(n_eff_values, dtype=float)
    k_pers_values = np.asarray(k_pers_values, dtype=float)
    n_exp = len(exposures)
    delta = np.zeros((n_exp, len(n_eff_values), len(k_pers_values)))
    boundary = np.zeros((n_exp, len(k_pers_values)))
    for e, exp in enumerate(exposures):
        delta[e] = margin_grid(n_eff_values, k_pers_values, k_infl=0.03, exposure_scale=exp)
        for j in range(len(k_pers_values)):
            boundary[e, j] = zero_crossing(n_eff_values, delta[e, :, j])
    return {
        "n_eff": n_eff_values, "k_pers": k_pers_values,
        "exposures": np.asarray(exposures, dtype=float),
        "delta": delta, "boundary_n_eff": boundary,
    }


def compute_tradeoff_map(n_eff_values, k_infl_values):
    """Delta over (n_eff x k_infl); mask where static is preferred; decomposition."""
    n_eff_values = np.asarray(n_eff_values, dtype=float)
    k_infl_values = np.asarray(k_infl_values, dtype=float)
    delta = np.zeros((len(n_eff_values), len(k_infl_values)))
    for i, n_eff in enumerate(n_eff_values):
        for j, k_infl in enumerate(k_infl_values):
            delta[i, j] = margin_at_point(n_eff, 0.01, k_infl).delta
    prefer_static = delta < 0.0
    # Decomposition at the most-inflammation-susceptible, mid-immune reference point.
    i_mid = len(n_eff_values) // 2
    ref = margin_at_point(n_eff_values[i_mid], 0.01, float(k_infl_values[-1]))
    decomposition = {
        "path_static": ref.path_static, "infl_static": ref.infl_static,
        "path_cidal": ref.path_cidal, "infl_cidal": ref.infl_cidal,
        "n_eff": float(n_eff_values[i_mid]), "k_infl": float(k_infl_values[-1]),
    }
    return {
        "n_eff": n_eff_values, "k_infl": k_infl_values,
        "delta": delta, "prefer_static": prefer_static, "decomposition": decomposition,
    }


def compute_robustness(reference, scv_midpoints, horizons):
    """Delta at a reference point vs SCV-switch midpoint, vs horizon, and terminal outcome."""
    n_eff, k_pers, k_infl = reference
    delta_vs_scv = [margin_at_point(n_eff, k_pers, k_infl, scv_midpoint=m).delta
                    for m in scv_midpoints]
    delta_vs_horizon = [margin_at_point(n_eff, k_pers, k_infl, t_span=tuple(h)).delta
                        for h in horizons]
    delta_terminal = margin_at_point(n_eff, k_pers, k_infl).delta_terminal
    return {
        "reference": np.asarray(reference, dtype=float),
        "scv_midpoints": np.asarray(scv_midpoints, dtype=float),
        "delta_vs_scv": np.asarray(delta_vs_scv, dtype=float),
        "horizons": np.asarray(horizons, dtype=float),
        "delta_vs_horizon": np.asarray(delta_vs_horizon, dtype=float),
        "delta_terminal": float(delta_terminal),
    }
