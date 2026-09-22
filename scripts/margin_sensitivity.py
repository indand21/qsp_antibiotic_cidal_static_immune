"""Variance-based (Sobol) sensitivity of the STRATEGY MARGIN itself.

Reviewer point 6: the decision-relevant question is not the sensitivity of cidal
peak damage (which is algebraically pinned under saturation) but the sensitivity
of the strategy-preference margin  Delta = peak_D_host(static) - peak_D_host(cidal)
-- i.e. which structural parameters move the cidal/static decision. We compute
first-order and total-effect Sobol indices of Delta at the hyperinflammatory
operating point (high immune capacity, high inflammation susceptibility, where
static is preferred) over the structural parameters the reviewer highlights:
inflammation coupling (alpha_cidal), injury half-saturation (I50), recovery and
inflammation rates (k_heal, k_infl), immune maintenance (k_deg_immune) and
relative potency (kill_C50, k_kill_max).

Evaluations are embarrassingly parallel and run across CPU cores. Writes
results/ijaa/margin_sensitivity.json.
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.analysis import _salib_numpy2_compat  # noqa: F401  (NumPy-2 SALib shim)
from src.core.parameters import (
    get_default_parameters, get_drug_pk_parameters,
)
from src.core.pd_model import create_ode_system
from src.core.pk_models import TwoCompartmentPKModel, DosingRegimen
from src.core.simulation import run_simulation

from SALib.sample import saltelli
from SALib.analyze import sobol

OUT = os.path.join("results", "ijaa", "margin_sensitivity.json")

# Hyperinflammatory operating point (static-preferred regime).
N_EFF = 5e7
HYPER_IC = {"B_rep": 1e6, "B_pers": 1e3, "B_SCV": 0, "N_eff": N_EFF,
            "Damage": 0, "IL6": 100, "TNF": 50, "PAMP": 0, "D_host": 0}
T_SPAN = (0, 96)
BASE_DOSE_MG, WEIGHT = 1000.0, 70.0

PROBLEM = {
    "num_vars": 7,
    "names": ["alpha_cidal", "I50", "k_heal", "k_infl",
              "k_deg_immune", "kill_C50", "k_kill_max"],
    "bounds": [[1.0, 5.0], [1.0, 20.0], [0.02, 0.30], [0.02, 0.10],
               [0.01, 0.20], [0.30, 1.20], [1.0, 8.0]],
}


def _peak(drug_class, vec):
    p = get_default_parameters()
    (alpha_cidal, I50, k_heal, k_infl, k_deg, killC50, kkill) = vec
    p["cytokine"].alpha_cidal = float(alpha_cidal)
    p["damage"].I50 = float(I50)
    p["damage"].k_heal = float(k_heal)
    p["damage"].k_infl = float(k_infl)
    p["immune"].k_deg_immune = float(k_deg)
    p["bacteria"].kill_C50 = float(killC50)
    p["bacteria"].k_kill_max = float(kkill)
    pk = get_drug_pk_parameters("meropenem")
    pkm = TwoCompartmentPKModel(CL=pk.CL, Vc=pk.Vc, Vp=pk.Vp, Q=pk.Q, Ka=pk.Ka,
                                Kp=pk.Kp, effect_site_model=True)
    n_doses = max(1, int(np.ceil((T_SPAN[1] - T_SPAN[0]) / 8)))
    reg = DosingRegimen(dose_mg=BASE_DOSE_MG, interval_hours=8, start_time=0,
                        n_doses=n_doses, infusion_duration_min=30)
    r = run_simulation(pkm, reg, create_ode_system(p), dict(HYPER_IC),
                       t_span=T_SPAN, drug_class=drug_class, weight_kg=WEIGHT)
    return float(np.max(r.get_host_damage()[1]))


def margin(vec):
    """Delta = peak_static - peak_cidal (Delta<0 => static preferred)."""
    return _peak("static", vec) - _peak("cidal", vec)


def main(n=128, processes=None):
    X = saltelli.sample(PROBLEM, n, calc_second_order=False)
    print(f"evaluating {len(X)} parameter sets x2 sims across "
          f"{processes or os.cpu_count()} processes ...")
    with Pool(processes=processes) as pool:
        Y = np.array(pool.map(margin, [tuple(row) for row in X]))
    Si = sobol.analyze(PROBLEM, Y, calc_second_order=False, print_to_console=False)
    out = {
        "operating_point": {"n_eff": N_EFF, "note": "hyperinflammatory (static-preferred)"},
        "names": PROBLEM["names"],
        "bounds": PROBLEM["bounds"],
        "n_base": n, "n_evaluations": int(len(X)),
        "margin_mean": float(Y.mean()), "margin_std": float(Y.std()),
        "margin_min": float(Y.min()), "margin_max": float(Y.max()),
        "S1": [float(x) for x in Si["S1"]],
        "ST": [float(x) for x in Si["ST"]],
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    order = np.argsort(out["ST"])[::-1]
    print(f"wrote {OUT}")
    print(f"margin Delta: mean={out['margin_mean']:+.3f} "
          f"[{out['margin_min']:+.3f}, {out['margin_max']:+.3f}]")
    print("driver          S1     ST")
    for i in order:
        print(f"  {out['names'][i]:13s} {out['S1'][i]:+.3f} {out['ST'][i]:+.3f}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    main(n=n)
