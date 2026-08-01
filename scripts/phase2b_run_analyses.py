"""Run the Phase 2b regime analyses and write JSON to results/phase2b/.

Usage: python scripts/phase2b_run_analyses.py [draft|final]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.regime_maps import (
    compute_threshold_surface, compute_tradeoff_map, compute_robustness,
)
from src.analysis.sensitivity_analysis import run_sensitivity_analysis

OUT_DIR = os.path.join("results", "phase2b")

RES = {
    "draft": dict(n=11, exposures=[0.5, 1.0, 2.0, 4.0], sobol_n=16),
    "final": dict(n=31, exposures=[0.5, 1.0, 2.0, 4.0], sobol_n=256),
}


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


def _dump(name, obj, res):
    payload = {"resolution": res, **_jsonable(obj)}
    with open(os.path.join(OUT_DIR, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("wrote", name)


def main(res="draft"):
    cfg = RES[res]
    n = cfg["n"]
    os.makedirs(OUT_DIR, exist_ok=True)

    n_eff = np.logspace(4, 8, n)            # 1e4 .. 1e8 effectors/mL
    k_pers = np.linspace(0.001, 0.05, n)    # per hour
    k_infl = np.linspace(0.005, 0.10, n)    # per hour

    thr = compute_threshold_surface(n_eff, k_pers, cfg["exposures"])
    _dump("threshold_surface", thr, res)

    trade = compute_tradeoff_map(n_eff, k_infl)
    _dump("tradeoff_map", trade, res)

    robust = compute_robustness(
        reference=(1e7, 0.01, 0.03),
        scv_midpoints=[0.2, 0.25, 0.3, 0.35, 0.4],
        horizons=[(0, 48), (0, 72), (0, 96), (0, 168)],
    )
    _dump("robustness", robust, res)

    sob = run_sensitivity_analysis(
        param_names=["k_pers", "k_infl", "k_path", "k_heal", "k_kill_base"],
        metric="peak_host_damage", drug_class="cidal",
        n_samples=cfg["sobol_n"], calc_second_order=False, print_progress=False,
    )
    sobol_out = {
        "names": sob["problem"]["names"],
        "S1": np.asarray(sob["Si"]["S1"]).tolist(),
        "ST": np.asarray(sob["Si"]["ST"]).tolist(),
    }
    _dump("sobol", sobol_out, res)
    print("Phase 2b analyses complete ->", OUT_DIR)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "draft")
