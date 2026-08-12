"""Render Phase 2b figures from results/phase2b/*.json to results/figures/manuscript/."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

plt.rcParams.update({"font.size": 9, "font.family": "sans-serif",
                     "axes.linewidth": 0.8, "savefig.dpi": 300,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.05})

IN_DIR = os.path.join("results", "phase2b")
OUT_DIR = os.path.join("results", "figures", "manuscript")


def _load(name):
    with open(os.path.join(IN_DIR, f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


def fig_threshold_surface():
    d = _load("threshold_surface")
    delta = np.array(d["delta"])            # (exposure, n_eff, k_pers)
    n_eff = np.array(d["n_eff"]); k_pers = np.array(d["k_pers"])
    exps = d["exposures"]
    fig, axes = plt.subplots(1, len(exps), figsize=(3.2 * len(exps), 3), squeeze=False)
    vmax = np.max(np.abs(delta))
    im = None
    for e, ax in enumerate(axes[0]):
        im = ax.pcolormesh(k_pers, np.log10(n_eff), delta[e],
                           cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        ax.set_title(f"exposure x{exps[e]:g}")
        ax.set_xlabel("k_pers (/h)")
        if e == 0:
            ax.set_ylabel("log10 immune capacity (N_eff)")
    fig.colorbar(im, ax=axes[0].tolist(), label="delta (static - cidal peak D_host)")
    fig.savefig(os.path.join(OUT_DIR, "fig03_threshold_surface.png"))
    plt.close(fig)


def fig_tradeoff_map():
    d = _load("tradeoff_map")
    delta = np.array(d["delta"]); n_eff = np.array(d["n_eff"]); k_infl = np.array(d["k_infl"])
    fig, ax = plt.subplots(figsize=(5, 4))
    vmax = np.max(np.abs(delta))
    im = ax.pcolormesh(k_infl, np.log10(n_eff), delta,
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    ax.contour(k_infl, np.log10(n_eff), delta, levels=[0.0], colors="k", linewidths=1.2)
    ax.set_xlabel("k_infl (inflammation susceptibility, /h)")
    ax.set_ylabel("log10 immune capacity (N_eff)")
    ax.set_title("Static preferred (blue, delta<0) vs cidal (red)")
    fig.colorbar(im, ax=ax, label="delta (static - cidal peak D_host)")
    fig.savefig(os.path.join(OUT_DIR, "fig04_tradeoff_map.png"))
    plt.close(fig)


def fig_sobol_robustness():
    s = _load("sobol"); r = _load("robustness")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.5))
    names = s["names"]; x = np.arange(len(names))
    a1.bar(x - 0.2, s["S1"], width=0.4, label="S1")
    a1.bar(x + 0.2, s["ST"], width=0.4, label="ST")
    a1.set_xticks(x); a1.set_xticklabels(names, rotation=45, ha="right")
    a1.set_ylabel("Sobol index"); a1.set_title("Peak host-damage sensitivity"); a1.legend()
    a2.plot(r["scv_midpoints"], r["delta_vs_scv"], "o-")
    a2.axhline(0.0, color="k", lw=0.6)
    a2.set_xlabel("SCV switch midpoint"); a2.set_ylabel("delta at reference")
    a2.set_title("Robustness to SCV threshold")
    fig.savefig(os.path.join(OUT_DIR, "fig05_sobol_robustness.png"))
    plt.close(fig)


def fig_representative_trajectories():
    from src.analysis.strategy_margin import run_one
    # The hyperinflammatory phenotype additionally starts from elevated baseline
    # cytokines (IL-6, TNF), matching its definition in the Model/Supplement.
    phenos = [("neutropenic", 1e5, None), ("immunosuppressed", 5e6, None),
              ("immunocompetent", 1e7, None),
              ("hyperinflammatory", 5e7, {"IL6": 100, "TNF": 50})]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3), sharey=True)
    for ax, (label, n_eff, ic_ovr) in zip(axes, phenos):
        for dc, color in (("cidal", "crimson"), ("static", "steelblue")):
            r = run_one(n_eff=n_eff, k_pers=0.01, k_infl=0.03, drug_class=dc,
                        init_overrides=ic_ovr)
            t, dh = r.get_host_damage()
            ax.plot(t, dh, color=color, label=dc)
        ax.set_title(label); ax.set_xlabel("time (h)")
    axes[0].set_ylabel("host damage D_host"); axes[0].legend()
    fig.savefig(os.path.join(OUT_DIR, "fig02_representative_trajectories.png"))
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fig_representative_trajectories()
    fig_threshold_surface()
    fig_tradeoff_map()
    fig_sobol_robustness()
    print("Phase 2b figures ->", OUT_DIR)


if __name__ == "__main__":
    main()
