"""IJAA PK/PD qualification: dose-fractionation time-dependence, stasis %fT>MIC,
and the three canonical unbound PK/PD indices for the meropenem-like cidal agent.

Reuses the qualification primitives in model_qualification.py (no re-tuning) and
adds the unbound Craig-index triplet via SimulationResult.get_pkpd_indices().
Writes results/ijaa/pkpd_qualification.json (source of truth for the IJAA
manuscript) and docs/IJAA_submission_package/figures/fig01_pkpd_qualification.png.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.model_qualification import (
    effective_mic, _logkill_regimen, _fT_over_mic, IMMUNE_OFF_IC, WEIGHT,
)
from src.core.parameters import get_default_parameters, get_drug_pk_parameters, get_drug_pd_parameters
from src.core.pd_model import create_ode_system
from src.core.pk_models import TwoCompartmentPKModel, DosingRegimen
from src.core.simulation import run_simulation

OUT_JSON = os.path.join("results", "ijaa", "pkpd_qualification.json")
OUT_FIG = os.path.join("docs", "IJAA_submission_package", "figures",
                       "fig01_pkpd_qualification.png")

# Fixed total daily dose (3 g), fractionated across intervals: the classic
# dose-fractionation design that isolates the time-dependence signature.
FRACTIONATION = [("q24", 3000.0, 24, 4), ("q12", 1500.0, 12, 8),
                 ("q8", 1000.0, 8, 12), ("q6", 750.0, 6, 16)]
STASIS_DOSES_Q8 = [30, 60, 100, 150, 200, 300, 450, 700, 1000]


def _indices(dose_mg, interval, n_doses, mic_es, fu, inf_min=30, t_end=96):
    """Unbound effect-site PK/PD indices for a regimen, referenced to the model's
    effective (effect-site) MIC."""
    p = get_drug_pk_parameters("meropenem")
    pkm = TwoCompartmentPKModel(CL=p.CL, Vc=p.Vc, Vp=p.Vp, Q=p.Q, Ka=p.Ka, Kp=p.Kp,
                                effect_site_model=True)
    reg = DosingRegimen(dose_mg=dose_mg, interval_hours=interval, start_time=0,
                        n_doses=n_doses, infusion_duration_min=inf_min)
    r = run_simulation(pkm, reg, create_ode_system(get_default_parameters()),
                       dict(IMMUNE_OFF_IC), t_span=(0, t_end),
                       drug_class="cidal", weight_kg=WEIGHT)
    return r.get_pkpd_indices(mic=mic_es, fraction_unbound=fu)


def main():
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)

    mero = get_drug_pd_parameters("meropenem")           # descriptor: cidal, time_dependent
    fu = mero.fraction_unbound
    mic_es, mic_pl = effective_mic()                     # emergent model MIC

    # Dose-fractionation (fixed 3 g/day): %fT>MIC, unbound indices, 24 h log-change.
    frac = []
    for label, dose, iv, n in FRACTIONATION:
        ft = _fT_over_mic(dose, iv, n, mic_pl)
        idx = _indices(dose, iv, n, mic_es, fu)
        lk = _logkill_regimen(dose, iv, n)
        frac.append(dict(regimen=label, dose_mg=dose, interval_h=iv,
                         fT_above_MIC_pct=round(ft, 1),
                         fAUC_MIC=round(idx["fAUC_MIC"], 1),
                         fCmax_MIC=round(idx["fCmax_MIC"], 1),
                         net_log10_change_24h=round(lk, 2)))

    # Stasis %fT>MIC (q8, sweep dose; interpolate %fT>MIC at net change = 0).
    pts = []
    for d in STASIS_DOSES_Q8:
        pts.append((_fT_over_mic(d, 8, 3, mic_pl), _logkill_regimen(d, 8, 3)))
    pts.sort()
    fts = np.array([p[0] for p in pts]); lks = np.array([p[1] for p in pts])
    stasis = None
    for i in range(len(lks) - 1):
        if (lks[i] >= 0 >= lks[i + 1] or lks[i] <= 0 <= lks[i + 1]) and lks[i] != lks[i + 1]:
            stasis = float(fts[i] + (0 - lks[i]) * (fts[i + 1] - fts[i]) / (lks[i + 1] - lks[i]))
            break

    payload = dict(
        agent="meropenem-like (cidal, time-dependent)",
        effect_mode=mero.effect_mode,
        fraction_unbound=fu,
        effective_mic_mg_L=dict(effect_site=round(mic_es, 3), plasma=round(mic_pl, 3)),
        fractionation=frac,
        stasis_fT_above_MIC_pct=(round(stasis, 0) if stasis is not None else None),
        craig_carbapenem_reference=dict(stasis_pct=20, one_log_kill_pct=40,
                                        source="Craig 1998; Andes & Craig 2002"),
        stasis_sweep=[dict(fT_above_MIC_pct=round(f, 1), net_log10_change_24h=round(l, 2))
                      for f, l in pts],
    )
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("wrote", OUT_JSON)
    print(json.dumps({k: payload[k] for k in
                      ("effective_mic_mg_L", "stasis_fT_above_MIC_pct")}, indent=2))
    for row in frac:
        print(f"  {row['regimen']:>4}  %fT>MIC={row['fT_above_MIC_pct']:5.1f}  "
              f"fAUC/MIC={row['fAUC_MIC']:6.1f}  fCmax/MIC={row['fCmax_MIC']:5.1f}  "
              f"24h dlog10={row['net_log10_change_24h']:+.2f}")

    # -------- Figure: (A) fractionation time-dependence, (B) stasis curve --------
    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8,
                         "savefig.dpi": 300, "savefig.bbox": "tight"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.4))

    labels = [r["regimen"] for r in frac]
    fT = [r["fT_above_MIC_pct"] for r in frac]
    lk = [r["net_log10_change_24h"] for r in frac]
    x = np.arange(len(labels))
    ax1b = ax1.twinx()
    ax1.bar(x, fT, width=0.55, color="#4C78A8", alpha=0.85, label="%fT>MIC")
    ax1b.plot(x, lk, "o-", color="#B4413C", label="24 h $\\Delta\\log_{10}$ burden")
    ax1b.axhline(0, color="grey", lw=0.8, ls=":")
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_xlabel("regimen (fixed 3 g/day)")
    ax1.set_ylabel("%fT>MIC", color="#4C78A8")
    ax1b.set_ylabel("24 h $\\Delta\\log_{10}$ burden", color="#B4413C")
    ax1.set_title("A  Dose-fractionation (time-dependence)")

    fs = np.array([p["fT_above_MIC_pct"] for p in payload["stasis_sweep"]])
    ls = np.array([p["net_log10_change_24h"] for p in payload["stasis_sweep"]])
    o = np.argsort(fs)
    ax2.plot(fs[o], ls[o], "o-", color="#333333")
    ax2.axhline(0, color="grey", lw=0.8, ls=":")
    if stasis is not None:
        ax2.axvline(stasis, color="#E45756", lw=1.2, ls="--")
        ax2.text(stasis + 1.5, ax2.get_ylim()[1] * 0.7,
                 f"stasis ~{stasis:.0f}% fT>MIC\n(Craig ~20%)", color="#E45756", fontsize=8)
    ax2.set_xlabel("%fT>MIC")
    ax2.set_ylabel("24 h $\\Delta\\log_{10}$ burden")
    ax2.set_title("B  Stasis target vs Craig")

    fig.tight_layout()
    fig.savefig(OUT_FIG)
    plt.close(fig)
    print("wrote", OUT_FIG)


if __name__ == "__main__":
    main()
