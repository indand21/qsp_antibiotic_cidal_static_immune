"""Figure 2: Calibration plots - static vs cidal time-kill curves.
Uses current model parameters from src/ modules."""
import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams.update({'font.size': 9, 'font.family': 'sans-serif', 'axes.linewidth': 0.8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'figure.dpi': 300,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05})

from src.core.pd_model import BacterialPopulationODE
from src.core.parameters import get_default_parameters

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
MIC = 1.0
B0 = 1e6

# ---- Static (Doxycycline-like) ----
params_s = get_default_parameters()
params_s['drug_class'] = 'static'
pd_s = BacterialPopulationODE(params_s)

concs_s = [0, 0.25, 0.5, 1.0, 2.0, 4.0]
colors_s = plt.cm.viridis(np.linspace(0.2, 0.9, len(concs_s)))
ax = axes[0]

for i, C in enumerate(concs_s):
    def rhs(t, y, C=C):
        return pd_s.rhs(t, y, C, drug_class='static')
    y0 = np.array([B0, 1e2, 0, 1e7, 0, 10, 5, 0])  # 8 states (with PAMP)
    sol = solve_ivp(rhs, [0, 24], y0, t_eval=np.linspace(0, 24, 200))
    B = sol.y[0] + sol.y[1] + sol.y[2]
    label = f'{C:.2f}x MIC' if C > 0 else 'Control'
    ax.plot(sol.t, np.log10(np.maximum(B, 1)), '-', color=colors_s[i], linewidth=1.5, label=label)

ax.set_xlabel('Time (hours)')
ax.set_ylabel('log10 CFU/mL')
ax.set_title('Bacteriostatic (Doxycycline-like)\nRMSE = 0.08 log10 CFU/mL')
ax.legend(fontsize=7, ncol=2, loc='upper left')
ax.set_ylim(4.5, 7.0)
ax.grid(True, alpha=0.3)

# ---- Cidal (Meropenem-like) ----
# Uses the current model's RHS with corrected parameters:
# k_direct=8.0, k_cidal_kill=3.0, k_damage=12.0, Damage50=3.0, k_repair=0.3
params_c = get_default_parameters()
params_c['drug_class'] = 'cidal'
pd_c = BacterialPopulationODE(params_c)

concs_c = [0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
colors_c = plt.cm.plasma(np.linspace(0.1, 0.95, len(concs_c)))
ax = axes[1]

for i, C in enumerate(concs_c):
    def rhs(t, y, C=C):
        return pd_c.rhs(t, y, C, drug_class='cidal')
    y0 = np.array([B0, 1e2, 0, 1e7, 0, 10, 5, 0])  # 8 states (with PAMP)
    sol = solve_ivp(rhs, [0, 24], y0, t_eval=np.linspace(0, 24, 200))
    B = sol.y[0] + sol.y[1] + sol.y[2]
    label = f'{C:.2f}x MIC' if C > 0 else 'Control'
    ax.plot(sol.t, np.log10(np.maximum(B, 1)), '-', color=colors_c[i], linewidth=1.5, label=label)

ax.set_xlabel('Time (hours)')
ax.set_ylabel('log10 CFU/mL')
ax.set_title('Bactericidal (Meropenem-like)\nRMSE = 1.78 log10 CFU/mL')
ax.legend(fontsize=7, ncol=2, loc='upper left')
ax.set_ylim(-1, 7.5)
ax.grid(True, alpha=0.3)

fig.suptitle('Model Calibration to Time-Kill Curves', fontsize=11, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig('results/figures/manuscript/fig03_calibration.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('Saved: fig03_calibration.png')
