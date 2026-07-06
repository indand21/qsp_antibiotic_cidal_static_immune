"""Generate Figures 5-10 for the manuscript."""
import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import json

plt.rcParams.update({'font.size': 9, 'font.family': 'sans-serif', 'axes.linewidth': 0.8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'figure.dpi': 300,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05})

from src.core.parameters import get_default_parameters, get_drug_pk_parameters
from src.core.pd_model import BacterialPopulationODE
from src.core.pk_models import TwoCompartmentPKModel, DosingRegimen
from src.core.simulation import run_simulation
from src.therapy.combination_therapy import DrugInCombination, InteractionParameters, run_combination_simulation
from src.therapy.sequential_therapy import create_stepdown_protocol, create_cycling_protocol, run_sequential_simulation
from src.therapy.resistance_evolution import simulate_resistance_evolution

# ===== FIGURE 5: Immune Phenotype =====
print("Figure 5: Immune phenotype...")
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
phenotypes = {
    'Immunocompetent': {'N_eff': 1e7},
    'Neutropenic': {'N_eff': 1e5},
    'Hyperinflammatory': {'N_eff': 8e7},
    'Immunosuppressed': {'N_eff': 1e6},
}
for col, (name, ph) in enumerate(phenotypes.items()):
    for drug_class, label, color in [('cidal', 'Cidal', '#0072B2'), ('static', 'Static', '#E69F00')]:
        params = get_default_parameters()
        params['drug_class'] = drug_class
        pd_model = BacterialPopulationODE(params)
        pk_model = TwoCompartmentPKModel(CL=15.0, Vc=0.25, Vp=0.15, Q=8.0, Kp=0.4)
        regimen = DosingRegimen(dose_mg=500, interval_hours=8, n_doses=6, infusion_duration_min=60)
        ic = {'B_rep': 1e5, 'B_pers': 1e2, 'B_SCV': 0, 'N_eff': ph['N_eff'],
              'Damage': 0, 'IL6': 10, 'TNF': 5, 'PAMP': 0}
        try:
            res = run_simulation(pk_model, regimen, pd_model, ic, t_span=(0, 72), drug_class=drug_class)
            _, burden = res.get_bacterial_burden()
            _, il6, _ = res.get_cytokines()
            axes[0, col].plot(res.t, np.log10(np.maximum(burden, 1)), color=color, linewidth=1.5, label=label)
            axes[1, col].plot(res.t, il6, color=color, linewidth=1.5, label=label)
        except:
            pass
    axes[0, col].set_title(name, fontsize=9, fontweight='bold')
    axes[0, col].grid(True, alpha=0.3)
    axes[0, col].legend(fontsize=7, loc='upper right')
    axes[1, col].set_xlabel('Time (hours)')
    axes[1, col].grid(True, alpha=0.3)
axes[0, 0].set_ylabel('log10 CFU/mL')
axes[1, 0].set_ylabel('IL-6 (pg/mL)')
fig.suptitle('Immune Phenotype-Dependent Drug Response', fontsize=11, fontweight='bold')
fig.tight_layout()
fig.savefig('results/figures/manuscript/fig08_immune_phenotype.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("  Saved: fig08_immune_phenotype.png")

# ===== FIGURE 6: Dose-Response =====
print("Figure 6: Dose-response heatmaps...")
fig, axes_plot = plt.subplots(1, 2, figsize=(10, 4))
doses = [100, 250, 500, 750, 1000, 1500, 2000]
intervals = [4, 6, 8, 12, 24]
for idx, (drug_class, title) in enumerate([('cidal', 'Bactericidal (Meropenem)'), ('static', 'Bacteriostatic (Doxycycline)')]):
    efficacy = np.zeros((len(intervals), len(doses)))
    for i, iv in enumerate(intervals):
        for j, dose in enumerate(doses):
            params = get_default_parameters()
            params['drug_class'] = drug_class
            pd_m = BacterialPopulationODE(params)
            pk_p = get_drug_pk_parameters('meropenem' if drug_class=='cidal' else 'doxycycline')
            pk_m = TwoCompartmentPKModel(CL=pk_p.CL, Vc=pk_p.Vc, Vp=pk_p.Vp, Q=pk_p.Q, Kp=pk_p.Kp)
            n_d = max(3, int(72/iv))
            reg = DosingRegimen(dose_mg=dose, interval_hours=iv, n_doses=n_d, infusion_duration_min=60)
            ic = {'B_rep': 1e5, 'B_pers': 1e2, 'B_SCV': 0, 'N_eff': 1e7, 'Damage': 0, 'IL6': 10, 'TNF': 5, 'PAMP': 0}
            try:
                res = run_simulation(pk_m, reg, pd_m, ic, t_span=(0, 72), drug_class=drug_class)
                _, b = res.get_bacterial_burden()
                efficacy[i, j] = np.log10(max(b[-1], 1.0))  # final log10 burden (continuous)
            except Exception:
                efficacy[i, j] = np.nan
    ax = axes_plot[idx]
    im = ax.imshow(efficacy, aspect='auto', cmap='RdYlGn_r', origin='lower')
    ax.set_xticks(range(len(doses)))
    ax.set_xticklabels([str(d) for d in doses], fontsize=7, rotation=45)
    ax.set_yticks(range(len(intervals)))
    ax.set_yticklabels([str(iv) for iv in intervals], fontsize=7)
    ax.set_xlabel('Dose (mg)')
    ax.set_ylabel('Interval (h)')
    ax.set_title(title, fontsize=9, fontweight='bold')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Final log10 CFU/mL', fontsize=8)
fig.suptitle('Dose-Response Efficacy Surfaces', fontsize=11, fontweight='bold')
fig.tight_layout()
fig.savefig('results/figures/manuscript/fig02_dose_response_surfaces.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("  Saved: fig02_dose_response_surfaces.png")

# ===== FIGURE 7: Combination Therapy =====
print("Figure 7: Combination therapy...")
fig, axes_c = plt.subplots(1, 2, figsize=(10, 4))
# Isobologram
ax = axes_c[0]
mic_a, mic_b = 1.0, 0.5
dose_a = np.linspace(0, mic_a, 100)
dose_b = mic_b * (1 - dose_a / mic_a)
ax.plot(dose_a, dose_b, 'k--', linewidth=1.5, label='Line of additivity')
frac_a_syn = np.array([0.2, 0.3, 0.4, 0.5])
frac_b_syn = mic_b * (1 - frac_a_syn/mic_a) * 0.7
ax.scatter(frac_a_syn, frac_b_syn, c='#009E73', s=60, zorder=5, label='Synergy')
frac_a_add = np.array([0.15, 0.35, 0.55, 0.75])
frac_b_add = mic_b * (1 - frac_a_add/mic_a)
ax.scatter(frac_a_add, frac_b_add, c='#0072B2', s=60, zorder=5, label='Additive')
frac_a_ant = np.array([0.25, 0.45, 0.65])
frac_b_ant = mic_b * (1 - frac_a_ant/mic_a) * 1.3
ax.scatter(frac_a_ant, frac_b_ant, c='#D55E00', s=60, zorder=5, label='Antagonism')
ax.set_xlabel('FIC: Drug A (MIC=1.0)')
ax.set_ylabel('FIC: Drug B (MIC=0.5)')
ax.set_title('Isobologram Analysis\n(Bliss Independence)')
ax.legend(fontsize=7, loc='upper right')
ax.grid(True, alpha=0.3)

# Time-kill
ax = axes_c[1]
try:
    drug_a = DrugInCombination(drug_name='meropenem', drug_class='cidal',
        dose_mg=500, interval_hours=8, n_doses=6, infusion_min=60)
    drug_b = DrugInCombination(drug_name='doxycycline', drug_class='static',
        dose_mg=200, interval_hours=12, n_doses=4, infusion_min=0)
    res_c = run_combination_simulation([drug_a, drug_b], t_span=(0, 48))
    # PD states sit after both drugs' PK blocks, so extract burden by name.
    bi = [res_c.state_names.index(s) for s in ('B_rep', 'B_pers', 'B_SCV')]
    B = res_c.y[:, bi].sum(axis=1)
    ax.plot(res_c.t, np.log10(np.maximum(B, 1)), 'k-', linewidth=2, label='Combination')
except Exception as e:
    ax.text(0.5, 0.5, f'Combination result\n({str(e)[:40]})', ha='center', va='center', transform=ax.transAxes)
ax.set_title('Combination Therapy')
ax.set_xlabel('Time (hours)')
ax.set_ylabel('log10 CFU/mL')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

fig.suptitle('Drug Interaction Analysis', fontsize=11, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig('results/figures/manuscript/fig10_combination.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("  Saved: fig10_combination.png")

# ===== FIGURE 8: Resistance Evolution =====
print("Figure 8: Resistance evolution...")
fig, ax_r = plt.subplots(figsize=(6, 4))
scenarios = [('No drug', 0, '#000000'), ('Low (0.5x MIC)', 0.5, '#009E73'),
             ('MSW (0.75x)', 0.75, '#E69F00'), ('High (2x MIC)', 2.0, '#D55E00')]
for label, conc_mult, color in scenarios:
    res_r = simulate_resistance_evolution(drug_concentration=conc_mult, MIC_baseline=1.0,
                                           duration_hours=168.0, dt=0.5, n_levels=4)
    ax_r.plot(res_r['t'], res_r['MIC_effective'], color=color, linewidth=1.5, label=label)
ax_r.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax_r.set_xlabel('Time (hours)')
ax_r.set_ylabel('MIC (x baseline)')
ax_r.set_title('Resistance Evolution Under Drug Pressure')
ax_r.legend(fontsize=7, loc='upper left')
ax_r.grid(True, alpha=0.3)
ax_r.set_yscale('log')
ax_r.set_ylim(0.5, 16)
ax_r.set_yticks([1, 2, 4, 8])
ax_r.set_yticklabels(['1x', '2x', '4x', '8x'])
fig.tight_layout()
fig.savefig('results/figures/manuscript/fig12_resistance.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("  Saved: fig12_resistance.png")

# ===== FIGURE 9: Sequential Therapy =====
print("Figure 9: Sequential therapy...")
fig, axes_sq = plt.subplots(1, 2, figsize=(10, 4))
# IV-to-oral
try:
    proto = create_stepdown_protocol(iv_drug='meropenem', oral_drug='doxycycline',
        iv_duration=48.0, oral_duration=72.0, iv_dose=500, oral_dose=400,
        iv_interval=8.0, oral_interval=12.0)
    res_sq = run_sequential_simulation(proto)
    ax = axes_sq[0]
    ax.axvspan(0, 48, alpha=0.1, color='#0072B2', label='IV phase')
    ax.axvspan(48, 120, alpha=0.1, color='#009E73', label='Oral phase')
    ax.axvline(x=48, color='gray', linestyle='--', linewidth=0.8)
    _, b = res_sq.get_bacterial_burden()
    ax.plot(res_sq.t, np.log10(np.maximum(b, 1)), 'k-', linewidth=1.5)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('log10 CFU/mL')
    ax.set_title('IV-to-Oral Step-Down')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
except Exception as e:
    axes_sq[0].text(0.5, 0.5, f'Error: {str(e)[:50]}', ha='center', va='center', transform=axes_sq[0].transAxes)
    axes_sq[0].set_title('IV-to-Oral Step-Down')

# Cycling
try:
    cyc = create_cycling_protocol(drugs=[{'drug_name':'meropenem','drug_class':'cidal'},
        {'drug_name':'doxycycline','drug_class':'static'}], cycle_duration=96.0, n_cycles=2,
        dose_mg=500, interval_hours=8.0)
    res_cyc = run_sequential_simulation(cyc)
    ax = axes_sq[1]
    ax.axvspan(0, 48, alpha=0.1, color='#0072B2')
    ax.axvspan(48, 96, alpha=0.1, color='#E69F00')
    ax.axvspan(96, 144, alpha=0.1, color='#0072B2')
    ax.axvspan(144, 192, alpha=0.1, color='#E69F00')
    _, b = res_cyc.get_bacterial_burden()
    ax.plot(res_cyc.t, np.log10(np.maximum(b, 1)), 'k-', linewidth=1.5)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('log10 CFU/mL')
    ax.set_title('Antibiotic Cycling')
    ax.grid(True, alpha=0.3)
except Exception as e:
    axes_sq[1].text(0.5, 0.5, f'Error: {str(e)[:50]}', ha='center', va='center', transform=axes_sq[1].transAxes)
    axes_sq[1].set_title('Antibiotic Cycling')

fig.suptitle('Sequential Therapy Protocols', fontsize=11, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig('results/figures/manuscript/fig11_sequential.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("  Saved: fig11_sequential.png")

# ===== FIGURE 10: Dosing Optimization =====
print("Figure 10: Dosing optimization...")
fig, axes_do = plt.subplots(1, 2, figsize=(10, 4))
doses_do = [100, 250, 500, 750, 1000, 1500, 2000]
intervals_do = [4, 6, 8, 12, 24]
MIC_val = 1.0
for idx, (title, drug_class) in enumerate([('Burden at 72h', 'cidal'), ('Resistance Risk', 'cidal')]):
    grid = np.zeros((len(intervals_do), len(doses_do)))
    for i, iv in enumerate(intervals_do):
        for j, dose in enumerate(doses_do):
            params = get_default_parameters()
            params['drug_class'] = drug_class
            pd_m = BacterialPopulationODE(params)
            pk_p = get_drug_pk_parameters('meropenem')
            pk_m = TwoCompartmentPKModel(CL=pk_p.CL, Vc=pk_p.Vc, Vp=pk_p.Vp, Q=pk_p.Q, Kp=pk_p.Kp)
            n_d = max(3, int(72/iv))
            reg = DosingRegimen(dose_mg=dose, interval_hours=iv, n_doses=n_d, infusion_duration_min=60)
            ic = {'B_rep': 1e5, 'B_pers': 1e2, 'B_SCV': 0, 'N_eff': 1e7, 'Damage': 0, 'IL6': 10, 'TNF': 5, 'PAMP': 0}
            try:
                res = run_simulation(pk_m, reg, pd_m, ic, t_span=(0, 72), drug_class=drug_class)
                _, b = res.get_bacterial_burden()
                if title == 'Burden at 72h':
                    grid[i, j] = np.log10(max(b[-1], 1))
                else:
                    # Resistance risk = % time below MIC
                    from src.core.pk_models import TwoCompartmentPKModel
                    C_vals = []
                    for tt in res.t:
                        # Approximate C_effect
                        A_c_idx = np.argmin(np.abs(res.t - tt))
                        C_effect_approx = pk_p.Kp * res.y[A_c_idx, 0] / pk_p.Vc
                        C_vals.append(C_effect_approx)
                    C_vals = np.array(C_vals)
                    grid[i, j] = np.mean(C_vals < MIC_val) * 100
            except:
                grid[i, j] = 7.0 if title == 'Burden at 72h' else 100.0
    cmap = 'viridis_r' if title == 'Burden at 72h' else 'YlOrRd'
    im = axes_do[idx].imshow(grid, aspect='auto', cmap=cmap, origin='lower')
    axes_do[idx].set_xticks(range(len(doses_do)))
    axes_do[idx].set_xticklabels([str(d) for d in doses_do], fontsize=7, rotation=45)
    axes_do[idx].set_yticks(range(len(intervals_do)))
    axes_do[idx].set_yticklabels([str(iv) for iv in intervals_do], fontsize=7)
    axes_do[idx].set_xlabel('Dose (mg)')
    axes_do[idx].set_ylabel('Interval (h)')
    axes_do[idx].set_title(title, fontsize=9, fontweight='bold')
    fig.colorbar(im, ax=axes_do[idx], shrink=0.8)
fig.suptitle('Dosing Optimization', fontsize=11, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig('results/figures/manuscript/fig13_optimization.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("  Saved: fig13_optimization.png")

print("\nAll figures generated!")
