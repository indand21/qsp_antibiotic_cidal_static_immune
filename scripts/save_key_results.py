"""Save key simulation outputs to back up manuscript claims."""
import sys
sys.path.insert(0, '.')

import numpy as np
import json
from src.core.parameters import get_default_parameters, get_drug_pk_parameters
from src.core.pd_model import BacterialPopulationODE
from src.core.pk_models import TwoCompartmentPKModel, DosingRegimen
from src.core.simulation import run_simulation

results = {}

# 1. Immunocompetent cidal - should clear
print("1. Immunocompetent cidal...")
params = get_default_parameters()
params['drug_class'] = 'cidal'
pd_m = BacterialPopulationODE(params)
pk_p = get_drug_pk_parameters('meropenem')
pk_m = TwoCompartmentPKModel(CL=pk_p.CL, Vc=pk_p.Vc, Vp=pk_p.Vp, Q=pk_p.Q, Kp=pk_p.Kp)
reg = DosingRegimen(dose_mg=500, interval_hours=8, n_doses=6, infusion_duration_min=60)
ic = {'B_rep': 1e5, 'B_pers': 1e2, 'B_SCV': 0, 'N_eff': 1e7, 'Damage': 0, 'IL6': 10, 'TNF': 5, 'PAMP': 0}
res = run_simulation(pk_m, reg, pd_m, ic, t_span=(0, 72), drug_class='cidal')
_, b = res.get_bacterial_burden()
_, il6, _ = res.get_cytokines()
results['immunocompetent_cidal'] = {
    'final_burden': float(b[-1]), 'final_log10': float(np.log10(max(b[-1], 1))),
    'peak_il6': float(np.max(il6)), 'cleared_48h': bool(b[96] < 1 if len(b) > 96 else b[-1] < 1),
}
print(f"  Final burden: {b[-1]:.2e} (log10={np.log10(max(b[-1],1)):.2f})")
print(f"  Peak IL-6: {np.max(il6):.1f}")

# 2. Immunocompetent static
print("2. Immunocompetent static...")
params['drug_class'] = 'static'
pd_m = BacterialPopulationODE(params)
pk_p = get_drug_pk_parameters('doxycycline')
pk_m = TwoCompartmentPKModel(CL=pk_p.CL, Vc=pk_p.Vc, Vp=pk_p.Vp, Q=pk_p.Q, Kp=pk_p.Kp)
reg = DosingRegimen(dose_mg=200, interval_hours=12, n_doses=6, infusion_duration_min=0)
res = run_simulation(pk_m, reg, pd_m, ic, t_span=(0, 72), drug_class='static')
_, b = res.get_bacterial_burden()
_, il6, _ = res.get_cytokines()
results['immunocompetent_static'] = {
    'final_burden': float(b[-1]), 'final_log10': float(np.log10(max(b[-1], 1))),
    'peak_il6': float(np.max(il6)),
}
print(f"  Final burden: {b[-1]:.2e} (log10={np.log10(max(b[-1],1)):.2f})")
print(f"  Peak IL-6: {np.max(il6):.1f}")

# 3. Neutropenic cidal
print("3. Neutropenic cidal...")
params['drug_class'] = 'cidal'
pd_m = BacterialPopulationODE(params)
pk_p = get_drug_pk_parameters('meropenem')
pk_m = TwoCompartmentPKModel(CL=pk_p.CL, Vc=pk_p.Vc, Vp=pk_p.Vp, Q=pk_p.Q, Kp=pk_p.Kp)
ic_neut = {**ic, 'N_eff': 1e5}
res = run_simulation(pk_m, reg, pd_m, ic_neut, t_span=(0, 72), drug_class='cidal')
_, b = res.get_bacterial_burden()
results['neutropenic_cidal'] = {
    'final_burden': float(b[-1]), 'final_log10': float(np.log10(max(b[-1], 1))),
}
print(f"  Final burden: {b[-1]:.2e} (log10={np.log10(max(b[-1],1)):.2f})")

# 4. Neutropenic static
print("4. Neutropenic static...")
params['drug_class'] = 'static'
pd_m = BacterialPopulationODE(params)
pk_p = get_drug_pk_parameters('doxycycline')
pk_m = TwoCompartmentPKModel(CL=pk_p.CL, Vc=pk_p.Vc, Vp=pk_p.Vp, Q=pk_p.Q, Kp=pk_p.Kp)
res = run_simulation(pk_m, reg, pd_m, ic_neut, t_span=(0, 72), drug_class='static')
_, b = res.get_bacterial_burden()
results['neutropenic_static'] = {
    'final_burden': float(b[-1]), 'final_log10': float(np.log10(max(b[-1], 1))),
}
print(f"  Final burden: {b[-1]:.2e} (log10={np.log10(max(b[-1],1)):.2f})")

# 5. Hyperinflammatory cidal
print("5. Hyperinflammatory cidal...")
params['drug_class'] = 'cidal'
pd_m = BacterialPopulationODE(params)
pk_p = get_drug_pk_parameters('meropenem')
pk_m = TwoCompartmentPKModel(CL=pk_p.CL, Vc=pk_p.Vc, Vp=pk_p.Vp, Q=pk_p.Q, Kp=pk_p.Kp)
ic_hyp = {**ic, 'N_eff': 8e7}
res = run_simulation(pk_m, reg, pd_m, ic_hyp, t_span=(0, 72), drug_class='cidal')
_, b = res.get_bacterial_burden()
results['hyperinflammatory_cidal'] = {
    'final_burden': float(b[-1]), 'final_log10': float(np.log10(max(b[-1], 1))),
}
print(f"  Final burden: {b[-1]:.2e} (log10={np.log10(max(b[-1],1)):.2f})")

# 6. Hyperinflammatory static
print("6. Hyperinflammatory static...")
params['drug_class'] = 'static'
pd_m = BacterialPopulationODE(params)
pk_p = get_drug_pk_parameters('doxycycline')
pk_m = TwoCompartmentPKModel(CL=pk_p.CL, Vc=pk_p.Vc, Vp=pk_p.Vp, Q=pk_p.Q, Kp=pk_p.Kp)
res = run_simulation(pk_m, reg, pd_m, ic_hyp, t_span=(0, 72), drug_class='static')
_, b = res.get_bacterial_burden()
results['hyperinflammatory_static'] = {
    'final_burden': float(b[-1]), 'final_log10': float(np.log10(max(b[-1], 1))),
}
print(f"  Final burden: {b[-1]:.2e} (log10={np.log10(max(b[-1],1)):.2f})")

# 7. Immunosuppressed cidal
print("7. Immunosuppressed cidal...")
params['drug_class'] = 'cidal'
pd_m = BacterialPopulationODE(params)
pk_p = get_drug_pk_parameters('meropenem')
pk_m = TwoCompartmentPKModel(CL=pk_p.CL, Vc=pk_p.Vc, Vp=pk_p.Vp, Q=pk_p.Q, Kp=pk_p.Kp)
ic_imm = {**ic, 'N_eff': 1e6}
res = run_simulation(pk_m, reg, pd_m, ic_imm, t_span=(0, 72), drug_class='cidal')
_, b = res.get_bacterial_burden()
results['immunosuppressed_cidal'] = {
    'final_burden': float(b[-1]), 'final_log10': float(np.log10(max(b[-1], 1))),
}
print(f"  Final burden: {b[-1]:.2e} (log10={np.log10(max(b[-1],1)):.2f})")

# 8. Immunosuppressed static
print("8. Immunosuppressed static...")
params['drug_class'] = 'static'
pd_m = BacterialPopulationODE(params)
pk_p = get_drug_pk_parameters('doxycycline')
pk_m = TwoCompartmentPKModel(CL=pk_p.CL, Vc=pk_p.Vc, Vp=pk_p.Vp, Q=pk_p.Q, Kp=pk_p.Kp)
res = run_simulation(pk_m, reg, pd_m, ic_imm, t_span=(0, 72), drug_class='static')
_, b = res.get_bacterial_burden()
results['immunosuppressed_static'] = {
    'final_burden': float(b[-1]), 'final_log10': float(np.log10(max(b[-1], 1))),
}
print(f"  Final burden: {b[-1]:.2e} (log10={np.log10(max(b[-1],1)):.2f})")

# Save
with open('results/trial_data/immune_phenotype_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to results/trial_data/immune_phenotype_results.json")

# Print summary table
print("\n=== Immune Phenotype Summary ===")
print(f"{'Phenotype':<25} {'Cidal log10':>12} {'Static log10':>13}")
print("-" * 50)
for name in ['immunocompetent', 'neutropenic', 'hyperinflammatory', 'immunosuppressed']:
    c_log = results[f'{name}_cidal']['final_log10']
    s_log = results[f'{name}_static']['final_log10']
    print(f"{name:<25} {c_log:>12.2f} {s_log:>13.2f}")
