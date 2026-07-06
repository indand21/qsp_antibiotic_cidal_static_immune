"""Supplementary S5: Resistance evolution under varied scenarios."""
import sys
sys.path.insert(0, '.')

import numpy as np
from src.therapy.resistance_evolution import simulate_resistance_evolution

print("=== Supplementary S5: Resistance Evolution Results ===\n")

# Scenario 1: Varying fitness costs
print("Scenario 1: Fitness cost effect on MIC evolution")
print("-" * 60)
for fc in [0.05, 0.10, 0.20]:
    res = simulate_resistance_evolution(drug_concentration=0.75, MIC_baseline=1.0,
                                         duration_hours=168.0, dt=0.5, n_levels=4,
                                         fitness_cost=fc, mu_resistance=1e-6)
    final_mic = res['MIC_effective'][-1]
    final_frac = res['resistant_fraction'][-1]
    print(f"  Fitness cost={fc:.0%}: Final MIC = {final_mic:.2f}x, Resistant fraction = {final_frac:.3f}")

# Scenario 2: Varying mutation rates
print("\nScenario 2: Mutation rate effect")
print("-" * 60)
for mu in [1e-7, 1e-6, 1e-5]:
    res = simulate_resistance_evolution(drug_concentration=0.75, MIC_baseline=1.0,
                                         duration_hours=168.0, dt=0.5, n_levels=4,
                                         fitness_cost=0.1, mu_resistance=mu)
    final_mic = res['MIC_effective'][-1]
    print(f"  mu={mu:.0e}: Final MIC = {final_mic:.2f}x")

# Scenario 3: High initial burden
print("\nScenario 3: High initial burden (1e8 CFU/mL)")
print("-" * 60)
res = simulate_resistance_evolution(drug_concentration=0.75, MIC_baseline=1.0,
                                     initial_burden=1e8, duration_hours=168.0, dt=0.5,
                                     n_levels=4, fitness_cost=0.1, mu_resistance=1e-6)
final_mic = res['MIC_effective'][-1]
print(f"  Initial burden 1e8: Final MIC = {final_mic:.2f}x")

# Scenario 4: Different drug concentrations
print("\nScenario 4: Drug concentration effect")
print("-" * 60)
for conc in [0, 0.25, 0.5, 0.75, 1.0, 2.0]:
    res = simulate_resistance_evolution(drug_concentration=conc, MIC_baseline=1.0,
                                         duration_hours=168.0, dt=0.5, n_levels=4,
                                         fitness_cost=0.1, mu_resistance=1e-6)
    final_mic = res['MIC_effective'][-1]
    max_frac = np.max(res['resistant_fraction'])
    print(f"  C={conc}x MIC: Final MIC = {final_mic:.2f}x, Max resistant frac = {max_frac:.3f}")

print("\n=== End S5 Results ===")
