"""
Create visual comparison of original vs fixed results
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
print('Loading data...')
original = pd.read_csv('in_silico_trial_results_ORIGINAL.csv')
fixed = pd.read_csv('in_silico_trial_results.csv')

# Create comprehensive comparison figure
fig = plt.figure(figsize=(16, 10))

# Overall success rates
ax1 = plt.subplot(2, 3, 1)
treatments = ['Doxycycline\\n(Static)', 'Meropenem\\n(Cidal)']
orig_success = [
    original[original['treatment'] == 'Doxycycline (Static)']['clinical_success'].mean(),
    original[original['treatment'] == 'Meropenem (Cidal)']['clinical_success'].mean()
]
fixed_success = [
    fixed[fixed['treatment'] == 'Doxycycline (Static)']['clinical_success'].mean(),
    fixed[fixed['treatment'] == 'Meropenem (Cidal)']['clinical_success'].mean()
]

x = np.arange(len(treatments))
width = 0.35
bars1 = ax1.bar(x - width/2, orig_success, width, label='Original (Broken)',
               color='lightcoral', alpha=0.8, edgecolor='black')
bars2 = ax1.bar(x + width/2, fixed_success, width, label='Fixed',
               color='mediumseagreen', alpha=0.8, edgecolor='black')

ax1.set_ylabel('Clinical Success Rate', fontsize=12, fontweight='bold')
ax1.set_title('Overall Clinical Success', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(treatments, fontsize=10)
ax1.legend(fontsize=10)
ax1.set_ylim([0, 1.1])
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{100*height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# By immune status - Immunocompetent
ax2 = plt.subplot(2, 3, 2)
immune_groups = ['Immunocompetent', 'Neutropenic', 'Hyperinflammatory', 'Immunosuppressed']
colors_immune = ['steelblue', 'orange', 'green', 'purple']

for idx, status in enumerate(['immunocompetent', 'neutropenic', 'hyperinflammatory', 'immunosuppressed']):
    # Cidal only (static unchanged)
    orig_cidal = original[(original['treatment'] == 'Meropenem (Cidal)') &
                         (original['immune_status'] == status)]['clinical_success'].mean()
    fixed_cidal = fixed[(fixed['treatment'] == 'Meropenem (Cidal)') &
                       (fixed['immune_status'] == status)]['clinical_success'].mean()

    ax2.bar([idx - 0.2], [orig_cidal], 0.35, color='lightcoral', alpha=0.6)
    ax2.bar([idx + 0.2], [fixed_cidal], 0.35, color='mediumseagreen', alpha=0.8)

ax2.set_ylabel('Meropenem Success Rate', fontsize=11, fontweight='bold')
ax2.set_title('Cidal Success by Immune Status', fontsize=13, fontweight='bold')
ax2.set_xticks(range(len(immune_groups)))
ax2.set_xticklabels(immune_groups, rotation=45, ha='right', fontsize=9)
ax2.set_ylim([0, 1.1])
ax2.grid(True, alpha=0.3, axis='y')
ax2.legend(['Original', 'Fixed'], fontsize=9)

# Time to 50% reduction
ax3 = plt.subplot(2, 3, 3)
time_orig = [
    original[original['treatment'] == 'Doxycycline (Static)']['time_to_50_percent_reduction'].median(),
    original[original['treatment'] == 'Meropenem (Cidal)']['time_to_50_percent_reduction'].median()
]
time_fixed = [
    fixed[fixed['treatment'] == 'Doxycycline (Static)']['time_to_50_percent_reduction'].median(),
    fixed[fixed['treatment'] == 'Meropenem (Cidal)']['time_to_50_percent_reduction'].median()
]

bars1 = ax3.bar(x - width/2, time_orig, width, label='Original',
               color='lightcoral', alpha=0.8, edgecolor='black')
bars2 = ax3.bar(x + width/2, time_fixed, width, label='Fixed',
               color='mediumseagreen', alpha=0.8, edgecolor='black')

ax3.set_ylabel('Time to 50% Reduction (hours)', fontsize=11, fontweight='bold')
ax3.set_title('Speed of Bacterial Clearance', fontsize=13, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(treatments, fontsize=10)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}h', ha='center', va='bottom', fontsize=9)

# Calibration quality comparison
ax4 = plt.subplot(2, 3, 4)
calib_names = ['Static\\n(Doxycycline)', 'Cidal Original\\n(Broken)', 'Cidal Fixed']
calib_rmse = [0.08, 5.41, 1.78]
colors_calib = ['steelblue', 'lightcoral', 'mediumseagreen']

bars = ax4.bar(range(len(calib_names)), calib_rmse, color=colors_calib,
              alpha=0.8, edgecolor='black')
ax4.set_ylabel('Calibration RMSE (lower=better)', fontsize=11, fontweight='bold')
ax4.set_title('Parameter Calibration Quality', fontsize=13, fontweight='bold')
ax4.set_xticks(range(len(calib_names)))
ax4.set_xticklabels(calib_names, fontsize=10)
ax4.set_ylim([0, 6])
ax4.axhline(y=2.0, color='gray', linestyle='--', linewidth=2, label='Acceptable threshold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, calib_rmse):
    ax4.text(bar.get_x() + bar.get_width()/2., val + 0.2,
            f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Peak IL-6 comparison
ax5 = plt.subplot(2, 3, 5)
il6_orig = [
    np.median(np.log10(original[original['treatment'] == 'Doxycycline (Static)']['peak_IL6'] + 1)),
    np.median(np.log10(original[original['treatment'] == 'Meropenem (Cidal)']['peak_IL6'] + 1))
]
il6_fixed = [
    np.median(np.log10(fixed[fixed['treatment'] == 'Doxycycline (Static)']['peak_IL6'] + 1)),
    np.median(np.log10(fixed[fixed['treatment'] == 'Meropenem (Cidal)']['peak_IL6'] + 1))
]

bars1 = ax5.bar(x - width/2, il6_orig, width, label='Original',
               color='lightcoral', alpha=0.8, edgecolor='black')
bars2 = ax5.bar(x + width/2, il6_fixed, width, label='Fixed',
               color='mediumseagreen', alpha=0.8, edgecolor='black')

ax5.set_ylabel('Log10(Peak IL-6) [pg/mL]', fontsize=11, fontweight='bold')
ax5.set_title('Inflammatory Response', fontsize=13, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(treatments, fontsize=10)
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3, axis='y')

# Final burden comparison
ax6 = plt.subplot(2, 3, 6)
burden_orig = [
    np.median(np.log10(original[original['treatment'] == 'Doxycycline (Static)']['final_bacterial_burden'] + 1)),
    np.median(np.log10(original[original['treatment'] == 'Meropenem (Cidal)']['final_bacterial_burden'] + 1))
]
burden_fixed = [
    np.median(np.log10(fixed[fixed['treatment'] == 'Doxycycline (Static)']['final_bacterial_burden'] + 1)),
    np.median(np.log10(fixed[fixed['treatment'] == 'Meropenem (Cidal)']['final_bacterial_burden'] + 1))
]

bars1 = ax6.bar(x - width/2, burden_orig, width, label='Original',
               color='lightcoral', alpha=0.8, edgecolor='black')
bars2 = ax6.bar(x + width/2, burden_fixed, width, label='Fixed',
               color='mediumseagreen', alpha=0.8, edgecolor='black')

ax6.set_ylabel('Log10(Final Burden) [CFU/mL]', fontsize=11, fontweight='bold')
ax6.set_title('Final Bacterial Burden', fontsize=13, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(treatments, fontsize=10)
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3, axis='y')

plt.suptitle('CIDAL CALIBRATION FIX: Before vs After Comparison\\n'
             'Original (Broken): RMSE=5.41, 21.5% success | Fixed: RMSE=1.78, 100.0% success',
             fontsize=15, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig('COMPARISON_ORIGINAL_VS_FIXED_COMPREHENSIVE.png', dpi=150, bbox_inches='tight')
print('Saved: COMPARISON_ORIGINAL_VS_FIXED_COMPREHENSIVE.png')

# Print summary statistics
print('\\n' + '='*80)
print('COMPARISON SUMMARY: ORIGINAL vs FIXED')
print('='*80)
print('\\nClinical Success Rates:')
print(f'  Static (unchanged): {100*orig_success[0]:.1f}% -> {100*fixed_success[0]:.1f}%')
print(f'  Cidal (FIXED): {100*orig_success[1]:.1f}% -> {100*fixed_success[1]:.1f}% (+{100*(fixed_success[1]-orig_success[1]):.1f}%)')
print('\\nCalibration Quality:')
print(f'  Static: RMSE = 0.08 (excellent)')
print(f'  Cidal Original: RMSE = 5.41 (POOR)')
print(f'  Cidal Fixed: RMSE = 1.78 (GOOD, 67% improvement)')
print('\\nBy Immune Status (Meropenem):')
for status in ['immunocompetent', 'neutropenic', 'hyperinflammatory', 'immunosuppressed']:
    orig = original[(original['treatment'] == 'Meropenem (Cidal)') &
                   (original['immune_status'] == status)]['clinical_success'].mean()
    fix = fixed[(fixed['treatment'] == 'Meropenem (Cidal)') &
               (fixed['immune_status'] == status)]['clinical_success'].mean()
    print(f'  {status.capitalize()}: {100*orig:.1f}% -> {100*fix:.1f}% (+{100*(fix-orig):.1f}%)')
print('='*80)
