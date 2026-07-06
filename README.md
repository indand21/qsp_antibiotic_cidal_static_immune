# QSP Antibiotics: Bactericidal vs Bacteriostatic under Host Immunity

> A mechanistic quantitative systems pharmacology (QSP) framework linking antibiotic
> pharmacokinetics, bacterial subpopulation dynamics, host immune effectors, and
> inflammatory signalling, to test when the bactericidal/bacteriostatic distinction
> is consequential across immune phenotypes (immunocompetent, neutropenic,
> immunosuppressed, hyperinflammatory).

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Run a single simulation
python cli.py simulate --drug meropenem --dose 1000 --interval 8 --drug-class cidal

# Run a virtual patient cohort
python cli.py cohort --n-patients 50 --drug meropenem

# Global sensitivity analysis (Saltelli/Sobol)
python cli.py sensitivity --samples 1024

# Literature plausibility benchmarking
python cli.py validate

# Curve-level external validation against digitized published data
python scripts/run_external_validation.py

# Run the test suite (339 tests)
python -m pytest tests/ -v
```

## Model overview

- **Pharmacokinetics**: two-compartment model with linear elimination,
  intercompartmental distribution, oral absorption, and an effect-site concentration.
- **Pharmacodynamics** (eight-state ODE): replicating bacteria, persisters,
  small-colony variants, immune effectors, cidal damage, IL-6, TNF, and
  pathogen-associated molecular patterns (PAMPs).
- **Drug mechanisms**: bacteriostatic Hill-type growth inhibition vs bactericidal
  direct killing plus damage accumulation, with lysis-driven cytokine amplification.
- **Host immunity**: burden-dependent effector recruitment and mass-action clearance,
  with clinically recognizable immune phenotypes encoded via the effector level.

## Project structure

```
├── src/                 # Core source code
│   ├── core/            # PK/PD model (pk_models, pd_model, simulation, parameters)
│   ├── analysis/        # Sensitivity, external validation, dosing optimization
│   ├── therapy/         # Combination, sequential, resistance evolution
│   ├── ml/              # Metamodels and in-silico trials
│   ├── calibration/     # Parameter calibration
│   └── utils/           # Config, checkpoint, parallel, cohort helpers
├── tests/               # Test suite (339 tests)
├── config/              # Drug library and model-parameter YAML/JSON
├── data/                # External-validation datasets (digitized CSVs + registry)
├── scripts/             # Analysis, simulation, and figure-generation scripts
├── notebooks/           # Core-model demonstration notebook
├── cli.py               # Command-line entry point
├── pyproject.toml       # Project configuration
└── requirements.txt     # Dependencies
```

## External-validation data

`data/external_validation/` holds **digitized numeric CSVs** (curve points extracted
from published figures) together with `registry.json`, which records the provenance,
citation, digitization notes, and metric space of each dataset. The original publisher
PDFs and figure images are **not redistributed** here for copyright reasons; use the
citations in `registry.json` to obtain the source articles.

## Reproducibility

Implemented in Python (NumPy, SciPy, pandas, matplotlib, SALib); ODE integration uses
SciPy `solve_ivp` with adaptive Runge-Kutta. Core dependencies and versions are pinned
in `requirements.txt` / `pyproject.toml`. Run `python -m pytest tests/ -v` to verify.

## Note on the manuscript

The associated manuscript and journal-submission materials are maintained separately
and are intentionally **not** part of this repository, which contains only the model
code, tests, configuration, and digitized validation data.

## License

MIT. See [LICENSE](LICENSE).
