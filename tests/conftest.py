"""
Pytest configuration and shared fixtures for QSP test suite.
"""
import pytest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.parameters import (
    get_default_parameters,
    get_drug_pk_parameters,
    normalize_pk_parameters,
    BacterialParameters,
    ImmuneParameters,
    CytokineParameters,
    PKParameters,
)
from src.core.pd_model import create_ode_system, BacterialPopulationODE
from src.core.pk_models import TwoCompartmentPKModel, DosingRegimen
from src.core.simulation import run_simulation


@pytest.fixture
def default_params():
    """Return default model parameters."""
    return get_default_parameters()


@pytest.fixture
def pd_model(default_params):
    """Return initialized PD model."""
    return create_ode_system(default_params)


@pytest.fixture
def pk_model_meropenem():
    """Return meropenem PK model for 70kg patient."""
    pk_raw = get_drug_pk_parameters("meropenem")
    pk = normalize_pk_parameters(pk_raw, weight_kg=70)
    return TwoCompartmentPKModel(**pk, effect_site_model=True)


@pytest.fixture
def pk_model_doxycycline():
    """Return doxycycline PK model for 70kg patient."""
    pk_raw = get_drug_pk_parameters("doxycycline")
    pk = normalize_pk_parameters(pk_raw, weight_kg=70)
    return TwoCompartmentPKModel(**pk, effect_site_model=True)


@pytest.fixture
def standard_regimen():
    """Return standard dosing regimen."""
    return DosingRegimen(
        dose_mg=1000,
        interval_hours=8,
        start_time=0,
        n_doses=3,
        infusion_duration_min=30,
    )


@pytest.fixture
def standard_init_cond():
    """Return standard initial conditions."""
    return {
        "B_rep": 1e6,
        "B_pers": 1e3,
        "B_SCV": 0,
        "N_eff": 1e7,
        "Damage": 0,
        "IL6": 10,
        "TNF": 5,
        "PAMP": 0,
    }


@pytest.fixture
def short_simulation_result(pk_model_meropenem, standard_regimen, pd_model, standard_init_cond):
    """Run a short simulation for testing (24h)."""
    return run_simulation(
        pk_model=pk_model_meropenem,
        regimen=standard_regimen,
        pd_model=pd_model,
        initial_conditions=standard_init_cond,
        t_span=(0, 24),
        drug_class="cidal",
        weight_kg=70,
    )
