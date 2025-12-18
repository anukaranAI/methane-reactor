"""
Anukaran AI - Experimental Data Storage
=======================================
Contains experimental data for model validation.
Methane decomposition: CH4 → C + 2H2
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

# ============================================================================
# EXPERIMENTAL DATA FROM LITERATURE
# ============================================================================

# Time on Stream (TOS) in minutes
TOS_MINUTES = np.array([0, 30, 60, 90, 120, 150, 180, 210])

# Experimental data at different temperatures
EXPERIMENTAL_DATA = {
    770: {
        "TOS_min": TOS_MINUTES,
        "H2_percent": np.array([11.41, 20.28, 16.1, 14.08, 12.81, 11.82, 11.12, 10.33]),
        "CH4_percent": np.array([24.87, 79.71, 83.89, 85.91, 87.18, 88.17, 88.87, 89.66]),
    },
    800: {
        "TOS_min": TOS_MINUTES,
        "H2_percent": np.array([24.13, 24.5, 20.51, 17.8, 16.74, 15.51, 14.44, 13.43]),
        "CH4_percent": np.array([65.0, 75.49, 79.48, 82.19, 83.25, 84.48, 85.55, 86.56]),
    },
    830: {
        "TOS_min": TOS_MINUTES,
        "H2_percent": np.array([40.69, 31.68, 25.23, 23.9, 22.29, 20.36, 19.8, 18.79]),
        "CH4_percent": np.array([44.64, 68.31, 74.76, 76.09, 77.7, 79.63, 80.19, 81.2]),
    },
}

# ============================================================================
# EXPERIMENTAL CONDITIONS
# ============================================================================

@dataclass
class ExperimentalConditions:
    """Experimental setup parameters"""
    reactor_diameter_cm: float = 2.5
    bed_height_cm: float = 10.0
    particle_diameter_um: float = 500.0
    catalyst_density_kg_m3: float = 2000.0
    particle_porosity: float = 0.5
    tortuosity: float = 3.0
    bed_porosity: float = 0.4
    catalyst_mass_g: float = 20.0
    inlet_pressure_bar: float = 1.0
    flow_rate_mL_min: float = 50.0
    y_CH4_in: float = 0.5
    y_H2_in: float = 0.0
    y_N2_in: float = 0.5
    pre_exponential: float = 1.0e6
    activation_energy_kJ_mol: float = 100.0
    beta: float = 0.0
    heat_of_reaction_kJ_mol: float = 74.87


DEFAULT_CONDITIONS = ExperimentalConditions()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_experimental_data(temperature_C: int) -> Optional[Dict]:
    """Get experimental data for a specific temperature."""
    return EXPERIMENTAL_DATA.get(temperature_C, None)


def get_available_temperatures() -> List[int]:
    """Get list of temperatures with experimental data"""
    return list(EXPERIMENTAL_DATA.keys())


def get_initial_values(temperature_C: int) -> Optional[Dict]:
    """Get initial (TOS=0) experimental values."""
    data = get_experimental_data(temperature_C)
    if data is None:
        return None
    return {
        "H2_percent": data["H2_percent"][0],
        "CH4_percent": data["CH4_percent"][0],
    }


def get_all_initial_values() -> Dict[int, Dict]:
    """Get initial values for all temperatures"""
    result = {}
    for temp in get_available_temperatures():
        result[temp] = get_initial_values(temp)
    return result


def interpolate_experimental(temperature_C: int, time_min: float) -> Optional[Dict]:
    """Interpolate experimental data at a specific time."""
    data = get_experimental_data(temperature_C)
    if data is None:
        return None
    
    H2_interp = np.interp(time_min, data["TOS_min"], data["H2_percent"])
    CH4_interp = np.interp(time_min, data["TOS_min"], data["CH4_percent"])
    
    return {"H2_percent": H2_interp, "CH4_percent": CH4_interp}
