"""
Anukaran AI - Experimental Data & Reactor Configurations
=========================================================
Methane decomposition: CH4 → C + 2H2

Temperature Focus: 800°C
Target H2 at outlet: 30%
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

# ============================================================================
# EXPERIMENTAL DATA
# ============================================================================

TOS_MINUTES = np.array([0, 30, 60, 90, 120, 150, 180, 210])

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
# LAB SCALE CONFIGURATION (GIVEN VALUES)
# ============================================================================

@dataclass
class LabScaleConfig:
    """Lab scale reactor - GIVEN values from experiments"""
    # Given values
    flow_rate_mL_min: float = 75.0           # mL/min
    inlet_CH4_percent: float = 100.0          # %
    reactor_diameter_cm: float = 1.8          # cm (18mm)
    bed_height_cm: float = 2.59               # cm
    catalyst_mass_g: float = 2.2              # g
    particle_density_kg_m3: float = 2821.56   # kg/m³
    particle_size_um: float = 10.0            # μm
    temperature_C: float = 800.0              # °C
    
    # Assumed values
    bed_porosity: float = 0.4                 # Assumed - typical packed bed
    particle_porosity: float = 0.5            # Assumed - porous catalyst
    tortuosity: float = 3.0                   # Assumed - typical
    inlet_pressure_bar: float = 1.0           # Assumed - atmospheric
    
    def get_assumptions(self) -> Dict[str, str]:
        """Return dictionary of assumed values for display"""
        return {
            "Bed porosity (ε)": f"{self.bed_porosity} (typical packed bed)",
            "Particle porosity (εp)": f"{self.particle_porosity} (porous catalyst)",
            "Tortuosity (τ)": f"{self.tortuosity} (typical value)",
            "Inlet pressure": f"{self.inlet_pressure_bar} bar (atmospheric)",
        }


# ============================================================================
# INDUSTRIAL SCALE CONFIGURATION (GIVEN + TO OPTIMIZE)
# ============================================================================

@dataclass
class IndustrialScaleConfig:
    """Industrial scale reactor - GIVEN values + optimization targets"""
    # Given values
    flow_rate_LPM: float = 1960.0             # LPM (liters per minute)
    inlet_CH4_percent: float = 100.0          # %
    catalyst_mass_kg: float = 57.0            # kg
    particle_density_kg_m3: float = 2821.56   # kg/m³
    particle_size_um: float = 10.0            # μm
    temperature_C: float = 800.0              # °C
    
    # Optimization constraints (given)
    LD_ratio_min: float = 1.0                 # L/D minimum
    LD_ratio_max: float = 3.0                 # L/D maximum
    target_H2_percent: float = 30.0           # Target H2 at outlet
    
    # Assumed values (same as lab for consistency)
    bed_porosity: float = 0.4
    particle_porosity: float = 0.5
    tortuosity: float = 3.0
    inlet_pressure_bar: float = 1.0
    max_pressure_drop_kPa: float = 50.0       # Assumed industrial limit
    
    def get_assumptions(self) -> Dict[str, str]:
        """Return dictionary of assumed values for display"""
        return {
            "Bed porosity (ε)": f"{self.bed_porosity} (same as lab)",
            "Particle porosity (εp)": f"{self.particle_porosity} (same as lab)",
            "Tortuosity (τ)": f"{self.tortuosity} (same as lab)",
            "Inlet pressure": f"{self.inlet_pressure_bar} bar (atmospheric)",
            "Max pressure drop": f"{self.max_pressure_drop_kPa} kPa (industrial limit)",
        }
    
    def calculate_bed_volume_m3(self) -> float:
        """Calculate required bed volume from catalyst mass"""
        # V_bed = m_cat / (ρ_cat * (1 - ε))
        return self.catalyst_mass_kg / (self.particle_density_kg_m3 * (1 - self.bed_porosity))
    
    def calculate_geometry(self, LD_ratio: float) -> Dict[str, float]:
        """Calculate D and L for given L/D ratio"""
        V_bed = self.calculate_bed_volume_m3()
        # V = π/4 * D² * L, and L = LD_ratio * D
        # V = π/4 * D² * LD_ratio * D = π/4 * LD_ratio * D³
        # D = (4V / (π * LD_ratio))^(1/3)
        D_m = (4 * V_bed / (np.pi * LD_ratio)) ** (1/3)
        L_m = LD_ratio * D_m
        return {
            "diameter_m": D_m,
            "diameter_cm": D_m * 100,
            "height_m": L_m,
            "height_cm": L_m * 100,
            "LD_ratio": LD_ratio,
            "volume_m3": V_bed,
            "volume_L": V_bed * 1000,
        }


# Default configurations
LAB_CONFIG = LabScaleConfig()
INDUSTRIAL_CONFIG = IndustrialScaleConfig()


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


def get_scale_comparison() -> Dict:
    """Get side-by-side comparison of lab vs industrial scale"""
    lab = LAB_CONFIG
    ind = INDUSTRIAL_CONFIG
    
    # Calculate scale factors
    flow_scale = (ind.flow_rate_LPM * 1000) / lab.flow_rate_mL_min
    mass_scale = (ind.catalyst_mass_kg * 1000) / lab.catalyst_mass_g
    
    return {
        "flow_scale_factor": flow_scale,
        "mass_scale_factor": mass_scale,
        "lab": lab,
        "industrial": ind,
    }
