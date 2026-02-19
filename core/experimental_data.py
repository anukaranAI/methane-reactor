"""
KinetiScale - Experimental Data & Reactor Configurations
=========================================================
Generic reactor geometry and scale configurations.
Reaction-specific experimental data is stored in reaction_templates.py.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ============================================================================
# LAB SCALE CONFIGURATION
# ============================================================================

@dataclass
class LabScaleConfig:
    """Lab scale reactor geometry and operating conditions."""
    flow_rate_mL_min: float = 75.0
    inlet_reactant_percent: float = 100.0
    reactor_diameter_cm: float = 1.8
    bed_height_cm: float = 2.59
    catalyst_mass_g: float = 2.2
    particle_density_kg_m3: float = 2821.56
    particle_size_um: float = 10.0
    temperature_C: float = 800.0

    # Assumed values
    bed_porosity: float = 0.4
    particle_porosity: float = 0.5
    tortuosity: float = 3.0
    inlet_pressure_bar: float = 1.0

    def get_assumptions(self) -> Dict[str, str]:
        """Return dictionary of assumed values for display"""
        return {
            "Bed porosity (\u03b5)": f"{self.bed_porosity} (typical packed bed)",
            "Particle porosity (\u03b5p)": f"{self.particle_porosity} (porous catalyst)",
            "Tortuosity (\u03c4)": f"{self.tortuosity} (typical value)",
            "Inlet pressure": f"{self.inlet_pressure_bar} bar (atmospheric)",
        }


# ============================================================================
# INDUSTRIAL SCALE CONFIGURATION
# ============================================================================

@dataclass
class IndustrialScaleConfig:
    """Industrial scale reactor geometry and operating conditions."""
    flow_rate_LPM: float = 1960.0
    inlet_reactant_percent: float = 100.0
    catalyst_mass_kg: float = 57.0
    particle_density_kg_m3: float = 2821.56
    particle_size_um: float = 10.0
    temperature_C: float = 800.0

    # Optimization constraints
    LD_ratio_min: float = 1.0
    LD_ratio_max: float = 3.0
    target_product_percent: float = 30.0

    # Assumed values
    bed_porosity: float = 0.4
    particle_porosity: float = 0.5
    tortuosity: float = 3.0
    inlet_pressure_bar: float = 1.0
    max_pressure_drop_kPa: float = 50.0

    def get_assumptions(self) -> Dict[str, str]:
        """Return dictionary of assumed values for display"""
        return {
            "Bed porosity (\u03b5)": f"{self.bed_porosity} (same as lab)",
            "Particle porosity (\u03b5p)": f"{self.particle_porosity} (same as lab)",
            "Tortuosity (\u03c4)": f"{self.tortuosity} (same as lab)",
            "Inlet pressure": f"{self.inlet_pressure_bar} bar (atmospheric)",
            "Max pressure drop": f"{self.max_pressure_drop_kPa} kPa (industrial limit)",
        }

    def calculate_bed_volume_m3(self) -> float:
        """Calculate required bed volume from catalyst mass"""
        return self.catalyst_mass_kg / (self.particle_density_kg_m3 * (1 - self.bed_porosity))

    def calculate_geometry(self, LD_ratio: float) -> Dict[str, float]:
        """Calculate D and L for given L/D ratio"""
        V_bed = self.calculate_bed_volume_m3()
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
# HELPER FUNCTIONS (for experimental data from templates)
# ============================================================================

def get_experimental_data_from_template(template, temperature_C: int) -> Optional[Dict]:
    """Get experimental data for a specific temperature from a reaction template."""
    if template.experimental_data is None:
        return None
    return template.experimental_data.get(temperature_C, None)


def get_available_temperatures_from_template(template) -> List[int]:
    """Get list of temperatures with experimental data from a template."""
    if template.experimental_data is None:
        return []
    return list(template.experimental_data.keys())
