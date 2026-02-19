"""
KinetiScale - Industrial Scale-Up Module
========================================
Generic scale-up calculations from lab to industrial reactor.
Works with any single catalytic reaction.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from .chemistry import Reaction
from .transient_model import ReactorConfig, SteadyStateReactor, TransientReactor, DeactivationModel, DeactivationParams


@dataclass
class ScaleUpResult:
    """Results from scale-up optimization"""
    optimal_diameter_m: float
    optimal_height_m: float
    optimal_LD_ratio: float
    predicted_product_percent: float
    predicted_conversion: float
    pressure_drop_kPa: float
    meets_target: bool
    meets_pressure_constraint: bool
    product_name: str = ""
    reactant_name: str = ""

    def to_dict(self) -> Dict:
        prod = self.product_name or "Product"
        react = self.reactant_name or "Reactant"
        return {
            'Diameter (cm)': self.optimal_diameter_m * 100,
            'Height (cm)': self.optimal_height_m * 100,
            'L/D Ratio': self.optimal_LD_ratio,
            f'{prod} at Outlet (%)': self.predicted_product_percent,
            f'{react} Conversion (%)': self.predicted_conversion,
            'Pressure Drop (kPa)': self.pressure_drop_kPa,
            f'Meets {prod} Target': '\u2705' if self.meets_target else '\u274c',
            'Pressure OK': '\u2705' if self.meets_pressure_constraint else '\u274c',
        }


class IndustrialScaleUp:
    """
    Generic industrial reactor scale-up optimizer.

    Uses calibrated kinetics from lab data to optimize
    industrial reactor geometry for any reaction.
    """

    def __init__(self,
                 reaction: Reaction,
                 # Lab scale params
                 lab_flow_rate_mL_min: float = 75.0,
                 lab_diameter_cm: float = 1.8,
                 lab_height_cm: float = 2.59,
                 lab_catalyst_mass_g: float = 2.2,
                 lab_particle_size_um: float = 10.0,
                 lab_particle_density: float = 2821.56,
                 # Industrial scale params
                 ind_flow_rate_LPM: float = 1960.0,
                 ind_catalyst_mass_kg: float = 57.0,
                 ind_particle_size_um: float = 3000.0,
                 ind_particle_density: float = 2821.56,
                 ind_LD_min: float = 1.0,
                 ind_LD_max: float = 3.0,
                 ind_target_product_pct: float = 30.0,
                 ind_max_dP_kPa: float = 50.0,
                 # Common params
                 temperature_K: float = 1073.15,
                 inlet_pressure_Pa: float = 101325.0,
                 inlet_mole_fractions: Dict[str, float] = None,
                 bed_porosity: float = 0.4,
                 particle_porosity: float = 0.5,
                 tortuosity: float = 3.0,
                 # Calibrated kinetics
                 calibrated_A: float = 5.0e4,
                 calibrated_Ea: float = 150000.0,
                 calibrated_kd: float = 0.008):

        self.reaction = reaction
        self.lab_flow_mL_min = lab_flow_rate_mL_min
        self.lab_diameter_cm = lab_diameter_cm
        self.lab_height_cm = lab_height_cm
        self.lab_catalyst_mass_g = lab_catalyst_mass_g
        self.lab_particle_size_um = lab_particle_size_um
        self.lab_particle_density = lab_particle_density

        self.ind_flow_LPM = ind_flow_rate_LPM
        self.ind_catalyst_mass_kg = ind_catalyst_mass_kg
        self.ind_particle_size_um = ind_particle_size_um
        self.ind_particle_density = ind_particle_density
        self.ind_LD_min = ind_LD_min
        self.ind_LD_max = ind_LD_max
        self.ind_target_product_pct = ind_target_product_pct
        self.ind_max_dP_kPa = ind_max_dP_kPa

        self.temperature_K = temperature_K
        self.inlet_pressure_Pa = inlet_pressure_Pa
        self.inlet_mole_fractions = inlet_mole_fractions or {}
        self.bed_porosity = bed_porosity
        self.particle_porosity = particle_porosity
        self.tortuosity = tortuosity

        self.A = calibrated_A
        self.Ea = calibrated_Ea
        self.kd = calibrated_kd

    def calculate_bed_volume(self) -> float:
        """Calculate required bed volume [m³]"""
        return self.ind_catalyst_mass_kg / (self.ind_particle_density * (1 - self.bed_porosity))

    def geometry_from_LD(self, LD_ratio: float) -> Tuple[float, float]:
        """Calculate D and L from L/D ratio"""
        V = self.calculate_bed_volume()
        D = (4 * V / (np.pi * LD_ratio)) ** (1/3)
        L = LD_ratio * D
        return D, L

    def _make_industrial_config(self, D: float, L: float) -> ReactorConfig:
        """Create ReactorConfig for industrial scale with given geometry."""
        return ReactorConfig(
            diameter_m=D,
            bed_height_m=L,
            particle_diameter_m=self.ind_particle_size_um * 1e-6,
            particle_density=self.ind_particle_density,
            particle_porosity=self.particle_porosity,
            tortuosity=self.tortuosity,
            bed_porosity=self.bed_porosity,
            catalyst_mass_kg=self.ind_catalyst_mass_kg,
            temperature_K=self.temperature_K,
            inlet_pressure_Pa=self.inlet_pressure_Pa,
            flow_rate_m3_s=self.ind_flow_LPM / 60 / 1000,
            reaction=self.reaction,
            inlet_mole_fractions=self.inlet_mole_fractions,
            pre_exponential=self.A,
            activation_energy=self.Ea,
        )

    def _make_lab_config(self) -> ReactorConfig:
        """Create ReactorConfig for lab scale."""
        return ReactorConfig(
            diameter_m=self.lab_diameter_cm / 100,
            bed_height_m=self.lab_height_cm / 100,
            particle_diameter_m=self.lab_particle_size_um * 1e-6,
            particle_density=self.lab_particle_density,
            particle_porosity=self.particle_porosity,
            tortuosity=self.tortuosity,
            bed_porosity=self.bed_porosity,
            catalyst_mass_kg=self.lab_catalyst_mass_g / 1000,
            temperature_K=self.temperature_K,
            inlet_pressure_Pa=self.inlet_pressure_Pa,
            flow_rate_m3_s=self.lab_flow_mL_min / 60 / 1e6,
            reaction=self.reaction,
            inlet_mole_fractions=self.inlet_mole_fractions,
            pre_exponential=self.A,
            activation_energy=self.Ea,
        )

    def evaluate_geometry(self, LD_ratio: float, activity: float = 1.0) -> Dict:
        """Evaluate reactor performance for given L/D ratio"""
        D, L = self.geometry_from_LD(LD_ratio)
        config = self._make_industrial_config(D, L)

        reactor = SteadyStateReactor(config)
        results = reactor.solve(activity=activity)

        return {
            'LD_ratio': LD_ratio,
            'diameter_m': D,
            'diameter_cm': D * 100,
            'height_m': L,
            'height_cm': L * 100,
            'product_percent': results['outlet_product_percent'],
            'conversion_percent': results['outlet_conversion'],
            'pressure_drop_kPa': results['pressure_drop_kPa'],
            'meets_target': results['outlet_product_percent'] >= self.ind_target_product_pct,
            'meets_pressure': results['pressure_drop_kPa'] <= self.ind_max_dP_kPa,
        }

    def optimize_geometry(self, n_points: int = 21) -> ScaleUpResult:
        """
        Find optimal L/D ratio that:
        1. Achieves target product %
        2. Minimizes pressure drop
        3. Stays within L/D constraints
        """
        LD_range = np.linspace(self.ind_LD_min, self.ind_LD_max, n_points)

        results = []
        for LD in LD_range:
            try:
                result = self.evaluate_geometry(LD)
                results.append(result)
            except Exception:
                continue

        if not results:
            raise ValueError("No valid geometries found")

        valid_configs = [r for r in results if r['meets_target'] and r['meets_pressure']]

        if valid_configs:
            best = min(valid_configs, key=lambda x: x['pressure_drop_kPa'])
        else:
            best = min(results, key=lambda x: abs(x['product_percent'] - self.ind_target_product_pct))

        return ScaleUpResult(
            optimal_diameter_m=best['diameter_m'],
            optimal_height_m=best['height_m'],
            optimal_LD_ratio=best['LD_ratio'],
            predicted_product_percent=best['product_percent'],
            predicted_conversion=best['conversion_percent'],
            pressure_drop_kPa=best['pressure_drop_kPa'],
            meets_target=best['meets_target'],
            meets_pressure_constraint=best['meets_pressure'],
            product_name=self.reaction.key_product,
            reactant_name=self.reaction.key_reactant,
        )

    def parametric_study(self, n_points: int = 11) -> List[Dict]:
        """Run parametric study across L/D range"""
        LD_range = np.linspace(self.ind_LD_min, self.ind_LD_max, n_points)

        results = []
        for LD in LD_range:
            try:
                result = self.evaluate_geometry(LD)
                results.append(result)
            except Exception:
                continue

        return results

    def transient_industrial(self, LD_ratio: float, t_final_min: float = 210.0) -> Dict:
        """Run transient simulation for industrial reactor"""
        D, L = self.geometry_from_LD(LD_ratio)
        config = self._make_industrial_config(D, L)

        deact = DeactivationModel('first_order', DeactivationParams(k_d=self.kd))
        reactor = TransientReactor(config, deact)

        return reactor.solve(t_final_min=t_final_min)

    def compare_scales(self, industrial_LD: float = 2.0) -> Dict:
        """Compare lab and industrial performance"""
        lab_config = self._make_lab_config()
        lab_reactor = SteadyStateReactor(lab_config)
        lab_results = lab_reactor.solve()

        D, L = self.geometry_from_LD(industrial_LD)
        ind_config = self._make_industrial_config(D, L)
        ind_reactor = SteadyStateReactor(ind_config)
        ind_results = ind_reactor.solve()

        prod = self.reaction.key_product
        react = self.reaction.key_reactant

        return {
            'lab': {
                'diameter_cm': self.lab_diameter_cm,
                'height_cm': self.lab_height_cm,
                'flow_rate': f"{self.lab_flow_mL_min} mL/min",
                'catalyst_mass': f"{self.lab_catalyst_mass_g} g",
                'product_percent': lab_results['outlet_product_percent'],
                'conversion': lab_results['outlet_conversion'],
                'pressure_drop_kPa': lab_results['pressure_drop_kPa'],
            },
            'industrial': {
                'diameter_cm': D * 100,
                'height_cm': L * 100,
                'LD_ratio': industrial_LD,
                'flow_rate': f"{self.ind_flow_LPM} LPM",
                'catalyst_mass': f"{self.ind_catalyst_mass_kg} kg",
                'product_percent': ind_results['outlet_product_percent'],
                'conversion': ind_results['outlet_conversion'],
                'pressure_drop_kPa': ind_results['pressure_drop_kPa'],
            },
            'scale_factors': {
                'flow': self.ind_flow_LPM * 1000 / self.lab_flow_mL_min,
                'catalyst': self.ind_catalyst_mass_kg * 1000 / self.lab_catalyst_mass_g,
            },
            'product_name': prod,
            'reactant_name': react,
        }


def get_assumptions_text(reaction: Reaction = None,
                         temperature_C: float = 800.0,
                         lab_flow: float = 75.0,
                         ind_flow: float = 1960.0) -> str:
    """Get formatted text of all assumptions for display"""
    rxn_name = reaction.display_name if reaction else "Catalytic Reaction"
    rxn_eq = reaction.format_equation() if reaction else "A \u2192 Products"
    key_react = reaction.species[reaction.key_reactant].formula if reaction else "Reactant"
    key_prod = reaction.species[reaction.key_product].formula if reaction else "Product"

    text = f"""
### Model Assumptions

**Reaction:** {rxn_eq}

#### Scale-Up Assumptions:
- Same kinetics at both scales (A, Ea, kd)
- Same catalyst particle properties
- Plug flow model (1D axial)
- Isothermal operation at {temperature_C:.0f}\u00b0C
- First-order reaction in {key_react}
- First-order catalyst deactivation

#### Assumed Parameters (Same for both scales):
| Parameter | Value | Justification |
|-----------|-------|---------------|
| Bed porosity (\u03b5) | 0.4 | Typical for randomly packed spheres |
| Particle porosity (\u03b5p) | 0.5 | Typical for porous catalysts |
| Tortuosity (\u03c4) | 3.0 | Typical for porous media |
| Inlet pressure | 1 bar | Atmospheric operation |
| Max pressure drop | 50 kPa | Industrial design limit |
"""
    return text
