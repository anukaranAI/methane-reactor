"""
Anukaran AI - Industrial Scale-Up Module
========================================
Scale-up calculations from lab to industrial reactor.

Temperature: 800°C
Target H2: 30%
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from .experimental_data import LAB_CONFIG, INDUSTRIAL_CONFIG, LabScaleConfig, IndustrialScaleConfig
from .transient_model import ReactorConfig, SteadyStateReactor, TransientReactor, DeactivationModel, DeactivationParams


@dataclass
class ScaleUpResult:
    """Results from scale-up optimization"""
    optimal_diameter_m: float
    optimal_height_m: float
    optimal_LD_ratio: float
    predicted_H2_percent: float
    predicted_conversion: float
    pressure_drop_kPa: float
    meets_target: bool
    meets_pressure_constraint: bool
    
    def to_dict(self) -> Dict:
        return {
            'Diameter (cm)': self.optimal_diameter_m * 100,
            'Height (cm)': self.optimal_height_m * 100,
            'L/D Ratio': self.optimal_LD_ratio,
            'H2 at Outlet (%)': self.predicted_H2_percent,
            'CH4 Conversion (%)': self.predicted_conversion,
            'Pressure Drop (kPa)': self.pressure_drop_kPa,
            'Meets H2 Target': '✅' if self.meets_target else '❌',
            'Pressure OK': '✅' if self.meets_pressure_constraint else '❌',
        }


class IndustrialScaleUp:
    """
    Industrial reactor scale-up optimizer.
    
    Uses calibrated kinetics from lab data to optimize
    industrial reactor geometry.
    """
    
    def __init__(self, 
                 lab_config: LabScaleConfig = None,
                 industrial_config: IndustrialScaleConfig = None,
                 calibrated_A: float = 5.0e4,
                 calibrated_Ea: float = 150000.0,
                 calibrated_kd: float = 0.008):
        
        self.lab_config = lab_config or LAB_CONFIG
        self.ind_config = industrial_config or INDUSTRIAL_CONFIG
        self.A = calibrated_A
        self.Ea = calibrated_Ea
        self.kd = calibrated_kd
    
    def calculate_bed_volume(self) -> float:
        """Calculate required bed volume [m³]"""
        m_cat = self.ind_config.catalyst_mass_kg
        rho_cat = self.ind_config.particle_density_kg_m3
        eps = self.ind_config.bed_porosity
        return m_cat / (rho_cat * (1 - eps))
    
    def geometry_from_LD(self, LD_ratio: float) -> Tuple[float, float]:
        """Calculate D and L from L/D ratio"""
        V = self.calculate_bed_volume()
        D = (4 * V / (np.pi * LD_ratio)) ** (1/3)
        L = LD_ratio * D
        return D, L
    
    def evaluate_geometry(self, LD_ratio: float, activity: float = 1.0) -> Dict:
        """Evaluate reactor performance for given L/D ratio"""
        D, L = self.geometry_from_LD(LD_ratio)
        
        config = ReactorConfig.from_industrial_config(
            self.ind_config, D, L, self.A, self.Ea
        )
        
        reactor = SteadyStateReactor(config)
        results = reactor.solve(activity=activity)
        
        return {
            'LD_ratio': LD_ratio,
            'diameter_m': D,
            'diameter_cm': D * 100,
            'height_m': L,
            'height_cm': L * 100,
            'H2_percent': results['outlet_H2_percent'],
            'conversion_percent': results['outlet_conversion'],
            'pressure_drop_kPa': results['pressure_drop_kPa'],
            'meets_H2_target': results['outlet_H2_percent'] >= self.ind_config.target_H2_percent,
            'meets_pressure': results['pressure_drop_kPa'] <= self.ind_config.max_pressure_drop_kPa,
        }
    
    def optimize_geometry(self, n_points: int = 21) -> ScaleUpResult:
        """
        Find optimal L/D ratio that:
        1. Achieves target H2 (30%)
        2. Minimizes pressure drop
        3. Stays within L/D = 1-3
        """
        LD_range = np.linspace(
            self.ind_config.LD_ratio_min,
            self.ind_config.LD_ratio_max,
            n_points
        )
        
        results = []
        for LD in LD_range:
            try:
                result = self.evaluate_geometry(LD)
                results.append(result)
            except Exception as e:
                continue
        
        if not results:
            raise ValueError("No valid geometries found")
        
        # Find configurations that meet H2 target
        valid_configs = [r for r in results if r['meets_H2_target'] and r['meets_pressure']]
        
        if valid_configs:
            # Among valid configs, pick one with lowest pressure drop
            best = min(valid_configs, key=lambda x: x['pressure_drop_kPa'])
        else:
            # If none meet target, pick closest to target H2
            best = min(results, key=lambda x: abs(x['H2_percent'] - self.ind_config.target_H2_percent))
        
        return ScaleUpResult(
            optimal_diameter_m=best['diameter_m'],
            optimal_height_m=best['height_m'],
            optimal_LD_ratio=best['LD_ratio'],
            predicted_H2_percent=best['H2_percent'],
            predicted_conversion=best['conversion_percent'],
            pressure_drop_kPa=best['pressure_drop_kPa'],
            meets_target=best['meets_H2_target'],
            meets_pressure_constraint=best['meets_pressure'],
        )
    
    def parametric_study(self, n_points: int = 11) -> List[Dict]:
        """Run parametric study across L/D range"""
        LD_range = np.linspace(
            self.ind_config.LD_ratio_min,
            self.ind_config.LD_ratio_max,
            n_points
        )
        
        results = []
        for LD in LD_range:
            try:
                result = self.evaluate_geometry(LD)
                results.append(result)
            except:
                continue
        
        return results
    
    def transient_industrial(self, LD_ratio: float, t_final_min: float = 210.0) -> Dict:
        """Run transient simulation for industrial reactor"""
        D, L = self.geometry_from_LD(LD_ratio)
        
        config = ReactorConfig.from_industrial_config(
            self.ind_config, D, L, self.A, self.Ea
        )
        
        deact = DeactivationModel('first_order', DeactivationParams(k_d=self.kd))
        reactor = TransientReactor(config, deact)
        
        return reactor.solve(t_final_min=t_final_min)
    
    def compare_scales(self, industrial_LD: float = 2.0) -> Dict:
        """Compare lab and industrial performance"""
        # Lab
        lab_reactor_config = ReactorConfig.from_lab_config(self.lab_config, self.A, self.Ea)
        lab_reactor = SteadyStateReactor(lab_reactor_config)
        lab_results = lab_reactor.solve()
        
        # Industrial
        D, L = self.geometry_from_LD(industrial_LD)
        ind_reactor_config = ReactorConfig.from_industrial_config(
            self.ind_config, D, L, self.A, self.Ea
        )
        ind_reactor = SteadyStateReactor(ind_reactor_config)
        ind_results = ind_reactor.solve()
        
        return {
            'lab': {
                'diameter_cm': self.lab_config.reactor_diameter_cm,
                'height_cm': self.lab_config.bed_height_cm,
                'flow_rate': f"{self.lab_config.flow_rate_mL_min} mL/min",
                'catalyst_mass': f"{self.lab_config.catalyst_mass_g} g",
                'H2_percent': lab_results['outlet_H2_percent'],
                'conversion': lab_results['outlet_conversion'],
                'pressure_drop_kPa': lab_results['pressure_drop_kPa'],
            },
            'industrial': {
                'diameter_cm': D * 100,
                'height_cm': L * 100,
                'LD_ratio': industrial_LD,
                'flow_rate': f"{self.ind_config.flow_rate_LPM} LPM",
                'catalyst_mass': f"{self.ind_config.catalyst_mass_kg} kg",
                'H2_percent': ind_results['outlet_H2_percent'],
                'conversion': ind_results['outlet_conversion'],
                'pressure_drop_kPa': ind_results['pressure_drop_kPa'],
            },
            'scale_factors': {
                'flow': self.ind_config.flow_rate_LPM * 1000 / self.lab_config.flow_rate_mL_min,
                'catalyst': self.ind_config.catalyst_mass_kg * 1000 / self.lab_config.catalyst_mass_g,
            }
        }


def get_assumptions_text() -> str:
    """Get formatted text of all assumptions for display"""
    text = """
### 📋 Model Assumptions

#### Given Parameters (Lab Scale @ 800°C):
| Parameter | Value | Source |
|-----------|-------|--------|
| Flow rate | 75 mL/min | Given |
| Inlet CH4 | 100% | Given |
| Reactor diameter | 18 mm | Given |
| Bed height | 2.59 cm | Given |
| Catalyst mass | 2.2 g | Given |
| Particle density | 2821.56 kg/m³ | Given |
| Particle size | 10 μm | Given |

#### Given Parameters (Industrial Scale @ 800°C):
| Parameter | Value | Source |
|-----------|-------|--------|
| Flow rate | 1960 LPM | Given |
| Inlet CH4 | 100% | Given |
| Catalyst mass | 57 kg | Given |
| Particle density | 2821.56 kg/m³ | Given |
| Particle size | 10 μm | Given |
| L/D ratio range | 1-3 | Given |
| **Target H2 outlet** | **30%** | **Given** |

#### Assumed Parameters (Same for both scales):
| Parameter | Value | Justification |
|-----------|-------|---------------|
| Bed porosity (ε) | 0.4 | Typical for randomly packed spheres |
| Particle porosity (εp) | 0.5 | Typical for porous catalysts |
| Tortuosity (τ) | 3.0 | Typical for porous media |
| Inlet pressure | 1 bar | Atmospheric operation |
| Max pressure drop | 50 kPa | Industrial design limit |

#### Kinetic Parameters (Calibrated from lab data):
| Parameter | Value | Method |
|-----------|-------|--------|
| Pre-exponential (A) | Calibrated | Fitted to match lab H2% at TOS=0 |
| Activation energy (Ea) | Calibrated | Fitted to match lab H2% at TOS=0 |
| Deactivation rate (kd) | Calibrated | Fitted to match lab TOS curve |

#### Scale-Up Assumptions:
- Same kinetics at both scales (A, Ea, kd)
- Same particle properties
- Plug flow model
- Isothermal operation at 800°C
- First-order reaction in CH4
- First-order catalyst deactivation
"""
    return text
