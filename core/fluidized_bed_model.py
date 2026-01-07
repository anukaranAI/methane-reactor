"""
╔══════════════════════════════════════════════════════════════════════════╗
║              FLUIDIZED BED REACTOR MODEL (CSTR)                          ║
║                                                                          ║
║   Simple CSTR model calibrated to experimental data                      ║
║   - Uses kinetics fitted from Arrhenius plot of experimental data        ║
║   - Matches H₂% at 770°C, 800°C, 830°C within 1%                        ║
║                                                                          ║
║   Reaction: CH₄ → C(s) + 2H₂                                             ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from enum import Enum

# ============================================================================
# CONSTANTS
# ============================================================================
R = 8.314  # J/(mol·K) - Universal gas constant

# Molar masses (kg/mol)
M_CH4 = 0.01604
M_H2 = 0.002016

# Standard conditions
V_MOLAR_STP = 0.0224  # m³/mol at STP


# ============================================================================
# CALIBRATED KINETICS (from Arrhenius fit of experimental data)
# ============================================================================
# These parameters were fitted to match experimental H₂% at 770°C, 800°C, 830°C
# Using CSTR model: X = k*τ / (1 + k*τ) where τ = W/F

CALIBRATED_KINETICS = {
    'A': 3.6607e10,      # Pre-exponential factor (1/s)
    'Ea': 266439.0,      # Activation energy (J/mol) = 266.4 kJ/mol
}

# Experimental data for validation
EXPERIMENTAL_DATA_FBR = {
    770: {'T_K': 1043, 'H2_percent': 11.41, 'X': 0.0605, 'r_W': 1.53e-3},
    800: {'T_K': 1073, 'H2_percent': 24.13, 'X': 0.137, 'r_W': 3.48e-3},
    830: {'T_K': 1103, 'H2_percent': 40.69, 'X': 0.255, 'r_W': 6.47e-3},
}


# ============================================================================
# REACTOR MODEL ENUM
# ============================================================================

class ReactorModelType(Enum):
    """Reactor model types"""
    CSTR = "CSTR"    # Continuous Stirred Tank Reactor (fluidized bed)
    PFR = "PFR"      # Plug Flow Reactor (packed bed)


# ============================================================================
# FLUIDIZED BED CONFIG
# ============================================================================

@dataclass
class FluidizedBedConfig:
    """Configuration for fluidized bed reactor"""
    # Required fields (no defaults) - must come first
    diameter_m: float
    bed_height_m: float
    catalyst_mass_kg: float
    particle_diameter_m: float
    temperature_C: float
    
    # Optional fields (with defaults)
    bulk_density_kg_m3: float = 334.2
    particle_density_kg_m3: float = 2800.0
    pressure_bar: float = 1.0
    flow_rate_LPM: float = None      # Volumetric flow (L/min at STP)
    flow_rate_mol_s: float = None    # Molar flow (mol/s)
    A: float = CALIBRATED_KINETICS['A']
    Ea: float = CALIBRATED_KINETICS['Ea']
    model_type: ReactorModelType = ReactorModelType.CSTR
    
    def __post_init__(self):
        """Calculate derived properties"""
        # Convert flow rates
        if self.flow_rate_LPM is not None and self.flow_rate_mol_s is None:
            self.flow_rate_mol_s = self.flow_rate_LPM / 1000 / 60 / V_MOLAR_STP
        elif self.flow_rate_mol_s is not None and self.flow_rate_LPM is None:
            self.flow_rate_LPM = self.flow_rate_mol_s * V_MOLAR_STP * 60 * 1000
    
    @property
    def temperature_K(self) -> float:
        return self.temperature_C + 273.15
    
    @property
    def pressure_Pa(self) -> float:
        return self.pressure_bar * 1e5
    
    @property
    def cross_section_area_m2(self) -> float:
        return np.pi * self.diameter_m**2 / 4
    
    @property
    def bed_volume_m3(self) -> float:
        return self.cross_section_area_m2 * self.bed_height_m
    
    @property
    def bed_porosity(self) -> float:
        return 1 - self.bulk_density_kg_m3 / self.particle_density_kg_m3
    
    @property
    def space_time_s(self) -> float:
        """Space time τ = W/F (s) - mass basis"""
        if self.flow_rate_mol_s is None or self.flow_rate_mol_s == 0:
            return float('inf')
        return self.catalyst_mass_kg / self.flow_rate_mol_s
    
    @property
    def superficial_velocity_m_s(self) -> float:
        """Superficial velocity at operating conditions"""
        if self.flow_rate_mol_s is None:
            return 0.0
        Q_operating = self.flow_rate_mol_s * R * self.temperature_K / self.pressure_Pa
        return Q_operating / self.cross_section_area_m2
    
    @property
    def rate_constant(self) -> float:
        """Rate constant k = A * exp(-Ea/RT)"""
        return self.A * np.exp(-self.Ea / (R * self.temperature_K))


# ============================================================================
# PREDEFINED CONFIGURATIONS
# ============================================================================

def get_lab_config(temperature_C: float = 800) -> FluidizedBedConfig:
    """Get lab scale configuration"""
    return FluidizedBedConfig(
        diameter_m=0.018,
        bed_height_m=0.03,
        catalyst_mass_kg=2.2e-3,
        particle_diameter_m=10e-6,
        temperature_C=temperature_C,
        pressure_bar=1.0,
        flow_rate_LPM=0.075,  # 75 mL/min
    )


def get_industrial_config_1(temperature_C: float = 800) -> FluidizedBedConfig:
    """Industrial config 1: Same superficial velocity as lab"""
    return FluidizedBedConfig(
        diameter_m=2.909845357,
        bed_height_m=0.0256471263,
        catalyst_mass_kg=57.0,
        particle_diameter_m=10e-6,
        temperature_C=temperature_C,
        pressure_bar=1.0,
        flow_rate_LPM=1960,
    )


def get_industrial_config_2(temperature_C: float = 800) -> FluidizedBedConfig:
    """Industrial config 2: Higher superficial velocity"""
    return FluidizedBedConfig(
        diameter_m=0.5,
        bed_height_m=0.8686373914,
        catalyst_mass_kg=57.0,
        particle_diameter_m=10e-6,
        temperature_C=temperature_C,
        pressure_bar=1.0,
        flow_rate_LPM=1960,
    )


def get_industrial_config_3(temperature_C: float = 800) -> FluidizedBedConfig:
    """Industrial config 3: Intermediate design"""
    return FluidizedBedConfig(
        diameter_m=1.301322404,
        bed_height_m=0.1282356315,
        catalyst_mass_kg=57.0,
        particle_diameter_m=10e-6,
        temperature_C=temperature_C,
        pressure_bar=1.0,
        flow_rate_LPM=1960,
    )


# ============================================================================
# CONVERSION UTILITIES
# ============================================================================

def conversion_to_H2_percent(X: float) -> float:
    """Convert methane conversion to H₂ mole percent in product"""
    if X >= 1.0:
        return 100.0
    if X <= 0.0:
        return 0.0
    y_H2 = 2 * X / (1 + X)
    return y_H2 * 100


def H2_percent_to_conversion(H2_pct: float) -> float:
    """Convert H₂ percent to methane conversion"""
    y_H2 = H2_pct / 100
    if y_H2 >= 1.0:
        return 1.0
    if y_H2 <= 0.0:
        return 0.0
    X = y_H2 / (2 - y_H2)
    return X


# ============================================================================
# PRESSURE DROP (ERGUN EQUATION)
# ============================================================================

def calculate_pressure_drop(config: FluidizedBedConfig) -> Dict[str, float]:
    """
    Calculate pressure drop using Ergun equation.
    
    Returns dict with pressure drop values.
    """
    # Gas properties
    T_K = config.temperature_K
    P_Pa = config.pressure_Pa
    
    # Gas viscosity (approximate for CH4 at high T)
    mu_gas = 3.5e-5  # Pa·s
    
    # Gas density
    rho_gas = P_Pa * M_CH4 / (R * T_K)
    
    epsilon = config.bed_porosity
    d_p = config.particle_diameter_m
    Us = config.superficial_velocity_m_s
    L = config.bed_height_m
    
    if d_p <= 0 or epsilon >= 1 or Us <= 0:
        return {'dP_total_Pa': 0, 'dP_total_kPa': 0, 'dP_total_bar': 0}
    
    # Ergun equation
    term_viscous = 150 * mu_gas * (1 - epsilon)**2 * Us / (d_p**2 * epsilon**3)
    term_inertial = 1.75 * rho_gas * (1 - epsilon) * Us**2 / (d_p * epsilon**3)
    
    dP_dL = term_viscous + term_inertial
    dP_total = dP_dL * L
    
    return {
        'dP_dL_Pa_m': dP_dL,
        'dP_total_Pa': dP_total,
        'dP_total_kPa': dP_total / 1000,
        'dP_total_bar': dP_total / 1e5,
        'term_viscous_Pa_m': term_viscous,
        'term_inertial_Pa_m': term_inertial,
    }


# ============================================================================
# CSTR SOLVER
# ============================================================================

def solve_cstr(config: FluidizedBedConfig) -> Dict[str, float]:
    """
    Solve CSTR model for fluidized bed reactor.
    
    For first-order reaction:
    X = k·τ / (1 + k·τ)  where τ = W/F
    
    Returns dict with results.
    """
    tau = config.space_time_s
    k = config.rate_constant
    
    # CSTR solution
    k_tau = k * tau
    X = k_tau / (1 + k_tau)
    X = np.clip(X, 0.0, 0.999)
    
    # H₂ percent
    H2_percent = conversion_to_H2_percent(X)
    
    # Reaction rate
    r_W = config.flow_rate_mol_s * X / config.catalyst_mass_kg if config.catalyst_mass_kg > 0 else 0.0
    
    # Pressure drop
    dP = calculate_pressure_drop(config)
    
    return {
        'model': 'CSTR',
        'conversion': X,
        'conversion_percent': X * 100,
        'H2_percent': H2_percent,
        'CH4_percent': 100 - H2_percent,
        'rate_constant': k,
        'space_time_s': tau,
        'k_tau': k_tau,
        'r_W_mol_kg_s': r_W,
        'pressure_drop_Pa': dP['dP_total_Pa'],
        'pressure_drop_kPa': dP['dP_total_kPa'],
        'superficial_velocity_m_s': config.superficial_velocity_m_s,
        'temperature_C': config.temperature_C,
        'temperature_K': config.temperature_K,
    }


# ============================================================================
# PFR SOLVER
# ============================================================================

def solve_pfr(config: FluidizedBedConfig) -> Dict[str, float]:
    """
    Solve PFR model for packed bed reactor.
    
    For first-order reaction:
    X = 1 - exp(-k·τ)
    
    Returns dict with results.
    """
    tau = config.space_time_s
    k = config.rate_constant
    
    # PFR solution
    k_tau = k * tau
    X = 1 - np.exp(-k_tau)
    X = np.clip(X, 0.0, 0.999)
    
    # H₂ percent
    H2_percent = conversion_to_H2_percent(X)
    
    # Reaction rate
    r_W = config.flow_rate_mol_s * X / config.catalyst_mass_kg if config.catalyst_mass_kg > 0 else 0.0
    
    # Pressure drop
    dP = calculate_pressure_drop(config)
    
    return {
        'model': 'PFR',
        'conversion': X,
        'conversion_percent': X * 100,
        'H2_percent': H2_percent,
        'CH4_percent': 100 - H2_percent,
        'rate_constant': k,
        'space_time_s': tau,
        'k_tau': k_tau,
        'r_W_mol_kg_s': r_W,
        'pressure_drop_Pa': dP['dP_total_Pa'],
        'pressure_drop_kPa': dP['dP_total_kPa'],
        'superficial_velocity_m_s': config.superficial_velocity_m_s,
        'temperature_C': config.temperature_C,
        'temperature_K': config.temperature_K,
    }


# ============================================================================
# UNIFIED SOLVER
# ============================================================================

def solve_reactor(config: FluidizedBedConfig) -> Dict[str, float]:
    """
    Solve reactor based on model type in config.
    """
    if config.model_type == ReactorModelType.CSTR:
        return solve_cstr(config)
    else:
        return solve_pfr(config)


# ============================================================================
# VALIDATION
# ============================================================================

def validate_against_experimental(temperature_C: float = None) -> Dict:
    """
    Validate model against experimental data.
    
    Returns dict with experimental vs model comparison.
    """
    results = {}
    
    temps_to_check = [temperature_C] if temperature_C else [770, 800, 830]
    
    for T_C in temps_to_check:
        if T_C not in EXPERIMENTAL_DATA_FBR:
            continue
            
        exp_data = EXPERIMENTAL_DATA_FBR[T_C]
        
        # Create lab config at this temperature
        config = get_lab_config(T_C)
        model_result = solve_cstr(config)
        
        results[T_C] = {
            'experimental': {
                'H2_percent': exp_data['H2_percent'],
                'X': exp_data['X'],
                'r_W': exp_data['r_W'],
            },
            'model': {
                'H2_percent': model_result['H2_percent'],
                'X': model_result['conversion'],
                'r_W': model_result['r_W_mol_kg_s'],
            },
            'errors': {
                'H2_error_abs': abs(model_result['H2_percent'] - exp_data['H2_percent']),
                'H2_error_rel': abs(model_result['H2_percent'] - exp_data['H2_percent']) / exp_data['H2_percent'] * 100,
                'X_error_rel': abs(model_result['conversion'] - exp_data['X']) / exp_data['X'] * 100,
            }
        }
    
    return results


# ============================================================================
# SCALE-UP ANALYSIS
# ============================================================================

def analyze_scaleup(lab_config: FluidizedBedConfig, 
                    ind_config: FluidizedBedConfig) -> Dict:
    """
    Analyze scale-up from lab to industrial.
    """
    lab_result = solve_reactor(lab_config)
    ind_result = solve_reactor(ind_config)
    
    return {
        'lab': {
            'config': {
                'diameter_m': lab_config.diameter_m,
                'bed_height_m': lab_config.bed_height_m,
                'catalyst_mass_kg': lab_config.catalyst_mass_kg,
                'flow_rate_LPM': lab_config.flow_rate_LPM,
            },
            'result': lab_result,
        },
        'industrial': {
            'config': {
                'diameter_m': ind_config.diameter_m,
                'bed_height_m': ind_config.bed_height_m,
                'catalyst_mass_kg': ind_config.catalyst_mass_kg,
                'flow_rate_LPM': ind_config.flow_rate_LPM,
            },
            'result': ind_result,
        },
        'scale_factors': {
            'flow': ind_config.flow_rate_LPM / lab_config.flow_rate_LPM,
            'catalyst': ind_config.catalyst_mass_kg / lab_config.catalyst_mass_kg,
            'diameter': ind_config.diameter_m / lab_config.diameter_m,
            'tau_ratio': ind_result['space_time_s'] / lab_result['space_time_s'],
        },
        'comparison': {
            'H2_lab': lab_result['H2_percent'],
            'H2_ind': ind_result['H2_percent'],
            'H2_match': abs(lab_result['H2_percent'] - ind_result['H2_percent']) < 1.0,
            'dP_lab_kPa': lab_result['pressure_drop_kPa'],
            'dP_ind_kPa': ind_result['pressure_drop_kPa'],
        }
    }


# ============================================================================
# KINETICS CALIBRATION
# ============================================================================

def calibrate_kinetics_from_experimental() -> Dict[str, float]:
    """
    Calibrate kinetics from experimental data using Arrhenius fit.
    
    Returns calibrated A and Ea.
    """
    # Calculate k from experimental data using CSTR model
    # k = X / (τ * (1-X))
    
    tau = 39.4  # s (lab space time)
    
    T_K_list = []
    k_list = []
    
    for T_C, data in EXPERIMENTAL_DATA_FBR.items():
        T_K = data['T_K']
        X = data['X']
        k = X / (tau * (1 - X))
        T_K_list.append(T_K)
        k_list.append(k)
    
    T_K = np.array(T_K_list)
    k = np.array(k_list)
    
    # Arrhenius fit: ln(k) = ln(A) - Ea/(R*T)
    inv_T = 1 / T_K
    ln_k = np.log(k)
    
    # Linear regression
    coeffs = np.polyfit(inv_T, ln_k, 1)
    slope = coeffs[0]
    intercept = coeffs[1]
    
    Ea = -slope * R
    A = np.exp(intercept)
    
    return {'A': A, 'Ea': Ea, 'Ea_kJ_mol': Ea / 1000}


# ============================================================================
# MAIN TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FLUIDIZED BED REACTOR MODEL TEST")
    print("=" * 70)
    
    # Validate against experimental data
    print("\n📊 VALIDATION AGAINST EXPERIMENTAL DATA:")
    print("-" * 50)
    
    validation = validate_against_experimental()
    
    print(f"{'T (°C)':<10} {'H₂_exp':<12} {'H₂_model':<12} {'Error':<10}")
    print("-" * 45)
    for T_C, data in validation.items():
        print(f"{T_C:<10} {data['experimental']['H2_percent']:<12.2f} "
              f"{data['model']['H2_percent']:<12.2f} "
              f"{data['errors']['H2_error_abs']:<10.2f}")
    
    # Test lab config
    print("\n🔬 LAB SCALE (800°C):")
    print("-" * 50)
    lab = get_lab_config(800)
    lab_result = solve_cstr(lab)
    print(f"D = {lab.diameter_m*1000:.1f} mm, H = {lab.bed_height_m*100:.1f} cm")
    print(f"τ = {lab_result['space_time_s']:.1f} s")
    print(f"k = {lab_result['rate_constant']:.4e} s⁻¹")
    print(f"H₂ = {lab_result['H2_percent']:.2f}%")
    print(f"ΔP = {lab_result['pressure_drop_kPa']:.4f} kPa")
    
    # Test industrial configs
    print("\n🏭 INDUSTRIAL SCALE COMPARISON:")
    print("-" * 50)
    
    configs = [
        ("Industrial 1 (Same Us)", get_industrial_config_1(800)),
        ("Industrial 2 (High Us)", get_industrial_config_2(800)),
        ("Industrial 3 (Intermediate)", get_industrial_config_3(800)),
    ]
    
    print(f"{'Config':<25} {'τ (s)':<10} {'H₂ (%)':<10} {'ΔP (kPa)':<12} {'Feasible':<10}")
    print("-" * 70)
    
    for name, config in configs:
        result = solve_cstr(config)
        feasible = "✅" if result['pressure_drop_kPa'] < 100 else "❌"
        print(f"{name:<25} {result['space_time_s']:<10.1f} {result['H2_percent']:<10.2f} "
              f"{result['pressure_drop_kPa']:<12.2f} {feasible:<10}")
    
    print("\n✅ All industrial configs with τ ≈ 39s give same H₂% as lab!")
