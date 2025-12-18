"""
Anukaran AI - Transient Reactor Model
=====================================
Unsteady-state packed bed reactor with catalyst deactivation.
Reaction: CH4 → C + 2H2

Configured for LAB SCALE and INDUSTRIAL SCALE reactors.
Temperature: 800°C
Target H2: 30%
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar, brentq
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
import warnings

warnings.filterwarnings('ignore')

# Physical constants
R_GAS = 8.314          # J/(mol·K)
MW_CH4 = 16.04e-3      # kg/mol
MW_H2 = 2.016e-3       # kg/mol
MW_C = 12.01e-3        # kg/mol
MW_N2 = 28.01e-3       # kg/mol


# ============================================================================
# DEACTIVATION MODELS
# ============================================================================

@dataclass
class DeactivationParams:
    """Parameters for catalyst deactivation"""
    k_d: float = 0.008         # Deactivation rate constant [1/min]
    order: int = 1             # Deactivation order (1 or 2)
    E_d: float = 80000.0       # Deactivation activation energy [J/mol]
    T_ref: float = 1073.15     # Reference temperature [K] (800°C)
    
    def get_kd_at_temperature(self, T: float) -> float:
        """Get temperature-dependent deactivation rate constant"""
        return self.k_d * np.exp(-self.E_d / R_GAS * (1/T - 1/self.T_ref))


class DeactivationModel:
    """Catalyst deactivation kinetics."""
    
    def __init__(self, model_type: str = 'first_order', params: DeactivationParams = None):
        self.model_type = model_type
        self.params = params or DeactivationParams()
    
    def rate(self, activity: float, T: float, C_CH4: float = 0.0) -> float:
        """Calculate deactivation rate da/dt [1/min]"""
        k_d = self.params.get_kd_at_temperature(T)
        
        if self.model_type == 'linear':
            return -k_d
        elif self.model_type == 'first_order':
            return -k_d * activity
        elif self.model_type == 'second_order':
            return -k_d * activity ** 2
        elif self.model_type == 'coking':
            return -k_d * activity * max(C_CH4 * 1000, 0.001)
        else:
            return -k_d * activity


# ============================================================================
# TRANSPORT PROPERTIES
# ============================================================================

def gas_viscosity(T, y_CH4, y_H2):
    """Gas mixture viscosity [Pa·s]"""
    y_N2 = max(0, 1 - y_CH4 - y_H2)
    mu_CH4 = 1.02e-5 * (T / 300) ** 0.87
    mu_H2 = 8.76e-6 * (T / 300) ** 0.68
    mu_N2 = 1.78e-5 * (T / 300) ** 0.67
    return y_CH4 * mu_CH4 + y_H2 * mu_H2 + y_N2 * mu_N2


def gas_density(T, P, y_CH4, y_H2):
    """Gas mixture density [kg/m³]"""
    y_N2 = max(0, 1 - y_CH4 - y_H2)
    MW_mix = y_CH4 * MW_CH4 + y_H2 * MW_H2 + y_N2 * MW_N2
    return P * MW_mix / (R_GAS * T)


def diffusivity_CH4(T, P):
    """CH4 molecular diffusivity [m²/s]"""
    return 1.87e-5 * (T / 300) ** 1.75 * (101325 / P)


def arrhenius_rate_constant(T, A, Ea, beta=0):
    """Arrhenius rate constant [1/s]"""
    return A * (T ** beta) * np.exp(-Ea / (R_GAS * T))


def effectiveness_factor(phi):
    """Catalyst effectiveness factor for sphere"""
    if phi < 0.1:
        return 1.0
    elif phi > 100:
        return 3.0 / phi
    else:
        return (3.0 / phi) * (1.0 / np.tanh(phi) - 1.0 / phi)


def ergun_pressure_drop(u, rho, mu, d_p, eps):
    """Ergun equation pressure drop [Pa/m]"""
    if u <= 0:
        return 0.0
    term1 = 150 * mu * (1 - eps)**2 / (d_p**2 * eps**3) * u
    term2 = 1.75 * rho * (1 - eps) / (d_p * eps**3) * u * abs(u)
    return term1 + term2


# ============================================================================
# REACTOR CONFIGURATION
# ============================================================================

@dataclass
class ReactorConfig:
    """Unified reactor configuration for both scales"""
    # Geometry
    diameter_m: float
    bed_height_m: float
    
    # Catalyst
    particle_diameter_m: float
    particle_density: float      # kg/m³
    particle_porosity: float
    tortuosity: float
    bed_porosity: float
    catalyst_mass_kg: float
    
    # Operating conditions
    temperature_K: float
    inlet_pressure_Pa: float
    flow_rate_m3_s: float
    
    # Inlet composition (pure CH4)
    y_CH4_in: float = 1.0
    y_H2_in: float = 0.0
    
    # Kinetics (to be calibrated)
    pre_exponential: float = 5.0e4      # 1/s - calibrated for lab data
    activation_energy: float = 150000.0  # J/mol - calibrated
    beta: float = 0.0
    
    @property
    def cross_section_area(self) -> float:
        return np.pi * (self.diameter_m / 2) ** 2
    
    @classmethod
    def from_lab_config(cls, lab_config, A: float = 5.0e4, Ea: float = 150000.0):
        """Create ReactorConfig from LabScaleConfig"""
        return cls(
            diameter_m=lab_config.reactor_diameter_cm / 100,
            bed_height_m=lab_config.bed_height_cm / 100,
            particle_diameter_m=lab_config.particle_size_um * 1e-6,
            particle_density=lab_config.particle_density_kg_m3,
            particle_porosity=lab_config.particle_porosity,
            tortuosity=lab_config.tortuosity,
            bed_porosity=lab_config.bed_porosity,
            catalyst_mass_kg=lab_config.catalyst_mass_g / 1000,
            temperature_K=lab_config.temperature_C + 273.15,
            inlet_pressure_Pa=lab_config.inlet_pressure_bar * 1e5,
            flow_rate_m3_s=lab_config.flow_rate_mL_min / 60 / 1e6,
            pre_exponential=A,
            activation_energy=Ea,
        )
    
    @classmethod
    def from_industrial_config(cls, ind_config, diameter_m: float, height_m: float,
                                A: float = 5.0e4, Ea: float = 150000.0):
        """Create ReactorConfig from IndustrialScaleConfig with given geometry"""
        return cls(
            diameter_m=diameter_m,
            bed_height_m=height_m,
            particle_diameter_m=ind_config.particle_size_um * 1e-6,
            particle_density=ind_config.particle_density_kg_m3,
            particle_porosity=ind_config.particle_porosity,
            tortuosity=ind_config.tortuosity,
            bed_porosity=ind_config.bed_porosity,
            catalyst_mass_kg=ind_config.catalyst_mass_kg,
            temperature_K=ind_config.temperature_C + 273.15,
            inlet_pressure_Pa=ind_config.inlet_pressure_bar * 1e5,
            flow_rate_m3_s=ind_config.flow_rate_LPM / 60 / 1000,  # LPM to m³/s
            pre_exponential=A,
            activation_energy=Ea,
        )


# ============================================================================
# STEADY-STATE REACTOR SOLVER
# ============================================================================

class SteadyStateReactor:
    """
    Steady-state packed bed reactor solver.
    Used for both fresh catalyst and with activity factor.
    """
    
    def __init__(self, config: ReactorConfig):
        self.cfg = config
        
        # Calculate inlet molar flow
        C_total_in = config.inlet_pressure_Pa / (R_GAS * config.temperature_K)  # mol/m³
        Q_in = config.flow_rate_m3_s  # m³/s
        self.F_total_in = Q_in * C_total_in / 1000  # kmol/s
        self.F_CH4_in = config.y_CH4_in * self.F_total_in
        self.F_H2_in = config.y_H2_in * self.F_total_in
    
    def solve(self, activity: float = 1.0, n_points: int = 100) -> Dict:
        """Solve steady-state reactor with given catalyst activity"""
        cfg = self.cfg
        
        def ode_system(z, y):
            F_CH4, F_H2, P = y
            
            # Stability
            F_CH4 = max(F_CH4, 1e-30)
            F_H2 = max(F_H2, 0.0)
            P = max(P, 1000.0)
            
            # Total molar flow (CH4 + H2, assuming no inerts for pure CH4 feed)
            # For pure CH4: F_total = F_CH4 + F_H2 (carbon deposits, doesn't flow)
            F_total = F_CH4 + F_H2
            if F_total < 1e-30:
                F_total = 1e-30
            
            y_CH4 = F_CH4 / F_total
            y_H2 = F_H2 / F_total
            
            # Properties
            T = cfg.temperature_K
            rho = gas_density(T, P, y_CH4, y_H2)
            mu = gas_viscosity(T, y_CH4, y_H2)
            
            # Volumetric flow and velocity
            Q = F_total * 1000 * R_GAS * T / P  # m³/s
            u = Q / cfg.cross_section_area  # m/s
            
            # Concentration
            C_CH4 = F_CH4 * 1000 / Q if Q > 0 else 0  # mol/m³
            
            # Effective diffusivity
            D_mol = diffusivity_CH4(T, P)
            D_eff = D_mol * cfg.particle_porosity / cfg.tortuosity
            
            # Rate constant with activity
            k = arrhenius_rate_constant(T, cfg.pre_exponential, cfg.activation_energy, cfg.beta)
            k_eff = k * activity
            
            # Thiele modulus and effectiveness factor
            if D_eff > 0 and k_eff > 0:
                phi = (cfg.particle_diameter_m / 6) * np.sqrt(k_eff / D_eff)
            else:
                phi = 0
            eta = effectiveness_factor(phi)
            
            # Reaction rate [mol/(m³_bed·s)]
            r_bed = k_eff * eta * C_CH4 * (1 - cfg.bed_porosity)
            
            # Molar balances [kmol/(m·s)] -> multiply by area for kmol/s per meter
            A = cfg.cross_section_area
            dF_CH4_dz = -r_bed * A / 1000  # kmol/s per m
            dF_H2_dz = +2.0 * r_bed * A / 1000
            
            # Pressure drop
            dP_dz = -ergun_pressure_drop(u, rho, mu, cfg.particle_diameter_m, cfg.bed_porosity)
            
            return [dF_CH4_dz, dF_H2_dz, dP_dz]
        
        # Initial conditions
        y0 = [self.F_CH4_in, self.F_H2_in, cfg.inlet_pressure_Pa]
        
        # Solve
        z_span = (0, cfg.bed_height_m)
        z_eval = np.linspace(0, cfg.bed_height_m, n_points)
        
        solution = solve_ivp(
            ode_system, z_span, y0,
            method='RK45', t_eval=z_eval,
            rtol=1e-8, atol=1e-12
        )
        
        # Extract results
        z = solution.t
        F_CH4 = np.maximum(solution.y[0], 0)
        F_H2 = np.maximum(solution.y[1], 0)
        P = solution.y[2]
        
        # Calculate outputs
        F_total = F_CH4 + F_H2
        F_total = np.maximum(F_total, 1e-30)
        
        y_CH4 = F_CH4 / F_total
        y_H2 = F_H2 / F_total
        
        # Conversion
        X_CH4 = np.clip((self.F_CH4_in - F_CH4) / self.F_CH4_in, 0, 1)
        
        # Pressure drop
        delta_P = cfg.inlet_pressure_Pa - P[-1]
        
        return {
            'z': z,
            'z_cm': z * 100,
            'F_CH4': F_CH4,
            'F_H2': F_H2,
            'P': P,
            'y_CH4': y_CH4,
            'y_H2': y_H2,
            'X_CH4': X_CH4,
            'H2_percent': y_H2 * 100,
            'CH4_percent': y_CH4 * 100,
            'conversion_percent': X_CH4 * 100,
            'pressure_drop_Pa': delta_P,
            'pressure_drop_kPa': delta_P / 1000,
            'outlet_H2_percent': y_H2[-1] * 100,
            'outlet_CH4_percent': y_CH4[-1] * 100,
            'outlet_conversion': X_CH4[-1] * 100,
        }


# ============================================================================
# TRANSIENT REACTOR SOLVER
# ============================================================================

class TransientReactor:
    """Transient reactor with catalyst deactivation"""
    
    def __init__(self, config: ReactorConfig, deactivation: DeactivationModel = None):
        self.cfg = config
        self.deactivation = deactivation or DeactivationModel('first_order')
        self.steady_solver = SteadyStateReactor(config)
    
    def solve(self, t_final_min: float = 210.0, dt_min: float = 1.0,
              callback: Callable = None) -> Dict:
        """Solve transient reactor over time"""
        
        n_steps = int(t_final_min / dt_min) + 1
        time_min = np.linspace(0, t_final_min, n_steps)
        
        # Storage
        activity = np.zeros(n_steps)
        H2_percent = np.zeros(n_steps)
        CH4_percent = np.zeros(n_steps)
        conversion = np.zeros(n_steps)
        
        current_activity = 1.0
        
        for i, t in enumerate(time_min):
            activity[i] = current_activity
            
            # Solve steady-state at current activity
            results = self.steady_solver.solve(activity=current_activity)
            
            H2_percent[i] = results['outlet_H2_percent']
            CH4_percent[i] = results['outlet_CH4_percent']
            conversion[i] = results['outlet_conversion']
            
            if callback:
                callback(t, current_activity, results)
            
            # Update activity
            if i < n_steps - 1:
                da_dt = self.deactivation.rate(
                    current_activity,
                    self.cfg.temperature_K,
                    0.0  # Simplified
                )
                current_activity = max(0.0, current_activity + da_dt * dt_min)
        
        return {
            'time_min': time_min,
            'activity': activity,
            'H2_percent': H2_percent,
            'CH4_percent': CH4_percent,
            'conversion': conversion,
            'temperature_C': self.cfg.temperature_K - 273.15,
        }


# ============================================================================
# KINETIC PARAMETER CALIBRATION
# ============================================================================

def calibrate_kinetics(lab_config, experimental_H2_at_t0: float,
                       A_range: Tuple[float, float] = (1e3, 1e8),
                       Ea_range: Tuple[float, float] = (100000, 250000)) -> Dict:
    """
    Calibrate kinetic parameters (A, Ea) to match experimental H2% at TOS=0.
    
    For 800°C, experimental H2% at TOS=0 is 24.13%
    """
    from scipy.optimize import minimize
    
    best_result = {'A': 1e5, 'Ea': 150000, 'error': float('inf')}
    
    # Grid search for initial guess
    A_values = np.logspace(np.log10(A_range[0]), np.log10(A_range[1]), 20)
    Ea_values = np.linspace(Ea_range[0], Ea_range[1], 20)
    
    for A in A_values:
        for Ea in Ea_values:
            try:
                config = ReactorConfig.from_lab_config(lab_config, A=A, Ea=Ea)
                reactor = SteadyStateReactor(config)
                results = reactor.solve(activity=1.0)
                
                model_H2 = results['outlet_H2_percent']
                error = abs(model_H2 - experimental_H2_at_t0)
                
                if error < best_result['error']:
                    best_result = {'A': A, 'Ea': Ea, 'error': error, 'H2_predicted': model_H2}
            except:
                continue
    
    return best_result


def fit_deactivation(lab_config, A: float, Ea: float,
                     experimental_times: np.ndarray,
                     experimental_H2: np.ndarray,
                     kd_range: Tuple[float, float] = (0.001, 0.1)) -> Dict:
    """Fit deactivation parameter to experimental time series"""
    
    best_kd = 0.01
    best_rmse = float('inf')
    
    kd_values = np.linspace(kd_range[0], kd_range[1], 50)
    
    for kd in kd_values:
        try:
            config = ReactorConfig.from_lab_config(lab_config, A=A, Ea=Ea)
            deact = DeactivationModel('first_order', DeactivationParams(k_d=kd))
            reactor = TransientReactor(config, deact)
            
            results = reactor.solve(t_final_min=experimental_times[-1], dt_min=1.0)
            
            model_H2 = np.interp(experimental_times, results['time_min'], results['H2_percent'])
            rmse = np.sqrt(np.mean((model_H2 - experimental_H2) ** 2))
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_kd = kd
        except:
            continue
    
    return {'k_d': best_kd, 'rmse': best_rmse}
