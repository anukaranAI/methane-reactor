"""
Anukaran AI - Transient Reactor Model
=====================================
Unsteady-state packed bed reactor with catalyst deactivation.
Reaction: CH4 → C + 2H2
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
import warnings

warnings.filterwarnings('ignore')

# Physical constants
R_GAS = 8.314
MW_CH4 = 16.04e-3
MW_H2 = 2.016e-3
MW_C = 12.01e-3
MW_N2 = 28.01e-3


# ============================================================================
# DEACTIVATION MODELS
# ============================================================================

@dataclass
class DeactivationParams:
    """Parameters for catalyst deactivation"""
    k_d: float = 0.01          # Deactivation rate constant [1/min]
    order: int = 1             # Deactivation order (1 or 2)
    E_d: float = 50000.0       # Deactivation activation energy [J/mol]
    T_ref: float = 1073.15     # Reference temperature [K] (800°C)
    
    def get_kd_at_temperature(self, T: float) -> float:
        """Get temperature-dependent deactivation rate constant"""
        return self.k_d * np.exp(-self.E_d / R_GAS * (1/T - 1/self.T_ref))


class DeactivationModel:
    """
    Catalyst deactivation kinetics.
    
    Models available:
    - 'linear': da/dt = -k_d (constant rate)
    - 'first_order': da/dt = -k_d * a
    - 'second_order': da/dt = -k_d * a^2
    - 'coking': da/dt = -k_d * a * C_CH4
    """
    
    def __init__(self, model_type: str = 'first_order', params: DeactivationParams = None):
        self.model_type = model_type
        self.params = params or DeactivationParams()
    
    def rate(self, activity: float, T: float, C_CH4: float = 0.0) -> float:
        """
        Calculate deactivation rate da/dt.
        
        Args:
            activity: Current catalyst activity (0 to 1)
            T: Temperature [K]
            C_CH4: CH4 concentration [kmol/m³] (for coking model)
        
        Returns:
            da/dt [1/min] (negative value)
        """
        k_d = self.params.get_kd_at_temperature(T)
        
        if self.model_type == 'linear':
            return -k_d
        elif self.model_type == 'first_order':
            return -k_d * activity
        elif self.model_type == 'second_order':
            return -k_d * activity ** 2
        elif self.model_type == 'coking':
            return -k_d * activity * (C_CH4 * 1000)  # Convert to mol/m³
        else:
            return -k_d * activity  # Default to first order


# ============================================================================
# TRANSPORT PROPERTIES (Same as steady-state model)
# ============================================================================

def gas_viscosity(T, y_CH4, y_H2, y_N2):
    mu_CH4 = 1.02e-5 * (T / 300) ** 0.87
    mu_H2 = 8.76e-6 * (T / 300) ** 0.68
    mu_N2 = 1.78e-5 * (T / 300) ** 0.67
    return y_CH4 * mu_CH4 + y_H2 * mu_H2 + y_N2 * mu_N2


def gas_density(T, P, y_CH4, y_H2, y_N2):
    MW_mix = y_CH4 * MW_CH4 + y_H2 * MW_H2 + y_N2 * MW_N2
    return P * MW_mix / (R_GAS * T)


def diffusivity_CH4(T, P):
    return 1.87e-5 * (T / 300) ** 1.75 * (101325 / P)


def arrhenius_rate_constant(T, A, Ea, beta=0):
    return A * T ** beta * np.exp(-Ea / (R_GAS * T))


def effectiveness_factor(phi):
    if phi < 0.1:
        return 1.0
    elif phi > 100:
        return 3.0 / phi
    else:
        return (3.0 / phi) * (1.0 / np.tanh(phi) - 1.0 / phi)


def ergun_pressure_drop(u, rho, mu, d_p, eps):
    term1 = 150 * mu * (1 - eps)**2 / (d_p**2 * eps**3) * u
    term2 = 1.75 * rho * (1 - eps) / (d_p * eps**3) * u**2
    return term1 + term2


# ============================================================================
# TRANSIENT REACTOR CONFIG
# ============================================================================

@dataclass
class TransientReactorConfig:
    """Configuration for transient reactor simulation"""
    # Geometry
    diameter: float              # Reactor diameter [m]
    bed_height: float            # Bed height [m]
    
    # Catalyst
    particle_diameter: float     # Particle diameter [m]
    catalyst_density: float      # Particle density [kg/m³]
    particle_porosity: float     # Intra-particle porosity [-]
    tortuosity: float            # Tortuosity [-]
    bed_porosity: float          # Bed void fraction [-]
    
    # Operating conditions
    temperature: float           # Operating temperature [K]
    inlet_pressure: float        # Inlet pressure [Pa]
    flow_rate: float             # Volumetric flow rate [m³/s]
    
    # Inlet composition
    y_CH4_in: float              # CH4 mole fraction
    y_H2_in: float               # H2 mole fraction
    y_N2_in: float               # N2 mole fraction
    
    # Kinetics
    pre_exponential: float       # Pre-exponential factor [1/s]
    activation_energy: float     # Activation energy [J/mol]
    beta: float = 0.0            # Temperature exponent
    
    @property
    def cross_section_area(self) -> float:
        return np.pi * (self.diameter / 2) ** 2


# ============================================================================
# TRANSIENT REACTOR SOLVER
# ============================================================================

class TransientReactor:
    """
    Unsteady-state packed bed reactor with catalyst deactivation.
    
    Uses pseudo-steady-state approach:
    1. At each time step, solve steady-state spatial profiles
    2. Update catalyst activity based on deactivation kinetics
    3. Repeat for next time step
    """
    
    def __init__(
        self,
        config: TransientReactorConfig,
        deactivation: DeactivationModel = None
    ):
        self.cfg = config
        self.deactivation = deactivation or DeactivationModel('first_order')
        
        # Calculate inlet molar quantities
        C_total_in = config.inlet_pressure / (R_GAS * config.temperature) / 1000  # kmol/m³
        self.F_total_in = config.flow_rate * C_total_in  # kmol/s
        self.F_CH4_in = config.y_CH4_in * self.F_total_in
        self.F_H2_in = config.y_H2_in * self.F_total_in
        self.F_N2_in = config.y_N2_in * self.F_total_in
    
    def _solve_steady_state(self, activity: float) -> Dict:
        """
        Solve steady-state reactor at given catalyst activity.
        
        Args:
            activity: Catalyst activity (0 to 1)
        
        Returns:
            Dictionary with outlet conditions
        """
        cfg = self.cfg
        
        def ode_system(z, y):
            F_CH4, F_H2, P = y
            
            # Stability clamps
            F_CH4 = max(F_CH4, 1e-30)
            F_H2 = max(F_H2, 0.0)
            P = max(P, 1000.0)
            
            # Mole fractions
            F_total = F_CH4 + F_H2 + self.F_N2_in
            y_CH4 = F_CH4 / F_total
            y_H2 = F_H2 / F_total
            y_N2 = self.F_N2_in / F_total
            
            # Properties
            T = cfg.temperature
            rho = gas_density(T, P, y_CH4, y_H2, y_N2)
            mu = gas_viscosity(T, y_CH4, y_H2, y_N2)
            
            # Velocity and concentration
            Q = F_total * 1000 * R_GAS * T / P
            u = Q / cfg.cross_section_area
            C_CH4 = F_CH4 / Q
            
            # Diffusivity and effectiveness
            D_mol = diffusivity_CH4(T, P)
            D_eff = D_mol * cfg.particle_porosity / cfg.tortuosity
            
            # Rate constant with activity
            k = arrhenius_rate_constant(T, cfg.pre_exponential, cfg.activation_energy, cfg.beta)
            k_eff = k * activity  # Multiply by catalyst activity
            
            # Thiele modulus and effectiveness factor
            phi = (cfg.particle_diameter / 6) * np.sqrt(k_eff / D_eff) if D_eff > 0 else 0
            eta = effectiveness_factor(phi)
            
            # Reaction rate
            r_bed = k_eff * eta * C_CH4 * (1 - cfg.bed_porosity)
            
            # Derivatives
            A = cfg.cross_section_area
            dF_CH4_dz = -r_bed * A
            dF_H2_dz = +2.0 * r_bed * A
            dP_dz = -ergun_pressure_drop(u, rho, mu, cfg.particle_diameter, cfg.bed_porosity)
            
            return [dF_CH4_dz, dF_H2_dz, dP_dz]
        
        # Initial conditions
        y0 = [self.F_CH4_in, self.F_H2_in, cfg.inlet_pressure]
        
        # Solve ODE
        solution = solve_ivp(
            ode_system,
            (0, cfg.bed_height),
            y0,
            method='RK45',
            dense_output=True,
            rtol=1e-6,
            atol=1e-10
        )
        
        # Extract outlet values
        F_CH4_out = max(solution.y[0, -1], 0)
        F_H2_out = max(solution.y[1, -1], 0)
        P_out = solution.y[2, -1]
        
        # Calculate compositions
        F_total_out = F_CH4_out + F_H2_out + self.F_N2_in
        y_CH4_out = F_CH4_out / F_total_out
        y_H2_out = F_H2_out / F_total_out
        y_N2_out = self.F_N2_in / F_total_out
        
        # Conversion
        X_CH4 = (self.F_CH4_in - F_CH4_out) / self.F_CH4_in if self.F_CH4_in > 0 else 0
        
        # Average concentration for deactivation calculation
        Q_avg = F_total_out * 1000 * R_GAS * cfg.temperature / P_out
        C_CH4_avg = (F_CH4_out + self.F_CH4_in) / 2 / Q_avg
        
        return {
            'F_CH4': F_CH4_out,
            'F_H2': F_H2_out,
            'P': P_out,
            'y_CH4': y_CH4_out,
            'y_H2': y_H2_out,
            'y_N2': y_N2_out,
            'X_CH4': X_CH4,
            'C_CH4_avg': C_CH4_avg,
            'H2_percent': y_H2_out * 100,
            'CH4_percent': y_CH4_out * 100,
        }
    
    def solve(
        self,
        t_final_min: float = 210.0,
        dt_min: float = 1.0,
        callback: Callable = None
    ) -> Dict:
        """
        Solve transient reactor over time.
        
        Args:
            t_final_min: Final time [minutes]
            dt_min: Time step [minutes]
            callback: Optional callback function(t, activity, results)
        
        Returns:
            Dictionary with time-dependent results
        """
        # Time array
        n_steps = int(t_final_min / dt_min) + 1
        time_min = np.linspace(0, t_final_min, n_steps)
        
        # Initialize storage
        activity = np.zeros(n_steps)
        H2_percent = np.zeros(n_steps)
        CH4_percent = np.zeros(n_steps)
        conversion = np.zeros(n_steps)
        
        # Initial activity = 1 (fresh catalyst)
        current_activity = 1.0
        
        for i, t in enumerate(time_min):
            # Store current activity
            activity[i] = current_activity
            
            # Solve steady-state at current activity
            results = self._solve_steady_state(current_activity)
            
            # Store results
            H2_percent[i] = results['H2_percent']
            CH4_percent[i] = results['CH4_percent']
            conversion[i] = results['X_CH4'] * 100
            
            # Callback for progress updates
            if callback:
                callback(t, current_activity, results)
            
            # Update activity for next time step (Euler method)
            if i < n_steps - 1:
                da_dt = self.deactivation.rate(
                    current_activity,
                    self.cfg.temperature,
                    results['C_CH4_avg']
                )
                current_activity = max(0.0, current_activity + da_dt * dt_min)
        
        return {
            'time_min': time_min,
            'activity': activity,
            'H2_percent': H2_percent,
            'CH4_percent': CH4_percent,
            'conversion': conversion,
            'temperature_C': self.cfg.temperature - 273.15,
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_transient_config(
    temperature_C: float = 800.0,
    reactor_diameter_cm: float = 2.5,
    bed_height_cm: float = 10.0,
    particle_diameter_um: float = 500.0,
    flow_rate_mL_min: float = 50.0,
    inlet_pressure_bar: float = 1.0,
    y_CH4: float = 0.5,
    y_N2: float = 0.5,
    pre_exponential: float = 1.0e6,
    activation_energy_kJ_mol: float = 100.0,
) -> TransientReactorConfig:
    """Create transient reactor config with user-friendly units."""
    
    return TransientReactorConfig(
        diameter=reactor_diameter_cm / 100,
        bed_height=bed_height_cm / 100,
        particle_diameter=particle_diameter_um * 1e-6,
        catalyst_density=2000.0,
        particle_porosity=0.5,
        tortuosity=3.0,
        bed_porosity=0.4,
        temperature=temperature_C + 273.15,
        inlet_pressure=inlet_pressure_bar * 1e5,
        flow_rate=flow_rate_mL_min / 60 / 1e6,
        y_CH4_in=y_CH4,
        y_H2_in=0.0,
        y_N2_in=y_N2,
        pre_exponential=pre_exponential,
        activation_energy=activation_energy_kJ_mol * 1000,
    )


def fit_deactivation_parameter(
    experimental_H2: np.ndarray,
    experimental_time: np.ndarray,
    config: TransientReactorConfig,
    k_d_range: Tuple[float, float] = (0.001, 0.1),
    n_trials: int = 20
) -> Tuple[float, float]:
    """
    Fit deactivation rate constant to experimental data.
    
    Args:
        experimental_H2: Experimental H2 composition [%]
        experimental_time: Time on stream [min]
        config: Reactor configuration
        k_d_range: Range of k_d values to try
        n_trials: Number of trials
    
    Returns:
        Tuple of (best_k_d, best_rmse)
    """
    best_k_d = k_d_range[0]
    best_rmse = float('inf')
    
    k_d_values = np.linspace(k_d_range[0], k_d_range[1], n_trials)
    
    for k_d in k_d_values:
        # Create deactivation model
        deact_params = DeactivationParams(k_d=k_d)
        deact_model = DeactivationModel('first_order', deact_params)
        
        # Run simulation
        reactor = TransientReactor(config, deact_model)
        results = reactor.solve(t_final_min=experimental_time[-1], dt_min=1.0)
        
        # Interpolate model results at experimental times
        model_H2 = np.interp(experimental_time, results['time_min'], results['H2_percent'])
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((model_H2 - experimental_H2) ** 2))
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_k_d = k_d
    
    return best_k_d, best_rmse


def run_validation_at_temperature(
    temperature_C: int,
    k_d: float = 0.01,
    experimental_data: Dict = None
) -> Dict:
    """
    Run transient simulation and compare with experimental data.
    
    Args:
        temperature_C: Temperature in Celsius
        k_d: Deactivation rate constant
        experimental_data: Optional experimental data dict
    
    Returns:
        Dictionary with model results and comparison metrics
    """
    from core.experimental_data import get_experimental_data
    
    # Get experimental data if not provided
    if experimental_data is None:
        experimental_data = get_experimental_data(temperature_C)
    
    if experimental_data is None:
        raise ValueError(f"No experimental data for {temperature_C}°C")
    
    # Create config
    config = create_transient_config(temperature_C=temperature_C)
    
    # Create deactivation model
    deact_params = DeactivationParams(k_d=k_d)
    deact_model = DeactivationModel('first_order', deact_params)
    
    # Run simulation
    reactor = TransientReactor(config, deact_model)
    results = reactor.solve(t_final_min=210.0, dt_min=1.0)
    
    # Interpolate at experimental times
    exp_times = experimental_data['TOS_min']
    model_H2 = np.interp(exp_times, results['time_min'], results['H2_percent'])
    model_CH4 = np.interp(exp_times, results['time_min'], results['CH4_percent'])
    
    # Calculate errors
    exp_H2 = experimental_data['H2_percent']
    exp_CH4 = experimental_data['CH4_percent']
    
    rmse_H2 = np.sqrt(np.mean((model_H2 - exp_H2) ** 2))
    rmse_CH4 = np.sqrt(np.mean((model_CH4 - exp_CH4) ** 2))
    
    return {
        'model_results': results,
        'experimental_times': exp_times,
        'model_H2_at_exp_times': model_H2,
        'model_CH4_at_exp_times': model_CH4,
        'experimental_H2': exp_H2,
        'experimental_CH4': exp_CH4,
        'rmse_H2': rmse_H2,
        'rmse_CH4': rmse_CH4,
        'temperature_C': temperature_C,
        'k_d': k_d,
    }
