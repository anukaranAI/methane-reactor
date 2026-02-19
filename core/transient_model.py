"""
KinetiScale - Generic Transient Reactor Model
==============================================
Unsteady-state packed bed reactor with catalyst deactivation.
Supports any single gas-phase catalytic reaction defined via the
Reaction dataclass from core.chemistry.
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
import warnings

from .chemistry import Reaction, R_GAS
from .transport import (
    gas_mixture_viscosity,
    gas_mixture_density,
    effective_diffusivity,
    ergun_pressure_drop,
)

warnings.filterwarnings('ignore')


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

    def rate(self, activity: float, T: float, C_reactant: float = 0.0) -> float:
        """Calculate deactivation rate da/dt [1/min]"""
        k_d = self.params.get_kd_at_temperature(T)

        if self.model_type == 'linear':
            return -k_d
        elif self.model_type == 'first_order':
            return -k_d * activity
        elif self.model_type == 'second_order':
            return -k_d * activity ** 2
        elif self.model_type == 'coking':
            return -k_d * activity * max(C_reactant * 1000, 0.001)
        else:
            return -k_d * activity


# ============================================================================
# KINETIC FUNCTIONS
# ============================================================================

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


# ============================================================================
# REACTOR CONFIGURATION
# ============================================================================

@dataclass
class ReactorConfig:
    """Unified reactor configuration for any single catalytic reaction."""
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

    # Reaction definition
    reaction: Reaction = None

    # Inlet composition: {species_name: mole_fraction}
    # Species not listed are assumed to be 0. Inert fills remainder to 1.0.
    inlet_mole_fractions: Dict[str, float] = field(default_factory=dict)

    # Kinetics (to be calibrated)
    pre_exponential: float = 5.0e4      # 1/s
    activation_energy: float = 150000.0  # J/mol
    beta: float = 0.0

    @property
    def cross_section_area(self) -> float:
        return np.pi * (self.diameter_m / 2) ** 2

    def get_inlet_y_array(self) -> np.ndarray:
        """
        Build ordered mole fraction array for all gas species.
        Inert species fills the remainder to sum to 1.0.
        """
        gas_sp = self.reaction.gas_species
        y = np.zeros(len(gas_sp))
        assigned_sum = 0.0
        inert_idx = None

        for i, sp_name in enumerate(gas_sp):
            if sp_name in self.inlet_mole_fractions:
                y[i] = self.inlet_mole_fractions[sp_name]
                assigned_sum += y[i]
            elif sp_name == self.reaction.inert:
                inert_idx = i

        # Fill inert with remainder
        if inert_idx is not None and assigned_sum < 1.0:
            y[inert_idx] = max(0.0, 1.0 - assigned_sum)

        # Normalize if needed
        y_sum = y.sum()
        if y_sum > 0:
            y = y / y_sum

        return y


# ============================================================================
# STEADY-STATE REACTOR SOLVER
# ============================================================================

class SteadyStateReactor:
    """
    Generic steady-state packed bed reactor solver.
    Supports any single reaction defined by a Reaction object.
    """

    def __init__(self, config: ReactorConfig):
        self.cfg = config
        rxn = config.reaction
        self.rxn = rxn
        self.gas_sp = rxn.gas_species
        self.n_gas = rxn.n_gas
        self.stoich = rxn.gas_stoich_vector
        self.key_idx = rxn.gas_species_index(rxn.key_reactant)
        self.species_objects = [rxn.species[s] for s in self.gas_sp]
        self.key_species = rxn.species[rxn.key_reactant]

        # Calculate inlet molar flows
        C_total_in = config.inlet_pressure_Pa / (R_GAS * config.temperature_K)  # mol/m³
        Q_in = config.flow_rate_m3_s  # m³/s
        self.F_total_in = Q_in * C_total_in / 1000  # kmol/s

        y_in = config.get_inlet_y_array()
        self.F_in = y_in * self.F_total_in  # kmol/s per species

    def solve(self, activity: float = 1.0, n_points: int = 100) -> Dict:
        """Solve steady-state reactor with given catalyst activity"""
        cfg = self.cfg
        n_gas = self.n_gas
        stoich = self.stoich
        key_idx = self.key_idx
        species_objects = self.species_objects
        key_species = self.key_species

        def ode_system(z, y):
            F = np.maximum(y[:n_gas], 0.0)       # molar flows [kmol/s]
            P = max(y[n_gas], 1000.0)             # pressure [Pa]

            F_total = max(np.sum(F), 1e-30)
            y_frac = F / F_total                  # mole fractions

            T = cfg.temperature_K
            rho = gas_mixture_density(T, P, y_frac, species_objects)
            mu = gas_mixture_viscosity(T, y_frac, species_objects)

            # Volumetric flow and velocity
            Q = F_total * 1000 * R_GAS * T / P  # m³/s
            u = Q / cfg.cross_section_area        # m/s

            # Key reactant concentration
            C_key = F[key_idx] * 1000 / Q if Q > 0 else 0  # mol/m³

            # Effective diffusivity
            D_eff = effective_diffusivity(
                T, P, key_species,
                cfg.particle_porosity, cfg.tortuosity
            )

            # Rate constant with activity
            k = arrhenius_rate_constant(T, cfg.pre_exponential, cfg.activation_energy, cfg.beta)
            k_eff = k * activity

            # Thiele modulus and effectiveness factor
            if D_eff > 0 and k_eff > 0:
                phi = (cfg.particle_diameter_m / 6) * np.sqrt(k_eff / D_eff)
            else:
                phi = 0
            eta = effectiveness_factor(phi)

            # Reaction rate [mol/(m³_bed·s)] — first order in key reactant
            r_bed = k_eff * eta * C_key * (1 - cfg.bed_porosity)

            # Generic species balances [kmol/(m·s)]
            A = cfg.cross_section_area
            dF_dz = stoich * r_bed * A / 1000

            # Pressure drop
            dP_dz = -ergun_pressure_drop(u, rho, mu, cfg.particle_diameter_m, cfg.bed_porosity)

            return np.concatenate([dF_dz, [dP_dz]])

        # Initial conditions: [F_1, F_2, ..., F_n, P]
        y0 = np.concatenate([self.F_in, [cfg.inlet_pressure_Pa]])

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
        F_values = {}
        y_values = {}
        for i, sp_name in enumerate(self.gas_sp):
            F_values[sp_name] = np.maximum(solution.y[i], 0)
        P = solution.y[n_gas]

        # Total flow and mole fractions
        F_total = sum(F_values.values())
        F_total = np.maximum(F_total, 1e-30)

        for sp_name in self.gas_sp:
            y_values[sp_name] = F_values[sp_name] / F_total

        # Key reactant conversion
        key_name = self.rxn.key_reactant
        prod_name = self.rxn.key_product
        X_key = np.clip((self.F_in[self.key_idx] - F_values[key_name]) / self.F_in[self.key_idx], 0, 1)

        # Pressure drop
        delta_P = cfg.inlet_pressure_Pa - P[-1]

        # Build results dict with generic keys
        results = {
            'z': z,
            'z_cm': z * 100,
            'P': P,
            'conversion_percent': X_key * 100,
            'pressure_drop_Pa': delta_P,
            'pressure_drop_kPa': delta_P / 1000,
            # Outlet summary
            'outlet_conversion': X_key[-1] * 100,
            f'outlet_{prod_name}_percent': y_values[prod_name][-1] * 100,
            f'outlet_{key_name}_percent': y_values[key_name][-1] * 100,
        }

        # Per-species profiles
        for sp_name in self.gas_sp:
            results[f'F_{sp_name}'] = F_values[sp_name]
            results[f'y_{sp_name}'] = y_values[sp_name]
            results[f'{sp_name}_percent'] = y_values[sp_name] * 100

        # Convenience aliases for product/reactant
        results['product_percent'] = y_values[prod_name] * 100
        results['reactant_percent'] = y_values[key_name] * 100
        results['outlet_product_percent'] = y_values[prod_name][-1] * 100
        results['outlet_reactant_percent'] = y_values[key_name][-1] * 100

        return results


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

        rxn = self.cfg.reaction
        prod_name = rxn.key_product
        react_name = rxn.key_reactant

        # Storage
        activity = np.zeros(n_steps)
        product_percent = np.zeros(n_steps)
        reactant_percent = np.zeros(n_steps)
        conversion = np.zeros(n_steps)

        current_activity = 1.0

        for i, t in enumerate(time_min):
            activity[i] = current_activity

            # Solve steady-state at current activity
            results = self.steady_solver.solve(activity=current_activity)

            product_percent[i] = results['outlet_product_percent']
            reactant_percent[i] = results['outlet_reactant_percent']
            conversion[i] = results['outlet_conversion']

            if callback:
                callback(t, current_activity, results)

            # Update activity
            if i < n_steps - 1:
                da_dt = self.deactivation.rate(
                    current_activity,
                    self.cfg.temperature_K,
                    0.0
                )
                current_activity = max(0.0, current_activity + da_dt * dt_min)

        return {
            'time_min': time_min,
            'activity': activity,
            'product_percent': product_percent,
            'reactant_percent': reactant_percent,
            'conversion': conversion,
            'temperature_C': self.cfg.temperature_K - 273.15,
            # Named keys for backward compat and clarity
            f'{prod_name}_percent': product_percent,
            f'{react_name}_percent': reactant_percent,
        }


# ============================================================================
# SOLID DEPOSITION CALCULATION
# ============================================================================

def calculate_solid_deposition(conversion: float, flow_rate_mol_s: float,
                                catalyst_mass_kg: float, reaction: Reaction) -> Dict:
    """
    Calculate solid product deposition rate for reactions that produce solids.

    Returns dict of {solid_species_name: rate_g_per_h_per_kg_cat}
    """
    rates = {}
    key_stoich = abs(reaction.stoichiometry[reaction.key_reactant])
    mol_reacted = conversion * flow_rate_mol_s

    for solid_name in reaction.solid_species:
        solid_sp = reaction.species[solid_name]
        solid_stoich = reaction.stoichiometry.get(solid_name, 0)
        mol_solid = mol_reacted * abs(solid_stoich) / key_stoich
        # g/h per kg catalyst
        rate = mol_solid * solid_sp.MW * 1000 * 3600 / catalyst_mass_kg if catalyst_mass_kg > 0 else 0
        rates[solid_name] = rate

    return rates


# ============================================================================
# KINETIC PARAMETER CALIBRATION
# ============================================================================

def calibrate_kinetics(config: ReactorConfig,
                       experimental_product_at_t0: float,
                       A_range: Tuple[float, float] = (1e3, 1e8),
                       Ea_range: Tuple[float, float] = (100000, 250000)) -> Dict:
    """
    Calibrate kinetic parameters (A, Ea) to match experimental product% at TOS=0.
    """
    from scipy.optimize import minimize

    best_result = {'A': 1e5, 'Ea': 150000, 'error': float('inf')}

    A_values = np.logspace(np.log10(A_range[0]), np.log10(A_range[1]), 20)
    Ea_values = np.linspace(Ea_range[0], Ea_range[1], 20)

    for A in A_values:
        for Ea in Ea_values:
            try:
                cfg_copy = ReactorConfig(
                    diameter_m=config.diameter_m,
                    bed_height_m=config.bed_height_m,
                    particle_diameter_m=config.particle_diameter_m,
                    particle_density=config.particle_density,
                    particle_porosity=config.particle_porosity,
                    tortuosity=config.tortuosity,
                    bed_porosity=config.bed_porosity,
                    catalyst_mass_kg=config.catalyst_mass_kg,
                    temperature_K=config.temperature_K,
                    inlet_pressure_Pa=config.inlet_pressure_Pa,
                    flow_rate_m3_s=config.flow_rate_m3_s,
                    reaction=config.reaction,
                    inlet_mole_fractions=config.inlet_mole_fractions,
                    pre_exponential=A,
                    activation_energy=Ea,
                )
                reactor = SteadyStateReactor(cfg_copy)
                results = reactor.solve(activity=1.0)

                model_product = results['outlet_product_percent']
                error = abs(model_product - experimental_product_at_t0)

                if error < best_result['error']:
                    best_result = {'A': A, 'Ea': Ea, 'error': error, 'product_predicted': model_product}
            except Exception:
                continue

    return best_result


def fit_deactivation(config: ReactorConfig, A: float, Ea: float,
                     experimental_times: np.ndarray,
                     experimental_product: np.ndarray,
                     kd_range: Tuple[float, float] = (0.001, 0.1)) -> Dict:
    """Fit deactivation parameter to experimental time series"""

    best_kd = 0.01
    best_rmse = float('inf')

    kd_values = np.linspace(kd_range[0], kd_range[1], 50)

    for kd in kd_values:
        try:
            cfg_copy = ReactorConfig(
                diameter_m=config.diameter_m,
                bed_height_m=config.bed_height_m,
                particle_diameter_m=config.particle_diameter_m,
                particle_density=config.particle_density,
                particle_porosity=config.particle_porosity,
                tortuosity=config.tortuosity,
                bed_porosity=config.bed_porosity,
                catalyst_mass_kg=config.catalyst_mass_kg,
                temperature_K=config.temperature_K,
                inlet_pressure_Pa=config.inlet_pressure_Pa,
                flow_rate_m3_s=config.flow_rate_m3_s,
                reaction=config.reaction,
                inlet_mole_fractions=config.inlet_mole_fractions,
                pre_exponential=A,
                activation_energy=Ea,
            )
            deact = DeactivationModel('first_order', DeactivationParams(k_d=kd))
            reactor = TransientReactor(cfg_copy, deact)

            results = reactor.solve(t_final_min=experimental_times[-1], dt_min=1.0)

            model_product = np.interp(experimental_times, results['time_min'], results['product_percent'])
            rmse = np.sqrt(np.mean((model_product - experimental_product) ** 2))

            if rmse < best_rmse:
                best_rmse = rmse
                best_kd = kd
        except Exception:
            continue

    return {'k_d': best_kd, 'rmse': best_rmse}
