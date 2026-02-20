"""
KinetiScale - Bayesian Optimizer
================================
Generic optimization engine for catalytic reactor parameters.
Uses scikit-optimize (Gaussian Process + Expected Improvement)
with random search fallback.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


@dataclass
class OptimizationResult:
    """Container for optimization results."""
    best_params: Dict[str, float]
    best_value: float
    all_trials: List[Dict]
    convergence: List[float]
    best_so_far: List[float]
    variable_names: List[str]
    success: bool = True
    message: str = ""


@dataclass
class OptimizationConfig:
    """Configuration for optimization run."""
    variable_names: List[str]
    bounds: List[Tuple[float, float]]
    n_iterations: int = 25
    n_initial_points: int = 5
    random_state: int = 42
    maximize: bool = True

    def validate(self) -> Tuple[bool, str]:
        if len(self.variable_names) != len(self.bounds):
            return False, "Variable names and bounds must have same length"
        if self.n_iterations < 5:
            return False, "Need at least 5 iterations"
        if self.n_initial_points >= self.n_iterations:
            return False, "Initial points must be less than total iterations"
        for name, (low, high) in zip(self.variable_names, self.bounds):
            if low >= high:
                return False, f"Invalid bounds for {name}: [{low}, {high}]"
        return True, "OK"


class BayesianOptimizer:
    """
    Bayesian Optimization using Gaussian Process surrogate model.
    Uses Expected Improvement (EI) acquisition function.
    Falls back to random search if scikit-optimize is unavailable.
    """

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.history: List[Dict] = []
        self.best_value = float('-inf') if config.maximize else float('inf')
        self.best_params = None
        self.convergence = []
        self.best_so_far = []

    def optimize(
        self,
        objective_fn: Callable[[Dict[str, float]], float],
        callback: Optional[Callable[[int, Dict, float], None]] = None
    ) -> OptimizationResult:
        try:
            from skopt import gp_minimize
            return self._optimize_with_skopt(objective_fn, callback)
        except ImportError:
            return self._optimize_simple(objective_fn, callback)

    def _optimize_with_skopt(self, objective_fn, callback) -> OptimizationResult:
        from skopt import gp_minimize
        from skopt.space import Real

        space = [
            Real(low, high, name=name)
            for name, (low, high) in zip(self.config.variable_names, self.config.bounds)
        ]

        iteration_count = [0]

        def wrapped_objective(params_list):
            params_dict = {
                name: val
                for name, val in zip(self.config.variable_names, params_list)
            }
            value = objective_fn(params_dict)
            self.history.append({'params': params_dict.copy(), 'value': value})
            self.convergence.append(value)

            if self.config.maximize:
                if value > self.best_value:
                    self.best_value = value
                    self.best_params = params_dict.copy()
            else:
                if value < self.best_value:
                    self.best_value = value
                    self.best_params = params_dict.copy()

            self.best_so_far.append(self.best_value)
            iteration_count[0] += 1
            if callback:
                callback(iteration_count[0], params_dict, value)

            return -value if self.config.maximize else value

        gp_minimize(
            wrapped_objective,
            space,
            n_calls=self.config.n_iterations,
            n_initial_points=self.config.n_initial_points,
            random_state=self.config.random_state,
            acq_func='EI',
            verbose=False
        )

        return OptimizationResult(
            best_params=self.best_params,
            best_value=self.best_value,
            all_trials=self.history,
            convergence=self.convergence,
            best_so_far=self.best_so_far,
            variable_names=self.config.variable_names,
            success=True,
            message=f"Bayesian optimization completed: {len(self.history)} evaluations"
        )

    def _optimize_simple(self, objective_fn, callback) -> OptimizationResult:
        """Latin Hypercube + random search fallback."""
        np.random.seed(self.config.random_state)

        for i in range(self.config.n_iterations):
            params_dict = {}
            for name, (low, high) in zip(self.config.variable_names, self.config.bounds):
                params_dict[name] = np.random.uniform(low, high)

            value = objective_fn(params_dict)
            self.history.append({'params': params_dict.copy(), 'value': value})
            self.convergence.append(value)

            if self.config.maximize:
                if value > self.best_value:
                    self.best_value = value
                    self.best_params = params_dict.copy()
            else:
                if value < self.best_value:
                    self.best_value = value
                    self.best_params = params_dict.copy()

            self.best_so_far.append(self.best_value)
            if callback:
                callback(i + 1, params_dict, value)

        return OptimizationResult(
            best_params=self.best_params,
            best_value=self.best_value,
            all_trials=self.history,
            convergence=self.convergence,
            best_so_far=self.best_so_far,
            variable_names=self.config.variable_names,
            success=True,
            message="Random search fallback (install scikit-optimize for GP-based optimization)"
        )


class SensitivityAnalyzer:
    """One-at-a-time sensitivity analysis."""

    def __init__(self, variable_names: List[str], bounds: List[Tuple[float, float]]):
        self.variable_names = variable_names
        self.bounds = bounds

    def analyze(
        self,
        objective_fn: Callable[[Dict[str, float]], float],
        n_samples: int = 20,
        callback: Optional[Callable] = None
    ) -> Dict[str, float]:
        base_params = {
            name: (low + high) / 2
            for name, (low, high) in zip(self.variable_names, self.bounds)
        }

        importance_scores = {}
        total_iterations = len(self.variable_names) * n_samples
        current_iter = 0

        for var_name, (low, high) in zip(self.variable_names, self.bounds):
            values = []
            test_values = np.linspace(low, high, n_samples)

            for test_val in test_values:
                params = base_params.copy()
                params[var_name] = test_val
                result = objective_fn(params)
                values.append(result)

                current_iter += 1
                if callback:
                    callback(current_iter, total_iterations, var_name, test_val, result)

            importance_scores[var_name] = np.std(values)

        total = sum(importance_scores.values())
        if total > 0:
            importance_scores = {k: v / total for k, v in importance_scores.items()}

        return importance_scores


# ============================================================================
# GENERIC OBJECTIVE FUNCTION FACTORY
# ============================================================================

# Available optimization variables with display names and unit conversions
OPTIMIZATION_VARIABLES = {
    'temperature_C': {
        'display': 'Temperature (C)',
        'default_bounds': (600, 1000),
        'unit': 'C',
    },
    'flow_rate_mL_min': {
        'display': 'Flow Rate (mL/min)',
        'default_bounds': (20, 200),
        'unit': 'mL/min',
    },
    'particle_size_um': {
        'display': 'Particle Size (um)',
        'default_bounds': (100, 5000),
        'unit': 'um',
    },
    'bed_height_cm': {
        'display': 'Bed Height (cm)',
        'default_bounds': (1.0, 20.0),
        'unit': 'cm',
    },
    'catalyst_mass_g': {
        'display': 'Catalyst Mass (g)',
        'default_bounds': (0.5, 50.0),
        'unit': 'g',
    },
    'pre_exponential_log': {
        'display': 'Pre-exponential (log10 A)',
        'default_bounds': (3.0, 8.0),
        'unit': 'log10',
    },
    'activation_energy_kJ': {
        'display': 'Activation Energy (kJ/mol)',
        'default_bounds': (80, 250),
        'unit': 'kJ/mol',
    },
    'bed_porosity': {
        'display': 'Bed Porosity',
        'default_bounds': (0.3, 0.5),
        'unit': '-',
    },
    'LD_ratio': {
        'display': 'L/D Ratio',
        'default_bounds': (1.0, 3.0),
        'unit': '-',
    },
    'ind_flow_rate_LPM': {
        'display': 'Ind. Flow Rate (LPM)',
        'default_bounds': (500, 5000),
        'unit': 'LPM',
    },
    'ind_particle_size_um': {
        'display': 'Ind. Particle Size (um)',
        'default_bounds': (500, 5000),
        'unit': 'um',
    },
    'ind_catalyst_mass_kg': {
        'display': 'Ind. Catalyst Mass (kg)',
        'default_bounds': (10, 200),
        'unit': 'kg',
    },
}

OPTIMIZATION_OBJECTIVES = {
    'product_percent': 'Maximize Product (%)',
    'conversion_percent': 'Maximize Conversion (%)',
    'min_pressure_drop': 'Minimize Pressure Drop (kPa)',
    'product_per_dP': 'Maximize Product/dP Ratio',
}


def create_lab_objective(
    reaction,
    base_config_dict: dict,
    variable_names: List[str],
    target: str = 'product_percent',
):
    """
    Create objective function for lab-scale optimization.

    Args:
        reaction: Reaction object
        base_config_dict: dict with base reactor config values
        variable_names: list of variable names to optimize
        target: objective target key
    """
    from .transient_model import ReactorConfig, SteadyStateReactor

    def objective(params: Dict[str, float]) -> float:
        cfg = base_config_dict.copy()

        for name, value in params.items():
            if name == 'temperature_C':
                cfg['temperature_K'] = value + 273.15
            elif name == 'flow_rate_mL_min':
                cfg['flow_rate_m3_s'] = value / 60 / 1e6
            elif name == 'particle_size_um':
                cfg['particle_diameter_m'] = value * 1e-6
            elif name == 'bed_height_cm':
                cfg['bed_height_m'] = value / 100
            elif name == 'catalyst_mass_g':
                cfg['catalyst_mass_kg'] = value / 1000
            elif name == 'pre_exponential_log':
                cfg['pre_exponential'] = 10 ** value
            elif name == 'activation_energy_kJ':
                cfg['activation_energy'] = value * 1000
            elif name == 'bed_porosity':
                cfg['bed_porosity'] = value
            else:
                cfg[name] = value

        try:
            config = ReactorConfig(
                diameter_m=cfg['diameter_m'],
                bed_height_m=cfg['bed_height_m'],
                particle_diameter_m=cfg['particle_diameter_m'],
                particle_density=cfg['particle_density'],
                particle_porosity=cfg['particle_porosity'],
                tortuosity=cfg['tortuosity'],
                bed_porosity=cfg['bed_porosity'],
                catalyst_mass_kg=cfg['catalyst_mass_kg'],
                temperature_K=cfg['temperature_K'],
                inlet_pressure_Pa=cfg['inlet_pressure_Pa'],
                flow_rate_m3_s=cfg['flow_rate_m3_s'],
                reaction=reaction,
                inlet_mole_fractions=cfg['inlet_mole_fractions'],
                pre_exponential=cfg['pre_exponential'],
                activation_energy=cfg['activation_energy'],
            )
            reactor = SteadyStateReactor(config)
            results = reactor.solve(activity=1.0)

            if target == 'product_percent':
                return float(results['outlet_product_percent'])
            elif target == 'conversion_percent':
                return float(results['outlet_conversion'])
            elif target == 'min_pressure_drop':
                return float(-results['pressure_drop_kPa'])
            elif target == 'product_per_dP':
                dp = max(results['pressure_drop_kPa'], 0.01)
                return float(results['outlet_product_percent'] / dp)
            else:
                return float(results.get(target, 0))
        except Exception:
            return 0.0

    return objective


def create_industrial_objective(
    reaction,
    base_config_dict: dict,
    variable_names: List[str],
    target: str = 'product_percent',
):
    """
    Create objective function for industrial-scale optimization.
    """
    from .transient_model import ReactorConfig, SteadyStateReactor

    def objective(params: Dict[str, float]) -> float:
        cfg = base_config_dict.copy()

        for name, value in params.items():
            if name == 'temperature_C':
                cfg['temperature_K'] = value + 273.15
            elif name == 'ind_flow_rate_LPM':
                cfg['flow_rate_m3_s'] = value / 60 / 1000
            elif name == 'ind_particle_size_um':
                cfg['particle_diameter_m'] = value * 1e-6
            elif name == 'ind_catalyst_mass_kg':
                cfg['catalyst_mass_kg'] = value
            elif name == 'LD_ratio':
                V_bed = cfg['catalyst_mass_kg'] / (cfg['particle_density'] * (1 - cfg['bed_porosity']))
                D = (4 * V_bed / (np.pi * value)) ** (1/3)
                L = value * D
                cfg['diameter_m'] = D
                cfg['bed_height_m'] = L
            elif name == 'pre_exponential_log':
                cfg['pre_exponential'] = 10 ** value
            elif name == 'activation_energy_kJ':
                cfg['activation_energy'] = value * 1000
            elif name == 'bed_porosity':
                cfg['bed_porosity'] = value
            else:
                cfg[name] = value

        try:
            config = ReactorConfig(
                diameter_m=cfg['diameter_m'],
                bed_height_m=cfg['bed_height_m'],
                particle_diameter_m=cfg['particle_diameter_m'],
                particle_density=cfg['particle_density'],
                particle_porosity=cfg['particle_porosity'],
                tortuosity=cfg['tortuosity'],
                bed_porosity=cfg['bed_porosity'],
                catalyst_mass_kg=cfg['catalyst_mass_kg'],
                temperature_K=cfg['temperature_K'],
                inlet_pressure_Pa=cfg['inlet_pressure_Pa'],
                flow_rate_m3_s=cfg['flow_rate_m3_s'],
                reaction=reaction,
                inlet_mole_fractions=cfg['inlet_mole_fractions'],
                pre_exponential=cfg['pre_exponential'],
                activation_energy=cfg['activation_energy'],
            )
            reactor = SteadyStateReactor(config)
            results = reactor.solve(activity=1.0)

            if target == 'product_percent':
                return float(results['outlet_product_percent'])
            elif target == 'conversion_percent':
                return float(results['outlet_conversion'])
            elif target == 'min_pressure_drop':
                return float(-results['pressure_drop_kPa'])
            elif target == 'product_per_dP':
                dp = max(results['pressure_drop_kPa'], 0.01)
                return float(results['outlet_product_percent'] / dp)
            else:
                return float(results.get(target, 0))
        except Exception:
            return 0.0

    return objective
