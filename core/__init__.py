"""
Anukaran AI - Core Module
=========================
Reactor simulation, optimization, and scale-up.
"""

from .templates import (
    OPTIMIZATION_TEMPLATES,
    get_template,
    get_template_names,
    get_variable_bounds,
    get_variable_names
)

from .optimizer import (
    OptimizationResult,
    OptimizationConfig,
    BayesianOptimizer,
    SensitivityAnalyzer,
    create_objective_function,
    get_base_config_from_session
)

from .transient_model import (
    DeactivationParams,
    DeactivationModel,
    ReactorConfig,
    SteadyStateReactor,
    TransientReactor,
    calibrate_kinetics,
    fit_deactivation,
)

from .experimental_data import (
    EXPERIMENTAL_DATA,
    TOS_MINUTES,
    LabScaleConfig,
    IndustrialScaleConfig,
    LAB_CONFIG,
    INDUSTRIAL_CONFIG,
    get_experimental_data,
    get_available_temperatures,
    get_initial_values,
    get_all_initial_values,
    get_scale_comparison,
)

from .scaleup import (
    ScaleUpResult,
    IndustrialScaleUp,
    get_assumptions_text,
)

__all__ = [
    # Templates
    'OPTIMIZATION_TEMPLATES', 'get_template', 'get_template_names',
    'get_variable_bounds', 'get_variable_names',
    # Optimizer
    'OptimizationResult', 'OptimizationConfig', 'BayesianOptimizer',
    'SensitivityAnalyzer', 'create_objective_function', 'get_base_config_from_session',
    # Transient
    'DeactivationParams', 'DeactivationModel', 'ReactorConfig',
    'SteadyStateReactor', 'TransientReactor',
    'calibrate_kinetics', 'fit_deactivation',
    # Data
    'EXPERIMENTAL_DATA', 'TOS_MINUTES', 'LabScaleConfig', 'IndustrialScaleConfig',
    'LAB_CONFIG', 'INDUSTRIAL_CONFIG', 'get_experimental_data',
    'get_available_temperatures', 'get_initial_values', 'get_all_initial_values',
    'get_scale_comparison',
    # Scale-up
    'ScaleUpResult', 'IndustrialScaleUp', 'get_assumptions_text',
]
