"""
Anukaran AI - Core Module
=========================
Contains optimization engine, templates, and transient model.
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
    TransientReactorConfig,
    TransientReactor,
    create_transient_config,
    fit_deactivation_parameter,
    run_validation_at_temperature
)

from .experimental_data import (
    EXPERIMENTAL_DATA,
    TOS_MINUTES,
    ExperimentalConditions,
    DEFAULT_CONDITIONS,
    get_experimental_data,
    get_available_temperatures,
    get_initial_values,
    get_all_initial_values,
    interpolate_experimental
)

__all__ = [
    # Templates
    'OPTIMIZATION_TEMPLATES',
    'get_template',
    'get_template_names',
    'get_variable_bounds',
    'get_variable_names',
    # Optimizer
    'OptimizationResult',
    'OptimizationConfig',
    'BayesianOptimizer',
    'SensitivityAnalyzer',
    'create_objective_function',
    'get_base_config_from_session',
    # Transient model
    'DeactivationParams',
    'DeactivationModel',
    'TransientReactorConfig',
    'TransientReactor',
    'create_transient_config',
    'fit_deactivation_parameter',
    'run_validation_at_temperature',
    # Experimental data
    'EXPERIMENTAL_DATA',
    'TOS_MINUTES',
    'ExperimentalConditions',
    'DEFAULT_CONDITIONS',
    'get_experimental_data',
    'get_available_temperatures',
    'get_initial_values',
    'get_all_initial_values',
    'interpolate_experimental',
]
