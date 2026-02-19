"""
Generic transport property functions for KinetiScale.

These functions compute mixture properties from Species objects and mole fraction
arrays, replacing the hardcoded CH4/H2/N2-specific functions in the old code.
"""

import numpy as np
from typing import List
from .chemistry import Species, R_GAS


def gas_mixture_viscosity(T: float, y: np.ndarray, species_list: List[Species]) -> float:
    """
    Gas mixture viscosity by linear mole-fraction weighting.

    Args:
        T: Temperature [K]
        y: Mole fractions array (same order as species_list)
        species_list: List of Species objects

    Returns:
        Mixture viscosity [Pa·s]
    """
    mu = 0.0
    for y_i, sp in zip(y, species_list):
        mu += y_i * sp.viscosity(T)
    return max(mu, 1e-7)


def gas_mixture_density(T: float, P: float, y: np.ndarray, species_list: List[Species]) -> float:
    """
    Gas mixture density from ideal gas law.

    Args:
        T: Temperature [K]
        P: Pressure [Pa]
        y: Mole fractions array
        species_list: List of Species objects

    Returns:
        Mixture density [kg/m³]
    """
    MW_mix = sum(y_i * sp.MW for y_i, sp in zip(y, species_list))
    return P * MW_mix / (R_GAS * T)


def gas_mixture_Cp(T: float, y: np.ndarray, species_list: List[Species]) -> float:
    """
    Gas mixture molar heat capacity.

    Args:
        T: Temperature [K]
        y: Mole fractions array
        species_list: List of Species objects

    Returns:
        Mixture Cp [J/(mol·K)]
    """
    return sum(y_i * sp.heat_capacity(T) for y_i, sp in zip(y, species_list))


def effective_diffusivity(T: float, P: float, key_species: Species,
                          particle_porosity: float, tortuosity: float) -> float:
    """
    Effective diffusivity for key reactant through porous catalyst.

    D_eff = D_mol * ε_p / τ

    Args:
        T: Temperature [K]
        P: Pressure [Pa]
        key_species: The reactant Species object
        particle_porosity: Catalyst particle porosity [-]
        tortuosity: Catalyst pore tortuosity [-]

    Returns:
        Effective diffusivity [m²/s]
    """
    D_mol = key_species.diffusivity(T, P)
    return D_mol * particle_porosity / tortuosity


def ergun_pressure_drop(u: float, rho: float, mu: float,
                        d_p: float, epsilon: float) -> float:
    """
    Ergun equation for pressure drop in packed bed.

    -dP/dz = 150*mu*(1-eps)^2 / (d_p^2 * eps^3) * u
           + 1.75*rho*(1-eps) / (d_p * eps^3) * u^2

    Args:
        u: Superficial velocity [m/s]
        rho: Gas density [kg/m³]
        mu: Gas viscosity [Pa·s]
        d_p: Particle diameter [m]
        epsilon: Bed porosity [-]

    Returns:
        Pressure gradient dP/dz [Pa/m] (positive value)
    """
    if d_p <= 0 or epsilon <= 0 or epsilon >= 1:
        return 0.0

    viscous = 150 * mu * (1 - epsilon) ** 2 / (d_p ** 2 * epsilon ** 3) * abs(u)
    inertial = 1.75 * rho * (1 - epsilon) / (d_p * epsilon ** 3) * u ** 2
    return viscous + inertial
