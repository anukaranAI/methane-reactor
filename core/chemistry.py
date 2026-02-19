"""
Chemistry abstraction layer for KinetiScale.

Defines generic dataclasses for chemical species, reactions, and reaction templates
that allow the reactor simulator to work with any single catalytic reaction.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


R_GAS = 8.314  # J/(mol·K)


@dataclass
class Species:
    """Properties of a single chemical species."""
    name: str               # Short identifier, e.g. "CH4"
    display_name: str        # Human-readable, e.g. "Methane"
    formula: str             # Chemical formula, e.g. "CH₄"
    MW: float                # Molecular weight [kg/mol], e.g. 16.04e-3
    phase: str               # "gas" or "solid"

    # Viscosity correlation: mu = mu_ref * (T / mu_T_ref)^mu_exp  [Pa·s]
    mu_ref: float = 0.0
    mu_T_ref: float = 300.0
    mu_exp: float = 0.7

    # Heat capacity polynomial: Cp = Cp_a + Cp_b*T + Cp_c*T^2  [J/(mol·K)]
    Cp_a: float = 30.0
    Cp_b: float = 0.0
    Cp_c: float = 0.0

    # Diffusivity correlation: D = D_ref * (T/D_T_ref)^D_exp * (D_P_ref/P)  [m²/s]
    D_ref: float = 1.87e-5
    D_T_ref: float = 300.0
    D_exp: float = 1.75
    D_P_ref: float = 101325.0

    def viscosity(self, T: float) -> float:
        """Gas viscosity at temperature T [K]. Returns [Pa·s]."""
        if self.mu_ref <= 0:
            return 1.0e-5  # fallback
        return self.mu_ref * (T / self.mu_T_ref) ** self.mu_exp

    def heat_capacity(self, T: float) -> float:
        """Molar heat capacity at temperature T [K]. Returns [J/(mol·K)]."""
        return self.Cp_a + self.Cp_b * T + self.Cp_c * T ** 2

    def diffusivity(self, T: float, P: float) -> float:
        """Molecular diffusivity at T [K] and P [Pa]. Returns [m²/s]."""
        return self.D_ref * (T / self.D_T_ref) ** self.D_exp * (self.D_P_ref / P)


@dataclass
class Reaction:
    """
    Definition of a single catalytic reaction.

    Stoichiometric coefficients: negative for reactants, positive for products.
    Only gas-phase species participate in the ODE system.
    Solid products (e.g. carbon) are tracked for deposition but do not
    contribute to gas-phase molar flow.
    """
    name: str
    display_name: str                      # e.g. "Methane Decomposition"
    stoichiometry: Dict[str, float]        # {"CH4": -1, "C": +1, "H2": +2}
    species: Dict[str, Species]            # All participating species, keyed by name
    key_reactant: str                      # Species whose conversion is tracked
    key_product: str                       # Primary product of interest
    delta_H: float = 0.0                   # Heat of reaction [J/mol key_reactant], +ve = endothermic
    inert: str = "N2"                      # Inert/diluent species name

    @property
    def gas_species(self) -> List[str]:
        """Ordered list of gas-phase species names (reactants first, then products, then inert)."""
        reactants = [s for s, coeff in self.stoichiometry.items()
                     if coeff < 0 and self.species[s].phase == "gas"]
        products = [s for s, coeff in self.stoichiometry.items()
                    if coeff > 0 and self.species[s].phase == "gas"]
        inerts = [s for s in self.species
                  if self.species[s].phase == "gas" and s not in self.stoichiometry]
        # Also include inert if it's in stoichiometry with coeff 0
        inerts += [s for s, coeff in self.stoichiometry.items()
                   if coeff == 0 and self.species[s].phase == "gas" and s not in inerts]
        return reactants + products + inerts

    @property
    def solid_species(self) -> List[str]:
        """List of solid-phase species names."""
        return [s for s in self.species if self.species[s].phase == "solid"]

    @property
    def gas_stoich_vector(self) -> np.ndarray:
        """Stoichiometric coefficients for gas species only, in gas_species order."""
        gas = self.gas_species
        return np.array([self.stoichiometry.get(s, 0.0) for s in gas])

    @property
    def n_gas(self) -> int:
        return len(self.gas_species)

    def gas_species_index(self, name: str) -> int:
        return self.gas_species.index(name)

    def format_equation(self) -> str:
        """Format reaction as a readable equation string, e.g. 'CH₄ → C + 2H₂'."""
        reactant_parts = []
        product_parts = []
        for sp_name, coeff in self.stoichiometry.items():
            formula = self.species[sp_name].formula
            abs_coeff = abs(coeff)
            if abs_coeff == 1.0:
                part = formula
            elif abs_coeff == int(abs_coeff):
                part = f"{int(abs_coeff)}{formula}"
            else:
                part = f"{abs_coeff}{formula}"

            if coeff < 0:
                reactant_parts.append(part)
            elif coeff > 0:
                product_parts.append(part)

        return " + ".join(reactant_parts) + " → " + " + ".join(product_parts)


@dataclass
class ReactionTemplate:
    """Pre-built reaction with default parameters for quick start."""
    reaction: Reaction

    # Default kinetic parameters
    default_A: float = 5.0e4            # Pre-exponential factor [1/s]
    default_Ea: float = 150000.0        # Activation energy [J/mol]
    default_beta: float = 0.0           # Temperature exponent

    # Default deactivation
    default_kd: float = 0.008           # Deactivation rate constant [1/min]
    default_deact_order: int = 1        # Deactivation order

    # Default inlet composition: {species_name: mole_fraction}
    default_inlet: Dict[str, float] = field(default_factory=dict)

    # Default operating conditions
    default_T_C: float = 800.0          # Temperature [°C]
    default_P_bar: float = 1.0          # Pressure [bar]

    # Default lab scale
    default_lab_flow_mL_min: float = 75.0
    default_lab_diameter_cm: float = 1.8
    default_lab_height_cm: float = 2.59
    default_lab_catalyst_mass_g: float = 2.2
    default_lab_particle_size_um: float = 10.0
    default_lab_particle_density: float = 2821.56

    # Default industrial scale
    default_ind_flow_LPM: float = 1960.0
    default_ind_catalyst_mass_kg: float = 57.0
    default_ind_particle_size_um: float = 3000.0
    default_ind_particle_density: float = 2821.56
    default_ind_target_product_pct: float = 30.0

    # Optional experimental data for this reaction
    # Format: {temperature_C: {"TOS_min": [...], "product_percent": [...], "reactant_percent": [...]}}
    experimental_data: Optional[Dict] = None

    description: str = ""
