"""Compatibility aliases for the old UltraNeRF prototype module."""

from .field import PositionalEncoding, UltraNeRFField


NeuralRadianceField = UltraNeRFField


class RenderParameter:
    def __init__(self, parameter_dict):
        self.attenuation_coefficient = parameter_dict["alpha"]
        self.reflection_coefficient = parameter_dict["beta"]
        self.border_probability = parameter_dict["rho_b"]
        self.scattering_density_coefficient = parameter_dict["rho_s"]
        self.scattering_amplitude = parameter_dict["phi"]
