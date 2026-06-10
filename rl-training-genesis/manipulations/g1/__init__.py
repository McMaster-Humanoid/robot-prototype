"""
G1 Manipulation Environment Package
Genesis-based manipulation environment for Unitree G1 humanoid
"""

from .env import G1ManipulationEnv, make_env

__version__ = "0.1.0"
__all__ = ["G1ManipulationEnv", "make_env"]
