import taichi as ti
import pytest


def pytest_configure(config):
    """Initialise Taichi once for the entire test session."""
    ti.init(arch=ti.cpu)
