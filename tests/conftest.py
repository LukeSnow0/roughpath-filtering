"""Shared pytest configuration for the cubature-filtering test suite."""

import jax

# Match experiment scripts and examples: CRF numerics assume float64.
jax.config.update("jax_enable_x64", True)
