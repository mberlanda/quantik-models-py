"""Quantik policy/value networks, training, evaluation and play.

Published weights for the four architectures live on the Hugging Face Hub
rather than in this wheel — see `quantik_models.hub` and `docs/models.md`.
"""

# The single source of truth for the version. `pyproject.toml` reads this
# attribute statically (`[tool.setuptools.dynamic]`), so the number is
# declared once and cannot drift between the package and its metadata.
__version__ = "1.0.0"

__all__ = ["__version__"]
