"""Dataset materialization utilities for Quantik models."""

from .materialize import TrainingDatasetView, load_npz, write_npz

__all__ = ["TrainingDatasetView", "load_npz", "write_npz"]
