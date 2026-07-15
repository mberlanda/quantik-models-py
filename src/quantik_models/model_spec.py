"""Portable model shape notes kept executable for tests and tooling."""

TENSOR_SHAPE = (9, 4, 4)
ACTION_COUNT = 64
VALUE_RANGE = (-1.0, 1.0)
TARGET_MODEL_SIZE_MB = (50, 100)

MODEL_INPUT_CONTRACTS = ("tensor-board.v1", "bitboard.v1", "action-index.v1")
MODEL_OUTPUT_CONTRACT = "policy-value.v1"
