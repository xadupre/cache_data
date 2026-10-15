#!/usr/bin/env python3
"""Build a binary QMoE expert-selection tensor from the published routing data."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("iteration", "qmoe_number", "expert_number", "position")


def parse_args():
    """Parses command-line arguments for tensor construction."""
    parser = argparse.ArgumentParser(
        description="Build S[t, m, i], the binary QMoE expert-selection tensor, from the routing CSV archive."
    )
    parser.add_argument(
        "archive", type=Path, help="ZIP archive containing the selected-experts CSV."
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output compressed NumPy .npz file."
    )
    parser.add_argument("--qmoe-count", type=int, default=40)
    parser.add_argument("--expert-count", type=int, default=256)
    parser.add_argument("--selected-expert-count", type=int, default=8)
    return parser.parse_args()


def read_selected_experts(archive):
    """Reads the selected-expert table directly from its ZIP archive.

    Args:
        archive: Path to a ZIP archive containing one routing CSV file.

    Returns:
        A pandas DataFrame containing the four required integer columns.
    """
    return pd.read_csv(
        archive,
        compression="zip",
        dtype={column: np.int64 for column in REQUIRED_COLUMNS},
        usecols=list(REQUIRED_COLUMNS),
    )


def build_selection_tensor(
    selected_experts,
    qmoe_count=40,
    expert_count=256,
    selected_expert_count=8,
):
    """Builds the binary expert-selection tensor ``S[t, m, i]``.

    The input uses one-based decode iterations and zero-based QMoE and expert
    numbers. The returned tensor uses zero-based NumPy axes, so CSV iteration
    ``t`` is stored at ``S[t - 1]``.

    Args:
        selected_experts: Routing rows with the columns in ``REQUIRED_COLUMNS``.
        qmoe_count: Expected number of QMoE layers.
        expert_count: Number of experts in each QMoE layer.
        selected_expert_count: Expected selections per iteration and QMoE.

    Returns:
        A uint8 tensor of shape ``(iterations, qmoe_count, expert_count)``.
        An entry is one exactly when the corresponding expert was selected.

    Raises:
        ValueError: If the table is empty, incomplete, duplicated, or contains
            indices outside the configured tensor dimensions.
    """
    missing_columns = set(REQUIRED_COLUMNS) - set(selected_experts.columns)
    if missing_columns:
        raise ValueError(f"Missing CSV columns: {sorted(missing_columns)}.")
    if qmoe_count <= 0 or expert_count <= 0 or selected_expert_count <= 0:
        raise ValueError(
            "Tensor dimensions and selected expert count must be positive."
        )
    if selected_experts.empty:
        raise ValueError("The selected-expert table is empty.")

    iterations = selected_experts["iteration"].to_numpy()
    qmoes = selected_experts["qmoe_number"].to_numpy()
    experts = selected_experts["expert_number"].to_numpy()
    positions = selected_experts["position"].to_numpy()
    iteration_count = int(iterations.max())

    if iterations.min() != 1 or set(np.unique(iterations)) != set(
        range(1, iteration_count + 1)
    ):
        raise ValueError("Iterations must be consecutive and one-based.")
    if (
        qmoes.min() != 0
        or qmoes.max() >= qmoe_count
        or np.unique(qmoes).size != qmoe_count
    ):
        raise ValueError(f"QMoE numbers must cover [0, {qmoe_count}).")
    if experts.min() < 0 or experts.max() >= expert_count:
        raise ValueError(f"Expert numbers must be in [0, {expert_count}).")
    if positions.min() < 0 or positions.max() >= qmoe_count * expert_count:
        raise ValueError(
            f"Global positions must be in [0, {qmoe_count * expert_count})."
        )
    if selected_experts.duplicated(["iteration", "qmoe_number", "expert_number"]).any():
        raise ValueError(
            "An expert must not be selected twice by the same QMoE at one iteration."
        )

    selection_counts = selected_experts.groupby(
        ["iteration", "qmoe_number"], sort=False
    ).size()
    expected_group_count = iteration_count * qmoe_count
    if (
        selection_counts.size != expected_group_count
        or not (selection_counts == selected_expert_count).all()
    ):
        raise ValueError(
            f"Every iteration/QMoE pair must contain exactly {selected_expert_count} selected experts."
        )

    tensor = np.zeros((iteration_count, qmoe_count, expert_count), dtype=np.uint8)
    tensor[iterations - 1, qmoes, experts] = 1
    return tensor


def main():
    """Loads the routing archive, builds ``S``, and writes a compressed NPZ."""
    args = parse_args()
    selected_experts = read_selected_experts(args.archive)
    tensor = build_selection_tensor(
        selected_experts,
        qmoe_count=args.qmoe_count,
        expert_count=args.expert_count,
        selected_expert_count=args.selected_expert_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        S=tensor,
        iterations=np.arange(1, tensor.shape[0] + 1, dtype=np.int64),
        qmoes=np.arange(tensor.shape[1], dtype=np.int64),
        experts=np.arange(tensor.shape[2], dtype=np.int64),
    )
    print(f"S.shape={tensor.shape}")
    print(f"S.dtype={tensor.dtype}")
    print(f"S.sum={int(tensor.sum())}")
    print(args.output)


if __name__ == "__main__":
    main()
