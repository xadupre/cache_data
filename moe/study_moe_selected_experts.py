#!/usr/bin/env python3
"""Build a binary QMoE expert-selection tensor from the published routing data."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression

REQUIRED_COLUMNS = ("iteration", "qmoe_number", "expert_number", "position")
PROMPT_ITERATION_COUNTS = (216, 221, 253, 430, 407, 664, 450, 500, 646, 736)
RANDOM_BASELINE_SEEDS = (0, 1, 2)
OPTIMIZATION_ALPHA_MIN = np.nextafter(0.5, 1.0)
MATRIX_INITIAL_ALPHA = 0.719428897297


def parse_args(argv=None):
    """Parses command-line arguments for tensor construction.

    Args:
        argv: Command-line arguments without the program name. Pass ``None``
            to read them from ``sys.argv[1:]``.

    Returns:
        An ``argparse.Namespace`` containing:

        * ``archive``: input ZIP archive as a ``Path``;
        * ``qmoe_count``: expected number of QMoE layers;
        * ``expert_count``: expected number of experts per QMoE;
        * ``selected_expert_count``: expected selections per iteration/QMoE.
        * ``optimization``: selected optimization family;
        * ``alpha_grid_size``: number of coarse alpha candidates.
        * ``matrix_epochs``: number of matrix-regression training epochs;
        * ``matrix_learning_rate``: matrix-regression Adam learning rate.
        * ``markdown_output``: Markdown results file.

    Raises:
        SystemExit: If required arguments are missing or an argument is invalid.
    """
    parser = argparse.ArgumentParser(
        description="Build S[t, m, i], the binary QMoE expert-selection tensor, from the routing CSV archive.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  Run the scalar-alpha optimizations:
    python study_moe_selected_experts.py \\
      moe-no-offload-unbounded-expert-activation-correlations-selected-experts.zip

  Optimize alpha, W1, and W2 instead:
    python study_moe_selected_experts.py \\
      moe-no-offload-unbounded-expert-activation-correlations-selected-experts.zip \\
      --optimization matrix
""",
    )
    parser.add_argument(
        "archive",
        type=Path,
        help="ZIP archive containing one CSV with iteration, qmoe_number, expert_number, and position columns.",
    )
    parser.add_argument(
        "--qmoe-count",
        type=int,
        default=40,
        help="Expected number of QMoE layers (default: %(default)s).",
    )
    parser.add_argument(
        "--expert-count",
        type=int,
        default=256,
        help="Expected number of experts in each QMoE layer (default: %(default)s).",
    )
    parser.add_argument(
        "--selected-expert-count",
        type=int,
        default=8,
        help="Expected selected experts per iteration and QMoE (default: %(default)s).",
    )
    parser.add_argument(
        "--optimization",
        choices=("alpha", "matrix", "none"),
        default="alpha",
        help="Run scalar-alpha methods, joint alpha/W1/W2 matrix training, or only build S (default: %(default)s).",
    )
    parser.add_argument(
        "--alpha-grid-size",
        type=int,
        default=101,
        help="Alpha grid size used to locate candidate minima before bounded refinement (default: %(default)s).",
    )
    parser.add_argument(
        "--matrix-epochs",
        type=int,
        default=100,
        help="Training epochs for alpha, W1, and W2 matrix regression (default: %(default)s).",
    )
    parser.add_argument(
        "--matrix-learning-rate",
        type=float,
        default=0.005,
        help="Adam learning rate for alpha, W1, and W2 matrix regression (default: %(default)s).",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path(__file__).with_suffix(".md"),
        help="Markdown results file (default: the script path with extension .md).",
    )
    return parser.parse_args(argv)


def read_selected_experts(archive):
    """Reads the selected-expert table directly from its ZIP archive.

    The CSV stored in the archive begins with::

        iteration,qmoe_number,expert_number,position
        1,0,236,112
        1,0,43,12
        1,0,30,7
        1,0,191,95
        1,0,124,2850
        1,0,73,2827
        1,0,188,94
        1,0,238,114
        1,1,33,18
        1,1,86,31

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


def validate_selection_tensor(selection_tensor):
    """Validates and returns a binary selection tensor.

    Args:
        selection_tensor: Array expected to represent ``S[t, m, i]``.

    Returns:
        The input converted to a NumPy array without copying when possible.

    Raises:
        ValueError: If the tensor is not three-dimensional, is empty, or
            contains values other than zero and one.
    """
    selection_tensor = np.asarray(selection_tensor)
    if selection_tensor.ndim != 3 or 0 in selection_tensor.shape:
        raise ValueError(
            "The selection tensor must have non-empty (iteration, QMoE, expert) axes."
        )
    if not np.logical_or(selection_tensor == 0, selection_tensor == 1).all():
        raise ValueError("The selection tensor must contain only zero and one.")
    return selection_tensor


def popularity_error(selection_tensor, popularity_tensor):
    """Computes ``mean(t,m) sum(i) (P(t,m,i) - S(t,m,i))**2``.

    Args:
        selection_tensor: Binary tensor ``S`` with shape
            ``(iterations, qmoes, experts)``.
        popularity_tensor: Floating-point tensor ``P`` with the same shape as
            ``S``.

    Returns:
        The mean squared prediction error per iteration and QMoE. It lies in
        ``[0, expert_count]``.

    Raises:
        ValueError: If either tensor is invalid.
    """
    selection_tensor = validate_selection_tensor(selection_tensor)
    popularity_tensor = np.asarray(popularity_tensor)
    if popularity_tensor.shape != selection_tensor.shape:
        raise ValueError("P must have the same shape as S.")
    if not np.issubdtype(popularity_tensor.dtype, np.floating):
        raise ValueError("P must have a floating-point dtype.")
    if not np.isfinite(popularity_tensor).all():
        raise ValueError("P must contain only finite values.")

    error = 0.0
    for selection, popularity in zip(selection_tensor, popularity_tensor, strict=True):
        error += np.square(popularity - selection).sum()
    return float(error / (selection_tensor.shape[0] * selection_tensor.shape[1]))


def compute_popularity(selection_tensor, alpha):
    """Computes the popularity tensor ``P`` from ``S`` and ``alpha``.

    Popularity starts at zero and follows
    ``P(t) = alpha * P(t-1) + (1-alpha) * S(t-1)``. Consequently, ``P[0]`` is
    zero and has the same QMoE and expert axes as ``S[0]``.

    Args:
        selection_tensor: Binary tensor ``S`` with shape
            ``(iterations, qmoes, experts)``.
        alpha: Popularity coefficient in the closed interval ``[0, 1]``.

    Returns:
        Floating-point popularity tensor ``P`` with the same shape as ``S``.

    Raises:
        ValueError: If ``S`` is invalid or ``alpha`` is outside ``[0, 1]``.
    """
    selection_tensor = validate_selection_tensor(selection_tensor)
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and in [0, 1].")

    popularity_tensor = np.empty(selection_tensor.shape, dtype=np.float64)
    popularity = np.zeros(selection_tensor.shape[1:], dtype=np.float64)
    for iteration, selection in enumerate(selection_tensor):
        popularity_tensor[iteration] = popularity
        popularity *= alpha
        popularity += (1 - alpha) * selection
    return popularity_tensor


def moe_routing_error(
    selection_tensor,
    popularity_tensor,
    gpu_expert_count,
    iteration_ranges=None,
):
    """Computes the simulated QMoE routing error from ``S`` and ``P``.

    At every iteration, all experts in ``P[t, :, :]`` are sorted together by
    increasing popularity, with the flattened expert position breaking ties.
    The last ``gpu_expert_count`` experts, which have the highest popularity,
    are retained on GPU and define the binary prediction ``Q``.
    The error is evaluated only for the experts selected in ``S``:
    ``mean(t,m) sum(i) S(t,m,i) * |Q(t,m,i) - S(t,m,i)|``.

    Args:
        selection_tensor: Binary tensor ``S`` with shape
            ``(iterations, qmoes, experts)``.
        popularity_tensor: Floating-point tensor ``P`` with shape
            ``(iterations, qmoes, experts)``, or a fixed popularity
            distribution with shape ``(qmoes, experts)``.
        gpu_expert_count: Number ``K`` of highest-popularity experts retained
            on GPU across all QMoEs.
        iteration_ranges: Optional sequence of ``(start, end)`` iteration
            ranges. Pass ``None`` to average over every iteration.

    Returns:
        The mean number of selected experts missed by ``Q`` per iteration and
        QMoE. It lies in ``[0, selections_per_qmoe]``. When
        ``iteration_ranges`` is provided, returns one mean per range.

    Raises:
        ValueError: If either tensor, ``gpu_expert_count``, or an iteration
            range is invalid.
    """
    selection_tensor = validate_selection_tensor(selection_tensor)
    popularity_tensor = np.asarray(popularity_tensor)
    if popularity_tensor.shape not in (
        selection_tensor.shape,
        selection_tensor.shape[1:],
    ):
        raise ValueError("P must have the same shape as S or shape (qmoes, experts).")
    if not np.issubdtype(popularity_tensor.dtype, np.floating):
        raise ValueError("P must have a floating-point dtype.")
    if not np.isfinite(popularity_tensor).all():
        raise ValueError("P must contain only finite values.")
    experts_per_iteration = selection_tensor.shape[1] * selection_tensor.shape[2]
    if not 0 <= gpu_expert_count <= experts_per_iteration:
        raise ValueError(f"gpu_expert_count must be in [0, {experts_per_iteration}].")

    popularity_rows = popularity_tensor.reshape(-1, experts_per_iteration)
    sorted_experts = np.argsort(popularity_rows, axis=1, kind="stable")
    retained_experts = sorted_experts[:, experts_per_iteration - gpu_expert_count :]
    predicted_rows = np.zeros(popularity_rows.shape, dtype=np.uint8)
    np.put_along_axis(predicted_rows, retained_experts, 1, axis=1)

    selection_rows = selection_tensor.reshape(
        selection_tensor.shape[0], experts_per_iteration
    )
    missed_rows = (selection_rows == 1) & (predicted_rows == 0)
    errors = missed_rows.reshape(selection_tensor.shape).sum(axis=2)

    if iteration_ranges is None:
        return float(errors.mean())

    means = []
    for start, end in iteration_ranges:
        if not 0 <= start < end <= selection_tensor.shape[0]:
            raise ValueError("Every iteration range must be non-empty and within S.")
        means.append(float(errors[start:end].mean()))
    return tuple(means)


def optimize_popularity_alpha(
    selection_tensor, grid_size=101, xatol=1e-8, verbose=False
):
    """Finds the alpha that minimizes the mean squared popularity error.

    The objective is evaluated first on a uniform grid over ``(0.5, 1]``.
    Every grid-local minimum is then refined with bounded scalar minimization;
    the two interval endpoints are also retained as candidates. This avoids
    assuming that the recurrence objective is globally unimodal.

    Args:
        selection_tensor: Binary tensor ``S`` with shape
            ``(iterations, qmoes, experts)``.
        grid_size: Number of uniformly spaced alpha values used to locate
            candidate minima. It must be at least three.
        xatol: Absolute alpha tolerance used during bounded refinement.
        verbose: Whether to print grid and refinement progress.

    Returns:
        A tuple ``(alpha, error)`` containing the minimizing coefficient and
        its mean squared prediction error per iteration and QMoE.

    Raises:
        ValueError: If ``S`` is invalid, ``grid_size`` is smaller than three,
            or ``xatol`` is not positive.
    """
    selection_tensor = validate_selection_tensor(selection_tensor)
    if grid_size < 3:
        raise ValueError("grid_size must be at least 3.")
    if not np.isfinite(xatol) or xatol <= 0:
        raise ValueError("xatol must be finite and positive.")

    def objective(alpha):
        popularity = compute_popularity(selection_tensor, alpha)
        return popularity_error(selection_tensor, popularity)

    alphas = np.linspace(OPTIMIZATION_ALPHA_MIN, 1, grid_size)
    errors = np.empty(grid_size, dtype=np.float64)
    progress_step = max(1, grid_size // 10)
    for index, alpha in enumerate(alphas):
        errors[index] = objective(alpha)
        if verbose and (
            index == 0 or (index + 1) % progress_step == 0 or index + 1 == grid_size
        ):
            print(
                f"Alpha grid: {index + 1}/{grid_size} "
                f"(alpha={alpha:.6g}, mean_squared_error={errors[index]:.6g})",
                flush=True,
            )
    candidates = [
        (float(alphas[0]), float(errors[0])),
        (float(alphas[-1]), float(errors[-1])),
    ]
    intervals = []
    if errors[0] <= errors[1]:
        intervals.append((alphas[0], alphas[1]))
    for index in range(1, grid_size - 1):
        if errors[index] <= errors[index - 1] and errors[index] <= errors[index + 1]:
            intervals.append((alphas[index - 1], alphas[index + 1]))
    if errors[-1] <= errors[-2]:
        intervals.append((alphas[-2], alphas[-1]))
    for interval_index, (lower_bound, upper_bound) in enumerate(intervals, start=1):
        if verbose:
            print(
                f"Refining candidate {interval_index}/{len(intervals)} "
                f"in [{lower_bound:.6g}, {upper_bound:.6g}]...",
                flush=True,
            )
        result = minimize_scalar(
            objective,
            bounds=(lower_bound, upper_bound),
            method="bounded",
            options={"xatol": xatol},
        )
        candidates.append((float(result.x), float(result.fun)))
    return min(candidates, key=lambda candidate: (candidate[1], candidate[0]))


def optimize_routing_alpha(
    selection_tensor,
    gpu_expert_count=6144,
    grid_size=101,
    xatol=1e-5,
    verbose=False,
):
    """Finds the alpha minimizing routing error over all iterations.

    Args:
        selection_tensor: Binary tensor ``S`` with shape
            ``(iterations, qmoes, experts)``.
        gpu_expert_count: Number ``K`` of experts retained on GPU.
        grid_size: Number of uniformly spaced alpha values in ``(0.5, 1]``.
        xatol: Absolute tolerance for refinement around the best grid point.
        verbose: Whether to print grid progress.

    Returns:
        A tuple ``(alpha, error)`` containing the best coefficient and its mean
        routing error over all iterations and QMoEs.

    Raises:
        ValueError: If ``S`` is invalid, ``grid_size`` is smaller than three,
            or ``xatol`` is not positive.
    """
    selection_tensor = validate_selection_tensor(selection_tensor)
    if grid_size < 3:
        raise ValueError("grid_size must be at least 3.")
    if not np.isfinite(xatol) or xatol <= 0:
        raise ValueError("xatol must be finite and positive.")

    def objective(alpha):
        popularity = compute_popularity(selection_tensor, alpha)
        return moe_routing_error(
            selection_tensor,
            popularity,
            gpu_expert_count=gpu_expert_count,
        )

    alphas = np.linspace(OPTIMIZATION_ALPHA_MIN, 1, grid_size)
    errors = np.empty(grid_size, dtype=np.float64)
    progress_step = max(1, grid_size // 10)
    for index, alpha in enumerate(alphas):
        errors[index] = objective(alpha)
        if verbose and (
            index == 0 or (index + 1) % progress_step == 0 or index + 1 == grid_size
        ):
            print(
                f"Routing alpha grid: {index + 1}/{grid_size} "
                f"(alpha={alpha:.6g}, error={errors[index]:.6g})",
                flush=True,
            )

    best_index = min(range(grid_size), key=lambda index: (errors[index], alphas[index]))
    candidates = [(float(alphas[best_index]), float(errors[best_index]))]
    if 0 < best_index < grid_size - 1:
        lower_bound = float(alphas[best_index - 1])
        upper_bound = float(alphas[best_index + 1])
        if verbose:
            print(
                f"Refining routing alpha in [{lower_bound:.6g}, {upper_bound:.6g}]...",
                flush=True,
            )
        result = minimize_scalar(
            objective,
            bounds=(lower_bound, upper_bound),
            method="bounded",
            options={"xatol": xatol},
        )
        candidates.append((float(result.x), float(result.fun)))
    return min(candidates, key=lambda candidate: (candidate[1], candidate[0]))


def optimize_regression_alpha(
    selection_tensor,
    initial_alpha=0.9,
    tolerance=1e-10,
    max_iterations=100,
    verbose=False,
):
    """Fits alpha as a linear-regression coefficient until convergence.

    For a fixed popularity tensor, the regression model is
    ``S(t) - S(t-1) = alpha * (P(t-1) - S(t-1))``. Since ``P`` itself depends
    on alpha, the function alternates between computing ``P`` and fitting the
    no-intercept least-squares coefficient. Alpha is constrained to
    ``0.5 < alpha <= 1``.

    Args:
        selection_tensor: Binary tensor ``S`` with shape
            ``(iterations, qmoes, experts)``.
        initial_alpha: Coefficient used to construct the first ``P``.
        tolerance: Absolute convergence tolerance for alpha.
        max_iterations: Maximum number of regression iterations.
        verbose: Whether to print each fitted coefficient.

    Returns:
        The converged regression coefficient.

    Raises:
        ValueError: If an argument is invalid.
        RuntimeError: If the coefficient does not converge.
    """
    selection_tensor = validate_selection_tensor(selection_tensor)
    if (
        not np.isfinite(initial_alpha)
        or not OPTIMIZATION_ALPHA_MIN <= initial_alpha <= 1
    ):
        raise ValueError("initial_alpha must be finite and in (0.5, 1].")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")

    regression_target = (
        selection_tensor[1:].astype(np.float64) - selection_tensor[:-1]
    ).reshape(-1)
    alpha = float(initial_alpha)
    for iteration in range(1, max_iterations + 1):
        popularity = compute_popularity(selection_tensor, alpha)
        regression_input = (popularity[:-1] - selection_tensor[:-1]).reshape(-1, 1)
        model = LinearRegression(fit_intercept=False, copy_X=False)
        model.fit(regression_input, regression_target)
        fitted_alpha = float(np.clip(model.coef_[0], OPTIMIZATION_ALPHA_MIN, 1))
        if verbose:
            print(
                f"Regression iteration {iteration}: alpha={fitted_alpha:.12g}",
                flush=True,
            )
        if abs(fitted_alpha - alpha) <= tolerance:
            return fitted_alpha
        alpha = fitted_alpha

    raise RuntimeError(
        f"Regression alpha did not converge after {max_iterations} iterations."
    )


def optimize_matrix_regression(
    selection_tensor,
    initial_alpha=MATRIX_INITIAL_ALPHA,
    epochs=100,
    learning_rate=0.005,
    device=None,
    verbose=False,
):
    """Optimizes alpha and per-QMoE W1/W2 matrices with PyTorch.

    The prediction follows
    ``P[t,m] = alpha * P[t-1,m] + W1[m] @ S[t-1,m] + W2[m] @ S[t,m-1]``.
    ``P[-1,m]``, ``S[-1,m]``, and ``S[t,-1]`` are zero. The loss is
    ``mean(t,m) sum(i) (P[t,m,i] - S[t,m,i])**2``.

    Args:
        selection_tensor: Binary tensor ``S`` with shape
            ``(iterations, qmoes, experts)``.
        initial_alpha: Initial alpha, constrained throughout training to
            ``0.5 < alpha <= 1``.
        epochs: Number of full-batch Adam training epochs.
        learning_rate: Adam learning rate.
        device: PyTorch device. Pass ``None`` to prefer CUDA when available.
        verbose: Whether to print training progress.

    Returns:
        A tuple ``(alpha, W1, W2, P, error)``. ``W1`` and ``W2`` have shape
        ``(qmoes, experts, experts)``, ``P`` has the same shape as ``S``, and
        ``error`` is the final mean squared prediction error.

    Raises:
        ValueError: If an argument is invalid.
    """
    selection_tensor = validate_selection_tensor(selection_tensor)
    if (
        not np.isfinite(initial_alpha)
        or not OPTIMIZATION_ALPHA_MIN <= initial_alpha < 1
    ):
        raise ValueError("initial_alpha must be finite and in (0.5, 1).")
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    selection = torch.from_numpy(selection_tensor).to(
        device=device, dtype=torch.float32
    )
    iterations, qmoes, experts = selection.shape
    previous_selection = torch.zeros_like(selection)
    previous_selection[1:] = selection[:-1]
    previous_layer_selection = torch.zeros_like(selection)
    previous_layer_selection[:, 1:] = selection[:, :-1]

    alpha_ratio = (initial_alpha - 0.5) / 0.5
    raw_alpha = torch.nn.Parameter(
        torch.tensor(
            np.log(alpha_ratio / (1 - alpha_ratio)),
            dtype=torch.float32,
            device=device,
        )
    )
    weight1 = torch.nn.Parameter(
        torch.eye(experts, dtype=torch.float32, device=device)
        .repeat(qmoes, 1, 1)
        .mul_(1 - initial_alpha)
    )
    weight2 = torch.nn.Parameter(
        torch.zeros(qmoes, experts, experts, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(
        (raw_alpha, weight1, weight2),
        lr=learning_rate,
    )

    progress_step = max(1, epochs // 10)
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        alpha = 0.5 + 0.5 * torch.sigmoid(raw_alpha)
        drive = torch.einsum(
            "mij,tmj->tmi", weight1, previous_selection
        ) + torch.einsum("mij,tmj->tmi", weight2, previous_layer_selection)
        popularity = torch.zeros((qmoes, experts), dtype=torch.float32, device=device)
        loss = torch.zeros((), dtype=torch.float32, device=device)
        for iteration in range(iterations):
            popularity = alpha * popularity + drive[iteration]
            loss = loss + (popularity - selection[iteration]).square().sum()
        loss = loss / (iterations * qmoes)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((raw_alpha, weight1, weight2), 1.0)
        optimizer.step()
        if verbose and (
            epoch == 0 or (epoch + 1) % progress_step == 0 or epoch + 1 == epochs
        ):
            print(
                f"Matrix regression: {epoch + 1}/{epochs} "
                f"(alpha={alpha.detach().item():.12g}, "
                f"error={loss.detach().item():.12g})",
                flush=True,
            )

    with torch.no_grad():
        alpha = 0.5 + 0.5 * torch.sigmoid(raw_alpha)
        drive = torch.einsum(
            "mij,tmj->tmi", weight1, previous_selection
        ) + torch.einsum("mij,tmj->tmi", weight2, previous_layer_selection)
        popularity = torch.zeros((qmoes, experts), dtype=torch.float32, device=device)
        popularity_tensor = torch.empty_like(selection)
        for iteration in range(iterations):
            popularity = alpha * popularity + drive[iteration]
            popularity_tensor[iteration] = popularity
        error = (popularity_tensor - selection).square().sum(dim=2).mean()

    return (
        float(alpha.item()),
        weight1.detach().cpu().numpy(),
        weight2.detach().cpu().numpy(),
        popularity_tensor.cpu().numpy(),
        float(error.item()),
    )


def print_routing_error_table(
    label,
    selection_tensor,
    popularity_tensor,
    random_popularities,
    first_two_end,
    last_five_start,
    markdown_output,
):
    """Prints routing CSV and appends its Markdown table to a file.

    Args:
        label: Name identifying how ``alpha`` was optimized.
        selection_tensor: Binary selection tensor ``S``.
        popularity_tensor: Moving-average popularity tensor ``P``.
        random_popularities: Three fixed random popularity distributions.
        first_two_end: Iteration immediately after the first two prompts.
        last_five_start: First iteration of the last five prompts.
        markdown_output: Markdown file receiving the formatted table.
    """
    iteration_ranges = (
        (0, selection_tensor.shape[0]),
        (0, first_two_end),
        (last_five_start, selection_tensor.shape[0]),
    )
    rows = []
    for gpu_expert_count in range(0, 1024 * 11, 1024):
        errors = moe_routing_error(
            selection_tensor,
            popularity_tensor,
            gpu_expert_count=gpu_expert_count,
            iteration_ranges=iteration_ranges,
        )
        random_errors = tuple(
            moe_routing_error(
                selection_tensor[last_five_start:],
                random_popularity,
                gpu_expert_count=gpu_expert_count,
            )
            for random_popularity in random_popularities
        )
        rows.append((gpu_expert_count, *errors, *random_errors))

    print(
        "alpha_objective,gpu_expert_count,all_prompts,first_2_prompts,"
        "last_5_prompts,random_1_last_5_prompts,"
        "random_2_last_5_prompts,random_3_last_5_prompts"
    )
    for row in rows:
        print(
            f"{label},{row[0]},"
            f"{row[1]:.12g},{row[2]:.12g},{row[3]:.12g},"
            f"{row[4]:.12g},{row[5]:.12g},{row[6]:.12g}"
        )

    markdown_lines = [
        f"## `{label}`",
        "",
        (
            "| Experts on GPU | All prompts | First 2 prompts | Last 5 prompts | "
            "Random 1, last 5 | Random 2, last 5 | Random 3, last 5 |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown_lines.append(
            f"| {row[0]} | {row[1]:.12g} | {row[2]:.12g} | "
            f"{row[3]:.12g} | {row[4]:.12g} | {row[5]:.12g} | "
            f"{row[6]:.12g} |"
        )
    with markdown_output.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(markdown_lines))
        stream.write("\n\n")


def append_markdown_values(markdown_output, heading, values):
    """Appends named scalar values to the Markdown results file."""
    with markdown_output.open("a", encoding="utf-8") as stream:
        stream.write(f"## {heading}\n\n")
        for name, value in values:
            stream.write(f"- `{name} = {value:.12g}`\n")
        stream.write("\n")


def run_alpha_optimizations(
    selection_tensor,
    grid_size,
    random_popularities,
    first_two_end,
    last_five_start,
    markdown_output,
):
    """Runs the three scalar-alpha methods and writes their routing results."""
    print("Fitting alpha as a linear-regression coefficient...", flush=True)
    regression_alpha = optimize_regression_alpha(selection_tensor, verbose=True)
    print(f"regression_alpha={regression_alpha:.12g}")
    append_markdown_values(
        markdown_output,
        "Linear regression",
        (("regression_alpha", regression_alpha),),
    )
    regression_popularity = compute_popularity(selection_tensor, regression_alpha)
    print_routing_error_table(
        "linear_regression",
        selection_tensor,
        regression_popularity,
        random_popularities,
        first_two_end,
        last_five_start,
        markdown_output,
    )

    print(f"Optimizing alpha with a {grid_size}-point grid...", flush=True)
    alpha, error = optimize_popularity_alpha(
        selection_tensor,
        grid_size=grid_size,
        verbose=True,
    )
    print(f"optimal_alpha={alpha:.12g}")
    print(f"minimum_mean_squared_error={error:.12g}")
    append_markdown_values(
        markdown_output,
        "Squared prediction error",
        (
            ("optimal_alpha", alpha),
            ("minimum_mean_squared_error", error),
        ),
    )
    popularity = compute_popularity(selection_tensor, alpha)
    print_routing_error_table(
        "squared_error",
        selection_tensor,
        popularity,
        random_popularities,
        first_two_end,
        last_five_start,
        markdown_output,
    )

    print("Optimizing alpha for routing error at K=6144...", flush=True)
    routing_alpha, routing_error = optimize_routing_alpha(
        selection_tensor,
        gpu_expert_count=6144,
        grid_size=grid_size,
        verbose=True,
    )
    print(f"routing_optimal_alpha_k6144={routing_alpha:.12g}")
    print(f"routing_minimum_error_k6144={routing_error:.12g}")
    append_markdown_values(
        markdown_output,
        "Routing error at K=6144",
        (
            ("routing_optimal_alpha_k6144", routing_alpha),
            ("routing_minimum_error_k6144", routing_error),
        ),
    )
    routing_popularity = compute_popularity(selection_tensor, routing_alpha)
    print_routing_error_table(
        "routing_error_k6144",
        selection_tensor,
        routing_popularity,
        random_popularities,
        first_two_end,
        last_five_start,
        markdown_output,
    )


def run_matrix_optimization(
    selection_tensor,
    epochs,
    learning_rate,
    random_popularities,
    first_two_end,
    last_five_start,
    markdown_output,
):
    """Jointly optimizes alpha, W1, and W2 and writes routing results."""
    print("Optimizing alpha, W1, and W2 with PyTorch...", flush=True)
    (
        matrix_alpha,
        _weight1,
        _weight2,
        matrix_popularity,
        matrix_error,
    ) = optimize_matrix_regression(
        selection_tensor,
        epochs=epochs,
        learning_rate=learning_rate,
        verbose=True,
    )
    print(f"matrix_regression_alpha={matrix_alpha:.12g}")
    print(f"matrix_regression_mean_squared_error={matrix_error:.12g}")
    append_markdown_values(
        markdown_output,
        "Matrix regression",
        (
            ("matrix_regression_alpha", matrix_alpha),
            ("matrix_regression_mean_squared_error", matrix_error),
        ),
    )
    print_routing_error_table(
        "matrix_regression",
        selection_tensor,
        matrix_popularity,
        random_popularities,
        first_two_end,
        last_five_start,
        markdown_output,
    )


def main():
    """Loads the routing archive, builds ``S``, and optionally optimizes alpha."""
    args = parse_args()
    print(f"Reading selected experts from {args.archive}...", flush=True)
    selected_experts = read_selected_experts(args.archive)
    print(f"Loaded {len(selected_experts):,} selected-expert rows.", flush=True)
    print(
        f"Building S with {args.qmoe_count} QMoE layers, "
        f"{args.expert_count} experts, and {args.selected_expert_count} selections...",
        flush=True,
    )
    tensor = build_selection_tensor(
        selected_experts,
        qmoe_count=args.qmoe_count,
        expert_count=args.expert_count,
        selected_expert_count=args.selected_expert_count,
    )
    print(f"Built S with shape {tensor.shape}.", flush=True)
    print(f"S.shape={tensor.shape}")
    print(f"S.dtype={tensor.dtype}")
    print(f"S.sum={int(tensor.sum())}")
    if args.optimization != "none":
        if tensor.shape[0] != sum(PROMPT_ITERATION_COUNTS):
            raise ValueError(
                f"Expected {sum(PROMPT_ITERATION_COUNTS)} iterations for the "
                f"10 prompts, got {tensor.shape[0]}."
            )
        first_two_end = sum(PROMPT_ITERATION_COUNTS[:2])
        last_five_start = sum(PROMPT_ITERATION_COUNTS[:-5])
        random_popularities = tuple(
            np.random.default_rng(seed).random(tensor.shape[1:])
            for seed in RANDOM_BASELINE_SEEDS
        )
        args.markdown_output.write_text("# QMoE routing results\n\n", encoding="utf-8")
        print(f"Writing Markdown results to {args.markdown_output}.", flush=True)

        if args.optimization == "alpha":
            run_alpha_optimizations(
                tensor,
                args.alpha_grid_size,
                random_popularities,
                first_two_end,
                last_five_start,
                args.markdown_output,
            )
        else:
            run_matrix_optimization(
                tensor,
                args.matrix_epochs,
                args.matrix_learning_rate,
                random_popularities,
                first_two_end,
                last_five_start,
                args.markdown_output,
            )


if __name__ == "__main__":
    main()
