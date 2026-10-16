#!/usr/bin/env python3
"""Build a binary QMoE expert-selection tensor from the published routing data."""

import argparse
from pathlib import Path

REQUIRED_COLUMNS = ("iteration", "qmoe_number", "expert_number", "position")
PROMPT_ITERATION_COUNTS = (216, 221, 253, 430, 407, 664, 450, 500, 646, 736)
RANDOM_BASELINE_SEEDS = (0, 1, 2)
OPTIMIZATION_ALPHA_MIN = 0.5000000000000001
MATRIX_INITIAL_ALPHA = 0.719428897297
EARLY_STOPPING_PATIENCE = 10
NEURAL_EARLY_STOPPING_PATIENCE = 20
NEURAL_RETENTION_WARMUP_EPOCHS = 100
PENALTY_HISTORY_LENGTH = 64
FUTURE_RETENTION_RADIUS = 16
RETENTION_LAMBDA = 100.0
np = None


def load_numpy():
    """Loads NumPy after argument parsing so ``--help`` stays fast."""
    global np
    if np is None:
        import numpy

        np = numpy


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
        * ``alpha_epochs``: number of PyTorch alpha-training epochs;
        * ``alpha_learning_rate``: PyTorch alpha-training learning rate;
        * ``alpha_early_stopping_tolerance``: minimum relative improvement;
        * ``alpha_early_stopping_patience``: epochs without improvement.
        * ``matrix_epochs``: number of matrix-regression training epochs;
        * ``matrix_learning_rate``: matrix-regression Adam learning rate.
        * ``matrix_early_stopping_tolerance``: minimum relative improvement;
        * ``matrix_early_stopping_patience``: epochs without improvement.
        * ``retention_lambda``: history-retention penalty coefficient;
        * ``neural_epochs``: maximum neural-model training epochs;
        * ``neural_learning_rate``: neural-model Adam learning rate;
        * ``neural_hidden_size``: neural-model hidden width;
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

  Optimize W0, W1, and W2 instead:
    python study_moe_selected_experts.py \\
      moe-no-offload-unbounded-expert-activation-correlations-selected-experts.zip \\
      --optimization matrix

  Optimize W0, W1, and W2 while retaining previously selected experts:
    python study_moe_selected_experts.py \\
      moe-no-offload-unbounded-expert-activation-correlations-selected-experts.zip \\
      --optimization penalty

  Optimize W0 and W1 with the same retention penalty:
    python study_moe_selected_experts.py \\
      moe-no-offload-unbounded-expert-activation-correlations-selected-experts.zip \\
      --optimization short

  Optimize W0 and W1 with retention over offsets -16..-1 and 1..16:
    python study_moe_selected_experts.py \\
      moe-no-offload-unbounded-expert-activation-correlations-selected-experts.zip \\
      --optimization future

  Optimize a one-layer neural recurrence instead:
    python study_moe_selected_experts.py \\
      moe-no-offload-unbounded-expert-activation-correlations-selected-experts.zip \\
      --optimization neural
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
        "-o",
        "--optimization",
        choices=("alpha", "matrix", "penalty", "short", "future", "neural", "none"),
        default="alpha",
        help=(
            "Run scalar-alpha, matrix, penalty, short, future, neural training, or only build S (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--alpha-grid-size",
        type=int,
        default=101,
        help="Alpha grid size used for the GPU routing-error search (default: %(default)s).",
    )
    parser.add_argument(
        "--alpha-epochs",
        type=int,
        default=200,
        help="Maximum PyTorch epochs for squared-error alpha optimization (default: %(default)s).",
    )
    parser.add_argument(
        "--alpha-learning-rate",
        type=float,
        default=0.01,
        help="PyTorch learning rate for squared-error alpha optimization (default: %(default)s).",
    )
    parser.add_argument(
        "--alpha-early-stopping-tolerance",
        type=float,
        default=0.0001,
        help="Minimum relative error improvement for alpha training (default: %(default)s).",
    )
    parser.add_argument(
        "--alpha-early-stopping-patience",
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help="Stop alpha training after this many epochs without sufficient improvement (default: %(default)s).",
    )
    parser.add_argument(
        "--matrix-epochs",
        type=int,
        default=200,
        help="Maximum epochs for W0, W1, and W2 matrix regression (default: %(default)s).",
    )
    parser.add_argument(
        "--matrix-learning-rate",
        type=float,
        default=0.005,
        help="Adam learning rate for W0, W1, and W2 matrix regression (default: %(default)s).",
    )
    parser.add_argument(
        "--matrix-early-stopping-tolerance",
        type=float,
        default=0.0001,
        help="Minimum relative error improvement for matrix training (default: %(default)s).",
    )
    parser.add_argument(
        "--matrix-early-stopping-patience",
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help="Stop matrix training after this many epochs without sufficient improvement (default: %(default)s).",
    )
    parser.add_argument(
        "--retention-lambda",
        type=float,
        default=RETENTION_LAMBDA,
        help="Lambda in the lambda/(k+lambda) history-retention weight for penalty and neural (default: %(default)s).",
    )
    parser.add_argument(
        "--neural-epochs",
        type=int,
        default=200,
        help="Maximum epochs for one-layer neural recurrence training (default: %(default)s).",
    )
    parser.add_argument(
        "--neural-learning-rate",
        type=float,
        default=0.0005,
        help="Adam learning rate for neural recurrence training (default: %(default)s).",
    )
    parser.add_argument(
        "--neural-hidden-size",
        type=int,
        default=256,
        help="Width of the neural recurrence hidden layer (default: %(default)s).",
    )
    parser.add_argument(
        "--neural-early-stopping-tolerance",
        type=float,
        default=0.0001,
        help="Minimum relative error improvement for neural training (default: %(default)s).",
    )
    parser.add_argument(
        "--neural-early-stopping-patience",
        type=int,
        default=NEURAL_EARLY_STOPPING_PATIENCE,
        help="Stop neural training after this many epochs without sufficient improvement (default: %(default)s).",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        help="Markdown results file (default: <script>.<optimization>.md).",
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
    import pandas as pd

    load_numpy()
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
    load_numpy()
    missing_columns = set(REQUIRED_COLUMNS) - set(selected_experts.columns)
    if missing_columns:
        raise ValueError(f"Missing CSV columns: {sorted(missing_columns)}.")
    if qmoe_count <= 0 or expert_count <= 0 or selected_expert_count <= 0:
        raise ValueError("Tensor dimensions and selected expert count must be positive.")
    if selected_experts.empty:
        raise ValueError("The selected-expert table is empty.")

    iterations = selected_experts["iteration"].to_numpy()
    qmoes = selected_experts["qmoe_number"].to_numpy()
    experts = selected_experts["expert_number"].to_numpy()
    positions = selected_experts["position"].to_numpy()
    iteration_count = int(iterations.max())

    if iterations.min() != 1 or set(np.unique(iterations)) != set(range(1, iteration_count + 1)):
        raise ValueError("Iterations must be consecutive and one-based.")
    if qmoes.min() != 0 or qmoes.max() >= qmoe_count or np.unique(qmoes).size != qmoe_count:
        raise ValueError(f"QMoE numbers must cover [0, {qmoe_count}).")
    if experts.min() < 0 or experts.max() >= expert_count:
        raise ValueError(f"Expert numbers must be in [0, {expert_count}).")
    if positions.min() < 0 or positions.max() >= qmoe_count * expert_count:
        raise ValueError(f"Global positions must be in [0, {qmoe_count * expert_count}).")
    if selected_experts.duplicated(["iteration", "qmoe_number", "expert_number"]).any():
        raise ValueError("An expert must not be selected twice by the same QMoE at one iteration.")

    selection_counts = selected_experts.groupby(["iteration", "qmoe_number"], sort=False).size()
    expected_group_count = iteration_count * qmoe_count
    if selection_counts.size != expected_group_count or not (selection_counts == selected_expert_count).all():
        raise ValueError(f"Every iteration/QMoE pair must contain exactly {selected_expert_count} selected experts.")

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
    load_numpy()
    selection_tensor = np.asarray(selection_tensor)
    if selection_tensor.ndim != 3 or 0 in selection_tensor.shape:
        raise ValueError("The selection tensor must have non-empty (iteration, QMoE, expert) axes.")
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


def compute_popularity(selection_tensor, alpha, device=None):
    """Computes the popularity tensor ``P`` from ``S`` and ``alpha`` with PyTorch.

    Popularity starts at zero and follows
    ``P(t) = alpha * P(t-1) + (1-alpha) * S(t-1)``. Consequently, ``P[0]`` is
    zero and has the same QMoE and expert axes as ``S[0]``.

    Args:
        selection_tensor: Binary tensor ``S`` with shape
            ``(iterations, qmoes, experts)``.
        alpha: Popularity coefficient in the closed interval ``[0, 1]``.
        device: PyTorch device. Pass ``None`` to prefer CUDA when available.

    Returns:
        Floating-point popularity tensor ``P`` with the same shape as ``S``.

    Raises:
        ValueError: If ``S`` is invalid or ``alpha`` is outside ``[0, 1]``.
    """
    import torch
    from torch._higher_order_ops import scan

    selection_tensor = validate_selection_tensor(selection_tensor)
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and in [0, 1].")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    selection = torch.from_numpy(selection_tensor).to(device=device, dtype=torch.float32)
    previous_selection = torch.zeros_like(selection)
    previous_selection[1:] = selection[:-1]
    alpha_tensor = torch.tensor(alpha, dtype=torch.float32, device=device)

    def recurrence(popularity, current_selection):
        next_popularity = alpha_tensor * popularity + (1 - alpha_tensor) * current_selection
        return next_popularity, next_popularity.clone()

    with torch.no_grad():
        popularity = scan(recurrence, torch.zeros_like(selection[0]), previous_selection)[1]
    return popularity.cpu().numpy()


def moe_routing_error(
    selection_tensor,
    popularity_tensor,
    gpu_expert_count,
    iteration_ranges=None,
    device=None,
):
    """Computes the simulated QMoE routing error from ``S`` and ``P`` on GPU.

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
        device: PyTorch device. Pass ``None`` to prefer CUDA when available.

    Returns:
        The mean number of selected experts missed by ``Q`` per iteration and
        QMoE. It lies in ``[0, selections_per_qmoe]``. When
        ``iteration_ranges`` is provided, returns one mean per range.

    Raises:
        ValueError: If either tensor, ``gpu_expert_count``, or an iteration
            range is invalid.
    """
    import torch

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
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    selection = torch.from_numpy(selection_tensor).to(device=device)
    popularity = torch.from_numpy(popularity_tensor).to(device=device, dtype=torch.float32)
    popularity_rows = popularity.reshape(-1, experts_per_iteration)
    sorted_experts = torch.argsort(popularity_rows, dim=1, stable=True)
    retained_experts = sorted_experts[:, experts_per_iteration - gpu_expert_count :]
    predicted_rows = torch.zeros(popularity_rows.shape, dtype=torch.bool, device=device)
    predicted_rows.scatter_(1, retained_experts, True)

    selection_rows = selection.reshape(selection.shape[0], experts_per_iteration)
    missed_rows = selection_rows.bool() & ~predicted_rows
    errors = missed_rows.reshape(selection.shape).sum(dim=2)

    if iteration_ranges is None:
        return float(errors.float().mean().item())

    means = []
    for start, end in iteration_ranges:
        if not 0 <= start < end <= selection_tensor.shape[0]:
            raise ValueError("Every iteration range must be non-empty and within S.")
        means.append(float(errors[start:end].float().mean().item()))
    return tuple(means)


def moe_routing_error_with_swap_limit(
    selection_tensor,
    popularity_tensor,
    gpu_expert_count,
    max_swaps=2,
    iteration_ranges=None,
    device=None,
):
    """Computes routing error with a persistent placement and a swap limit.

    The first ``gpu_expert_count`` flattened experts start on GPU. At each
    subsequent iteration, up to ``max_swaps`` highest-popularity CPU experts
    replace the same number of lowest-popularity GPU experts, provided each
    replacement increases popularity. Placement remains continuous across
    prompt boundaries.

    Returns:
        A tuple ``(error, swap_counts)``. ``error`` follows the return shape of
        ``moe_routing_error`` and ``swap_counts[t]`` is the number of swaps
        from iteration ``t`` to ``t + 1``.
    """
    import torch

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
    if max_swaps < 0:
        raise ValueError("max_swaps must be non-negative.")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    selection = torch.from_numpy(selection_tensor).to(device=device)
    popularity = torch.from_numpy(popularity_tensor).to(device=device, dtype=torch.float32)
    popularity_rows = popularity.reshape(-1, experts_per_iteration)
    if popularity_rows.shape[0] == 1:
        popularity_rows = popularity_rows.expand(selection.shape[0], -1)
    selection_rows = selection.reshape(selection.shape[0], experts_per_iteration)

    on_gpu = torch.zeros(experts_per_iteration, dtype=torch.bool, device=device)
    on_gpu[:gpu_expert_count] = True
    errors = torch.empty(
        (selection.shape[0], selection.shape[1]),
        dtype=torch.int64,
        device=device,
    )
    swap_counts = torch.zeros(max(0, selection.shape[0] - 1), dtype=torch.int64, device=device)
    swaps_per_iteration = min(max_swaps, gpu_expert_count, experts_per_iteration - gpu_expert_count)

    for iteration in range(selection.shape[0]):
        if iteration > 0 and swaps_per_iteration > 0:
            scores = popularity_rows[iteration]
            entering_scores, entering = torch.topk(
                scores.masked_fill(on_gpu, -torch.inf),
                swaps_per_iteration,
                sorted=True,
            )
            leaving_scores, leaving = torch.topk(
                scores.masked_fill(~on_gpu, torch.inf),
                swaps_per_iteration,
                largest=False,
                sorted=True,
            )
            beneficial = entering_scores > leaving_scores
            on_gpu[leaving[beneficial]] = False
            on_gpu[entering[beneficial]] = True
            swap_counts[iteration - 1] = beneficial.sum()
        missed = selection_rows[iteration].bool() & ~on_gpu
        errors[iteration] = missed.reshape(selection.shape[1:]).sum(dim=1)

    if iteration_ranges is None:
        error = float(errors.float().mean().item())
    else:
        means = []
        for start, end in iteration_ranges:
            if not 0 <= start < end <= selection.shape[0]:
                raise ValueError("Every iteration range must be non-empty and within S.")
            means.append(float(errors[start:end].float().mean().item()))
        error = tuple(means)
    return error, swap_counts.cpu().numpy()


def _routing_metrics_for_gpu_counts(
    selection_tensor,
    popularity_tensor,
    gpu_expert_counts,
    iteration_ranges,
    device=None,
):
    """Computes unbounded routing metrics for several K values with one sort."""
    import torch

    load_numpy()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    selection = torch.from_numpy(selection_tensor).to(device=device)
    popularity = torch.from_numpy(np.asarray(popularity_tensor)).to(device=device, dtype=torch.float32)
    iterations, qmoes, experts = selection.shape
    total_experts = qmoes * experts
    selection_rows = selection.reshape(iterations, total_experts)
    popularity_rows = popularity.reshape(-1, total_experts)
    sorted_experts = torch.argsort(popularity_rows, dim=1, stable=True)
    selected_per_iteration = selection_rows.sum(dim=1)
    metrics = []

    for gpu_expert_count in gpu_expert_counts:
        retained = sorted_experts[:, total_experts - gpu_expert_count :]
        if popularity_rows.shape[0] == 1:
            on_gpu = torch.zeros(total_experts, dtype=torch.bool, device=device)
            on_gpu[retained[0]] = True
            selected_on_gpu = selection_rows[:, on_gpu].sum(dim=1)
            swap_counts = torch.zeros(max(0, iterations - 1), dtype=torch.int64, device=device)
        else:
            selected_on_gpu = torch.gather(selection_rows, 1, retained).sum(dim=1)
            on_gpu = torch.zeros((iterations, total_experts), dtype=torch.bool, device=device)
            on_gpu.scatter_(1, retained, True)
            swap_counts = (on_gpu[1:] & ~on_gpu[:-1]).sum(dim=1)
        missed = selected_per_iteration - selected_on_gpu
        errors = tuple(float(missed[start:end].float().mean().item() / qmoes) for start, end in iteration_ranges)
        metrics.append((errors, swap_counts.cpu().numpy()))
    return metrics


def _limited_routing_metrics_batch(
    selection_tensor,
    popularity_rows,
    gpu_expert_counts,
    iteration_ranges,
    temporal_popularity,
    max_swaps=2,
    device=None,
):
    """Simulates several swap-limited placements in one CUDA loop."""
    import torch

    load_numpy()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    selection = torch.from_numpy(selection_tensor).to(device=device)
    iterations, qmoes, experts = selection.shape
    total_experts = qmoes * experts
    selection_rows = selection.reshape(iterations, total_experts).bool()
    popularity = torch.from_numpy(np.asarray(popularity_rows)).to(device=device, dtype=torch.float32)
    popularity = popularity.reshape(popularity.shape[0], total_experts)
    counts = torch.tensor(gpu_expert_counts, dtype=torch.int64, device=device)
    batch_size = counts.shape[0]
    expert_indices = torch.arange(total_experts, device=device)
    on_gpu = expert_indices.unsqueeze(0) < counts.unsqueeze(1)
    errors = torch.empty((iterations, batch_size), dtype=torch.int64, device=device)
    swap_counts = torch.zeros((max(0, iterations - 1), batch_size), dtype=torch.int64, device=device)

    for iteration in range(iterations):
        if iteration > 0 and max_swaps > 0:
            scores = popularity[iteration].expand(batch_size, -1) if temporal_popularity else popularity
            entering_scores, entering = torch.topk(
                scores.masked_fill(on_gpu, -torch.inf),
                max_swaps,
                dim=1,
                sorted=True,
            )
            leaving_scores, leaving = torch.topk(
                scores.masked_fill(~on_gpu, torch.inf),
                max_swaps,
                dim=1,
                largest=False,
                sorted=True,
            )
            beneficial = entering_scores > leaving_scores
            leaving_was_on_gpu = torch.gather(on_gpu, 1, leaving)
            entering_was_on_gpu = torch.gather(on_gpu, 1, entering)
            on_gpu.scatter_(1, leaving, leaving_was_on_gpu & ~beneficial)
            on_gpu.scatter_(1, entering, entering_was_on_gpu | beneficial)
            swap_counts[iteration - 1] = beneficial.sum(dim=1)
        errors[iteration] = (selection_rows[iteration].unsqueeze(0) & ~on_gpu).sum(dim=1)

    range_errors = tuple(errors[start:end].float().mean(dim=0) / qmoes for start, end in iteration_ranges)
    return (
        torch.stack(range_errors, dim=1).cpu().numpy(),
        swap_counts.cpu().numpy(),
    )


def optimize_popularity_alpha(
    selection_tensor,
    epochs=200,
    learning_rate=0.01,
    early_stopping_tolerance=0.0001,
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    device=None,
    verbose=False,
):
    """Optimizes alpha for mean squared popularity error with PyTorch.

    Args:
        selection_tensor: Binary tensor ``S`` with shape
            ``(iterations, qmoes, experts)``.
        epochs: Maximum number of Adam optimization epochs.
        learning_rate: Adam learning rate.
        early_stopping_tolerance: Minimum relative loss improvement that resets
            early-stopping patience.
        early_stopping_patience: Number of consecutive epochs without sufficient
            improvement before training stops.
        device: PyTorch device. Pass ``None`` to prefer CUDA when available.
        verbose: Whether to display epoch progress.

    Returns:
        A tuple ``(alpha, error)`` containing the minimizing coefficient and
        its mean squared prediction error per iteration and QMoE.

    Raises:
        ValueError: If ``S`` or an optimization argument is invalid.
    """
    import torch
    from torch._higher_order_ops import scan
    from tqdm.auto import trange

    selection_tensor = validate_selection_tensor(selection_tensor)
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    if not np.isfinite(early_stopping_tolerance) or early_stopping_tolerance < 0:
        raise ValueError("early_stopping_tolerance must be finite and non-negative.")
    if early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive.")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    selection = torch.from_numpy(selection_tensor).to(device=device, dtype=torch.float32)
    previous_selection = torch.zeros_like(selection)
    previous_selection[1:] = selection[:-1]
    alpha = torch.nn.Parameter(torch.tensor(0.9, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam((alpha,), lr=learning_rate)
    epoch_iterator = trange(epochs, desc="Alpha squared error", unit="epoch") if verbose else range(epochs)

    def compute_torch_popularity():
        def recurrence(popularity, current_selection):
            next_popularity = alpha * popularity + (1 - alpha) * current_selection
            return next_popularity, next_popularity.clone()

        return scan(recurrence, torch.zeros_like(selection[0]), previous_selection)[1]

    best_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in epoch_iterator:
        optimizer.zero_grad(set_to_none=True)
        popularity = compute_torch_popularity()
        loss = (popularity - selection).square().sum(dim=2).mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            alpha.clamp_(OPTIMIZATION_ALPHA_MIN, 1)
        current_loss = loss.detach().item()
        relative_improvement = (
            float("inf")
            if not np.isfinite(best_loss)
            else (best_loss - current_loss) / max(abs(best_loss), np.finfo(np.float64).eps)
        )
        if relative_improvement >= early_stopping_tolerance:
            best_loss = current_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if verbose:
            epoch_iterator.set_postfix(
                alpha=f"{alpha.detach().item():.3f}",
                error=f"{current_loss:.3f}",
                patience=f"{epochs_without_improvement}/{early_stopping_patience}",
            )
        if epochs_without_improvement >= early_stopping_patience:
            if verbose:
                epoch_iterator.update(1)
                epoch_iterator.write(
                    f"Early stopping after {epoch + 1} epochs: relative error "
                    f"improvement remained below {early_stopping_tolerance:.3g} "
                    f"for {early_stopping_patience} epochs."
                )
            break

    with torch.no_grad():
        popularity = compute_torch_popularity()
        loss = (popularity - selection).square().sum(dim=2).mean()
    return float(alpha.item()), float(loss.item())


def optimize_routing_alpha(
    selection_tensor,
    gpu_expert_count=6144,
    grid_size=101,
    xatol=1e-5,
    device=None,
    verbose=False,
):
    """Finds alpha by evaluating routing error with PyTorch.

    Args:
        selection_tensor: Binary tensor ``S`` with shape
            ``(iterations, qmoes, experts)``.
        gpu_expert_count: Number ``K`` of experts retained on GPU.
        grid_size: Number of uniformly spaced alpha values in ``(0.5, 1]``.
        xatol: Absolute tolerance for refinement around the best grid point.
        device: PyTorch device. Pass ``None`` to prefer CUDA when available.
        verbose: Whether to print grid progress.

    Returns:
        A tuple ``(alpha, error)`` containing the best coefficient and its mean
        routing error over all iterations and QMoEs.

    Raises:
        ValueError: If ``S`` is invalid, ``grid_size`` is smaller than three,
            or ``xatol`` is not positive.
    """
    import torch
    from scipy.optimize import minimize_scalar
    from torch._higher_order_ops import scan

    selection_tensor = validate_selection_tensor(selection_tensor)
    if grid_size < 3:
        raise ValueError("grid_size must be at least 3.")
    if not np.isfinite(xatol) or xatol <= 0:
        raise ValueError("xatol must be finite and positive.")
    total_experts = selection_tensor.shape[1] * selection_tensor.shape[2]
    if not 0 <= gpu_expert_count <= total_experts:
        raise ValueError(f"gpu_expert_count must be in [0, {total_experts}], got {gpu_expert_count}.")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    selection = torch.from_numpy(selection_tensor).to(device=device, dtype=torch.float32)
    previous_selection = torch.zeros_like(selection)
    previous_selection[1:] = selection[:-1]
    flattened_selection = selection.flatten(1)

    @torch.no_grad()
    def objective(alpha):
        alpha_tensor = torch.tensor(alpha, dtype=torch.float32, device=device)

        def recurrence(popularity, current_selection):
            next_popularity = alpha_tensor * popularity + (1 - alpha_tensor) * current_selection
            return next_popularity, next_popularity.clone()

        popularity = scan(recurrence, torch.zeros_like(selection[0]), previous_selection)[1].flatten(1)
        if gpu_expert_count == 0:
            return float(flattened_selection.sum().item() / selection.shape[0] / selection.shape[1])
        if gpu_expert_count == total_experts:
            return 0.0

        selected_on_gpu = torch.zeros((), dtype=torch.float32, device=device)
        for popularity_chunk, selection_chunk in zip(
            popularity.split(256), flattened_selection.split(256), strict=True
        ):
            gpu_indices = torch.topk(popularity_chunk, gpu_expert_count, dim=1, sorted=False).indices
            selected_on_gpu += torch.gather(selection_chunk, 1, gpu_indices).sum()
        selected_total = flattened_selection.sum()
        return float(((selected_total - selected_on_gpu) / selection.shape[0] / selection.shape[1]).item())

    alphas = np.linspace(OPTIMIZATION_ALPHA_MIN, 1, grid_size)
    errors = np.empty(grid_size, dtype=np.float64)
    progress_step = max(1, grid_size // 10)
    for index, alpha in enumerate(alphas):
        errors[index] = objective(alpha)
        if verbose and (index == 0 or (index + 1) % progress_step == 0 or index + 1 == grid_size):
            print(
                f"Routing alpha grid: {index + 1}/{grid_size} (alpha={alpha:.3f}, error={errors[index]:.3f})",
                flush=True,
            )

    best_index = min(range(grid_size), key=lambda index: (errors[index], alphas[index]))
    candidates = [(float(alphas[best_index]), float(errors[best_index]))]
    if 0 < best_index < grid_size - 1:
        lower_bound = float(alphas[best_index - 1])
        upper_bound = float(alphas[best_index + 1])
        if verbose:
            print(
                f"Refining routing alpha in [{lower_bound:.3f}, {upper_bound:.3f}]...",
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
    from sklearn.linear_model import LinearRegression

    selection_tensor = validate_selection_tensor(selection_tensor)
    if not np.isfinite(initial_alpha) or not OPTIMIZATION_ALPHA_MIN <= initial_alpha <= 1:
        raise ValueError("initial_alpha must be finite and in (0.5, 1].")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")

    regression_target = (selection_tensor[1:].astype(np.float64) - selection_tensor[:-1]).reshape(-1)
    alpha = float(initial_alpha)
    for iteration in range(1, max_iterations + 1):
        popularity = compute_popularity(selection_tensor, alpha)
        regression_input = (popularity[:-1] - selection_tensor[:-1]).reshape(-1, 1)
        model = LinearRegression(fit_intercept=False, copy_X=False)
        model.fit(regression_input, regression_target)
        fitted_alpha = float(np.clip(model.coef_[0], OPTIMIZATION_ALPHA_MIN, 1))
        if verbose:
            print(
                f"Regression iteration {iteration}: alpha={fitted_alpha:.3f}",
                flush=True,
            )
        if abs(fitted_alpha - alpha) <= tolerance:
            return fitted_alpha
        alpha = fitted_alpha

    raise RuntimeError(f"Regression alpha did not converge after {max_iterations} iterations.")


def _weighted_selection_history(selection, retention_lambda):
    """Returns the lambda/(k+lambda)-weighted selection history."""
    history = selection.new_zeros(selection.shape)
    for lag in range(1, min(PENALTY_HISTORY_LENGTH, selection.shape[0] - 1) + 1):
        history[lag:] += selection[:-lag] * (retention_lambda / (lag + retention_lambda))
    return history


def _weighted_selection_neighborhood(
    selection,
    retention_lambda,
    radius=FUTURE_RETENTION_RADIUS,
    segment_lengths=None,
):
    """Returns weighted selections at nonzero offsets from ``-radius`` to ``radius``."""
    if segment_lengths is None:
        segment_lengths = (selection.shape[0],)
    if sum(segment_lengths) != selection.shape[0]:
        raise ValueError("segment_lengths must sum to the number of iterations.")

    neighborhood = selection.new_zeros(selection.shape)
    start = 0
    for length in segment_lengths:
        end = start + length
        for lag in range(1, min(radius, length - 1) + 1):
            weight = retention_lambda / (lag + retention_lambda)
            neighborhood[start + lag : end] += selection[start : end - lag] * weight
            neighborhood[start : end - lag] += selection[start + lag : end] * weight
        start = end
    return neighborhood


def matrix_diagonal_statistics(weight):
    """Measures how strongly a batch of square matrices is diagonal."""
    weight = np.asarray(weight)
    if weight.ndim != 3 or weight.shape[1] != weight.shape[2]:
        raise ValueError("weight must have shape (matrices, rows, rows).")
    diagonal = np.diagonal(weight, axis1=1, axis2=2)
    diagonal_sum = float(diagonal.sum())
    total_sum = float(weight.sum())
    off_diagonal_count = weight.shape[0] * weight.shape[1] * (weight.shape[2] - 1)
    diagonal_mass_ratio = diagonal_sum / total_sum if total_sum > 0 else float("nan")
    off_diagonal_mean = (total_sum - diagonal_sum) / off_diagonal_count if off_diagonal_count > 0 else 0.0
    diagonal_is_largest = diagonal >= weight.max(axis=2)
    return (
        diagonal_mass_ratio,
        float(diagonal.mean()),
        off_diagonal_mean,
        float(diagonal_is_largest.mean()),
    )


def optimize_matrix_regression(
    selection_tensor,
    initial_decay=MATRIX_INITIAL_ALPHA,
    epochs=100,
    learning_rate=0.005,
    retention_penalty=False,
    use_previous_layer=True,
    retention_window="past",
    prompt_iteration_counts=None,
    retention_lambda=RETENTION_LAMBDA,
    early_stopping_tolerance=0.0001,
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    device=None,
    verbose=False,
):
    """Optimizes per-QMoE W0/W1/W2 matrices with PyTorch.

    The prediction follows
    ``P[t,m] = W0[m] @ P[t-1,m] + W1[m] @ S[t-1,m] + W2[m] @ S[t,m-1]``.
    All coefficients are constrained to be non-negative.
    ``P[-1,m]``, ``S[-1,m]``, and ``S[t,-1]`` are zero. The loss is
    ``mean(t,m) sum(i) (P[t,m,i] - S[t,m,i])**2``.

    Args:
        selection_tensor: Binary tensor ``S`` with shape
            ``(iterations, qmoes, experts)``.
        initial_decay: Scalar used to initialize ``W0`` to
            ``initial_decay * I`` and ``W1`` to ``(1-initial_decay) * I``.
        epochs: Number of full-batch Adam training epochs.
        learning_rate: Adam learning rate.
        retention_penalty: Whether to add
            ``mean(t,m) sum(i,k=1..history_length)
            retention_lambda/(k+retention_lambda) * S[t-k,m,i] *
            (P[t,m,i] - S[t-k,m,i])**2``.
        use_previous_layer: Whether to include ``W2[m] @ S[t,m-1]``.
        retention_window: ``"past"`` for lags 1 through 64, or ``"future"``
            for offsets -16 through -1 and 1 through 16.
        prompt_iteration_counts: Prompt lengths used to keep the future window
            within each prompt. ``None`` treats the input as one sequence.
        retention_lambda: History-retention penalty coefficient.
        early_stopping_tolerance: Minimum relative loss improvement that resets
            early-stopping patience.
        early_stopping_patience: Number of consecutive epochs without sufficient
            improvement before training stops.
        device: PyTorch device. Pass ``None`` to prefer CUDA when available.
        verbose: Whether to print training progress.

    Returns:
        A tuple ``(W0, W1, W2, P, error, retention_error)``. The three weights have shape
        ``(qmoes, experts, experts)``, ``P`` has the same shape as ``S``, and
        ``error`` is the final mean squared prediction error.

    Raises:
        ValueError: If an argument is invalid.
    """
    import torch
    from torch._higher_order_ops import scan
    from tqdm.auto import trange

    selection_tensor = validate_selection_tensor(selection_tensor)
    if not np.isfinite(initial_decay) or not 0 < initial_decay < 1:
        raise ValueError("initial_decay must be finite and in (0, 1).")
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    if not np.isfinite(retention_lambda) or retention_lambda < 0:
        raise ValueError("retention_lambda must be finite and non-negative.")
    if retention_window not in ("past", "future"):
        raise ValueError("retention_window must be 'past' or 'future'.")
    if not np.isfinite(early_stopping_tolerance) or early_stopping_tolerance < 0:
        raise ValueError("early_stopping_tolerance must be finite and non-negative.")
    if early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    selection = torch.from_numpy(selection_tensor).to(device=device, dtype=torch.float32)
    iterations, qmoes, experts = selection.shape
    previous_selection = torch.zeros_like(selection)
    previous_selection[1:] = selection[:-1]
    previous_layer_selection = torch.zeros_like(selection)
    previous_layer_selection[:, 1:] = selection[:, :-1]
    if retention_penalty:
        selection_history = (
            _weighted_selection_neighborhood(
                selection,
                retention_lambda,
                segment_lengths=prompt_iteration_counts,
            )
            if retention_window == "future"
            else _weighted_selection_history(selection, retention_lambda)
        )
    else:
        selection_history = None

    identity = torch.eye(experts, dtype=torch.float32, device=device).repeat(qmoes, 1, 1)
    weight0 = torch.nn.Parameter(identity.mul(initial_decay))
    weight1 = torch.nn.Parameter(
        torch.eye(experts, dtype=torch.float32, device=device).repeat(qmoes, 1, 1).mul_(1 - initial_decay)
    )
    weight2 = torch.nn.Parameter(torch.zeros(qmoes, experts, experts, dtype=torch.float32, device=device))
    optimized_weights = (weight0, weight1, weight2) if use_previous_layer else (weight0, weight1)
    optimizer = torch.optim.Adam(optimized_weights, lr=learning_rate)

    epoch_iterator = trange(epochs, desc="Matrix regression", unit="epoch") if verbose else range(epochs)
    best_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in epoch_iterator:
        optimizer.zero_grad(set_to_none=True)
        drive = torch.einsum("mij,tmj->tmi", weight1, previous_selection)
        if use_previous_layer:
            drive = drive + torch.einsum("mij,tmj->tmi", weight2, previous_layer_selection)

        def recurrence(popularity, current_drive):
            next_popularity = torch.bmm(weight0, popularity.unsqueeze(2)).squeeze(2) + current_drive
            return next_popularity, next_popularity.clone()

        _, popularity_tensor = scan(
            recurrence,
            torch.zeros((qmoes, experts), dtype=torch.float32, device=device),
            drive,
        )
        prediction_loss = (popularity_tensor - selection).square().sum(dim=2).mean()
        if retention_penalty:
            retention_loss = (selection_history * (popularity_tensor - 1).square()).sum(dim=2).mean()
            loss = prediction_loss + retention_loss
        else:
            retention_loss = torch.zeros_like(prediction_loss)
            loss = prediction_loss
        if not torch.isfinite(loss):
            raise RuntimeError("Matrix regression produced a non-finite loss. Reduce --matrix-learning-rate.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(optimized_weights, 1.0)
        optimizer.step()
        with torch.no_grad():
            weight0.clamp_(min=0)
            weight1.clamp_(min=0)
            weight2.clamp_(min=0)
            row_norm = weight0.abs().sum(dim=2).amax(dim=1)
            scale = torch.clamp(row_norm / 0.999, min=1)
            weight0.div_(scale[:, None, None])
        current_loss = loss.detach().item()
        relative_improvement = (
            float("inf")
            if not np.isfinite(best_loss)
            else (best_loss - current_loss) / max(abs(best_loss), np.finfo(np.float64).eps)
        )
        if relative_improvement >= early_stopping_tolerance:
            best_loss = current_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if verbose:
            progress = {
                "error": f"{prediction_loss.detach().item():.3f}",
                "patience": f"{epochs_without_improvement}/{early_stopping_patience}",
            }
            if retention_penalty:
                progress["retention"] = f"{retention_loss.detach().item():.3f}"
                progress["loss"] = f"{current_loss:.3f}"
            epoch_iterator.set_postfix(progress)
        if epochs_without_improvement >= early_stopping_patience:
            if verbose:
                epoch_iterator.update(1)
                epoch_iterator.write(
                    f"Early stopping after {epoch + 1} epochs: relative error "
                    f"improvement remained below {early_stopping_tolerance:.3g} "
                    f"for {early_stopping_patience} epochs."
                )
            break

    with torch.no_grad():
        drive = torch.einsum("mij,tmj->tmi", weight1, previous_selection)
        if use_previous_layer:
            drive = drive + torch.einsum("mij,tmj->tmi", weight2, previous_layer_selection)
        popularity = torch.zeros((qmoes, experts), dtype=torch.float32, device=device)
        popularity_tensor = torch.empty_like(selection)
        for iteration in range(iterations):
            popularity = torch.bmm(weight0, popularity.unsqueeze(2)).squeeze(2) + drive[iteration]
            popularity_tensor[iteration] = popularity
        error = (popularity_tensor - selection).square().sum(dim=2).mean()
        final_retention_error = (
            (selection_history * (popularity_tensor - 1).square()).sum(dim=2).mean()
            if retention_penalty
            else torch.zeros_like(error)
        )

    return (
        weight0.detach().cpu().numpy(),
        weight1.detach().cpu().numpy(),
        weight2.detach().cpu().numpy(),
        popularity_tensor.cpu().numpy(),
        float(error.item()),
        float(final_retention_error.item()),
    )


def optimize_neural_popularity(
    selection_tensor,
    epochs=200,
    learning_rate=0.0005,
    hidden_size=256,
    retention_lambda=RETENTION_LAMBDA,
    early_stopping_tolerance=0.0001,
    early_stopping_patience=NEURAL_EARLY_STOPPING_PATIENCE,
    device=None,
    verbose=False,
):
    """Optimizes a two-layer network, enabling retention after 100 epochs."""
    import torch
    from tqdm.auto import trange

    selection_tensor = validate_selection_tensor(selection_tensor)
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive.")
    if not np.isfinite(retention_lambda) or retention_lambda < 0:
        raise ValueError("retention_lambda must be finite and non-negative.")
    if not np.isfinite(early_stopping_tolerance) or early_stopping_tolerance < 0:
        raise ValueError("early_stopping_tolerance must be finite and non-negative.")
    if early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    selection = torch.from_numpy(selection_tensor).to(device=device, dtype=torch.float32)
    _iterations, qmoes, experts = selection.shape
    previous_selection = torch.zeros_like(selection)
    previous_selection[1:] = selection[:-1]
    previous_layer_selection = torch.zeros_like(selection)
    previous_layer_selection[:, 1:] = selection[:, :-1]
    neural_inputs = torch.cat((previous_selection, previous_layer_selection), dim=2)
    selection_history = _weighted_selection_history(selection, retention_lambda)

    hidden_weight = torch.nn.Parameter(torch.empty(qmoes, hidden_size, 3 * experts, dtype=torch.float32, device=device))
    output_weight = torch.nn.Parameter(torch.empty(qmoes, experts, hidden_size, dtype=torch.float32, device=device))
    selection_rate = selection.mean().clamp(1e-6, 1 - 1e-6)
    for qmoe_weight in hidden_weight:
        torch.nn.init.xavier_uniform_(qmoe_weight)
    with torch.no_grad():
        output_weight.fill_((torch.logit(selection_rate) / (0.5 * hidden_size)).item())
    optimizer = torch.optim.Adam((hidden_weight, output_weight), lr=learning_rate)

    def compute_torch_popularity():
        popularity = torch.zeros((qmoes, experts), dtype=torch.float32, device=device)
        popularity_sequence = []
        for current_inputs in neural_inputs:
            features = torch.cat((popularity, current_inputs), dim=1)
            hidden = torch.sigmoid(torch.bmm(hidden_weight, features.unsqueeze(2)).squeeze(2))
            popularity = torch.sigmoid(torch.bmm(output_weight, hidden.unsqueeze(2)).squeeze(2))
            popularity_sequence.append(popularity)
        return torch.stack(popularity_sequence)

    epoch_iterator = trange(epochs, desc="Neural popularity", unit="epoch") if verbose else range(epochs)
    best_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in epoch_iterator:
        optimizer.zero_grad(set_to_none=True)
        popularity_tensor = compute_torch_popularity()
        prediction_loss = (popularity_tensor - selection).square().sum(dim=2).mean()
        retention_active = epoch >= NEURAL_RETENTION_WARMUP_EPOCHS
        if retention_active:
            retention_loss = (selection_history * (popularity_tensor - 1).square()).sum(dim=2).mean()
        else:
            retention_loss = torch.zeros_like(prediction_loss)
        loss = prediction_loss + retention_loss
        if not torch.isfinite(loss):
            raise RuntimeError("Neural popularity training produced a non-finite loss. Reduce --neural-learning-rate.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_((hidden_weight, output_weight), 1.0)
        optimizer.step()

        current_loss = loss.detach().item()
        relative_improvement = (
            float("inf")
            if not np.isfinite(best_loss)
            else (best_loss - current_loss) / max(abs(best_loss), np.finfo(np.float64).eps)
        )
        if retention_active:
            if relative_improvement >= early_stopping_tolerance:
                best_loss = current_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        if verbose:
            epoch_iterator.set_postfix(
                phase="retention" if retention_active else "prediction",
                error=f"{prediction_loss.detach().item():.3f}",
                retention=f"{retention_loss.detach().item():.3f}",
                loss=f"{current_loss:.3f}",
                patience=f"{epochs_without_improvement}/{early_stopping_patience}",
            )
        if retention_active and epochs_without_improvement >= early_stopping_patience:
            if verbose:
                epoch_iterator.update(1)
                epoch_iterator.write(
                    f"Early stopping after {epoch + 1} epochs: relative error "
                    f"improvement remained below {early_stopping_tolerance:.3g} "
                    f"for {early_stopping_patience} epochs."
                )
            break

    with torch.no_grad():
        popularity_tensor = compute_torch_popularity()
        error = (popularity_tensor - selection).square().sum(dim=2).mean()
        final_retention_error = (selection_history * (popularity_tensor - 1).square()).sum(dim=2).mean()
    return (
        hidden_weight.detach().cpu().numpy(),
        output_weight.detach().cpu().numpy(),
        popularity_tensor.cpu().numpy(),
        float(error.item()),
        float(final_retention_error.item()),
    )


def gpu_expert_swap_counts(popularity_tensor, gpu_expert_count):
    """Counts GPU expert replacements between consecutive iterations."""
    load_numpy()
    popularity_tensor = np.asarray(popularity_tensor)
    if popularity_tensor.ndim != 3:
        raise ValueError("popularity_tensor must have shape (iterations, qmoes, experts).")
    total_experts = popularity_tensor.shape[1] * popularity_tensor.shape[2]
    if not 0 <= gpu_expert_count <= total_experts:
        raise ValueError(f"gpu_expert_count must be in [0, {total_experts}], got {gpu_expert_count}.")
    if popularity_tensor.shape[0] < 2 or gpu_expert_count in (0, total_experts):
        return np.zeros(max(0, popularity_tensor.shape[0] - 1), dtype=np.int64)

    flattened = popularity_tensor.reshape(popularity_tensor.shape[0], total_experts)
    gpu_indices = np.argpartition(flattened, total_experts - gpu_expert_count, axis=1)[:, -gpu_expert_count:]
    on_gpu = np.zeros(flattened.shape, dtype=bool)
    np.put_along_axis(on_gpu, gpu_indices, True, axis=1)
    return np.count_nonzero(on_gpu[1:] & ~on_gpu[:-1], axis=1)


def required_expert_swap_counts(
    selection_tensor,
    popularity_tensor,
    gpu_expert_counts,
    start_iteration,
    device=None,
):
    """Counts swaps required to keep every selected expert on GPU.

    At the end of the preceding iteration, each placement contains its K most
    popular experts. For every subsequent iteration, all missing selected
    experts enter and the least-popular resident experts that are not selected
    at that iteration leave. Counts are undefined when K cannot hold every
    selected expert and are returned as NaN.
    """
    import torch

    selection_tensor = validate_selection_tensor(selection_tensor)
    popularity_tensor = np.asarray(popularity_tensor)
    if popularity_tensor.shape != selection_tensor.shape:
        raise ValueError("P must have the same shape as S.")
    if not 0 < start_iteration < selection_tensor.shape[0]:
        raise ValueError("start_iteration must be within S and greater than zero.")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    selection = torch.from_numpy(selection_tensor).to(device=device).bool()
    popularity = torch.from_numpy(popularity_tensor).to(device=device, dtype=torch.float32)
    iterations, qmoes, experts = selection.shape
    total_experts = qmoes * experts
    selection_rows = selection.reshape(iterations, total_experts)
    popularity_rows = popularity.reshape(iterations, total_experts)
    counts = torch.tensor(gpu_expert_counts, dtype=torch.int64, device=device)
    selected_count = int(selection_rows[start_iteration:].sum(dim=1).max().item())
    valid = counts >= selected_count

    boundary_ranking = torch.argsort(popularity_rows[start_iteration - 1], stable=True)
    rank_positions = torch.empty_like(boundary_ranking)
    rank_positions[boundary_ranking] = torch.arange(total_experts, device=device)
    on_gpu = rank_positions.unsqueeze(0) >= (total_experts - counts).unsqueeze(1)
    swap_counts = torch.full(
        (counts.shape[0], iterations - start_iteration),
        torch.nan,
        dtype=torch.float32,
        device=device,
    )

    for output_index, iteration in enumerate(range(start_iteration, iterations)):
        selected = selection_rows[iteration].unsqueeze(0)
        missing = selected & ~on_gpu
        missing_counts = missing.sum(dim=1)
        evictable = on_gpu & ~selected
        leaving_scores = popularity_rows[iteration].unsqueeze(0).expand_as(on_gpu)
        leaving = torch.topk(
            leaving_scores.masked_fill(~evictable, torch.inf),
            selected_count,
            dim=1,
            largest=False,
            sorted=True,
        ).indices
        accepted = (
            torch.arange(selected_count, device=device).unsqueeze(0) < missing_counts.unsqueeze(1)
        ) & valid.unsqueeze(1)
        leaving_was_on_gpu = torch.gather(on_gpu, 1, leaving)
        on_gpu.scatter_(1, leaving, leaving_was_on_gpu & ~accepted)
        on_gpu |= selected & valid.unsqueeze(1)
        swap_counts[:, output_index] = torch.where(
            valid,
            missing_counts.float(),
            torch.nan,
        )
    return swap_counts.cpu().numpy()


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
    gpu_expert_counts = tuple(range(0, 1024 * 11, 1024))
    required_swap_counts = required_expert_swap_counts(
        selection_tensor,
        popularity_tensor,
        gpu_expert_counts,
        last_five_start,
    )
    required_swaps_last_five = np.array(
        [float(counts.mean()) if np.isfinite(counts).any() else np.nan for counts in required_swap_counts]
    )
    routing_metrics = _routing_metrics_for_gpu_counts(
        selection_tensor,
        popularity_tensor,
        gpu_expert_counts,
        iteration_ranges,
    )
    random_metrics = tuple(
        _routing_metrics_for_gpu_counts(
            selection_tensor[last_five_start:],
            random_popularity,
            gpu_expert_counts,
            ((0, selection_tensor.shape[0] - last_five_start),),
        )
        for random_popularity in random_popularities
    )
    rows = []
    for index, gpu_expert_count in enumerate(gpu_expert_counts):
        errors, swap_counts = routing_metrics[index]
        random_errors = tuple(metrics[index][0][0] for metrics in random_metrics)
        mean_swaps_first_five = float(swap_counts[: last_five_start - 1].mean())
        mean_swaps_last_five = float(swap_counts[last_five_start:].mean())
        rows.append(
            (
                gpu_expert_count,
                *errors,
                mean_swaps_first_five,
                mean_swaps_last_five,
                required_swaps_last_five[index],
                float(np.mean(random_errors)),
            )
        )

    print(
        "alpha_objective,gpu_expert_count,all_prompts,first_2_prompts,"
        "last_5_prompts,mean_swaps_first_5_prompts,"
        "mean_swaps_last_5_prompts,required_swaps_last_5_prompts,"
        "random_mean_last_5_prompts"
    )
    for row in rows:
        print(
            f"{label},{row[0]},{row[1]:.3f},{row[2]:.3f},"
            f"{row[3]:.3f},{row[4]:.3f},{row[5]:.3f},"
            f"{row[6]:.3f},{row[7]:.3f}"
        )

    markdown_lines = [
        f"**Routing results: `{label}`**",
        "",
        (
            "| Experts on GPU | All prompts | First 2 prompts | Last 5 prompts | "
            "Mean swaps, first 5 | Mean swaps, last 5 | "
            "Required swaps, last 5 | Random mean, last 5 |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown_lines.append(
            f"| {row[0]} | {row[1]:.3f} | {row[2]:.3f} | "
            f"{row[3]:.3f} | {row[4]:.3f} | {row[5]:.3f} | "
            f"{row[6]:.3f} | {row[7]:.3f} |"
        )

    random_popularity_rows = np.stack(random_popularities).reshape(len(random_popularities), -1)
    for max_swaps in (2, 6):
        limited_errors, limited_swap_counts = _limited_routing_metrics_batch(
            selection_tensor,
            popularity_tensor.reshape(selection_tensor.shape[0], -1),
            gpu_expert_counts,
            iteration_ranges,
            temporal_popularity=True,
            max_swaps=max_swaps,
        )
        random_limited_errors, _random_limited_swap_counts = _limited_routing_metrics_batch(
            selection_tensor,
            np.repeat(random_popularity_rows, len(gpu_expert_counts), axis=0),
            gpu_expert_counts * len(random_popularities),
            ((last_five_start, selection_tensor.shape[0]),),
            temporal_popularity=False,
            max_swaps=max_swaps,
        )
        random_limited_errors = random_limited_errors.reshape(len(random_popularities), len(gpu_expert_counts))
        limited_rows = []
        for index, gpu_expert_count in enumerate(gpu_expert_counts):
            errors = limited_errors[index]
            swap_counts = limited_swap_counts[:, index]
            mean_swaps_first_five = float(swap_counts[: last_five_start - 1].mean())
            mean_swaps_last_five = float(swap_counts[last_five_start:].mean())
            limited_rows.append(
                (
                    gpu_expert_count,
                    *errors,
                    mean_swaps_first_five,
                    mean_swaps_last_five,
                    required_swaps_last_five[index],
                    float(random_limited_errors[:, index].mean()),
                )
            )

        print(
            "alpha_objective,gpu_expert_count,all_prompts,first_2_prompts,"
            "last_5_prompts,mean_swaps_first_5_prompts,"
            "mean_swaps_last_5_prompts,required_swaps_last_5_prompts,"
            "random_mean_last_5_prompts"
        )
        for row in limited_rows:
            print(
                f"{label}_swap_limit_{max_swaps},{row[0]},{row[1]:.3f},"
                f"{row[2]:.3f},{row[3]:.3f},{row[4]:.3f},{row[5]:.3f},"
                f"{row[6]:.3f},{row[7]:.3f}"
            )

        markdown_lines.extend(
            (
                "",
                f"**Routing results with at most {max_swaps} swaps: `{label}`**",
                "",
                (
                    "| Experts on GPU | All prompts | First 2 prompts | Last 5 prompts | "
                    "Mean swaps, first 5 | Mean swaps, last 5 | "
                    "Required swaps, last 5 | Random mean, last 5 |"
                ),
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            )
        )
        for row in limited_rows:
            markdown_lines.append(
                f"| {row[0]} | {row[1]:.3f} | {row[2]:.3f} | "
                f"{row[3]:.3f} | {row[4]:.3f} | {row[5]:.3f} | "
                f"{row[6]:.3f} | {row[7]:.3f} |"
            )
    with markdown_output.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(markdown_lines))
        stream.write("\n\n")


def append_markdown_values(markdown_output, heading, values):
    """Appends named scalar values to the Markdown results file."""
    with markdown_output.open("a", encoding="utf-8") as stream:
        stream.write(f"## {heading}\n\n")
        for name, value in values:
            stream.write(f"- `{name} = {value:.3f}`\n")
        stream.write("\n")


def append_markdown_text(markdown_output, text):
    """Appends preformatted text to the Markdown results file."""
    with markdown_output.open("a", encoding="utf-8") as stream:
        stream.write(text.rstrip())
        stream.write("\n\n")


def run_alpha_optimizations(
    selection_tensor,
    grid_size,
    epochs,
    learning_rate,
    early_stopping_tolerance,
    early_stopping_patience,
    random_popularities,
    first_two_end,
    last_five_start,
    markdown_output,
):
    """Runs the three scalar-alpha methods and writes their routing results."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    execution_device = f"GPU ({torch.cuda.get_device_name()})" if device == "cuda" else "CPU"
    append_markdown_text(
        markdown_output,
        f"""## Configuration

- Optimization: `alpha`
- Search interval: `0.5 < alpha <= 1`
- PyTorch device: `{execution_device}`
- Squared-error epochs: `{epochs}`
- Squared-error learning rate: `{learning_rate}`
- Squared-error early-stopping relative tolerance: `{early_stopping_tolerance}`
- Squared-error early-stopping patience: `{early_stopping_patience}`
- Popularity: `P[t] = alpha * P[t-1] + (1-alpha) * S[t-1]`""",
    )
    print("Fitting alpha as a linear-regression coefficient...", flush=True)
    regression_alpha = optimize_regression_alpha(selection_tensor, verbose=True)
    print(f"regression_alpha={regression_alpha:.3f}")
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

    print(
        f"Optimizing squared-error alpha with PyTorch on {execution_device}: "
        f"at most {epochs} epochs; early stopping "
        f"tolerance={early_stopping_tolerance:.3g}, "
        f"patience={early_stopping_patience}.",
        flush=True,
    )
    alpha, error = optimize_popularity_alpha(
        selection_tensor,
        epochs=epochs,
        learning_rate=learning_rate,
        early_stopping_tolerance=early_stopping_tolerance,
        early_stopping_patience=early_stopping_patience,
        device=device,
        verbose=True,
    )
    print(f"optimal_alpha={alpha:.3f}")
    print(f"minimum_mean_squared_error={error:.3f}")
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

    print(
        f"Optimizing routing alpha with a {grid_size}-point grid on {execution_device} at K=6144...",
        flush=True,
    )
    routing_alpha, routing_error = optimize_routing_alpha(
        selection_tensor,
        gpu_expert_count=6144,
        grid_size=grid_size,
        device=device,
        verbose=True,
    )
    print(f"routing_optimal_alpha_k6144={routing_alpha:.3f}")
    print(f"routing_minimum_error_k6144={routing_error:.3f}")
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
    retention_penalty,
    use_previous_layer,
    retention_lambda,
    early_stopping_tolerance,
    early_stopping_patience,
    random_popularities,
    first_two_end,
    last_five_start,
    markdown_output,
    retention_window="past",
):
    """Jointly optimizes W0, W1, and W2 and writes routing results."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimization = (
        "future"
        if retention_window == "future"
        else ("short" if not use_previous_layer else ("penalty" if retention_penalty else "matrix"))
    )
    result_label = optimization
    loss_formula = "loss = mean(t,m) sum(i) (P[t,m,i] - S[t,m,i])**2"
    if retention_penalty:
        if retention_window == "future":
            loss_formula += (
                f"\n     + mean(t,m) sum(i,k=-{FUTURE_RETENTION_RADIUS}..-1,1..{FUTURE_RETENTION_RADIUS}) "
                f"{retention_lambda}/(|k|+{retention_lambda}) "
                "* S[t-k,m,i] * (P[t,m,i] - S[t-k,m,i])**2"
            )
        else:
            loss_formula += (
                f"\n     + mean(t,m) sum(i,k=1..{PENALTY_HISTORY_LENGTH}) "
                f"{retention_lambda}/(k+{retention_lambda}) "
                "* S[t-k,m,i] * (P[t,m,i] - S[t-k,m,i])**2"
            )
    retention_configuration = f"- Retention lambda: `{retention_lambda}`\n" if retention_penalty else ""
    previous_layer_formula = "\n       + W2[m] @ S[t,m-1]" if use_previous_layer else ""
    append_markdown_text(
        markdown_output,
        f"""## Configuration

- Optimization: `{optimization}`
- Device: `{device}`
- Maximum epochs: `{epochs}`
- Learning rate: `{learning_rate}`
{retention_configuration.rstrip()}
- Matrix coefficients: non-negative
- Early-stopping relative tolerance: `{early_stopping_tolerance}`
- Early-stopping patience: `{early_stopping_patience}`

```text
P[t,m] = W0[m] @ P[t-1,m]
       + W1[m] @ S[t-1,m]{previous_layer_formula}

{loss_formula}
```""",
    )
    if device == "cuda":
        execution_device = f"GPU ({torch.cuda.get_device_name()})"
    else:
        execution_device = "CPU"
    optimized_weight_names = "W0, W1, and W2" if use_previous_layer else "W0 and W1"
    print(
        f"Optimizing {optimized_weight_names} with PyTorch on {execution_device}: "
        f"at most {epochs} epochs, {selection_tensor.shape[0]} iterations per epoch; "
        f"early stopping tolerance={early_stopping_tolerance:.3g}, "
        f"patience={early_stopping_patience}.",
        flush=True,
    )
    (
        weight0,
        _weight1,
        _weight2,
        matrix_popularity,
        matrix_error,
        retention_error,
    ) = optimize_matrix_regression(
        selection_tensor,
        epochs=epochs,
        learning_rate=learning_rate,
        retention_penalty=retention_penalty,
        use_previous_layer=use_previous_layer,
        retention_window=retention_window,
        prompt_iteration_counts=(PROMPT_ITERATION_COUNTS if retention_window == "future" else None),
        retention_lambda=retention_lambda,
        early_stopping_tolerance=early_stopping_tolerance,
        early_stopping_patience=early_stopping_patience,
        device=device,
        verbose=True,
    )
    total_loss = matrix_error + retention_error if retention_penalty else matrix_error
    (
        diagonal_mass_ratio,
        diagonal_mean,
        off_diagonal_mean,
        diagonal_row_max_ratio,
    ) = matrix_diagonal_statistics(weight0)
    print(f"{result_label}_mean_squared_error={matrix_error:.3f}")
    print(f"{result_label}_retention_error={retention_error:.3f}")
    print(f"{result_label}_total_loss={total_loss:.3f}")
    print(f"{result_label}_w0_diagonal_mass_ratio={diagonal_mass_ratio:.6f}")
    print(f"{result_label}_w0_diagonal_mean={diagonal_mean:.6f}")
    print(f"{result_label}_w0_off_diagonal_mean={off_diagonal_mean:.6f}")
    print(f"{result_label}_w0_diagonal_row_max_ratio={diagonal_row_max_ratio:.6f}")
    first_five_swap_counts = gpu_expert_swap_counts(matrix_popularity[:last_five_start], 6144)
    last_five_swap_counts = gpu_expert_swap_counts(matrix_popularity[last_five_start:], 6144)
    swap_values = (
        ("expert_swaps_k6144_first_5_mean", float(first_five_swap_counts.mean())),
        (
            "expert_swaps_k6144_first_5_median",
            float(np.median(first_five_swap_counts)),
        ),
        ("expert_swaps_k6144_first_5_min", float(first_five_swap_counts.min())),
        ("expert_swaps_k6144_first_5_max", float(first_five_swap_counts.max())),
        ("expert_swaps_k6144_last_5_mean", float(last_five_swap_counts.mean())),
        (
            "expert_swaps_k6144_last_5_median",
            float(np.median(last_five_swap_counts)),
        ),
        ("expert_swaps_k6144_last_5_min", float(last_five_swap_counts.min())),
        ("expert_swaps_k6144_last_5_max", float(last_five_swap_counts.max())),
    )
    for name, value in swap_values:
        print(f"{name}={value:.3f}")
    append_markdown_values(
        markdown_output,
        (
            "Short matrix regression with bidirectional retention penalty"
            if optimization == "future"
            else (
                "Short matrix regression with previous-selection penalty"
                if optimization == "short"
                else ("Matrix regression with previous-selection penalty" if retention_penalty else "Matrix regression")
            )
        ),
        (
            (f"{result_label}_mean_squared_error", matrix_error),
            (f"{result_label}_retention_error", retention_error),
            (f"{result_label}_total_loss", total_loss),
            (f"{result_label}_w0_diagonal_mass_ratio", diagonal_mass_ratio),
            (f"{result_label}_w0_diagonal_mean", diagonal_mean),
            (f"{result_label}_w0_off_diagonal_mean", off_diagonal_mean),
            (f"{result_label}_w0_diagonal_row_max_ratio", diagonal_row_max_ratio),
            *swap_values,
        ),
    )
    print_routing_error_table(
        result_label,
        selection_tensor,
        matrix_popularity,
        random_popularities,
        first_two_end,
        last_five_start,
        markdown_output,
    )


def run_neural_optimization(
    selection_tensor,
    epochs,
    learning_rate,
    hidden_size,
    retention_lambda,
    early_stopping_tolerance,
    early_stopping_patience,
    random_popularities,
    first_two_end,
    last_five_start,
    markdown_output,
):
    """Optimizes a two-layer neural recurrence and writes routing results."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    execution_device = f"GPU ({torch.cuda.get_device_name()})" if device == "cuda" else "CPU"
    append_markdown_text(
        markdown_output,
        f"""## Configuration

- Optimization: `neural`
- Device: `{execution_device}`
- Maximum epochs: `{epochs}`
- Learning rate: `{learning_rate}`
- Hidden size: `{hidden_size}`
- Retention lambda: `{retention_lambda}`
- Retention schedule: disabled for the first `{NEURAL_RETENTION_WARMUP_EPOCHS}` epochs, enabled afterwards
- Early-stopping relative tolerance: `{early_stopping_tolerance}`
- Early-stopping patience: `{early_stopping_patience}`
- Network per QMoE: `Linear(768, {hidden_size}, bias=False) + Sigmoid + Linear({hidden_size}, 256, bias=False) + Sigmoid`

```text
H[t,m,:] = sigmoid(W1[m] @ concat(P[t-1,m,:], S[t-1,m,:], S[t,m-1,:]))
P[t,m,:] = sigmoid(W2[m] @ H[t,m,:])

loss = mean(t,m) sum(i) (P[t,m,i] - S[t,m,i])**2
     + mean(t,m) sum(i,k=1..{PENALTY_HISTORY_LENGTH}) {retention_lambda}/(k+{retention_lambda}) * S[t-k,m,i] * (P[t,m,i] - S[t-k,m,i])**2
```""",
    )
    print(
        f"Optimizing the two-layer neural recurrence with PyTorch on "
        f"{execution_device}: at most {epochs} epochs, "
        f"{selection_tensor.shape[0]} iterations per epoch; early stopping "
        f"starts after {NEURAL_RETENTION_WARMUP_EPOCHS} prediction-only epochs, "
        f"tolerance={early_stopping_tolerance:.3g}, "
        f"patience={early_stopping_patience}.",
        flush=True,
    )
    (
        _hidden_weight,
        _output_weight,
        neural_popularity,
        neural_error,
        retention_error,
    ) = optimize_neural_popularity(
        selection_tensor,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        retention_lambda=retention_lambda,
        early_stopping_tolerance=early_stopping_tolerance,
        early_stopping_patience=early_stopping_patience,
        device=device,
        verbose=True,
    )
    total_loss = neural_error + retention_error
    print(f"neural_mean_squared_error={neural_error:.3f}")
    print(f"neural_retention_error={retention_error:.3f}")
    print(f"neural_total_loss={total_loss:.3f}")
    append_markdown_values(
        markdown_output,
        "Neural popularity",
        (
            ("neural_mean_squared_error", neural_error),
            ("neural_retention_error", retention_error),
            ("neural_total_loss", total_loss),
        ),
    )
    print_routing_error_table(
        "neural",
        selection_tensor,
        neural_popularity,
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
                f"Expected {sum(PROMPT_ITERATION_COUNTS)} iterations for the 10 prompts, got {tensor.shape[0]}."
            )
        first_two_end = sum(PROMPT_ITERATION_COUNTS[:2])
        last_five_start = sum(PROMPT_ITERATION_COUNTS[:-5])
        random_popularities = tuple(
            np.random.default_rng(seed).random(tensor.shape[1:]) for seed in RANDOM_BASELINE_SEEDS
        )
        markdown_output = args.markdown_output
        if markdown_output is None:
            markdown_output = Path(__file__).with_suffix(f".{args.optimization}.md")
        temporary_markdown_output = markdown_output.with_name(f"{markdown_output.name}.tmp")
        temporary_markdown_output.write_text("# QMoE routing results\n\n", encoding="utf-8")
        print(f"Markdown results will be published to {markdown_output}.", flush=True)

        if args.optimization == "alpha":
            run_alpha_optimizations(
                tensor,
                args.alpha_grid_size,
                args.alpha_epochs,
                args.alpha_learning_rate,
                args.alpha_early_stopping_tolerance,
                args.alpha_early_stopping_patience,
                random_popularities,
                first_two_end,
                last_five_start,
                temporary_markdown_output,
            )
        elif args.optimization in ("matrix", "penalty", "short", "future"):
            run_matrix_optimization(
                tensor,
                args.matrix_epochs,
                args.matrix_learning_rate,
                args.optimization in ("penalty", "short", "future"),
                args.optimization not in ("short", "future"),
                args.retention_lambda,
                args.matrix_early_stopping_tolerance,
                args.matrix_early_stopping_patience,
                random_popularities,
                first_two_end,
                last_five_start,
                temporary_markdown_output,
                retention_window=("future" if args.optimization == "future" else "past"),
            )
        else:
            run_neural_optimization(
                tensor,
                args.neural_epochs,
                args.neural_learning_rate,
                args.neural_hidden_size,
                args.retention_lambda,
                args.neural_early_stopping_tolerance,
                args.neural_early_stopping_patience,
                random_popularities,
                first_two_end,
                last_five_start,
                temporary_markdown_output,
            )
        temporary_markdown_output.replace(markdown_output)
        print(f"Published Markdown results to {markdown_output}.", flush=True)


if __name__ == "__main__":
    main()
