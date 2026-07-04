"""
Compute RA / LA / RecA / BTI for the recurring permuted-MNIST setting, adapted to
MERLIN's output format (the essence of the original La-MAML metric script).

Metric definitions (unchanged from the reference):
  - Retained Accuracy (RA):   average of the LAST snapshot row across the kept tasks.
  - Learned Accuracy (LA):     average accuracy of each task at its FIRST encounter.
  - Recurring Accuracy (RecA): average accuracy of each task at its SECOND encounter.
  - BTI:                       average (end - first) across the kept tasks.

What differs from the reference:
  MERLIN already evaluates over the distinct recurring subset, so its
  `merlin_results.pt` stores a (n_presentations x n_subset) matrix together with
  `result_t`, the per-presentation original task id (the real train order).
  We therefore derive each task's first/second encounter rows directly from
  `result_t` instead of hardcoding SPLIT_TRAIN_ORDER, and we map a task to its
  column using the eval-task labels. This makes the script robust to any subset
  / order, including the schedule produced by config/permMNIST_recurring_local.yml.

Supported inputs:
  - <run>/pickles/merlin_results.pt   (preferred: tuple (result_t, result_a, stats))
  - .npy / .npz                       (matrix only; see --col-tasks / fallback below)
  - .txt / .csv / .tsv                (matrix only)

When only a matrix is given (no result_t), the script falls back to the original
reference behaviour: if it has >=20 columns it keeps even task ids [0,2,..,18],
reorders to first-appearance order, and derives encounters from SPLIT_TRAIN_ORDER.

Usage:
  python compute_recurring_metrics.py output/<run>/pickles/merlin_results.pt --per-task
  python compute_recurring_metrics.py results.txt
  python compute_recurring_metrics.py results.npz --npz-key acc --output report.txt
"""

from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
from datetime import datetime


# La-MAML split-view train order (used only for the matrix-only fallback).
SPLIT_TRAIN_ORDER = [8, 28, 9, 29, 12, 32, 13, 33, 16, 36, 17, 37, 0, 20, 1, 21, 4, 24, 5, 25]
# First-appearance order of the (even) original task ids (matrix-only fallback).
FIRST_APPEAR_ORIG_TASK_ORDER = [4, 14, 6, 16, 8, 18, 0, 10, 2, 12]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_merlin_pt(path: Path):
    """Load MERLIN's (result_t, result_a, stats) tuple. Returns (result_t, mat)."""
    import torch
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not (isinstance(obj, (tuple, list)) and len(obj) >= 2):
        raise ValueError(f"{path}: expected a (result_t, result_a, ...) tuple.")
    result_t = np.asarray(obj[0], dtype=float).reshape(-1)
    mat = np.asarray(obj[1], dtype=float)
    if mat.ndim == 1:
        mat = mat[None, :]
    return result_t, mat


def load_matrix(path: Path, *, npz_key: str | None = None) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        data = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        z = np.load(path, allow_pickle=False)
        try:
            if npz_key is not None:
                data = z[npz_key]
            else:
                keys = list(z.keys())
                if len(keys) != 1:
                    raise ValueError(f"{path} contains multiple arrays {keys}. Provide --npz-key.")
                data = z[keys[0]]
        finally:
            z.close()
    else:
        data = np.loadtxt(path, dtype=float)
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[None, :]
    return data


def read_col_labels(pt_path: Path) -> list[int] | None:
    """Best-effort: read 'Col labels:' from the sibling logs/merlin_eval.txt."""
    eval_txt = pt_path.parent.parent / "logs" / "merlin_eval.txt"
    if not eval_txt.exists():
        return None
    for line in eval_txt.read_text().splitlines():
        if line.startswith("Col labels:"):
            return [int(x) for x in line.split(":", 1)[1].split()]
    return None


# ---------------------------------------------------------------------------
# Metric core
# ---------------------------------------------------------------------------

def _metrics_from_encounters(mat, tasks, col_of, first_rows, second_rows):
    end_row = mat.shape[0] - 1
    kept_cols = [col_of[t] for t in tasks]
    retained = float(mat[end_row, kept_cols].mean())

    learned_vals, recurring_vals, bti_vals, per_task = [], [], [], {}
    for tid in tasks:
        c = col_of[tid]
        a1 = float(mat[first_rows[tid], c])
        a2 = float(mat[second_rows[tid], c])
        a_end = float(mat[end_row, c])
        learned_vals.append(a1)
        recurring_vals.append(a2)
        bti_vals.append(a_end - a1)
        per_task[tid] = {
            "first_row": int(first_rows[tid]),
            "second_row": int(second_rows[tid]),
            "learned": a1,
            "recurring": a2,
            "end": a_end,
            "bti": a_end - a1,
        }

    return {
        "retained_accuracy": retained,
        "learned_accuracy": float(np.mean(learned_vals)),
        "recurring_accuracy": float(np.mean(recurring_vals)),
        "bti": float(np.mean(bti_vals)),
        "tasks_order": list(tasks),
        "per_task": per_task,
    }


def compute_metrics_merlin(result_t: np.ndarray, mat: np.ndarray, col_tasks=None) -> dict:
    """Derive encounters from result_t (the actual presentation order)."""
    if mat.shape[0] != result_t.shape[0]:
        raise ValueError(f"result_a rows ({mat.shape[0]}) != result_t length ({result_t.shape[0]}).")

    presentation_tids = [int(round(x)) for x in result_t]

    # Column -> task-id mapping.
    if col_tasks is None:
        # Default: columns are the distinct subset tasks in sorted order, matching
        # eval_tasks = sorted(subset) used by learn_continually().
        col_tasks = sorted(set(presentation_tids))
    if len(col_tasks) != mat.shape[1]:
        raise ValueError(f"col labels ({len(col_tasks)}) != result_a columns ({mat.shape[1]}).")
    col_of = {int(t): i for i, t in enumerate(col_tasks)}

    # Tasks in first-appearance order; first/second encounter rows from result_t.
    tasks, first_rows, second_rows = [], {}, {}
    for tid in presentation_tids:
        if tid not in first_rows:
            tasks.append(tid)
            positions = [i for i, t in enumerate(presentation_tids) if t == tid]
            if len(positions) < 2:
                raise ValueError(f"Task {tid} appears {len(positions)} time(s); expected >= 2 (recurring).")
            first_rows[tid] = positions[0]
            second_rows[tid] = positions[1]

    missing = [t for t in tasks if t not in col_of]
    if missing:
        raise ValueError(f"Tasks {missing} have no matching evaluation column {col_tasks}.")

    return _metrics_from_encounters(mat, tasks, col_of, first_rows, second_rows)


def compute_metrics_fallback(mat: np.ndarray) -> dict:
    """Original reference behaviour for a bare 20-column matrix (no result_t)."""
    if mat.shape[1] < 20:
        raise ValueError(f"Matrix-only input needs >= 20 columns (orig tasks 0..19); got {mat.shape[1]}.")

    even_orig_tids = list(range(0, 20, 2))
    mat_even = mat[:, even_orig_tids]
    reorder_idx = [tid // 2 for tid in FIRST_APPEAR_ORIG_TASK_ORDER]
    mat_re = mat_even[:, reorder_idx]  # column j == FIRST_APPEAR_ORIG_TASK_ORDER[j]

    orig_seq = [sid // 2 for sid in SPLIT_TRAIN_ORDER]
    tasks = FIRST_APPEAR_ORIG_TASK_ORDER[:]
    col_of = {tid: j for j, tid in enumerate(tasks)}

    first_rows, second_rows = {}, {}
    for tid in tasks:
        positions = [i for i, t in enumerate(orig_seq) if t == tid]
        if len(positions) < 2:
            raise ValueError(f"Task {tid} appears {len(positions)} time(s) in SPLIT_TRAIN_ORDER; expected 2.")
        first_rows[tid], second_rows[tid] = positions[0], positions[1]

    max_needed = max(max(first_rows.values()), max(second_rows.values()))
    if mat_re.shape[0] <= max_needed:
        raise ValueError(f"Not enough rows: have {mat_re.shape[0]}, need >= {max_needed + 1}.")

    return _metrics_from_encounters(mat_re, tasks, col_of, first_rows, second_rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_report_lines(path: Path, mat: np.ndarray, out: dict, include_per_task: bool) -> list[str]:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# ---- recurring-metrics appended {ts} ----",
        f"# File: {path}",
        f"# Rows x Cols: {mat.shape[0]} x {mat.shape[1]}",
        f"# Tasks (first-appearance order): {out['tasks_order']}",
        f"# RA  (retained, last row avg):        {out['retained_accuracy']:.6f}",
        f"# LA  (learned, first encounter avg):  {out['learned_accuracy']:.6f}",
        f"# RecA(recurring, second enc. avg):    {out['recurring_accuracy']:.6f}",
        f"# BTI (end - first, avg):              {out['bti']:.6f}",
    ]
    if include_per_task:
        lines.append("# Per-task details:")
        for tid in out["tasks_order"]:
            d = out["per_task"][tid]
            lines.append(
                f"# tid={tid:2d}  first_row={d['first_row']:2d}  second_row={d['second_row']:2d}  "
                f"learned={d['learned']:.6f}  recurring={d['recurring']:.6f}  "
                f"end={d['end']:.6f}  bti={d['bti']:+.6f}"
            )
    lines.append("# ---- end recurring-metrics ----")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_path", type=str, help="merlin_results.pt, or a .txt/.npy/.npz matrix")
    ap.add_argument("--npz-key", type=str, default=None)
    ap.add_argument("--col-tasks", type=str, default=None,
                    help="Comma-separated eval-task ids for the columns (overrides inference).")
    ap.add_argument("--output", type=str, default=None,
                    help="Where to append the report (default: input if text; else <input>.metrics.txt).")
    ap.add_argument("--per-task", action="store_true")
    args = ap.parse_args()

    in_path = Path(args.results_path)
    suffix = in_path.suffix.lower()
    col_tasks = [int(x) for x in args.col_tasks.split(",")] if args.col_tasks else None

    if suffix == ".pt":
        result_t, mat = load_merlin_pt(in_path)
        if col_tasks is None:
            col_tasks = read_col_labels(in_path)  # may be None -> inferred
        out = compute_metrics_merlin(result_t, mat, col_tasks=col_tasks)
    else:
        mat = load_matrix(in_path, npz_key=args.npz_key)
        # If a 10-column subset matrix is supplied with explicit col-tasks, treat
        # it like MERLIN output but without result_t -> not enough info; require .pt.
        out = compute_metrics_fallback(mat)

    report_lines = format_report_lines(in_path, mat, out, include_per_task=bool(args.per_task))
    for ln in report_lines:
        print(ln)

    if args.output is not None:
        out_path = Path(args.output)
    elif suffix in {".npy", ".npz", ".pt"}:
        out_path = in_path.with_suffix(in_path.suffix + ".metrics.txt")
    else:
        out_path = in_path
    with out_path.open("a", encoding="utf-8") as f:
        f.write("\n" + "\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
