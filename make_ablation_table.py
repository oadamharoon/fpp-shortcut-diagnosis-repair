"""
Aggregate the per-variant metric JSONs from eval_phicalnet_ablation.py into a
single Markdown table, reference row first, columns ordered to match the
paper's Table 8 (object) and Table 14 (pixel) so the numbers paste directly.

Usage:
    python make_ablation_table.py --results_dir ablation_results
"""
import argparse
import json
from pathlib import Path

# Display order: reference at top, then the single-term-dropped variants.
ROW_ORDER = [
    ("reference", "Full loss (reference)"),
    ("drop_circular", "w_circ = 0  (drop $L_{\\mathrm{circ}}$, primary phase)"),
    ("drop_grad", "w_grad = 0  (drop $L_{\\nabla}$, smoothness)"),
    ("drop_depth", "w_depth = 0  (drop $L_d$, depth bridge)"),
]

HEADER = [
    "Variant", "Obj MAE (mm)", "Obj RMSE (mm)",
    "Pix MAE (mm)", "Pix RMSE (mm)",
    "P50 (mm)", "P90 (mm)", "P99 (mm)", "P99.9 (mm)",
    "px > 100 mm", "% > 100 mm",
]


def fmt_row(label, m):
    p = m["pixel_pctl_mm"]
    return [
        label,
        f"{m['object_mae_mm']:.2f}",
        f"{m['object_rmse_mm']:.2f}",
        f"{m['pixel_mae_mm']:.2f}",
        f"{m['pixel_rmse_mm']:.2f}",
        f"{p['50']:.2f}",
        f"{p['90']:.2f}",
        f"{p['99']:.2f}",
        f"{p['99.9']:.1f}",
        f"{m['n_err_gt_100mm']:,}",
        f"{m['pct_err_gt_100mm']:.3f}",
    ]


def main(args):
    results_dir = Path(args.results_dir)
    rows = []
    for key, label in ROW_ORDER:
        jp = results_dir / f"{key}_metrics.json"
        if not jp.exists():
            print(f"WARN: missing {jp}; skipping row '{label}'")
            continue
        with open(jp) as f:
            rows.append(fmt_row(label, json.load(f)))

    if not rows:
        print("No metric JSONs found. Run eval_phicalnet_ablation.py first.")
        return

    widths = [max(len(HEADER[i]), max(len(r[i]) for r in rows)) for i in range(len(HEADER))]

    def line(cells):
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    sep = "| " + " | ".join("-" * widths[i] for i in range(len(HEADER))) + " |"

    table = "\n".join([line(HEADER), sep] + [line(r) for r in rows])
    print("\n" + table + "\n")

    out_path = results_dir / "ablation_table.md"
    with open(out_path, "w") as f:
        f.write("# PhiCalNet loss-term ablation\n\n")
        f.write("All variants retrained from the reference PhiCalNet "
                "(`train_phase_A1_refined_v2.py`) with one loss weight zeroed; "
                "reference weights (circular, grad, depth) = (1.0, 0.5, 0.1). "
                "Reference row evaluated from the existing checkpoint (not retrained). "
                "Metrics computed with the canonical raw-depth pipeline "
                "(`full_unwrap` + calibration layer), matching Tables 8 and 14.\n\n")
        f.write(table + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="ablation_results")
    args = parser.parse_args()
    main(args)
