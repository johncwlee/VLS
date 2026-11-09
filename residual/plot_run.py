"""PNG dashboards from a run's metrics.jsonl, one figure per metric group (e.g. update/critic, eval).

    ../../envs/vls/bin/python -m residual.plot_run <run_dir>     -> <run_dir>/plots/*.png
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from residual.monitor import AXES  # noqa: E402

AXIS_KEYS = {"update_step", "env_step", "round", "stage"}


def main(run_dir: Path) -> None:
    series = defaultdict(lambda: ([], []))  # metric -> (x, y)
    for line in (run_dir / "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        for key, value in row.items():
            if key in AXIS_KEYS or not isinstance(value, (int, float)):
                continue
            axis = AXES.get(key.split("/")[0])
            if axis in row:
                series[key][0].append(row[axis])
                series[key][1].append(value)
    groups = defaultdict(list)
    for key in sorted(series):
        parts = key.split("/")
        groups["_".join(parts[:2]) if len(parts) > 2 else parts[0]].append(key)
    out = run_dir / "plots"
    out.mkdir(exist_ok=True)
    for group, keys in groups.items():
        cols = min(4, len(keys))
        rows = math.ceil(len(keys) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
        for ax, key in zip(axes.flat, keys, strict=False):  # grid may have spare cells
            xs, ys = series[key]
            ax.plot(xs, ys, marker="." if len(xs) < 50 else None, linewidth=1)
            ax.set_title(key.split("/", 1)[1] if "/" in key else key, fontsize=8)
            ax.set_xlabel(AXES.get(key.split("/")[0], ""), fontsize=7)
            ax.tick_params(labelsize=7)
        for ax in list(axes.flat)[len(keys):]:
            ax.axis("off")
        fig.suptitle(group)
        fig.tight_layout()
        fig.savefig(out / f"{group}.png", dpi=100)
        plt.close(fig)
    print(f"wrote {len(groups)} figures to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
