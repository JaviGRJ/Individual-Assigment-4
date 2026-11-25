"""Generate labelled Assignment 4 SVG charts from summary.csv."""

from __future__ import annotations

import csv
import html
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_summary() -> list[dict[str, str]]:
    with (ROOT / "summary.csv").open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def chart(
    rows: list[dict[str, str]],
    value_key: str,
    title: str,
    y_label: str,
    filename: str,
    *,
    log_scale: bool = False,
) -> None:
    valid_rows = [
        row for row in rows
        if row.get(value_key) not in (None, "")
        and row.get("status") == "ok"
    ]
    width, height = 1400, 760
    left, top, right, bottom = 95, 70, 30, 230
    plot_width, plot_height = width - left - right, height - top - bottom
    values = [float(row[value_key]) for row in valid_rows]
    maximum = max(values, default=1.0)
    scale_max = maximum if maximum > 0 else 1.0
    bar_width = plot_width / max(len(valid_rows), 1)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<style>text{font-family:Arial,sans-serif;font-size:12px;fill:#17202a}.title{font-size:22px;font-weight:bold}.axis{stroke:#17202a;stroke-width:2}.grid{stroke:#d9e0e6}.bar{fill:#357edd}</style>',
        f'<text x="{width/2}" y="34" text-anchor="middle" class="title">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" class="axis"/>',
        f'<line x1="{left}" y1="{top+plot_height}" x2="{left+plot_width}" y2="{top+plot_height}" class="axis"/>',
        f'<text x="22" y="{top+plot_height/2}" transform="rotate(-90 22 {top+plot_height/2})" text-anchor="middle">{html.escape(y_label)}</text>',
    ]
    for index in range(6):
        fraction = index / 5
        y = top + plot_height - fraction * plot_height
        label = f"{scale_max * fraction:.2f}"
        lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot_width}" y2="{y:.2f}" class="grid"/>'
        )
        lines.append(
            f'<text x="{left-8}" y="{y+4:.2f}" text-anchor="end">{label}</text>'
        )
    for index, row in enumerate(valid_rows):
        value = float(row[value_key])
        bar_height = value / scale_max * plot_height
        x = left + index * bar_width + bar_width * 0.15
        y = top + plot_height - bar_height
        workers = row.get("workers") or "n/a"
        label = f'{row["method"]} n={row["size"]} w={workers}'
        lines.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width*0.7:.2f}" height="{bar_height:.2f}" class="bar"><title>{html.escape(label)}: {value:.6f}</title></rect>'
        )
        label_x, label_y = x + bar_width * 0.35, top + plot_height + 18
        lines.append(
            f'<text x="{label_x:.2f}" y="{label_y}" transform="rotate(58 {label_x:.2f} {label_y})">{html.escape(label)}</text>'
        )
    lines.extend(
        [
            f'<text x="{left+plot_width/2}" y="{height-18}" text-anchor="middle">Configuration</text>',
            "</svg>",
        ]
    )
    (ROOT / filename).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = read_summary()
    chart(
        rows,
        "median_end_to_end_ms",
        "Assignment 4 end-to-end runtime",
        "Median time (ms)",
        "runtime.svg",
    )
    chart(
        rows,
        "speedup_vs_sequential",
        "Assignment 4 speedup vs sequential",
        "Speedup (ratio)",
        "speedup.svg",
    )
    chart(
        rows,
        "median_worker_rss_total_sampled_mib",
        "Assignment 4 sampled worker memory",
        "Sum of sampled worker RSS (MiB)",
        "worker_memory.svg",
    )


if __name__ == "__main__":
    main()
