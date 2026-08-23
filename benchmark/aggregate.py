"""
benchmark/aggregate.py
========================
Read the per-condition JSON result files produced by _run_stdio_pre.py /
_run_http_post.py and render a Markdown comparison table.

Usage:
    python3 benchmark/aggregate.py <label1>=<file1.json> <label2>=<file2.json> ...
"""
import json
import statistics
import sys


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def percentile(sorted_vals, pct):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * pct
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def summarize(samples):
    vals = sorted(s["elapsed_s"] for s in samples if s.get("ok"))
    if not vals:
        return None
    return {
        "n": len(vals),
        "mean_ms": round(statistics.mean(vals) * 1000, 1),
        "p50_ms": round(percentile(vals, 0.50) * 1000, 1),
        "p95_ms": round(percentile(vals, 0.95) * 1000, 1),
    }


def main(entries):
    conditions = {}
    for entry in entries:
        label, path = entry.split("=", 1)
        conditions[label] = load(path)

    scenario_names = []
    for data in conditions.values():
        for name in data.get("results", {}):
            if name not in scenario_names:
                scenario_names.append(name)

    lines = []
    lines.append("# pyATS MCP transport benchmark\n")
    lines.append("Connection setup time per condition:\n")
    lines.append("| condition | connect_elapsed_s |")
    lines.append("|---|---|")
    for label, data in conditions.items():
        lines.append(f"| {label} | {data.get('connect_elapsed_s', 'n/a'):.4f} |")
    lines.append("")

    for scenario in scenario_names:
        lines.append(f"## {scenario}\n")
        lines.append("| condition | n | mean (ms) | p50 (ms) | p95 (ms) |")
        lines.append("|---|---|---|---|---|")
        for label, data in conditions.items():
            entry = data.get("results", {}).get(scenario)
            if entry is None:
                lines.append(f"| {label} | - | not run | - | - |")
                continue
            if entry.get("status") == "not_available":
                lines.append(f"| {label} | - | tool not available | - | - |")
                continue
            summary = summarize(entry.get("samples", []))
            if summary is None:
                lines.append(f"| {label} | 0 | all calls failed | - | - |")
                continue
            lines.append(
                f"| {label} | {summary['n']} | {summary['mean_ms']} | "
                f"{summary['p50_ms']} | {summary['p95_ms']} |"
            )
        lines.append("")

    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:])
