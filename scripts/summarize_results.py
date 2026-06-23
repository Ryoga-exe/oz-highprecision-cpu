#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_blas_libs(path: Path) -> str:
    if not path.exists():
        return ""
    for line in path.read_text().splitlines():
        if line.startswith("BLAS_LIBS="):
            return line.split("=", 1)[1]
    return ""


def as_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except ValueError:
        return default


def shape(row: Dict[str, str]) -> str:
    return f"{row.get('m', '')}x{row.get('n', '')}x{row.get('k', '')}"


def fmt(value: object) -> str:
    if isinstance(value, float):
        if value < 0:
            return ""
        if value == 0:
            return "0"
        if abs(value) < 0.001 or abs(value) >= 1000:
            return f"{value:.6g}"
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def print_table(title: str, rows: List[Dict[str, object]], columns: List[str]) -> None:
    print(f"## {title}")
    print()
    if not rows:
        print("_No rows found._")
        print()
        return

    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        print("| " + " | ".join(fmt(row.get(column, "")) for column in columns) + " |")
    print()


def write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def discover_backend_dirs(roots: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if (root / "quick.csv").exists():
            out.append(root)
            continue
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / "quick.csv").exists():
                out.append(child)
        for child in sorted(root.glob("*/*")):
            if child.is_dir() and (child / "quick.csv").exists() and child not in out:
                out.append(child)
    return sorted(out)


def run_id_for(backend_dir: Path) -> str:
    if backend_dir.parent.name == "results":
        return ""
    return backend_dir.parent.name


def backend_for(backend_dir: Path) -> str:
    return backend_dir.name


def collect_quick(backend_dirs: Iterable[Path]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for backend_dir in backend_dirs:
        blas_libs = read_blas_libs(backend_dir / "blas-info.txt")
        for row in read_csv(backend_dir / "quick.csv"):
            reuse = as_float(row, "oz_reuse_seconds", -1.0)
            naive = as_float(row, "naive_hp_seconds", -1.0)
            fp64 = as_float(row, "fp64_seconds", -1.0)
            rows.append({
                "run": run_id_for(backend_dir),
                "backend": backend_for(backend_dir),
                "shape": shape(row),
                "blas_libs": blas_libs,
                "moduli": row.get("moduli", ""),
                "reuse_s": reuse,
                "fp64_s": fp64,
                "naive_s": naive,
                "speedup_vs_naive": (naive / reuse) if naive > 0 and reuse > 0 else "",
                "fp64_vs_reuse": (reuse / fp64) if fp64 > 0 and reuse > 0 else "",
            })
    return rows


def collect_precision(backend_dirs: Iterable[Path]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for backend_dir in backend_dirs:
        csv_rows = read_csv(backend_dir / "precision_64x64x64.csv")
        baselines: Dict[str, float] = {}
        for row in csv_rows:
            if row.get("target_bits") == "256":
                baselines[shape(row)] = as_float(row, "total_seconds", -1.0)
        for row in csv_rows:
            total = as_float(row, "total_seconds", -1.0)
            baseline = baselines.get(shape(row), -1.0)
            rows.append({
                "run": run_id_for(backend_dir),
                "backend": backend_for(backend_dir),
                "shape": shape(row),
                "target_bits": row.get("target_bits", ""),
                "moduli": row.get("moduli", ""),
                "planned_bits": row.get("planned_bits", ""),
                "total_s": total,
                "speedup_vs_256": (baseline / total) if baseline > 0 and total > 0 else "",
                "residue_fraction": as_float(row, "residue_fraction"),
                "blas_fraction": as_float(row, "blas_fraction"),
                "crt_fraction": as_float(row, "crt_fraction"),
            })
    return rows


def collect_best_block(backend_dirs: Iterable[Path]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for backend_dir in backend_dirs:
        csv_rows = read_csv(backend_dir / "block_sweep_64x512x64.csv")
        if not csv_rows:
            continue
        best = min(csv_rows, key=lambda row: as_float(row, "reuse_seconds", float("inf")))
        auto_row = next((row for row in csv_rows if row.get("requested_block") == "0"), None)
        best_time = as_float(best, "reuse_seconds", -1.0)
        auto_time = as_float(auto_row or {}, "reuse_seconds", -1.0)
        rows.append({
            "run": run_id_for(backend_dir),
            "backend": backend_for(backend_dir),
            "shape": shape(best),
            "best_requested_block": best.get("requested_block", ""),
            "best_effective_block": best.get("effective_block", ""),
            "best_s": best_time,
            "auto_s": auto_time,
            "speedup_vs_auto": (auto_time / best_time) if auto_time > 0 and best_time > 0 else "",
        })
    return rows


def collect_best_crt_threads(backend_dirs: Iterable[Path]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for backend_dir in backend_dirs:
        csv_rows = read_csv(backend_dir / "crt_threads_64x512x64.csv")
        if not csv_rows:
            continue
        best = min(csv_rows, key=lambda row: as_float(row, "total_seconds", float("inf")))
        auto_row = next((row for row in csv_rows
                         if row.get("requested_threads") == "0" and
                         row.get("auto_max_threads", "0") == "0"), None)
        best_time = as_float(best, "total_seconds", -1.0)
        auto_time = as_float(auto_row or {}, "total_seconds", -1.0)
        rows.append({
            "run": run_id_for(backend_dir),
            "backend": backend_for(backend_dir),
            "shape": shape(best),
            "best_requested_threads": best.get("requested_threads", ""),
            "best_auto_max_threads": best.get("auto_max_threads", ""),
            "best_effective_threads": best.get("effective_threads", ""),
            "best_s": best_time,
            "auto_s": auto_time,
            "speedup_vs_auto": (auto_time / best_time) if auto_time > 0 and best_time > 0 else "",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize saved oz-highprecision-cpu benchmark result directories.")
    parser.add_argument("roots", nargs="*", type=Path,
                        help="Result roots or backend directories. Defaults to results/.")
    parser.add_argument("--csv-dir", type=Path,
                        help="Optional directory where aggregate CSV files are written.")
    args = parser.parse_args()

    roots = args.roots or [Path("results")]
    backend_dirs = discover_backend_dirs(roots)

    quick = collect_quick(backend_dirs)
    precision = collect_precision(backend_dirs)
    blocks = collect_best_block(backend_dirs)
    crt_threads = collect_best_crt_threads(backend_dirs)

    print_table("Quick Benchmarks", quick, [
        "run", "backend", "shape", "moduli", "reuse_s", "fp64_s", "naive_s",
        "speedup_vs_naive", "fp64_vs_reuse", "blas_libs"])
    print_table("Precision Sweep", precision, [
        "run", "backend", "shape", "target_bits", "moduli", "planned_bits",
        "total_s", "speedup_vs_256", "residue_fraction", "blas_fraction", "crt_fraction"])
    print_table("Best Residue Block", blocks, [
        "run", "backend", "shape", "best_requested_block", "best_effective_block",
        "best_s", "auto_s", "speedup_vs_auto"])
    print_table("Best CRT Threads", crt_threads, [
        "run", "backend", "shape", "best_requested_threads", "best_auto_max_threads",
        "best_effective_threads", "best_s", "auto_s", "speedup_vs_auto"])

    if args.csv_dir:
        write_csv(args.csv_dir / "quick_summary.csv", quick, [
            "run", "backend", "shape", "moduli", "reuse_s", "fp64_s", "naive_s",
            "speedup_vs_naive", "fp64_vs_reuse", "blas_libs"])
        write_csv(args.csv_dir / "precision_summary.csv", precision, [
            "run", "backend", "shape", "target_bits", "moduli", "planned_bits",
            "total_s", "speedup_vs_256", "residue_fraction", "blas_fraction", "crt_fraction"])
        write_csv(args.csv_dir / "block_summary.csv", blocks, [
            "run", "backend", "shape", "best_requested_block", "best_effective_block",
            "best_s", "auto_s", "speedup_vs_auto"])
        write_csv(args.csv_dir / "crt_thread_summary.csv", crt_threads, [
            "run", "backend", "shape", "best_requested_threads", "best_auto_max_threads",
            "best_effective_threads", "best_s", "auto_s", "speedup_vs_auto"])


if __name__ == "__main__":
    main()
