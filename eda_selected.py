import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPEC = {
    "flow_duration": ["flow_duration", "flow duration", "duration", "flowdur"],
    "fwd_pkts_per_sec": ["fwd_pkts_per_sec", "fwd pkts/s", "fwdpktspers", "fwd_pkts_ps"],
    "payload_bps": [
        "payload_bytes_per_second", "payload bytes per second", "payload_bps",

    ],
    "total_fwd_bytes": ["total_fwd_bytes", "fwd bytes", "tot fwd bytes"],
    "total_bwd_bytes": ["total_bwd_bytes", "bwd bytes", "tot bwd bytes"],
}


def norm(s: str) -> str:
    return s.lower().replace(" ", "").replace(".", "").replace("_", "")


def find_first_match(cols: List[str], candidates: List[str], allow_substring: bool = True) -> Optional[str]:
    cols_norm = {norm(c): c for c in cols}

    for cand in candidates:
        n = norm(cand)
        if n in cols_norm:
            return cols_norm[n]
    if not allow_substring:
        return None

    for cand in candidates:
        n = norm(cand)
        hits = [orig for k, orig in cols_norm.items() if n in k]
        if hits:
            return sorted(hits, key=len)[0]
    return None


def read_any_csv(path: Path) -> pd.DataFrame:
    if path.is_dir():
        frames = [pd.read_csv(p, low_memory=False) for p in sorted(path.glob("*.csv"))]
        if not frames:
            raise FileNotFoundError("Brak plików CSV w katalogu.")
        df = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    else:
        df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def ensure_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def compute_stats(series: pd.Series) -> Dict[str, float]:
    s = series.dropna()
    return {
        "count": int(s.shape[0]),
        "missing": int(series.isna().sum()),
        "missing_ratio": float(series.isna().mean()),
        "mean": float(s.mean()) if len(s) else np.nan,
        "median": float(s.median()) if len(s) else np.nan,
        "min": float(s.min()) if len(s) else np.nan,
        "max": float(s.max()) if len(s) else np.nan,
        "std": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
    }


def histogram(figpath: Path, series: pd.Series, title: str, xlabel: str, bins: int = 50, sample: Optional[int] = None):
    data = series.dropna()
    if sample and len(data) > sample:
        data = data.sample(sample, random_state=42)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(data, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Częstość")
    plt.tight_layout()
    fig.savefig(figpath, dpi=150)
    plt.close(fig)


def boxplot(figpath: Path, series: pd.Series, title: str, ylabel: str):
    data = series.dropna().values
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(data, vert=True, whis=1.5, showfliers=True)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    fig.savefig(figpath, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="eda_top3")
    ap.add_argument("--label-col", default=None, help="np. Attack_type (opcjonalnie, tylko do boxplotów per klasa)")
    ap.add_argument("--per-class", action="store_true", help="Dodatkowe boxploty wg klasy dla TOP-K klas")
    ap.add_argument("--max-classes", type=int, default=8)
    ap.add_argument("--bins", type=int, default=50)
    ap.add_argument("--sample", type=int, default=None, help="Próbka do histogramów (dla bardzo dużych danych)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = read_any_csv(Path(args.input))
    cols = list(df.columns)

    flow_duration = find_first_match(cols, SPEC["flow_duration"])
    fwd_pkts_per_sec = find_first_match(cols, SPEC["fwd_pkts_per_sec"])
    payload_bps = find_first_match(cols, SPEC["payload_bps"])

    if payload_bps is None:
        total_fwd = find_first_match(cols, SPEC["total_fwd_bytes"])
        total_bwd = find_first_match(cols, SPEC["total_bwd_bytes"])
        if total_fwd and total_bwd and flow_duration:
            ensure_numeric(df, [total_fwd, total_bwd, flow_duration])
            eps = 0.001
            bytes_total = df[total_fwd].fillna(0) + df[total_bwd].fillna(0)
            denom = np.maximum(df[flow_duration].fillna(0).values, eps)
            df["payload_bytes_per_second_auto"] = bytes_total.values / denom
            payload_bps = "payload_bytes_per_second_auto"

    for c in [flow_duration, fwd_pkts_per_sec, payload_bps]:
        if c:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    mapping = {
        "flow_duration": flow_duration,
        "fwd_pkts_per_sec": fwd_pkts_per_sec,
        "payload_bytes_per_second": payload_bps
    }
    missing = [k for k, v in mapping.items() if v is None]

    rows = []
    for logical, col in mapping.items():
        if col is None:
            continue
        st = compute_stats(df[col])
        st["logical_name"] = logical
        st["column"] = col
        rows.append(st)

    stats_df = pd.DataFrame(rows)[
        ["logical_name", "column", "count", "missing", "missing_ratio", "mean", "median", "min", "max", "std"]
    ].sort_values(by="logical_name")
    stats_df.to_csv(outdir / "stats_top3.csv", index=False)
    (outdir / "stats_top3.md").write_text(
        "" + stats_df.to_markdown(index=False),
        encoding="utf-8"
    )

    plots = outdir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    for logical, col in mapping.items():
        if col is None:
            continue
        histogram(plots / f"hist_{logical}.png", df[col], f"Histogram: {logical}", logical, bins=args.bins,
                  sample=args.sample)
        boxplot(plots / f"box_{logical}.png", df[col], f"Boxplot: {logical}", logical)

    if args.per_class and args.label_col and args.label_col in df.columns:
        label_col = args.label_col
        df[label_col] = df[label_col].astype(str)
        class_counts = df[label_col].value_counts()
        top_classes = class_counts.index.astype(str).tolist()[:args.max_classes]
        per_class_dir = plots / "by_class"
        per_class_dir.mkdir(parents=True, exist_ok=True)
        for logical, col in mapping.items():
            if col is None:
                continue
            data = [df[df[label_col] == cls][col].dropna().values for cls in top_classes]
            fig, ax = plt.subplots(figsize=(max(8, len(top_classes) * 0.8), 5))
            ax.boxplot(data, vert=True, whis=1.5, showfliers=True)
            ax.set_title(f"{logical} — boxplot wg klasy (TOP-{len(top_classes)})")
            ax.set_xticks(np.arange(1, len(top_classes) + 1))
            ax.set_xticklabels(top_classes, rotation=90)
            ax.set_ylabel(logical)
            plt.tight_layout()
            fig.savefig(per_class_dir / f"box_by_class_{logical}.png", dpi=150)
            plt.close(fig)
        class_counts.to_csv(outdir / "class_distribution.csv")

    meta = {
        "input": str(Path(args.input).resolve()),
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "selected": mapping,
        "missing_selected": missing,
        "args": vars(args),
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== EDA (top3) zakończone ===")
    print("Wybrane kolumny:", mapping)
    if missing:
        print("UWAGA — nie znaleziono:", missing)


if __name__ == "__main__":
    main()
