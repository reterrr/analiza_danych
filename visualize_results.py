#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wizualizacja i raport z eksperymentów:
- Wczytuje outdir/results.csv
- Buduje tabelę najlepszych wyników per model (wg val_macro_f1)
- Generuje wykresy:
  1) Best val_macro_f1 per model (bar)
  2) Scatter: train_time vs val_macro_f1 (wszystkie uruchomienia)
  3) Krzywe strojenia:
      - SVM: val_macro_f1 vs C (log)
      - DecisionTree: val_macro_f1 vs max_depth
      - XGBoost: val_macro_f1 vs n_estimators (dla best learning_rate)
- Zapisuje:
  - outdir/summary_table.csv
  - outdir/summary_table.tex
  - outdir/figures/*.png
  - outdir/report.md
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_params(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        return {}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expdir", default="experiments")
    ap.add_argument("--select-metric", default="val_macro_f1",
                    choices=["val_macro_f1", "val_balanced_accuracy", "val_accuracy"])
    args = ap.parse_args()

    expdir = Path(args.expdir)
    results_path = expdir / "results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Brak {results_path}. Najpierw uruchom run_experiments.py")

    df = pd.read_csv(results_path)
    df["params"] = df["params_json"].apply(parse_params)

    figures_dir = expdir / "figures"
    ensure_dir(figures_dir)

    metric = args.select_metric

    # -------------------------
    # 1) Tabela najlepszych wyników per model
    # -------------------------
    best_rows = []
    for model_name, g in df.groupby("model"):
        g_sorted = g.sort_values(by=metric, ascending=False).head(1)
        best_rows.append(g_sorted.iloc[0])

    best_df = pd.DataFrame(best_rows).copy()
    best_df = best_df.sort_values(by=metric, ascending=False)

    # Uporządkowane kolumny do prezentacji
    summary_cols = [
        "model", "stage",
        "train_time_s",
        "val_macro_f1", "val_balanced_accuracy", "val_accuracy",
        "test_macro_f1", "test_balanced_accuracy", "test_accuracy",
        "best_iteration", "best_score",
        "params_json"
    ]
    summary_df = best_df[summary_cols]

    summary_df.to_csv(expdir / "summary_table.csv", index=False, encoding="utf-8")
    # LaTeX do pracy
    summary_df.to_latex(expdir / "summary_table.tex", index=False, float_format="%.4f")

    # -------------------------
    # 2) Wykres: najlepszy wynik metryki na walidacji per model
    # -------------------------
    plt.figure()
    plt.bar(summary_df["model"], summary_df[metric])
    plt.ylabel(metric)
    plt.title(f"Najlepszy wynik {metric} (walidacja) dla każdej metody")
    plt.tight_layout()
    plt.savefig(figures_dir / "best_metric_by_model.png", dpi=200)
    plt.close()

    # -------------------------
    # 3) Scatter: czas treningu vs metryka walidacyjna
    # -------------------------
    plt.figure()
    for model_name, g in df.groupby("model"):
        plt.scatter(g["train_time_s"], g[metric], label=model_name)
    plt.xlabel("Czas treningu [s]")
    plt.ylabel(metric)
    plt.title(f"Zależność jakości ({metric}) od czasu treningu")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "metric_vs_train_time_scatter.png", dpi=200)
    plt.close()

    # -------------------------
    # 4) Krzywe strojenia: SVM (val_macro_f1 vs C)
    # -------------------------
    svm_df = df[df["model"].isin(["LinearSVC", "SVC_RBF"])].copy()
    if not svm_df.empty:
        svm_df["C"] = svm_df["params"].apply(lambda d: float(d.get("C", np.nan)))
        svm_df = svm_df.dropna(subset=["C"])

        plt.figure()
        # agregacja: średnia po stage/ewentualnie powtórzeniach (tu zwykle brak powtórzeń)
        svm_agg = svm_df.groupby("C")[metric].mean().reset_index().sort_values("C")
        plt.plot(svm_agg["C"], svm_agg[metric], marker="o")
        plt.xscale("log")
        plt.xlabel("C (skala log)")
        plt.ylabel(metric)
        plt.title(f"SVM: wpływ C na {metric}")
        plt.tight_layout()
        plt.savefig(figures_dir / "svm_metric_vs_C.png", dpi=200)
        plt.close()

    # -------------------------
    # 5) Krzywe strojenia: DecisionTree (val_macro_f1 vs max_depth)
    # -------------------------
    dt_df = df[df["model"] == "DecisionTree"].copy()
    if not dt_df.empty:
        dt_df["max_depth"] = dt_df["params"].apply(lambda d: d.get("max_depth", None))
        # None mapujemy na np.nan, żeby móc rysować
        dt_df["max_depth_num"] = dt_df["max_depth"].apply(lambda v: np.nan if v is None else float(v))
        dt_df = dt_df.dropna(subset=["max_depth_num"])

        plt.figure()
        dt_agg = dt_df.groupby("max_depth_num")[metric].mean().reset_index().sort_values("max_depth_num")
        plt.plot(dt_agg["max_depth_num"], dt_agg[metric], marker="o")
        plt.xlabel("max_depth")
        plt.ylabel(metric)
        plt.title(f"DecisionTree: wpływ max_depth na {metric}")
        plt.tight_layout()
        plt.savefig(figures_dir / "dt_metric_vs_max_depth.png", dpi=200)
        plt.close()

    # -------------------------
    # 6) Krzywe strojenia: XGBoost (val_macro_f1 vs n_estimators)
    # -------------------------
    xgb_df = df[df["model"] == "XGBoost"].copy()
    if not xgb_df.empty:
        xgb_df["n_estimators"] = xgb_df["params"].apply(lambda d: float(d.get("n_estimators", np.nan)))
        xgb_df["learning_rate"] = xgb_df["params"].apply(lambda d: float(d.get("learning_rate", np.nan)))
        xgb_df = xgb_df.dropna(subset=["n_estimators", "learning_rate"])

        # wybieramy learning_rate z najlepszego uruchomienia, żeby wykres był czytelny
        best_xgb = xgb_df.sort_values(metric, ascending=False).iloc[0]
        best_lr = float(best_xgb["learning_rate"])

        xgb_slice = xgb_df[np.isclose(xgb_df["learning_rate"], best_lr)]
        plt.figure()
        xgb_agg = xgb_slice.groupby("n_estimators")[metric].mean().reset_index().sort_values("n_estimators")
        plt.plot(xgb_agg["n_estimators"], xgb_agg[metric], marker="o")
        plt.xlabel("n_estimators")
        plt.ylabel(metric)
        plt.title(f"XGBoost: wpływ n_estimators na {metric} (dla learning_rate={best_lr})")
        plt.tight_layout()
        plt.savefig(figures_dir / "xgb_metric_vs_n_estimators.png", dpi=200)
        plt.close()

    # -------------------------
    # 7) Raport Markdown (gotowy do wklejenia do rozdziału)
    # -------------------------
    report_md = []
    report_md.append("# Wyniki eksperymentów klasyfikacji\n")
    report_md.append("## Zestawienie najlepszych wyników\n")
    report_md.append("Tabela: `summary_table.csv` (oraz `summary_table.tex` do LaTeX).\n")
    report_md.append("Wykres: `figures/best_metric_by_model.png`.\n\n")
    report_md.append("### Najlepsze konfiguracje (wg walidacji)\n")

    for _, row in summary_df.iterrows():
        report_md.append(f"**{row['model']}** (stage={row['stage']}):\n")
        report_md.append(f"- {metric}: {row[metric]:.4f}\n")
        report_md.append(f"- test_macro_f1: {row['test_macro_f1']:.4f}\n")
        report_md.append(f"- train_time_s: {row['train_time_s']:.2f}\n")
        report_md.append(f"- params: {row['params_json']}\n")

        best_json = expdir / f"best_{row['model']}.json"
        if best_json.exists():
            report_md.append(f"- pełne metryki: `{best_json.name}`\n")
        report_md.append("\n")

    report_md.append("## Wykresy strojenia parametrów\n")
    report_md.append("- SVM: `figures/svm_metric_vs_C.png`\n")
    report_md.append("- DecisionTree: `figures/dt_metric_vs_max_depth.png`\n")
    report_md.append("- XGBoost: `figures/xgb_metric_vs_n_estimators.png`\n")
    report_md.append("\n")
    report_md.append("## Zależność jakości od czasu treningu\n")
    report_md.append("Wykres: `figures/metric_vs_train_time_scatter.png`\n")

    (expdir / "report.md").write_text("\n".join(report_md), encoding="utf-8")

    print("Zapisano:")
    print("-", (expdir / "summary_table.csv").resolve())
    print("-", (expdir / "summary_table.tex").resolve())
    print("-", (expdir / "report.md").resolve())
    print("-", figures_dir.resolve())


if __name__ == "__main__":
    main()
