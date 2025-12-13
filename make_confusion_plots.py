#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generowanie macierzy pomyłek (confusion matrices) dla najlepszych modeli zapisanych w best_*.json.

Wejście:
  - katalog eksperymentów, np. experiments/
  - pliki: best_*.json (np. best_XGBoost.json, best_DecisionTree.json, ...)

Wyjście:
  - experiments/figures/confusion/:
      * confusion_<MODEL>_<NORMALIZE>.png (i opcjonalnie .pdf)
      * confusion_<MODEL>_counts.csv
      * confusion_<MODEL>_<NORMALIZE>.csv

Normalizacja:
  - none  : surowe liczności
  - true  : normalizacja wierszami (po klasie rzeczywistej) -> "recall-like"
  - pred  : normalizacja kolumnami (po klasie przewidzianej) -> "precision-like"
  - all   : normalizacja globalna (po sumie wszystkich elementów)
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", s.strip())


def load_best_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_labels(mapping: dict, n: int) -> List[str]:
    """
    Próbuje odtworzyć nazwy klas w kolejności indeksów 0..n-1.
    mapping pochodzi z class_mapping.json zapisanej w best_*.json jako "class_mapping".
    """
    id2class = mapping.get("id2class", {})
    labels = []
    for i in range(n):
        # id2class może mieć klucze int lub str
        if str(i) in id2class:
            labels.append(str(id2class[str(i)]))
        elif i in id2class:
            labels.append(str(id2class[i]))
        else:
            labels.append(str(i))
    return labels


def normalize_cm(cm: np.ndarray, mode: str) -> np.ndarray:
    cm = cm.astype(float, copy=True)
    if mode == "none":
        return cm
    if mode == "all":
        s = cm.sum()
        return cm / s if s > 0 else cm
    if mode == "true":
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return cm / row_sums
    if mode == "pred":
        col_sums = cm.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        return cm / col_sums
    raise ValueError(f"Nieznany tryb normalizacji: {mode}")


def supports_from_report(report: dict) -> Dict[int, float]:
    """
    classification_report(output_dict=True) ma klucze klas jako stringi liczb, plus 'accuracy', 'macro avg', itd.
    Zwraca support per class_id.
    """
    supports: Dict[int, float] = {}
    for k, v in report.items():
        if isinstance(k, str) and k.isdigit() and isinstance(v, dict) and "support" in v:
            supports[int(k)] = float(v["support"])
    return supports


def topk_indices(cm: np.ndarray, report: Optional[dict], k: int) -> List[int]:
    n = cm.shape[0]
    if k <= 0 or k >= n:
        return list(range(n))

    if report:
        sup = supports_from_report(report)
        # jeśli z jakiegoś powodu brak supportu, fallback na sumę wiersza cm
        scores = [(i, sup.get(i, float(cm[i, :].sum()))) for i in range(n)]
    else:
        scores = [(i, float(cm[i, :].sum())) for i in range(n)]

    scores.sort(key=lambda x: x[1], reverse=True)
    return sorted([i for i, _ in scores[:k]])


def reduce_cm_with_other(
    cm: np.ndarray, labels: List[str], keep: List[int], add_other: bool
) -> Tuple[np.ndarray, List[str]]:
    """
    Redukuje macierz do klas z 'keep'. Opcjonalnie dodaje klasę 'Other'
    agregując wszystkie pozostałe wiersze/kolumny.
    """
    n = cm.shape[0]
    keep_set = set(keep)
    drop = [i for i in range(n) if i not in keep_set]

    cm_keep = cm[np.ix_(keep, keep)]
    labels_keep = [labels[i] for i in keep]

    if not add_other or len(drop) == 0:
        return cm_keep, labels_keep

    # agregacja: Other jako ostatni wiersz/kolumna
    # - wiersz Other: sumy rzeczywistych klas "drop" względem przewidywań keep + drop
    # - kolumna Other: sumy przewidywań "drop" dla klas keep + drop
    # Robimy pełną macierz (keep + other) x (keep + other)
    other_row = cm[drop, :][:, keep].sum(axis=0)          # true=Other, pred=keep
    other_col = cm[:, drop][keep, :].sum(axis=1)          # true=keep, pred=Other
    other_other = cm[np.ix_(drop, drop)].sum()            # true=Other, pred=Other

    cm_new = np.zeros((len(keep) + 1, len(keep) + 1), dtype=cm.dtype)
    cm_new[:len(keep), :len(keep)] = cm_keep
    cm_new[-1, :len(keep)] = other_row
    cm_new[:len(keep), -1] = other_col
    cm_new[-1, -1] = other_other

    labels_new = labels_keep + ["Other"]
    return cm_new, labels_new


def plot_cm(
    cm_values: np.ndarray,
    labels: List[str],
    title: str,
    outpath: Path,
    annotate: bool,
    is_normalized: bool,
):
    n = cm_values.shape[0]

    # Rozmiar figury zależny od liczby klas, ale z limitem
    w = min(22, max(8, 0.45 * n))
    h = min(22, max(6, 0.45 * n))
    fig, ax = plt.subplots(figsize=(w, h))

    im = ax.imshow(cm_values, aspect="auto")
    ax.set_title(title)

    ax.set_xlabel("Klasa przewidziana")
    ax.set_ylabel("Klasa rzeczywista")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # Adnotacje w komórkach
    if annotate:
        # Dostosuj format do trybu
        fmt = "{:.2f}" if is_normalized else "{:d}"
        for i in range(n):
            for j in range(n):
                val = cm_values[i, j]
                if is_normalized:
                    s = fmt.format(float(val))
                else:
                    s = fmt.format(int(val))
                ax.text(j, i, s, ha="center", va="center", fontsize=max(6, int(14 - 0.2 * n)))

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expdir", default="experiments", help="Katalog z best_*.json oraz figures/")
    ap.add_argument(
        "--normalize",
        default="true",
        choices=["none", "true", "pred", "all"],
        help="Tryb normalizacji macierzy pomyłek.",
    )
    ap.add_argument("--annotate", action="store_true", help="Wypisuj wartości w komórkach.")
    ap.add_argument(
        "--topk",
        type=int,
        default=0,
        help="Jeśli >0, rysuj tylko TOP-K najczęstszych klas (czytelniejsze przy wielu klasach).",
    )
    ap.add_argument(
        "--add-other",
        action="store_true",
        help="Gdy używasz --topk, dodaj dodatkową klasę 'Other' agregując resztę.",
    )
    ap.add_argument(
        "--pdf",
        action="store_true",
        help="Zapisuj dodatkowo wykresy w PDF (przydatne do pracy).",
    )
    args = ap.parse_args()

    expdir = Path(args.expdir)
    best_files = sorted(expdir.glob("best_*.json"))
    if not best_files:
        raise FileNotFoundError(f"Nie znaleziono plików best_*.json w {expdir}. Uruchom najpierw run_experiments.py")

    outdir = expdir / "figures" / "confusion"
    outdir.mkdir(parents=True, exist_ok=True)

    for bf in best_files:
        payload = load_best_json(bf)
        model_name = payload.get("model_name", bf.stem.replace("best_", ""))
        model_tag = safe_name(model_name)

        cm_counts = np.array(payload["confusion_matrix_test"], dtype=int)
        mapping = payload.get("class_mapping", {})
        labels_full = extract_labels(mapping, cm_counts.shape[0])

        report = payload.get("classification_report_test", None)

        # TOP-K redukcja (opcjonalna)
        cm_plot = cm_counts
        labels_plot = labels_full

        if args.topk and args.topk > 0:
            keep = topk_indices(cm_counts, report, args.topk)
            cm_plot, labels_plot = reduce_cm_with_other(cm_counts, labels_full, keep, args.add_other)

        # Zapis CSV z licznościami
        df_counts = pd.DataFrame(cm_plot, index=labels_plot, columns=labels_plot)
        df_counts.to_csv(outdir / f"confusion_{model_tag}_counts.csv", encoding="utf-8")

        # Normalizacja + CSV
        cm_norm = normalize_cm(cm_plot, args.normalize)
        df_norm = pd.DataFrame(cm_norm, index=labels_plot, columns=labels_plot)
        df_norm.to_csv(outdir / f"confusion_{model_tag}_{args.normalize}.csv", encoding="utf-8")

        # Wykresy
        title = f"Macierz pomyłek (TEST) – {model_name} – normalize={args.normalize}"
        png_path = outdir / f"confusion_{model_tag}_{args.normalize}.png"
        plot_cm(
            cm_values=cm_norm if args.normalize != "none" else cm_plot,
            labels=labels_plot,
            title=title,
            outpath=png_path,
            annotate=args.annotate,
            is_normalized=(args.normalize != "none"),
        )

        if args.pdf:
            pdf_path = outdir / f"confusion_{model_tag}_{args.normalize}.pdf"
            # ponownie zapis do PDF (matplotlib zapisze wg rozszerzenia)
            plot_cm(
                cm_values=cm_norm if args.normalize != "none" else cm_plot,
                labels=labels_plot,
                title=title,
                outpath=pdf_path,
                annotate=args.annotate,
                is_normalized=(args.normalize != "none"),
            )

        print(f"[OK] {model_name}: zapisano {png_path.name} + CSV w {outdir}")

    print("Gotowe. Wyniki w:", outdir.resolve())


if __name__ == "__main__":
    main()
