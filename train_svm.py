#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trening SVM na CPU na danych przygotowanych przez skrypt SVM.

Zgodne z prepare_svm.py:
- Jeśli w katalogu są X_train.npz / X_val.npz / X_test.npz:
    -> używany jest LinearSVC (dobry dla danych rzadkich).
- Jeśli w katalogu są X_train.npy / X_val.npy / X_test.npy:
    -> używany jest SVC (domyślnie kernel RBF) na danych gęstych.

Wejście (domyślnie w prepared/svm):
    - X_train.(npz/npy), X_val.(npz/npy), X_test.(npz/npy)
    - y_train.npy, y_val.npy, y_test.npy
    - class_mapping.json

Wyjście:
    - svm_cpu_model.joblib
    - svm_cpu_metrics.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC, SVC


def load_data_auto(data_dir: Path):
    """
    Automatycznie wczytuje dane:
    - jeśli istnieje X_train.npz -> wariant rzadki (csr)
    - jeśli istnieje X_train.npy -> wariant gęsty (ndarray)
    """
    sparse_path = data_dir / "X_train.npz"
    dense_path = data_dir / "X_train.npy"

    if sparse_path.exists():
        from scipy import sparse

        print("Wykryto zapis rzadki (npz) – użyję LinearSVC.")
        X_train = sparse.load_npz(data_dir / "X_train.npz")
        X_val = sparse.load_npz(data_dir / "X_val.npz")
        X_test = sparse.load_npz(data_dir / "X_test.npz")
        is_sparse = True
    elif dense_path.exists():
        print("Wykryto zapis gęsty (npy) – użyję SVC.")
        X_train = np.load(data_dir / "X_train.npy")
        X_val = np.load(data_dir / "X_val.npy")
        X_test = np.load(data_dir / "X_test.npy")
        is_sparse = False
    else:
        raise FileNotFoundError(
            "Nie znaleziono ani X_train.npz, ani X_train.npy w katalogu "
            f"{data_dir}. Upewnij się, że prepare_svm.py został uruchomiony."
        )

    y_train = np.load(data_dir / "y_train.npy")
    y_val = np.load(data_dir / "y_val.npy")
    y_test = np.load(data_dir / "y_test.npy")

    mapping_path = data_dir / "class_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Brak pliku {mapping_path}")

    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    # class_weights: {str(label_id): float}
    class_weights = {
        int(k): float(v) for k, v in mapping.get("class_weights", {}).items()
    }

    return X_train, X_val, X_test, y_train, y_val, y_test, class_weights, mapping, is_sparse


def build_model(is_sparse: bool, class_weights: dict[int, float], args: argparse.Namespace):
    """
    Buduje model SVM na CPU:
    - dla danych rzadkich (is_sparse=True) -> LinearSVC
    - dla danych gęstych (is_sparse=False) -> SVC (domyślnie kernel RBF)
    """
    cw = class_weights if args.use_class_weights and class_weights else None

    if is_sparse:
        # LinearSVC dobrze współpracuje z danymi rzadkimi
        model = LinearSVC(
            C=args.C,
            class_weight=cw,
            max_iter=args.max_iter,
            dual="auto",           # automatyczny wybór dual/primal
            random_state=args.random_state,
        )
        model_type = "LinearSVC"
    else:
        # Klasyczny SVC z jądrem RBF (lub innym, jeśli podano w args.kernel)
        model = SVC(
            C=args.C,
            kernel=args.kernel,
            gamma=args.gamma,
            class_weight=cw,
            probability=args.probability,
            cache_size=args.cache_size,
            max_iter=args.max_iter,
            random_state=args.random_state,
        )
        model_type = "SVC"

    return model, model_type


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data-dir",
        default="prepared/svm",
        help="Katalog z X_*, y_*, class_mapping.json (domyślnie: prepared/svm)",
    )
    ap.add_argument(
        "--model-out",
        default="svm_cpu_model.joblib",
        help="Nazwa pliku z modelem (joblib, zapisany w data-dir).",
    )
    ap.add_argument(
        "--metrics-out",
        default="svm_cpu_metrics.json",
        help="Nazwa pliku z metrykami (JSON, zapisany w data-dir).",
    )

    # Hiperparametry wspólne
    ap.add_argument("--C", type=float, default=1.0, help="Współczynnik regularyzacji C.")
    ap.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maksymalna liczba iteracji (LinearSVC/SVC).",
    )
    ap.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Ziarno losowe dla reproducowalności.",
    )

    # Hiperparametry specyficzne dla SVC (gdy dane są gęste)
    ap.add_argument(
        "--kernel",
        default="rbf",
        choices=["linear", "rbf", "poly", "sigmoid"],
        help="Jądro SVM (używane tylko dla SVC przy danych gęstych).",
    )
    ap.add_argument(
        "--gamma",
        default="scale",
        help="Parametr gamma dla SVC (float, 'scale' lub 'auto').",
    )
    ap.add_argument(
        "--probability",
        action="store_true",
        help="Włącz obliczanie prawdopodobieństw w SVC (wolniejsze, więcej RAM).",
    )
    ap.add_argument(
        "--cache-size",
        type=float,
        default=1024.0,
        help="Rozmiar cache (MB) dla SVC (domyślnie 1024).",
    )

    # Wagi klas
    ap.add_argument(
        "--no-class-weights",
        dest="use_class_weights",
        action="store_false",
        help="Nie używaj wag klas z class_mapping.json (class_weight=None).",
    )
    ap.set_defaults(use_class_weights=True)

    args = ap.parse_args()
    data_dir = Path(args.data_dir)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        class_weights,
        mapping,
        is_sparse,
    ) = load_data_auto(data_dir)

    print("Rozmiary zbiorów:")
    print("  X_train:", X_train.shape, "y_train:", y_train.shape)
    print("  X_val:  ", X_val.shape, "y_val:", y_val.shape)
    print("  X_test: ", X_test.shape, "y_test:", y_test.shape)
    print("Liczba klas:", len(np.unique(y_train)))
    if class_weights:
        print("Wagi klas (z training set):", class_weights)

    # Budowa modelu
    model, model_type = build_model(is_sparse, class_weights, args)
    print(f"\nTrenuję model: {model_type} (CPU)")

    model.fit(X_train, y_train)
    print("Trening zakończony.\n")

    # Ewaluacja – walidacja
    y_val_pred = model.predict(X_val)
    acc_val = accuracy_score(y_val, y_val_pred)
    report_val_str = classification_report(y_val, y_val_pred)
    conf_val = confusion_matrix(y_val, y_val_pred)

    # Ewaluacja – test
    y_test_pred = model.predict(X_test)
    acc_test = accuracy_score(y_test, y_test_pred)
    report_test_str = classification_report(y_test, y_test_pred)
    conf_test = confusion_matrix(y_test, y_test_pred)

    print(f"Accuracy (val):  {acc_val:.4f}")
    print(f"Accuracy (test): {acc_test:.4f}")

    # Zapis modelu
    model_path = data_dir / args.model_out
    dump(model, model_path)
    print("Zapisano model do:", model_path.resolve())

    # Przygotowanie metryk do JSON (unikamy typów numpy w JSON)
    metrics = {
        "data_type": "sparse" if is_sparse else "dense",
        "model_type": model_type,
        "C": float(args.C),
        "kernel": args.kernel if not is_sparse else None,
        "gamma": args.gamma if not is_sparse else None,
        "accuracy_val": float(acc_val),
        "accuracy_test": float(acc_test),
        "classification_report_val": report_val_str,
        "classification_report_test": report_test_str,
        "confusion_matrix_val": conf_val.tolist(),
        "confusion_matrix_test": conf_test.tolist(),
        "class_mapping": mapping,
    }

    metrics_path = data_dir / args.metrics_out
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Zapisano metryki do:", metrics_path.resolve())


if __name__ == "__main__":
    main()
