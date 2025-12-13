#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trening modelu XGBoost (drzewa decyzyjne) na danych przygotowanych przez skrypt
"przygotowanie danych dla Drzewa Decyzyjnego".

Dostosowane do xgboost==3.1.2:
- GPU: tree_method="hist", device="cuda" (zamiast przestarzałego gpu_hist)
- eval_metric i early_stopping_rounds przekazywane do konstruktora XGBClassifier,
  NIE do metody fit().

Wejście (domyślnie w prepared/decision_tree):
    - X_train.npy, X_val.npy, X_test.npy
    - y_train.npy, y_val.npy, y_test.npy
    - class_mapping.json

Wyjście:
    - xgb_model.json     (zapisany model XGBoost)
    - xgb_metrics.json   (metryki: accuracy / raporty / macierze pomyłek)
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import xgboost as xgb


# ----------------------------------------------------------------------
# Pomocnicze funkcje
# ----------------------------------------------------------------------
def load_data(data_dir: Path):
    """Wczytuje X_*.npy, y_*.npy oraz class_mapping.json z podanego katalogu."""
    X_train = np.load(data_dir / "X_train.npy")
    X_val = np.load(data_dir / "X_val.npy")
    X_test = np.load(data_dir / "X_test.npy")

    y_train = np.load(data_dir / "y_train.npy")
    y_val = np.load(data_dir / "y_val.npy")
    y_test = np.load(data_dir / "y_test.npy")

    mapping_path = data_dir / "class_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Brak pliku {mapping_path}")

    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    # class_weights zapisane jako {str(klasa_id): waga}
    class_weights = {
        int(k): float(v) for k, v in mapping.get("class_weights", {}).items()
    }

    return X_train, X_val, X_test, y_train, y_val, y_test, class_weights, mapping


def make_sample_weights(y: np.ndarray, class_weights: dict[int, float]) -> np.ndarray:
    """Tworzy wektor wag próbek na podstawie wag klas."""
    if not class_weights:
        return np.ones_like(y, dtype=float)

    w = np.ones_like(y, dtype=float)
    for cls, cw in class_weights.items():
        w[y == cls] = cw
    return w


def build_model(num_classes: int, args: argparse.Namespace) -> xgb.XGBClassifier:
    """
    Buduje model XGBClassifier skonfigurowany pod xgboost 3.1.2.

    - GPU: tree_method="hist", device="cuda" (lub inne urządzenie z args.device)
    - wczesne zatrzymanie: early_stopping_rounds ustawione w konstruktorze
    - metryka: mlogloss (wieloklasowa klasyfikacja)
    """
    objective = "multi:softprob"

    model = xgb.XGBClassifier(
        # podstawowe parametry boostingu
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        gamma=args.gamma,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,

        # wieloklasowa klasyfikacja
        objective=objective,
        num_class=num_classes,

        # GPU / CPU
        tree_method="hist",    # z device="cuda" -> GPU, z device="cpu" -> CPU
        device=args.device,    # np. "cuda" / "cuda:0" / "cpu"

        # wczesne zatrzymanie + metryka
        early_stopping_rounds=args.early_stopping_rounds,

        # inne
        n_jobs=args.n_jobs,
        random_state=args.random_state,
    )
    return model


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()

    # Ścieżki i nazwy plików
    ap.add_argument(
        "--data-dir",
        default="prepared/decision_tree",
        help="Katalog z X_*.npy, y_*.npy, class_mapping.json (domyślnie: prepared/decision_tree)",
    )
    ap.add_argument(
        "--model-out",
        default="xgb_model.json",
        help="Nazwa pliku z modelem (JSON, zapisany w data-dir).",
    )
    ap.add_argument(
        "--metrics-out",
        default="xgb_metrics.json",
        help="Nazwa pliku z metrykami (JSON, zapisany w data-dir).",
    )

    # Hiperparametry modelu
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample-bytree", type=float, default=0.8)
    ap.add_argument("--min-child-weight", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--reg-lambda", type=float, default=1.0)
    ap.add_argument("--reg-alpha", type=float, default=0.0)

    ap.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=50,
        help="Liczba rund bez poprawy na walidacji do zatrzymania (ustawiane w konstruktorze XGBClassifier).",
    )

    # Parametry wykonania
    ap.add_argument(
        "--device",
        default="cuda",
        help='Urządzenie XGBoost, np. "cuda", "cuda:0" lub "cpu". Domyślnie "cuda" (GPU).',
    )
    ap.add_argument(
        "--n-jobs",
        type=int,
        default=0,
        help="Liczba wątków CPU (0 = wszystkie dostępne).",
    )
    ap.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Ziarno generatora losowego.",
    )
    ap.add_argument(
        "--no-class-weights",
        dest="use_class_weights",
        action="store_false",
        help="Nie używaj wag klas z class_mapping.json (sample_weight=None).",
    )
    ap.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Poziom logowania XGBoost (0 = cicho).",
    )
    ap.set_defaults(use_class_weights=True)

    args = ap.parse_args()
    data_dir = Path(args.data_dir)

    # Wczytanie danych
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        class_weights,
        mapping,
    ) = load_data(data_dir)

    # Dla bezpieczeństwa można rzutować na float32 (lepsza wydajność na GPU)
    X_train = X_train.astype(np.float32, copy=False)
    X_val = X_val.astype(np.float32, copy=False)
    X_test = X_test.astype(np.float32, copy=False)

    num_classes = int(len(np.unique(y_train)))
    print(f"Liczba klas: {num_classes}")
    print("Rozmiary zbiorów:")
    print("  X_train:", X_train.shape, "y_train:", y_train.shape)
    print("  X_val:  ", X_val.shape, "y_val:", y_val.shape)
    print("  X_test: ", X_test.shape, "y_test:", y_test.shape)

    if class_weights:
        print("Wagi klas (z training set):", class_weights)

    # Wagi próbek na podstawie wag klas
    sample_weight = None
    if args.use_class_weights and class_weights:
        sample_weight = make_sample_weights(y_train, class_weights)

    # Budowa modelu XGBoost 3.1.2
    model = build_model(num_classes, args)

    # Trening z walidacją (eval_set) – early_stopping_rounds w konstruktorze
    eval_set = [(X_train, y_train), (X_val, y_val)]
    eval_names = ["train", "val"]
    print(f"\nStart treningu XGBoost na urządzeniu: {args.device}")
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=eval_set,
           # eval_metric już ustawione w konstruktorze
        verbose=args.verbose,
    )
    print("Trenowanie zakończone.\n")

    # Najlepsza iteracja / wynik (dla early stopping)
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None:
        best_iteration = getattr(model, "best_iteration_", None)

    best_score = getattr(model, "best_score", None)
    if best_score is None:
        best_score = getattr(model, "best_score_", None)

    if best_iteration is not None:
        print(f"Najlepsza iteracja (best_iteration): {best_iteration}")
    if best_score is not None:
        print(f"Najlepszy wynik walidacji (best_score): {best_score}")

    # Ewaluacja na walidacji
    y_val_pred = model.predict(X_val)
    acc_val = accuracy_score(y_val, y_val_pred)
    report_val = classification_report(y_val, y_val_pred, output_dict=True)
    conf_val = confusion_matrix(y_val, y_val_pred)

    # Ewaluacja na teście
    y_test_pred = model.predict(X_test)
    acc_test = accuracy_score(y_test, y_test_pred)
    report_test = classification_report(y_test, y_test_pred, output_dict=True)
    conf_test = confusion_matrix(y_test, y_test_pred)

    print(f"\nAccuracy (val):  {acc_val:.4f}")
    print(f"Accuracy (test): {acc_test:.4f}")

    # Zapis modelu (JSON – zalecany format od 3.x)
    model_path = data_dir / args.model_out
    model.save_model(str(model_path))
    print("Zapisano model do:", model_path.resolve())

    # Parametry modelu w formie JSON-friendy
    params_raw = model.get_xgb_params()
    params_json = {}
    for k, v in params_raw.items():
        if isinstance(v, (np.floating, np.integer)):
            params_json[k] = float(v)
        else:
            params_json[k] = v

    # Zapis metryk
    metrics = {
        "num_classes": num_classes,
        "classes": sorted([int(c) for c in np.unique(y_train)]),
        "accuracy_val": float(acc_val),
        "accuracy_test": float(acc_test),
        "classification_report_val": report_val,
        "classification_report_test": report_test,
        "confusion_matrix_val": conf_val.tolist(),
        "confusion_matrix_test": conf_test.tolist(),
        "best_iteration": int(best_iteration) if best_iteration is not None else None,
        "best_score": float(best_score) if best_score is not None else None,
        "params": params_json,
        "class_mapping": mapping,
        "eval_sets": eval_names,
    }

    metrics_path = data_dir / args.metrics_out
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Zapisano metryki do:", metrics_path.resolve())


if __name__ == "__main__":
    main()
