#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_test_data(data_dir: Path):
    """Wczytuje X_test (npy albo npz) oraz y_test i mapping (jeśli jest)."""
    x_test_npz = data_dir / "X_test.npz"
    x_test_npy = data_dir / "X_test.npy"
    y_test_npy = data_dir / "y_test.npy"
    mapping_json = data_dir / "class_mapping.json"

    if not y_test_npy.exists():
        raise FileNotFoundError(f"Brak pliku: {y_test_npy}")

    # X_test: sparse lub dense
    if x_test_npz.exists():
        from scipy import sparse
        X_test = sparse.load_npz(x_test_npz)
        is_sparse = True
    elif x_test_npy.exists():
        X_test = np.load(x_test_npy)
        is_sparse = False
    else:
        raise FileNotFoundError(f"Brak X_test.npz ani X_test.npy w: {data_dir}")

    y_test = np.load(y_test_npy)

    mapping = None
    if mapping_json.exists():
        with open(mapping_json, "r", encoding="utf-8") as f:
            mapping = json.load(f)

    return X_test, y_test, mapping, is_sparse


def pretty_label(idx: int, mapping: dict | None) -> str:
    """Zwraca nazwę klasy, jeśli mapping zawiera informacje (fallback: id)."""
    if not mapping:
        return str(idx)

    # Różne projekty trzymają różne klucze – obsługa kilku typowych wariantów.
    # Jeśli nie pasuje, zostaje id.
    for key in ["id_to_label", "id_to_class", "id_to_name"]:
        if key in mapping and isinstance(mapping[key], dict):
            return str(mapping[key].get(str(idx), idx))

    # Czasem mapping ma np. "classes": ["Normal", "DoS", ...]
    if "classes" in mapping and isinstance(mapping["classes"], list):
        if 0 <= idx < len(mapping["classes"]):
            return str(mapping["classes"][idx])

    return str(idx)


def load_model(model_path: Path):
    """Wczytuje model: joblib (sklearn) albo json (XGBoost)."""
    if not model_path.exists():
        raise FileNotFoundError(f"Brak modelu: {model_path}")

    if model_path.suffix.lower() == ".joblib":
        from joblib import load
        return load(model_path), "sklearn"

    if model_path.suffix.lower() == ".json":
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
        return model, "xgboost"

    raise ValueError(f"Nieobsługiwany format modelu: {model_path.suffix} (użyj .joblib lub .json)")


def predict_with_optional_proba(model, X):
    """Zwraca y_pred oraz (opcjonalnie) proba."""
    proba = None

    # XGBoost ma predict_proba; SVC ma predict_proba tylko jeśli trenowany z probability=True
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
        except Exception:
            proba = None

    y_pred = model.predict(X)
    return y_pred, proba


def main():
    ap = argparse.ArgumentParser(description="Demo: wczytaj model i uruchom predykcję na zbiorze testowym.")
    ap.add_argument("--data-dir", required=True, help="Katalog z X_test.* i y_test.npy (np. prepared/svm).")
    ap.add_argument("--model", required=True, help="Ścieżka do modelu (.joblib albo .json).")
    ap.add_argument("--show", type=int, default=10, help="Ile pierwszych predykcji wypisać (domyślnie 10).")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    model_path = Path(args.model)

    X_test, y_test, mapping, is_sparse = load_test_data(data_dir)
    model, model_kind = load_model(model_path)

    # Dla wydajności (zwłaszcza XGB na GPU) – jeśli dense, rzutuj do float32
    if not is_sparse and isinstance(X_test, np.ndarray) and X_test.dtype != np.float32:
        X_test = X_test.astype(np.float32, copy=False)

    y_pred, proba = predict_with_optional_proba(model, X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    print("=" * 80)
    print(f"Model: {model_path} ({model_kind})")
    print(f"Dane testowe: {data_dir}")
    print(f"X_test shape: {getattr(X_test, 'shape', None)}, y_test shape: {y_test.shape}")
    print(f"Accuracy(test): {acc:.4f}")
    print("-" * 80)
    print("Classification report (test):")
    print(report)
    print("-" * 80)
    print("Confusion matrix (test):")
    print(cm)

    # Mała prezentacja: kilka pierwszych predykcji
    n = min(args.show, len(y_test))
    print("-" * 80)
    print(f"Przykładowe predykcje (pierwsze {n}):")
    for i in range(n):
        yt = int(y_test[i])
        yp = int(y_pred[i])

        yt_name = pretty_label(yt, mapping)
        yp_name = pretty_label(yp, mapping)

        if proba is not None:
            # confidence = max probability
            conf = float(np.max(proba[i]))
            print(f"[{i:04d}] true={yt}({yt_name})  pred={yp}({yp_name})  conf={conf:.3f}")
        else:
            print(f"[{i:04d}] true={yt}({yt_name})  pred={yp}({yp_name})")

    print("=" * 80)


if __name__ == "__main__":
    main()
