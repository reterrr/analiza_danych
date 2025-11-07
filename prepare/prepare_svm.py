#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Przygotowanie danych dla SVM.
- Wczytanie CSV (plik/katalog)
- Czyszczenie i uzupełnianie braków
- ColumnTransformer: StandardScaler (num, with_mean=False) + OneHotEncoder (cat)
- Podział strat. 70/15/15
- Domyślnie zapis jako macierze rzadkie (npz) — dobre dla LinearSVC
- Opcja --dense-output do zapisu gęstego (npy) dla SVC z jądrem RBF (uwaga na pamięć)
- Zapis: X_*, y_*, class_mapping.json, preprocessor.joblib
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

LABEL_CANDIDATES = ["Attack_type", "Label", "label", "attack_type", "class"]

def read_any_csv(path: Path) -> pd.DataFrame:
    if path.is_dir():
        frames = [pd.read_csv(p, low_memory=False) for p in sorted(path.glob("*.csv"))]
        if not frames:
            raise FileNotFoundError("Brak plików CSV w katalogu.")
        df = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    else:
        df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df

def guess_label(df: pd.DataFrame, user_label: str | None) -> str:
    if user_label and user_label in df.columns:
        return user_label
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError("Nie znaleziono kolumny etykiety. Użyj --label-col.")

def split_sets(X, y, test_size=0.15, val_size=0.15):
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=(test_size + val_size), stratify=y, random_state=42
    )
    rel_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=(1 - rel_val), stratify=y_tmp, random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="prepared/svm")
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--drop-cols", nargs="*", default=["uid","ts","id.orig_h","id.resp_h","src_ip","dst_ip","flow_id","timestamp"])
    ap.add_argument("--dense-output", action="store_true", help="Wymuś gęsty zapis (npy) — przydatne dla SVC RBF, ryzyko dużego zużycia RAM.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = read_any_csv(Path(args.input))
    df = df.drop(columns=[c for c in args.drop_cols if c in df.columns], errors="ignore")

    y_col = guess_label(df, args.label_col)
    y_raw = df[y_col].astype(str)
    X = df.drop(columns=[y_col])

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("missing")

    classes = np.unique(y_raw)
    class2id = {cls: i for i, cls in enumerate(sorted(classes))}
    y = y_raw.map(class2id).astype(int).values

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        sparse_threshold=1.0,
    )

    X_enc = pre.fit_transform(X)  # sparse CSR

    X_tr, X_va, X_te, y_tr, y_va, y_te = split_sets(X_enc, y)

    class_weights = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    id2class = {v: k for k, v in class2id.items()}

    if args.dense_output:
        # UWAGA: może zająć bardzo dużo pamięci dla szerokich macierzy
        X_tr_d = X_tr.toarray() if hasattr(X_tr, "toarray") else X_tr
        X_va_d = X_va.toarray() if hasattr(X_va, "toarray") else X_va
        X_te_d = X_te.toarray() if hasattr(X_te, "toarray") else X_te

        np.save(outdir / "X_train.npy", X_tr_d)
        np.save(outdir / "X_val.npy", X_va_d)
        np.save(outdir / "X_test.npy", X_te_d)
    else:
        from scipy import sparse
        sparse.save_npz(outdir / "X_train.npz", X_tr)
        sparse.save_npz(outdir / "X_val.npz", X_va)
        sparse.save_npz(outdir / "X_test.npz", X_te)

    np.save(outdir / "y_train.npy", y_tr)
    np.save(outdir / "y_val.npy", y_va)
    np.save(outdir / "y_test.npy", y_te)

    with open(outdir / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"class2id": class2id, "id2class": id2class,
                   "class_weights": dict(zip(map(str, np.unique(y_tr)), class_weights.astype(float)))},
                  f, ensure_ascii=False, indent=2)

    dump(pre, outdir / "preprocessor.joblib")

    print("Zapisano przygotowane dane w:", outdir.resolve())
    print("Kształty:", {
        "X_train": X_tr.shape, "X_val": X_va.shape, "X_test": X_te.shape,
        "dense_output": args.dense_output
    })

if __name__ == "__main__":
    main()
