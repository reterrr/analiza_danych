#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Przygotowanie danych RT-IoT2022 dla XGBoost.
- Wczytanie CSV (pojedynczy plik lub katalog *.csv)
- Czyszczenie i uzupełnianie braków
- One-Hot Encoding dla cech kategorycznych, bez skalowania
- Podział strat. 70/15/15
- Zapis: X_train.npz, X_val.npz, X_test.npz (sparse); y_*.npy; feature_names.txt; class_mapping.json; preprocessor.joblib
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight

LABEL_CANDIDATES = ["Attack_type", "Label", "label", "attack_type", "class"]

def read_any_csv(path: Path) -> pd.DataFrame:
    if path.is_dir():
        frames = []
        for p in sorted(path.glob("*.csv")):
            frames.append(pd.read_csv(p, low_memory=False))
        if not frames:
            raise FileNotFoundError("Brak plików CSV w katalogu.")
        df = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    else:
        df = pd.read_csv(path, low_memory=False)
    # porządkowanie nagłówków
    df.columns = [c.strip() for c in df.columns]
    return df

def guess_label(df: pd.DataFrame, user_label: str | None) -> str:
    if user_label and user_label in df.columns:
        return user_label
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError("Nie znaleziono kolumny etykiety. Użyj --label-col.")

def split_sets(X, y, random_state=42, test_size=0.15, val_size=0.15):
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
    ap.add_argument("--input", required=True, help="Ścieżka do pliku CSV lub katalogu z CSV.")
    ap.add_argument("--outdir", default="prepared/xgboost", help="Katalog wyjściowy.")
    ap.add_argument("--label-col", default=None, help="Nazwa kolumny etykiety.")
    ap.add_argument("--drop-cols", nargs="*", default=["uid","ts","id.orig_h","id.resp_h","src_ip","dst_ip","flow_id","timestamp"],
                    help="Kolumny do usunięcia (jeśli są).")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = read_any_csv(Path(args.input))
    # usuwanie niepotrzebnych identyfikatorów/adresów jeśli występują
    drop_candidates = [c for c in args.drop_cols if c in df.columns]
    df = df.drop(columns=drop_candidates, errors="ignore")

    # wykrycie kolumny etykiety
    y_col = guess_label(df, args.label_col)

    # separacja X/y
    y_raw = df[y_col].astype(str)
    X = df.drop(columns=[y_col])

    # typy: kategorie vs numeryczne
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # konwersja numeryków z wymuszeniem, czyszczenie NaN
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    # fillna: num -> mediana, kat -> 'missing'
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("missing")

    # mapowanie klas -> int
    classes = np.unique(y_raw)
    class2id = {cls: i for i, cls in enumerate(sorted(classes))}
    y = y_raw.map(class2id).astype(int).values

    # preprocessor: OneHot dla kat., passthrough numeryki
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        sparse_threshold=1.0,
    )

    X_enc = pre.fit_transform(X)

    # podział
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_sets(X_enc, y)

    # wagi klas (przydatne przy trenowaniu)
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_tr), y=y_tr
    )
    id2class = {v: k for k, v in class2id.items()}

    # zapisy
    sparse.save_npz(outdir / "X_train.npz", X_tr)
    sparse.save_npz(outdir / "X_val.npz", X_va)
    sparse.save_npz(outdir / "X_test.npz", X_te)
    np.save(outdir / "y_train.npy", y_tr)
    np.save(outdir / "y_val.npy", y_va)
    np.save(outdir / "y_test.npy", y_te)

    # nazwy cech po one-hot
    feat_names = []
    try:
        feat_names = pre.get_feature_names_out().tolist()
    except Exception:
        # wersje sklearn < 1.0
        pass
    with open(outdir / "feature_names.txt", "w", encoding="utf-8") as f:
        for n in feat_names:
            f.write(n + "\n")

    with open(outdir / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"class2id": class2id, "id2class": id2class,
                   "class_weights": dict(zip(map(str, np.unique(y_tr)), class_weights.astype(float)))},
                  f, ensure_ascii=False, indent=2)

    dump(pre, outdir / "preprocessor.joblib")

    print("Zapisano przygotowane dane w:", outdir.resolve())
    print("Kształty:", {
        "X_train": X_tr.shape, "X_val": X_va.shape, "X_test": X_te.shape
    })

if __name__ == "__main__":
    main()
