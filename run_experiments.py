#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Eksperymenty klasyfikacji (min. 3 metody): DecisionTree, SVM (LinearSVC/SVC), XGBoost.
- Uruchamia serię eksperymentów z różnymi parametrami (Stage 1: coarse grid)
- Następnie automatycznie zawęża zakres (Stage 2: refine wokół najlepszego ustawienia)
- Zapisuje wyniki do CSV: outdir/results.csv
- Zapisuje najlepsze modele + pełne metryki (report + confusion matrix) do outdir/best_*

Wymagania:
  pip install numpy pandas scipy scikit-learn joblib matplotlib xgboost
"""

import argparse
import json
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from joblib import dump
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC

import xgboost as xgb
from scipy import sparse
from xgboost import XGBClassifier


# -----------------------------
# Data loading
# -----------------------------
def load_class_weights(mapping_path: Path) -> Tuple[dict[int, float], dict]:
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    cw = {int(k): float(v) for k, v in mapping.get("class_weights", {}).items()}
    return cw, mapping


def load_dense_dir(data_dir: Path):
    X_tr = np.load(data_dir / "X_train.npy")
    X_va = np.load(data_dir / "X_val.npy")
    X_te = np.load(data_dir / "X_test.npy")
    y_tr = np.load(data_dir / "y_train.npy")
    y_va = np.load(data_dir / "y_val.npy")
    y_te = np.load(data_dir / "y_test.npy")
    cw, mapping = load_class_weights(data_dir / "class_mapping.json")
    return X_tr, X_va, X_te, y_tr, y_va, y_te, cw, mapping


def load_sparse_or_dense_dir(data_dir: Path):
    # Prefer sparse if present
    if (data_dir / "X_train.npz").exists():
        X_tr = sparse.load_npz(data_dir / "X_train.npz")
        X_va = sparse.load_npz(data_dir / "X_val.npz")
        X_te = sparse.load_npz(data_dir / "X_test.npz")
    elif (data_dir / "X_train.npy").exists():
        X_tr = np.load(data_dir / "X_train.npy")
        X_va = np.load(data_dir / "X_val.npy")
        X_te = np.load(data_dir / "X_test.npy")
    else:
        raise FileNotFoundError(f"Brak X_train.(npz|npy) w {data_dir}")

    y_tr = np.load(data_dir / "y_train.npy")
    y_va = np.load(data_dir / "y_val.npy")
    y_te = np.load(data_dir / "y_test.npy")
    cw, mapping = load_class_weights(data_dir / "class_mapping.json")
    return X_tr, X_va, X_te, y_tr, y_va, y_te, cw, mapping


# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def class_weight_dict(cw: dict[int, float], use: bool) -> Optional[dict[int, float]]:
    return cw if (use and cw) else None


def sample_weight_vector(y: np.ndarray, cw: dict[int, float], use: bool) -> Optional[np.ndarray]:
    if not (use and cw):
        return None
    w = np.ones_like(y, dtype=float)
    for cls, w_cls in cw.items():
        w[y == cls] = w_cls
    return w


# -----------------------------
# Experiment definitions
# -----------------------------
@dataclass
class RunResult:
    model_name: str
    stage: str
    params: Dict[str, Any]
    train_time_s: float
    predict_time_val_s: float
    predict_time_test_s: float
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    best_iteration: Optional[int] = None
    best_score: Optional[float] = None


def dt_param_grid_stage1() -> List[Dict[str, Any]]:
    max_depths = [5, 10, 20, None]
    min_leaf = [1, 5, 10]
    criteria = ["gini", "entropy"]
    grid = []
    for md, ml, cr in product(max_depths, min_leaf, criteria):
        grid.append({"max_depth": md, "min_samples_leaf": ml, "criterion": cr})
    return grid


def dt_param_grid_refine(best: Dict[str, Any]) -> List[Dict[str, Any]]:
    # refine: zawężamy wokół best max_depth
    md = best.get("max_depth", None)
    ml = best.get("min_samples_leaf", 1)
    cr = best.get("criterion", "gini")

    max_depths = []
    if md is None:
        max_depths = [None, 20, 30]
    else:
        # np. 10 -> [7,10,13] z ograniczeniem min=2
        max_depths = sorted({max(2, md - 3), md, md + 3})

    min_leafs = sorted({max(1, ml - 2), ml, ml + 2})
    criteria = [cr]  # zostawiamy najlepsze kryterium
    return [{"max_depth": m, "min_samples_leaf": l, "criterion": c}
            for m, l, c in product(max_depths, min_leafs, criteria)]


def svm_param_grid_stage1() -> List[Dict[str, Any]]:
    # Dla LinearSVC/SVC: kluczowy parametr C
    Cs = [0.01, 0.1, 1.0, 10.0]
    return [{"C": C} for C in Cs]


def svm_param_grid_refine(best: Dict[str, Any]) -> List[Dict[str, Any]]:
    C = float(best.get("C", 1.0))
    # refinement: mnożniki wokół najlepszego C
    mult = [0.5, 1.0, 2.0]
    Cs = sorted({max(1e-4, C * m) for m in mult})
    return [{"C": c} for c in Cs]


def xgb_param_grid_stage1() -> List[Dict[str, Any]]:
    max_depths = [4, 8, 12]
    learning_rates = [0.1, 0.05]
    n_estimators = [200, 500]
    subsamples = [0.8, 1.0]
    colsample = [0.8, 1.0]

    grid = []
    for md, lr, ne, ss, cs in product(max_depths, learning_rates, n_estimators, subsamples, colsample):
        grid.append({
            "max_depth": md,
            "learning_rate": lr,
            "n_estimators": ne,
            "subsample": ss,
            "colsample_bytree": cs,
        })
    return grid


def xgb_param_grid_refine(best: Dict[str, Any]) -> List[Dict[str, Any]]:
    md = int(best.get("max_depth", 8))
    lr = float(best.get("learning_rate", 0.05))
    ne = int(best.get("n_estimators", 500))

    max_depths = sorted({max(2, md - 2), md, md + 2})
    learning_rates = sorted({max(0.005, lr * 0.5), lr, lr * 1.5})
    n_estimators = sorted({max(50, int(ne * 0.6)), ne, int(ne * 1.4)})

    # zostawiamy subsample/colsample z best
    ss = float(best.get("subsample", 0.8))
    cs = float(best.get("colsample_bytree", 0.8))

    grid = []
    for m, l, n in product(max_depths, learning_rates, n_estimators):
        grid.append({
            "max_depth": m,
            "learning_rate": l,
            "n_estimators": n,
            "subsample": ss,
            "colsample_bytree": cs,
        })
    return grid


# -----------------------------
# Model training wrappers
# -----------------------------
def run_decision_tree(X_tr, y_tr, X_va, y_va, X_te, y_te, cw, use_cw, seed, params) -> tuple[
    RunResult, DecisionTreeClassifier]:
    model = DecisionTreeClassifier(
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        criterion=params["criterion"],
        class_weight=class_weight_dict(cw, use_cw),
        random_state=seed,
    )

    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_va_pred = model.predict(X_va)
    pred_val_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_te_pred = model.predict(X_te)
    pred_test_time = time.perf_counter() - t0

    return RunResult(
        model_name="DecisionTree",
        stage="",
        params=params,
        train_time_s=train_time,
        predict_time_val_s=pred_val_time,
        predict_time_test_s=pred_test_time,
        val_metrics=compute_metrics(y_va, y_va_pred),
        test_metrics=compute_metrics(y_te, y_te_pred),
    ), model


def run_svm(X_tr, y_tr, X_va, y_va, X_te, y_te, cw, use_cw, seed, params, is_sparse) -> tuple[
    RunResult, LinearSVC | SVC]:
    # Jeśli dane są sparse -> LinearSVC, jeśli dense -> SVC (RBF domyślnie)
    if is_sparse:
        model = LinearSVC(
            C=float(params["C"]),
            class_weight=class_weight_dict(cw, use_cw),
            max_iter=5000,
            dual="auto",
            random_state=seed,
        )
        model_name = "LinearSVC"
    else:
        model = SVC(
            C=float(params["C"]),
            kernel="rbf",
            gamma="scale",
            class_weight=class_weight_dict(cw, use_cw),
            cache_size=1024.0,
            max_iter=5000,
            random_state=seed,
        )
        model_name = "SVC_RBF"

    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_va_pred = model.predict(X_va)
    pred_val_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_te_pred = model.predict(X_te)
    pred_test_time = time.perf_counter() - t0

    return RunResult(
        model_name=model_name,
        stage="",
        params=params,
        train_time_s=train_time,
        predict_time_val_s=pred_val_time,
        predict_time_test_s=pred_test_time,
        val_metrics=compute_metrics(y_va, y_va_pred),
        test_metrics=compute_metrics(y_te, y_te_pred),
    ), model


def run_xgboost(X_tr, y_tr, X_va, y_va, X_te, y_te, cw, use_cw, seed, params, device, n_jobs, early_stopping_rounds, verbose) -> \
tuple[RunResult, XGBClassifier]:
    num_classes = int(len(np.unique(y_tr)))

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,

        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        n_estimators=int(params["n_estimators"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),

        tree_method="hist",
        device=device,
        n_jobs=n_jobs,
        random_state=seed,

        eval_metric="mlogloss",
        early_stopping_rounds=early_stopping_rounds,
    )

    sw = sample_weight_vector(y_tr, cw, use_cw)

    t0 = time.perf_counter()
    model.fit(
        X_tr,
        y_tr,
        sample_weight=sw,
        eval_set=[(X_tr, y_tr), (X_va, y_va)],
          # ustawione w konstruktorze
        verbose=verbose,
    )
    train_time = time.perf_counter() - t0

    best_iteration = getattr(model, "best_iteration", None)
    best_score = getattr(model, "best_score", None)

    t0 = time.perf_counter()
    y_va_pred = model.predict(X_va)
    pred_val_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_te_pred = model.predict(X_te)
    pred_test_time = time.perf_counter() - t0

    return RunResult(
        model_name="XGBoost",
        stage="",
        params=params,
        train_time_s=train_time,
        predict_time_val_s=pred_val_time,
        predict_time_test_s=pred_test_time,
        val_metrics=compute_metrics(y_va, y_va_pred),
        test_metrics=compute_metrics(y_te, y_te_pred),
        best_iteration=int(best_iteration) if best_iteration is not None else None,
        best_score=float(best_score) if best_score is not None else None,
    ), model


# -----------------------------
# Orchestration
# -----------------------------
def pick_best(results: List[RunResult], target_metric: str) -> RunResult:
    return max(results, key=lambda r: r.val_metrics.get(target_metric, -1.0))


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_best_artifacts(outdir: Path, run: RunResult, model_obj: Any, X_te, y_te, mapping: dict):
    # Full report + confusion matrix on TEST for best model
    y_pred = model_obj.predict(X_te)
    report = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
    conf = confusion_matrix(y_te, y_pred).tolist()

    payload = {
        "model_name": run.model_name,
        "params": run.params,
        "val_metrics": run.val_metrics,
        "test_metrics": run.test_metrics,
        "train_time_s": run.train_time_s,
        "predict_time_val_s": run.predict_time_val_s,
        "predict_time_test_s": run.predict_time_test_s,
        "best_iteration": run.best_iteration,
        "best_score": run.best_score,
        "classification_report_test": report,
        "confusion_matrix_test": conf,
        "class_mapping": mapping,
    }

    with open(outdir / f"best_{run.model_name}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # Save model
    if run.model_name == "XGBoost":
        model_obj.save_model(str(outdir / "best_XGBoost_model.json"))
    else:
        dump(model_obj, outdir / f"best_{run.model_name}_model.joblib")


def to_row(r: RunResult) -> Dict[str, Any]:
    return {
        "model": r.model_name,
        "stage": r.stage,
        "params_json": json.dumps(r.params, ensure_ascii=False),
        "train_time_s": r.train_time_s,
        "predict_time_val_s": r.predict_time_val_s,
        "predict_time_test_s": r.predict_time_test_s,
        "val_accuracy": r.val_metrics["accuracy"],
        "val_balanced_accuracy": r.val_metrics["balanced_accuracy"],
        "val_macro_f1": r.val_metrics["macro_f1"],
        "val_weighted_f1": r.val_metrics["weighted_f1"],
        "test_accuracy": r.test_metrics["accuracy"],
        "test_balanced_accuracy": r.test_metrics["balanced_accuracy"],
        "test_macro_f1": r.test_metrics["macro_f1"],
        "test_weighted_f1": r.test_metrics["weighted_f1"],
        "best_iteration": r.best_iteration,
        "best_score": r.best_score,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt-dir", default="prepared/decision_tree")
    ap.add_argument("--svm-dir", default="prepared/svm")
    ap.add_argument("--xgb-dir", default="prepared/xgboost")
    ap.add_argument("--outdir", default="experiments")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-metric", default="val_macro_f1", choices=["val_macro_f1", "val_balanced_accuracy", "val_accuracy"])
    ap.add_argument("--use-class-weights", action="store_true")
    ap.add_argument("--device", default="cuda", help='XGBoost device: "cuda", "cuda:0" lub "cpu".')
    ap.add_argument("--n-jobs", type=int, default=0)
    ap.add_argument("--early-stopping-rounds", type=int, default=50)
    ap.add_argument("--xgb-verbose", type=int, default=0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    # Normalizujemy wybór metryki do klucza w RunResult
    target_map = {
        "val_macro_f1": "macro_f1",
        "val_balanced_accuracy": "balanced_accuracy",
        "val_accuracy": "accuracy",
    }
    target_metric = target_map[args.target_metric]

    all_rows: List[Dict[str, Any]] = []

    # -------------------------
    # 1) Decision Tree
    # -------------------------
    Xdt_tr, Xdt_va, Xdt_te, ydt_tr, ydt_va, ydt_te, cw_dt, mapping_dt = load_dense_dir(Path(args.dt_dir))

    dt_stage1_runs: List[RunResult] = []
    dt_stage1_models: List[Any] = []

    for p in dt_param_grid_stage1():
        rr, m = run_decision_tree(Xdt_tr, ydt_tr, Xdt_va, ydt_va, Xdt_te, ydt_te, cw_dt, args.use_class_weights, args.seed, p)
        rr.stage = "stage1"
        dt_stage1_runs.append(rr)
        dt_stage1_models.append(m)
        all_rows.append(to_row(rr))

    dt_best1 = pick_best(dt_stage1_runs, target_metric)
    dt_ref_grid = dt_param_grid_refine(dt_best1.params)

    dt_stage2_runs: List[RunResult] = []
    dt_stage2_models: List[Any] = []

    for p in dt_ref_grid:
        rr, m = run_decision_tree(Xdt_tr, ydt_tr, Xdt_va, ydt_va, Xdt_te, ydt_te, cw_dt, args.use_class_weights, args.seed, p)
        rr.stage = "stage2"
        dt_stage2_runs.append(rr)
        dt_stage2_models.append(m)
        all_rows.append(to_row(rr))

    dt_best2 = pick_best(dt_stage2_runs, target_metric)
    dt_best_model = dt_stage2_models[dt_stage2_runs.index(dt_best2)]
    save_best_artifacts(outdir, dt_best2, dt_best_model, Xdt_te, ydt_te, mapping_dt)

    # -------------------------
    # 2) SVM (auto: sparse -> LinearSVC, dense -> SVC)
    # -------------------------
    Xsv_tr, Xsv_va, Xsv_te, ysv_tr, ysv_va, ysv_te, cw_svm, mapping_svm = load_sparse_or_dense_dir(Path(args.svm_dir))
    svm_is_sparse = sparse.issparse(Xsv_tr)

    svm_stage1_runs: List[RunResult] = []
    svm_stage1_models: List[Any] = []

    for p in svm_param_grid_stage1():
        rr, m = run_svm(Xsv_tr, ysv_tr, Xsv_va, ysv_va, Xsv_te, ysv_te, cw_svm, args.use_class_weights, args.seed, p, svm_is_sparse)
        rr.stage = "stage1"
        svm_stage1_runs.append(rr)
        svm_stage1_models.append(m)
        all_rows.append(to_row(rr))

    svm_best1 = pick_best(svm_stage1_runs, target_metric)
    svm_ref_grid = svm_param_grid_refine(svm_best1.params)

    svm_stage2_runs: List[RunResult] = []
    svm_stage2_models: List[Any] = []

    for p in svm_ref_grid:
        rr, m = run_svm(Xsv_tr, ysv_tr, Xsv_va, ysv_va, Xsv_te, ysv_te, cw_svm, args.use_class_weights, args.seed, p, svm_is_sparse)
        rr.stage = "stage2"
        svm_stage2_runs.append(rr)
        svm_stage2_models.append(m)
        all_rows.append(to_row(rr))

    svm_best2 = pick_best(svm_stage2_runs, target_metric)
    svm_best_model = svm_stage2_models[svm_stage2_runs.index(svm_best2)]
    save_best_artifacts(outdir, svm_best2, svm_best_model, Xsv_te, ysv_te, mapping_svm)

    # -------------------------
    # 3) XGBoost (sparse)
    # -------------------------
    Xx_tr, Xx_va, Xx_te, yx_tr, yx_va, yx_te, cw_xgb, mapping_xgb = load_sparse_or_dense_dir(Path(args.xgb_dir))

    xgb_stage1_runs: List[RunResult] = []
    xgb_stage1_models: List[Any] = []

    for p in xgb_param_grid_stage1():
        rr, m = run_xgboost(
            Xx_tr, yx_tr, Xx_va, yx_va, Xx_te, yx_te,
            cw_xgb, args.use_class_weights, args.seed,
            p, args.device, args.n_jobs, args.early_stopping_rounds, args.xgb_verbose
        )
        rr.stage = "stage1"
        xgb_stage1_runs.append(rr)
        xgb_stage1_models.append(m)
        all_rows.append(to_row(rr))

    xgb_best1 = pick_best(xgb_stage1_runs, target_metric)
    xgb_ref_grid = xgb_param_grid_refine(xgb_best1.params)

    xgb_stage2_runs: List[RunResult] = []
    xgb_stage2_models: List[Any] = []

    for p in xgb_ref_grid:
        rr, m = run_xgboost(
            Xx_tr, yx_tr, Xx_va, yx_va, Xx_te, yx_te,
            cw_xgb, args.use_class_weights, args.seed,
            p, args.device, args.n_jobs, args.early_stopping_rounds, args.xgb_verbose
        )
        rr.stage = "stage2"
        xgb_stage2_runs.append(rr)
        xgb_stage2_models.append(m)
        all_rows.append(to_row(rr))

    xgb_best2 = pick_best(xgb_stage2_runs, target_metric)
    xgb_best_model = xgb_stage2_models[xgb_stage2_runs.index(xgb_best2)]
    save_best_artifacts(outdir, xgb_best2, xgb_best_model, Xx_te, yx_te, mapping_xgb)

    # -------------------------
    # Save results
    # -------------------------
    df = pd.DataFrame(all_rows)
    df.to_csv(outdir / "results.csv", index=False, encoding="utf-8")
    print("Zapisano wyniki do:", (outdir / "results.csv").resolve())
    print("Zapisano najlepsze modele i metryki do:", outdir.resolve())


if __name__ == "__main__":
    main()
