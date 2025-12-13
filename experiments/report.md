# Wyniki eksperymentów klasyfikacji

## Zestawienie najlepszych wyników

Tabela: `summary_table.csv` (oraz `summary_table.tex` do LaTeX).

Wykres: `figures/best_metric_by_model.png`.


### Najlepsze konfiguracje (wg walidacji)

**XGBoost** (stage=stage2):

- val_macro_f1: 0.9865

- test_macro_f1: 0.9724

- train_time_s: 6.34

- params: {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 280, "subsample": 1.0, "colsample_bytree": 0.8}

- pełne metryki: `best_XGBoost.json`



**DecisionTree** (stage=stage1):

- val_macro_f1: 0.9657

- test_macro_f1: 0.9386

- train_time_s: 0.61

- params: {"max_depth": 20, "min_samples_leaf": 1, "criterion": "entropy"}

- pełne metryki: `best_DecisionTree.json`



**LinearSVC** (stage=stage1):

- val_macro_f1: 0.9074

- test_macro_f1: 0.9094

- train_time_s: 288.20

- params: {"C": 10.0}

- pełne metryki: `best_LinearSVC.json`



## Wykresy strojenia parametrów

- SVM: `figures/svm_metric_vs_C.png`

- DecisionTree: `figures/dt_metric_vs_max_depth.png`

- XGBoost: `figures/xgb_metric_vs_n_estimators.png`



## Zależność jakości od czasu treningu

Wykres: `figures/metric_vs_train_time_scatter.png`
