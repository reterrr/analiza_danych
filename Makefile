# --- Config ---
PYTHON ?= python3
DATA   ?= data.csv
LABEL  ?= Attack_type

PREPARE_DIR := prepare
OUT_DIR := $(PREPARE_DIR)/prepared
XGB_DIR := $(OUT_DIR)/xgboost
SVM_DIR := $(OUT_DIR)/svm
DT_DIR  := $(OUT_DIR)/decision_tree

.PHONY: all clean xgboost svm decision_tree

# Build everything
all: xgboost svm decision_tree

# Friendly phony targets
xgboost: $(XGB_DIR)
svm: $(SVM_DIR)
decision_tree: $(DT_DIR)

# Real targets use the directory itself (created if missing)
$(XGB_DIR): $(PREPARE_DIR)/prepare_xgboost.py $(DATA)
	mkdir -p $@
	$(PYTHON) $(PREPARE_DIR)/prepare_xgboost.py --input $(DATA) --outdir $@ --label-col $(LABEL)

$(SVM_DIR): $(PREPARE_DIR)/prepare_svm.py $(DATA)
	mkdir -p $@
	$(PYTHON) $(PREPARE_DIR)/prepare_svm.py --input $(DATA) --outdir $@ --label-col $(LABEL)

$(DT_DIR): $(PREPARE_DIR)/prepare_decision_tree.py $(DATA)
	mkdir -p $@
	$(PYTHON) $(PREPARE_DIR)/prepare_decision_tree.py --input $(DATA) --outdir $@ --label-col $(LABEL)

# Cleanup
clean:
	rm -rf $(OUT_DIR)
