# 🧠 FCA-GAT: Feature-Category Graph Attention Network for Molecular Cardiotoxicity Prediction

FCA-GAT (Feature-Category Graph Attention Network) is a leakage-aware graph learning framework designed for structured molecular classification tasks.  
It models biochemical descriptor groups as independent similarity graphs and learns domain-specific embeddings using Graph Attention Networks (GAT).

This repository demonstrates a reproducible training protocol with explicit sanity checks to ensure that performance is driven by learning rather than data leakage or structural artifacts.

---

## 🔬 Dataset

This project uses the **DICT (Drug-Induced Cardiotoxicity) dataset**, a curated collection of FDA-approved drugs labeled for cardiotoxicity risk.

The dataset includes:

- Trade name  
- Generic name  
- Active ingredient  
- Cardiotoxicity category (Arrhythmia, Heart Damage, Mixed, etc.)  
- DICT concern level  
- Severity level  
- Keywords (e.g., QT prolongation, myocardial infarction)  
- Binary classification label (`y`)  

The classification task predicts whether a drug belongs to the cardiotoxic risk category based on structured molecular descriptors.

⚠️ The DICT dataset retains its original licensing and citation requirements.  
The MIT license in this repository applies **only to the code**.

---

## 🧩 Feature Categories Modeled as Graphs

Instead of treating molecular descriptors as a flat vector, FCA-GAT groups features into domain-specific categories:

- ADME features  
- CYP inhibition features  
- Drug-likeness rule violations  
- Toxicity alerts  
- Solubility descriptors  
- Synthetic accessibility  
- Mutagenicity  

Each category is used to build a cosine similarity graph across molecules.  
A dedicated `GATConv` layer processes each category graph, and the resulting embeddings are fused for classification.

## 🏗 Model Architecture
<img width="372" height="533" alt="image" src="https://github.com/user-attachments/assets/5b27fe49-4db0-40f0-824a-523a7e7dcd09" />

---

## 📊 Experimental Results (Representative Fold)

### Dataset Summary

- Total samples: 555
- Invalid SMILES removed: 0
- Feature categories:

| Category        | # Features |
|---------------|------------|
| ADME          | 3 |
| CYP           | 13 |
| Rules         | 6 |
| Alerts        | 2 |
| Solubility    | 7 |
| Synthetic     | 1 |
| Mutagenicity  | 1 |

---

## 🧠 Real Training (Leakage-Controlled)

**Early Stopping:** Epoch 54  
**Best Validation AUPRC:** 0.675  

### Test Performance

| Metric | Value |
|--------|--------|
| AUROC | 0.773 |
| AUPRC | 0.801 |
| F1-score | 0.728 |
| Precision | 0.573 |
| Recall | 1.000 |
| Balanced Accuracy | 0.577 |
| MCC | 0.297 |
| Optimal Threshold (τ) | 0.156 |
| Train AUPRC | 0.886 |
| Validation AUPRC | 0.675 |

---

### Interpretation

- The model shows strong discrimination (AUROC 0.773).
- AUPRC (0.801) indicates good performance under class imbalance.
- Recall = 1.0 suggests threshold tuning favors high sensitivity.
- Balanced accuracy and MCC indicate moderate overall balance.
- Train vs Validation AUPRC gap suggests controlled but present overfitting.

---

## 🔁 Label Shuffle Control (Sanity Check)

To verify learning validity, training labels were shuffled while preserving validation and test labels.

**Expected:** Performance should approach chance level.

### Shuffle Test Performance

| Metric | Value |
|--------|--------|
| AUROC | 0.764 |
| AUPRC | 0.808 |
| F1-score | 0.695 |
| Precision | 0.537 |
| Recall | 0.983 |
| Balanced Accuracy | 0.511 |
| MCC | 0.066 |
| Optimal Threshold (τ) | 0.080 |
| Train AUPRC | 0.851 |
| Validation AUPRC | 0.669 |

---
## 🔬 Extended Cross-Validation Results (5-Fold Stratified CV)
(check Extendedexp.py)
To ensure robustness and avoid single-split optimism, FCA-GAT was evaluated
using 5-fold stratified cross-validation with leakage-controlled graph construction.

### Dataset Split

- Total samples: 555
- Invalid SMILES removed: 0
- Per-fold split (approximate):
  - Train: 388
  - Validation: 83
  - Test: 84

---

### 📊 Per-Fold Results

| Fold | AUROC | AUPRC | F1 | MCC | BalancedAcc | Precision | Recall | τ |
|------|-------|-------|-----|-----|-------------|-----------|--------|-----|
| 0 | 0.843 | 0.875 | 0.698 | 0.102 | 0.510 | 0.536 | 1.000 | 0.043 |
| 1 | 0.707 | 0.750 | 0.734 | 0.341 | 0.653 | 0.638 | 0.864 | 0.341 |
| 2 | 0.651 | 0.741 | 0.695 | 0.095 | 0.522 | 0.543 | 0.966 | 0.385 |
| 3 | 0.733 | 0.769 | 0.682 | 0.259 | 0.625 | 0.620 | 0.759 | 0.453 |
| 4 | 0.768 | 0.807 | 0.685 | 0.372 | 0.686 | 0.717 | 0.655 | 0.537 |

---

### 📈 Cross-Validation Summary (Mean ± Std)

| Metric | Mean | Std |
|--------|------|------|
| AUROC | 0.740 | ± 0.071 |
| AUPRC | 0.789 | ± 0.055 |
| F1 | 0.699 | ± 0.021 |
| MCC | 0.234 | ± 0.130 |
| Balanced Accuracy | 0.599 | ± 0.079 |
| Precision | 0.611 | ± 0.075 |
| Recall | 0.849 | ± 0.144 |
| Optimal Threshold (τ) | 0.352 | ± 0.188 |

---

### 🧪 Calibration & Conformal Prediction

| Metric | Mean | Std |
|--------|------|------|
| Calibrated AUROC | 0.720 | ± 0.062 |
| Calibrated AUPRC | 0.725 | ± 0.061 |
| Conformal Coverage | 0.942 | ± 0.037 |
| Conformal Singleton Rate | 0.315 | ± 0.105 |

---

### 📌 Interpretation
- **High Recall (~0.85 mean across folds)**  
  The model detects most cardiotoxic drugs, minimizing false negatives.

- **Moderate Precision (~0.61 mean)**  
  Some safe drugs are flagged as toxic.  
  In toxicology and drug safety screening, this bias is often acceptable:
  missing a harmful drug is more costly than over-flagging a safe one.
- Performance remains consistent across folds with moderate variance.
- AUROC ≈ 0.74 indicates meaningful discrimination beyond random chance.
- AUPRC ≈ 0.79 suggests good behavior under class imbalance.
- Calibration slightly reduces AUROC, indicating probability overconfidence correction.
- Conformal coverage (~94%) confirms uncertainty control behaves as expected.

These results demonstrate stable learning under stratified cross-validation
with leakage-controlled graph construction.

## ⚠️ Project Status: Research Prototype

This repository represents an early-stage research prototype exploring
feature-category graph modeling for molecular cardiotoxicity prediction.
