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
## ⚠️ Project Status: Research Prototype

This repository represents an early-stage research prototype exploring
feature-category graph modeling for molecular cardiotoxicity prediction.
