
# Requirements:
#   pip install torch torch_geometric scikit-learn rdkit-pypi (if supported) OR rdkit (on Colab)
# Notes:
#   - Uses only rdkit.Chem (no rdMolStandardize) → safe on Python 3.12
#   - If some categories have different names, edit the `categories` dict below.

import os, sys, math, numpy as np, pandas as pd
from collections import defaultdict

# -------- RDKit (Chem-only, quiet) --------
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
from rdkit.rdBase import BlockLogs
RDLogger.DisableLog('rdApp.error')

# -------- Torch / PyG --------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_undirected

# -------- Sklearn --------
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, matthews_corrcoef,
    balanced_accuracy_score, precision_score, recall_score, confusion_matrix,
    precision_recall_curve
)
from sklearn.isotonic import IsotonicRegression


# ---------------------------
# 0) Helpers: safe RDKit ops
# ---------------------------
def safe_mol_from_smiles(smi: str):
    """Parse SMILES defensively (Chem-only). Returns Mol or None."""
    if not isinstance(smi, str) or not smi.strip():
        return None
    with BlockLogs():
        mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        try:
            # Relax: skip SANITIZE_PROPERTIES (valence checks), keep the rest
            Chem.SanitizeMol(
                mol,
                sanitizeOps = Chem.SanitizeFlags.SANITIZE_ALL ^
                              Chem.SanitizeFlags.SANITIZE_PROPERTIES
            )
            return mol
        except Exception:
            return None

def mol_to_ecfp(mol, radius=2, nBits=2048):
    if mol is None:
        return np.zeros((nBits,), dtype=np.int8)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits), dtype=np.int8)

def safe_scaffold(mol):
    if mol is None: return None
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaf, isomericSmiles=True)
    except Exception:
        return None


# ------------------------------------
# 1) Load CSV, label-encode, clean
# ------------------------------------
CSV_PATH = "MoGAT_dataset_full_clean.csv"
df = pd.read_csv(CSV_PATH)

# Encode labels
if df["Label"].dtype != int:
    df["Label"] = df["Label"].map({"most": 1, "no": 0}).astype(int)

# Parse SMILES safely and drop invalid rows
bad_idx, mols = [], []
for i, smi in enumerate(df["SMILES"].tolist()):
    m = safe_mol_from_smiles(smi)
    if m is None:
        bad_idx.append(i)
    mols.append(m)

print(f"[INFO] Invalid SMILES dropped: {len(bad_idx)} / {len(df)}")
if bad_idx:
    pd.Series(bad_idx, name="dropped_row_indices").to_csv("dropped_smiles_indices.csv", index=False)
    df = df.drop(index=bad_idx).reset_index(drop=True)
    mols = [m for i,m in enumerate(mols) if i not in bad_idx]

# Extract features/labels
y = df["Label"].values
X_all = df.drop(columns=["Label", "SMILES"])


# ------------------------------------
# 2) Define feature categories (EDIT)
# ------------------------------------
categories = {
    "adme": ["Blood-Brain Barrier Penetration", "Oral Bioavailability", "Bioavailability Score"],
    "cyp":  [c for c in X_all.columns if "CYP" in c],
    "rules": ["Lipinski #violations", "Ghose #violations", "Veber #violations",
              "Egan #violations", "Muegge #violations", "Leadlikeness #violations"],
    "alerts": ["PAINS #alerts", "Brenk #alerts"],
    "solubility": [c for c in X_all.columns if "Solubility" in c],
    "synthetic": ["Synthetic Accessibility"],
    "mutagenicity": ["Mutagenicity"]
}
# Drop empty categories (if any names don't exist)
categories = {k:v for k,v in categories.items() if len([c for c in v if c in X_all.columns]) > 0}
# Ensure only existing cols
categories = {k:[c for c in v if c in X_all.columns] for k,v in categories.items()}

print("[INFO] Categories and sizes:")
for k,v in categories.items():
    print(f"  - {k}: {len(v)} features")


# ------------------------------------
# 3) Build per-category graphs (kNN)
# ------------------------------------
def build_category_graph(Xdf, feature_list, k=8):
    feats = Xdf[feature_list].values.astype(float)
    feats = StandardScaler().fit_transform(feats)
    sim = cosine_similarity(feats)
    np.fill_diagonal(sim, 0.0)

    rows, cols = [], []
    for i in range(sim.shape[0]):
        nbrs = np.argsort(sim[i])[-k:]  # top-k neighbors (can include i if equal; diag was zeroed)
        for j in nbrs:
            if sim[i, j] > 0:
                rows.append(i); cols.append(j)

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_index = to_undirected(edge_index)  # symmetric graph
    x = torch.tensor(feats, dtype=torch.float32)
    return Data(x=x, edge_index=edge_index)

graphs = {cat: build_category_graph(X_all, feats, k=8) for cat, feats in categories.items()}

# Optional: add an extra "ecfp" category head
# ecfp = np.vstack([mol_to_ecfp(m) for m in mols])
# graphs["ecfp"] = Data(x=torch.tensor(StandardScaler().fit_transform(ecfp), dtype=torch.float32),
#                       edge_index=to_undirected(torch.empty((2,0), dtype=torch.long)))


# ------------------------------------
# 4) Scaffold split (fallback: stratified)
# ------------------------------------
scaffolds = [safe_scaffold(m) for m in mols]
scaf2idx = defaultdict(list)
for i, sc in enumerate(scaffolds):
    scaf2idx[sc].append(i)
clusters = sorted(scaf2idx.values(), key=len, reverse=True)

N = len(df)
train_idx, val_idx, test_idx = [], [], []
for c in clusters:
    if len(train_idx) < 0.7*N: train_idx += c
    elif len(val_idx) < 0.85*N: val_idx += c
    else: test_idx += c

if min(len(train_idx),len(val_idx),len(test_idx)) == 0:
    print("[WARN] Fallback to stratified split.")
    idx = np.arange(N)
    train_idx, tmp_idx, y_tr, y_tmp = train_test_split(idx, y, test_size=0.3, stratify=y, random_state=42)
    val_idx, test_idx, _, _ = train_test_split(tmp_idx, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

train_idx = np.array(train_idx); val_idx = np.array(val_idx); test_idx = np.array(test_idx)
print(f"[INFO] Split sizes: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")


# ------------------------------------
# 5) FCA-GAT model (node-level output)
# ------------------------------------
class FCAGAT(nn.Module):
    def __init__(self, in_dims, hidden=64, heads=2, num_classes=1, gat_dropout=0.3, mlp_dropout=0.4):
        super().__init__()
        self.categories = list(in_dims.keys())
        self.gats = nn.ModuleDict()
        for cat in self.categories:
            self.gats[cat] = GATConv(
                in_channels=in_dims[cat],
                out_channels=hidden,
                heads=heads,
                dropout=gat_dropout,
                add_self_loops=True
            )
        fused_dim = hidden * heads * len(self.categories)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, batch_dict):
        node_embeds = []
        for cat, data in batch_dict.items():
            h = self.gats[cat](data.x, data.edge_index)  # [N, hidden*heads]
            h = F.elu(h)
            node_embeds.append(h)
        H = torch.cat(node_embeds, dim=-1)              # [N, fused_dim]
        return self.classifier(H)                       # [N, 1]


# ------------------------------------
# 6) Metrics & utils
# ------------------------------------
def eval_metrics(logits, y_true):
    probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
    y_true = y_true.detach().cpu().numpy().ravel()
    y_pred = (probs >= 0.5).astype(int)
    mets = {}
    try: mets["AUROC"] = roc_auc_score(y_true, probs)
    except: mets["AUROC"] = np.nan
    try: mets["AUPRC"] = average_precision_score(y_true, probs)
    except: mets["AUPRC"] = np.nan
    mets["Accuracy"] = (y_pred == y_true).mean()
    mets["BalancedAcc"] = balanced_accuracy_score(y_true, y_pred)
    mets["F1"] = f1_score(y_true, y_pred, zero_division=0)
    mets["Precision"] = precision_score(y_true, y_pred, zero_division=0)
    mets["Recall"] = recall_score(y_true, y_pred, zero_division=0)
    mets["MCC"] = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true))>1 else np.nan
    mets["ConfusionMatrix"] = confusion_matrix(y_true, y_pred).tolist()
    return mets

def tune_threshold_on_val(val_probs, val_true):
    prec, rec, thr = precision_recall_curve(val_true, val_probs)
    f1s = 2*prec*rec/(prec+rec+1e-12)
    # thr has len = len(prec)-1; align
    best_i = np.nanargmax(f1s)
    tau = thr[max(best_i-1, 0)] if len(thr) else 0.5
    return float(tau), float(np.nanmax(f1s))


# ==========================
# 7) Cross-Validation + Uncertainty Eval
# ==========================
from sklearn.model_selection import StratifiedKFold

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for cat in graphs:
    graphs[cat] = graphs[cat].to(device)

y_tensor = torch.tensor(y, dtype=torch.float32).view(-1,1).to(device)

def run_fold(train_idx, val_idx, test_idx, graphs, y_tensor, patience=20):
    """Train/Eval FCA-GAT on one split and return metrics."""
    in_dims = {cat: graphs[cat].num_features for cat in graphs}
    model = FCAGAT(in_dims, hidden=64, heads=2,
                   gat_dropout=0.3, mlp_dropout=0.4).to(device)

    # class weight
    train_pos = int(y_tensor[train_idx].sum().item())
    train_neg = len(train_idx) - train_pos
    pos_weight = torch.tensor(train_neg / max(train_pos,1),
                              dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)

    best_state, best_val, wait = None, -1.0, 0

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        logits_all = model(graphs)
        loss = criterion(logits_all[train_idx], y_tensor[train_idx])
        loss.backward(); optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_all = model(graphs)
            val_m = eval_metrics(logits_all[val_idx], y_tensor[val_idx])
            cur = val_m.get("AUPRC", np.nan)

        if cur > best_val:
            best_val, wait = cur, 0
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= patience: break

    model.load_state_dict(best_state)

    # --- final eval ---
    model.eval()
    with torch.no_grad():
        logits_all = model(graphs)
        val_probs = torch.sigmoid(logits_all[val_idx]).cpu().numpy().ravel()
        val_true  = y_tensor[val_idx].cpu().numpy().ravel()
        test_probs = torch.sigmoid(logits_all[test_idx]).cpu().numpy().ravel()
        test_true  = y_tensor[test_idx].cpu().numpy().ravel()

    tau, _ = tune_threshold_on_val(val_probs, val_true)
    test_pred = (test_probs >= tau).astype(int)

    fold_metrics = {
        "AUROC": roc_auc_score(test_true, test_probs),
        "AUPRC": average_precision_score(test_true, test_probs),
        "F1": f1_score(test_true, test_pred),
        "MCC": matthews_corrcoef(test_true, test_pred),
        "BalancedAcc": balanced_accuracy_score(test_true, test_pred),
        "Precision": precision_score(test_true, test_pred, zero_division=0),
        "Recall": recall_score(test_true, test_pred, zero_division=0),
        "tau": tau
    }

    # --- Calibration + Conformal ---
    try:
        iso = IsotonicRegression(out_of_bounds='clip').fit(val_probs, val_true)
        test_probs_cal = iso.predict(test_probs)
        fold_metrics["Cal_AUROC"] = roc_auc_score(test_true, test_probs_cal)
        fold_metrics["Cal_AUPRC"] = average_precision_score(test_true, test_probs_cal)

        alpha = 0.10
        cal_probs_val = iso.predict(val_probs)
        s1 = 1 - cal_probs_val[val_true==1]
        s0 =     cal_probs_val[val_true==0]
        q1 = np.quantile(s1, 1-alpha, method="higher")
        q0 = np.quantile(s0, 1-alpha, method="higher")

        def cp_set(p):
            S = []
            if (1-p) <= q1: S.append(1)
            if (p)   <= q0: S.append(0)
            return S

        test_sets = [cp_set(p) for p in test_probs_cal]
        coverage  = np.mean([test_true[i] in test_sets[i] for i in range(len(test_true))])
        singleton = np.mean([len(s)==1 for s in test_sets])
        fold_metrics["CP_coverage"] = coverage
        fold_metrics["CP_singleton"] = singleton
    except Exception as e:
        fold_metrics["Cal_AUROC"] = np.nan
        fold_metrics["Cal_AUPRC"] = np.nan
        fold_metrics["CP_coverage"] = np.nan
        fold_metrics["CP_singleton"] = np.nan

    return fold_metrics


# ==========================
# Run 5-Fold Stratified CV
# ==========================
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []
for fold, (trainval_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
    tr_idx, val_idx, _, _ = train_test_split(
        trainval_idx, y[trainval_idx],
        test_size=0.2, stratify=y[trainval_idx], random_state=fold
    )
    tr_idx = torch.tensor(tr_idx, dtype=torch.long, device=device)
    val_idx = torch.tensor(val_idx, dtype=torch.long, device=device)
    test_idx = torch.tensor(test_idx, dtype=torch.long, device=device)

    fold_m = run_fold(tr_idx, val_idx, test_idx, graphs, y_tensor)
    results.append(fold_m)
    print(f"[Fold {fold}] {fold_m}")

# summarize
df_res = pd.DataFrame(results)
print("\n=== Cross-Validation Summary ===")
print(df_res.mean().round(3).to_dict(), "±", df_res.std().round(3).to_dict())

