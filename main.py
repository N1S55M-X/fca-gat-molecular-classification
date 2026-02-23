
import os, time, random, math
import numpy as np
import pandas as pd

# ---------- RDKit (quiet) ----------
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
from rdkit.rdBase import BlockLogs
RDLogger.DisableLog('rdApp.error')

# ---------- Torch / PyG ----------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_undirected

# ---------- Sklearn ----------
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, matthews_corrcoef,
    balanced_accuracy_score, precision_score, recall_score, confusion_matrix,
    precision_recall_curve
)
from sklearn.isotonic import IsotonicRegression

# ---------- Plotting ----------
import matplotlib.pyplot as plt

# ---------------------------
# 0) Reproducibility
# ---------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# 1) Safe RDKit helpers
# ---------------------------
def safe_mol_from_smiles(smi: str):
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
            Chem.SanitizeMol(
                mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
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

# ---------------------------
# 2) Load CSV & basic prep
# ---------------------------
CSV_PATH = "MoGAT_dataset_full_clean.csv"
df = pd.read_csv(CSV_PATH)

# Label encode
if df["Label"].dtype != int:
    df["Label"] = df["Label"].map({"most": 1, "no": 0}).astype(int)

# Parse SMILES and drop invalids
bad_idx, mols = [], []
for i, smi in enumerate(df["SMILES"].tolist()):
    m = safe_mol_from_smiles(smi)
    if m is None: bad_idx.append(i)
    mols.append(m)

print(f"[INFO] Invalid SMILES dropped: {len(bad_idx)} / {len(df)}")
if bad_idx:
    df = df.drop(index=bad_idx).reset_index(drop=True)
    mols = [m for i, m in enumerate(mols) if i not in bad_idx]

y = df["Label"].values
X_all = df.drop(columns=["Label", "SMILES"])

# ---------------------------
# 3) Feature categories
# ---------------------------
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
# clean missing columns
categories = {k: [c for c in v if c in X_all.columns] for k, v in categories.items() if any(c in X_all.columns for c in v)}

print("[INFO] Categories and sizes:")
for k, v in categories.items():
    print(f"  - {k}: {len(v)} features")

# ---------------------------
# 4) Leakage-free graph builder (fit on TRAIN only)
# ---------------------------
def build_category_graph_from_train(Xdf, feature_list, train_idx, k=8):
    cols = [c for c in feature_list if c in Xdf.columns]
    if len(cols) == 0:
        # empty category guard
        return Data(x=torch.zeros((len(Xdf), 1), dtype=torch.float32),
                    edge_index=to_undirected(torch.empty((2,0), dtype=torch.long)))

    X = Xdf[cols].values.astype(float)

    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()

    X_tr = imputer.fit_transform(X[train_idx])
    X_tr = scaler.fit_transform(X_tr)

    X_all_tf = scaler.transform(imputer.transform(X))

    sim = cosine_similarity(X_all_tf)
    np.fill_diagonal(sim, 0.0)

    rows, cols_idx = [], []
    for i in range(sim.shape[0]):
        nbrs = np.argsort(sim[i])[-8:]
        for j in nbrs:
            if sim[i, j] > 0:
                rows.append(i); cols_idx.append(j)

    edge_index = torch.tensor([rows, cols_idx], dtype=torch.long)
    edge_index = to_undirected(edge_index)
    x = torch.tensor(X_all_tf, dtype=torch.float32)
    return Data(x=x, edge_index=edge_index)

def build_graphs_for_fold(categories, X_all, train_idx, k=8):
    gdict = {}
    for cat, feats in categories.items():
        gdict[cat] = build_category_graph_from_train(X_all, feats, train_idx, k=k)
    return gdict

# ---------------------------
# 5) FCA-GAT model
# ---------------------------
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

# ---------------------------
# 6) Metrics & utils
# ---------------------------
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
    best_i = np.nanargmax(f1s)
    tau = thr[max(best_i-1, 0)] if len(thr) else 0.5
    return float(tau), float(np.nanmax(f1s))

# ---------------------------
# 7) Training with history (proof of learning)
# ---------------------------
def run_fold_with_history(train_idx, val_idx, test_idx, graphs, y_tensor,
                          patience=20, max_epochs=200, fold_name="fold"):
    in_dims = {cat: graphs[cat].num_features for cat in graphs}
    model = FCAGAT(in_dims, hidden=64, heads=2, gat_dropout=0.3, mlp_dropout=0.4).to(device)

    train_pos = int(y_tensor[train_idx].sum().item())
    train_neg = len(train_idx) - train_pos
    pos_weight = torch.tensor(train_neg / max(train_pos,1), dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)

    history = {"epoch": [], "train_loss": [], "train_auprc": [], "val_auprc": [], "val_auroc": []}
    best_state, best_val, wait = None, -1.0, 0

    t0 = time.time()
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits_all = model(graphs)
        loss = criterion(logits_all[train_idx], y_tensor[train_idx])
        loss.backward(); optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_all = model(graphs)
            train_m = eval_metrics(logits_all[train_idx], y_tensor[train_idx])
            val_m   = eval_metrics(logits_all[val_idx],   y_tensor[val_idx])

        history["epoch"].append(epoch)
        history["train_loss"].append(float(loss.item()))
        history["train_auprc"].append(float(train_m.get("AUPRC", float("nan"))))
        history["val_auprc"].append(float(val_m.get("AUPRC", float("nan"))))
        history["val_auroc"].append(float(val_m.get("AUROC", float("nan"))))

        cur = val_m.get("AUPRC", float("nan"))
        if not np.isnan(cur) and cur > best_val:
            best_val, wait = cur, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1

        if epoch % 10 == 0:
            print(f"[{fold_name} {epoch:03d}] loss {loss.item():.4f} | "
                  f"train AUPRC {train_m['AUPRC']:.3f} | "
                  f"val AUPRC {val_m['AUPRC']:.3f} AUROC {val_m['AUROC']:.3f}")

        if wait >= patience:
            print(f"[{fold_name}] early stop at {epoch}; best Val AUPRC={best_val:.3f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test
    model.eval()
    with torch.no_grad():
        logits_all = model(graphs)
        val_probs  = torch.sigmoid(logits_all[val_idx]).cpu().numpy().ravel()
        val_true   = y_tensor[val_idx].cpu().numpy().ravel()
        test_probs = torch.sigmoid(logits_all[test_idx]).cpu().numpy().ravel()
        test_true  = y_tensor[test_idx].cpu().numpy().ravel()

    tau, _ = tune_threshold_on_val(val_probs, val_true)
    test_pred = (test_probs >= tau).astype(int)

    final = {
        "AUROC": roc_auc_score(test_true, test_probs),
        "AUPRC": average_precision_score(test_true, test_probs),
        "F1": f1_score(test_true, test_pred),
        "Precision": precision_score(test_true, test_pred, zero_division=0),
        "Recall": recall_score(test_true, test_pred, zero_division=0),
        "BalancedAcc": balanced_accuracy_score(test_true, test_pred),
        "MCC": matthews_corrcoef(test_true, test_pred),
        "tau": tau
    }

    # Overfit gap probe
    train_probs = torch.sigmoid(logits_all[train_idx]).cpu().numpy().ravel()
    train_true  = y_tensor[train_idx].cpu().numpy().ravel()
    final["Train_AUPRC"] = average_precision_score(train_true, train_probs)
    final["Val_AUPRC"]   = average_precision_score(val_true,  val_probs)

    # Save curves
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(f"{fold_name}_learning_curve.csv", index=False)

    plt.figure()
    plt.plot(hist_df["epoch"], hist_df["train_loss"], label="Train loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"{fold_name} - Training Loss"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{fold_name}_loss.png"); plt.close()

    plt.figure()
    plt.plot(hist_df["epoch"], hist_df["train_auprc"], label="Train AUPRC")
    plt.plot(hist_df["epoch"], hist_df["val_auprc"],   label="Val AUPRC")
    plt.plot(hist_df["epoch"], hist_df["val_auroc"],   label="Val AUROC")
    plt.xlabel("Epoch"); plt.ylabel("Metric"); plt.title(f"{fold_name} - Learning Curves"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{fold_name}_metrics.png"); plt.close()

    print(f"[{fold_name}] done in {time.time()-t0:.1f}s | Test: {final}")
    return final, hist_df

# ---------------------------
# 8) Label-shuffle control
# ---------------------------
def label_shuffle_control(train_idx, val_idx, test_idx, graphs, y_tensor, repeats=1, fold_name="fold"):
    rng = np.random.default_rng(123)
    y_np = y_tensor.detach().cpu().numpy().ravel()
    out = []
    for r in range(repeats):
        y_sh = y_np.copy()
        # Shuffle only TRAIN labels; keep val/test ground-truth for fair eval
        tr = train_idx.detach().cpu().numpy()
        rng.shuffle(y_sh[tr])
        y_sh_t = torch.tensor(y_sh, dtype=torch.float32, device=device).view(-1,1)

        res, _ = run_fold_with_history(
            train_idx, val_idx, test_idx, graphs, y_sh_t,
            patience=10, max_epochs=100, fold_name=f"{fold_name}_shuffle_r{r}"
        )
        out.append(res)

    df = pd.DataFrame(out)
    print("\n[Shuffle Control] Mean metrics (should be ~chance):")
    print(df.mean(numeric_only=True).round(3).to_dict())
    return df

# ---------------------------
# 9) Run: one representative fold (proof of learning)
# ---------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
(trainval_idx, test_idx_np) = next(skf.split(np.zeros(len(y)), y))
tr_idx_np, val_idx_np, _, _ = train_test_split(
    trainval_idx, y[trainval_idx], test_size=0.2,
    stratify=y[trainval_idx], random_state=SEED
)

# Rebuild leakage-free graphs for this fold
graphs_fold = build_graphs_for_fold(categories, X_all, tr_idx_np, k=8)
for cat in graphs_fold:
    graphs_fold[cat] = graphs_fold[cat].to(device)

tr_idx  = torch.tensor(tr_idx_np,  dtype=torch.long, device=device)
val_idx = torch.tensor(val_idx_np, dtype=torch.long, device=device)
test_idx= torch.tensor(test_idx_np, dtype=torch.long, device=device)
y_tensor_full = torch.tensor(y, dtype=torch.float32, device=device).view(-1,1)

print("\n=== REAL TRAINING (should learn) ===")
real_res, real_hist = run_fold_with_history(
    tr_idx, val_idx, test_idx, graphs_fold, y_tensor_full,
    patience=20, max_epochs=200, fold_name="real_fold0"
)
print(real_res)

print("\n=== LABEL SHUFFLE CONTROL (should be near chance) ===")
_ = label_shuffle_control(tr_idx, val_idx, test_idx, graphs_fold, y_tensor_full,
                          repeats=1, fold_name="real_fold0")
