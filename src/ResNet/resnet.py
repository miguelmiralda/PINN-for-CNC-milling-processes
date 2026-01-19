# ============================================================
#  MULTI-HEAD TOOL WEAR TRAINING (USING PRECOMPUTED EMBEDDINGS)
#  Model A: 2-head (flank_wear, adhesion) on TRAIN_SETS
#  Model B: 1-head (flank_wear+adhesion) on FWAD_SETS (separate experiment)
#
#  Dense pairing: all-vs-all (i<j) within the SAME wear type
#
#  Inputs expected per set:
#    data/processed/set{sid}/merged.csv
#    data/processed/set{sid}/image_embeddings.npz
#
#  image_embeddings.npz must contain:
#    embeddings: (N, 2048) float
#    image_id:   (N,)      stringable
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Config
# ----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Model A split (MATWI paper split)
TRAIN_SETS = [1, 2, 5, 7, 8, 10, 11]
VAL_SETS   = [3, 6, 12]
TEST_SETS  = [4, 9, 13, 14, 15]

# Model B split (sets containing flank_wear+adhesion) — separate experiment
FWAD_TRAIN_SETS = [3, 13, 16]
FWAD_VAL_SETS   = [12]
FWAD_TEST_SETS  = [17]

# Pairing logic: dense all-vs-all sorted by time within type
PAIR_BY_TIME_COL = "anchor_time"

# Cap for training (paper-style): exclude near/consecutive pairs.
# Keep (i,i) always, and keep (i,j) only if |j-i| > K after sorting by time.
# Set to None to disable.
MAX_INDEX_GAP_K = None

# Columns in merged.csv
WEAR_COL = "wear"
TYPE_COL = "type"
IMAGE_ID_COL = "image_id"

# Training hyperparams
BATCH_SIZE = 16
EPOCHS = 20
LR = 5e-4
EMBED_IN_DIM = 2048
HEAD_EMBED_DIM = 32

# Wear types (must match values in merged.csv "type" column)
TYPE_FW = "flank_wear"
TYPE_AD = "adhesion"
TYPE_FWAD = "flank_wear+adhesion"


# ----------------------------
# Utilities: IO
# ----------------------------

def load_embeddings_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)

    if "embeddings" not in data or "image_id" not in data:
        raise KeyError(
            f"{npz_path} must contain keys: 'embeddings' and 'image_id' "
            f"(found: {list(data.keys())})"
        )

    out = {
        "embeddings": data["embeddings"].astype(np.float32),  # (N, 2048)
        "image_id": data["image_id"].astype(str),            # (N,)
    }

    if "image_name" in data:
        out["image_name"] = data["image_name"].astype(str)

    return out


def build_imageid_to_embedding(emb_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    image_ids = emb_data["image_id"]
    emb = emb_data["embeddings"]

    out: Dict[str, np.ndarray] = {}
    for i in range(len(image_ids)):
        vec = emb[i]
        if np.isnan(vec).all():
            continue
        out[str(image_ids[i])] = vec
    return out


def build_imageid_to_imagename(emb_data: Dict[str, np.ndarray]) -> Dict[str, str]:
    if "image_name" not in emb_data:
        return {}
    ids = emb_data["image_id"]
    names = emb_data["image_name"]
    return {str(ids[i]): str(names[i]) for i in range(min(len(ids), len(names)))}


def load_merged_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if PAIR_BY_TIME_COL in df.columns:
        df[PAIR_BY_TIME_COL] = pd.to_datetime(df[PAIR_BY_TIME_COL], errors="coerce")

    for col in (IMAGE_ID_COL, WEAR_COL, TYPE_COL):
        if col not in df.columns:
            raise KeyError(f"{path} missing required column '{col}'")

    df[TYPE_COL] = df[TYPE_COL].astype(str).str.strip()
    df[IMAGE_ID_COL] = df[IMAGE_ID_COL].astype(str).str.strip()

    return df


# ----------------------------
# Pair building (dense)
# ----------------------------

@dataclass(frozen=True)
class Pair:
    E_ref: np.ndarray   # (2048,)
    E_cur: np.ndarray   # (2048,)
    d_wear: float       # scalar >= 0
    head_idx: int       # which head to train (0/1), or 0 for single-head


def _dense_pairs_from_df_for_type(
        df: pd.DataFrame,
        id2emb: Dict[str, np.ndarray],
        wear_type: str,
        head_idx: int,
) -> List[Pair]:
    sub = df[df[TYPE_COL] == wear_type].copy()
    if sub.empty:
        return []

    sub = sub[sub[IMAGE_ID_COL].isin(id2emb.keys())].copy()
    if sub.empty:
        return []

    if PAIR_BY_TIME_COL in sub.columns:
        sub = sub.sort_values(PAIR_BY_TIME_COL)
    sub = sub.drop_duplicates(subset=[IMAGE_ID_COL], keep="first").reset_index(drop=True)

    def _to_float(x) -> Optional[float]:
        try:
            return float(x)
        except Exception:
            return None

    wears: List[float] = []
    embs: List[np.ndarray] = []

    for _, r in sub.iterrows():
        w = _to_float(r[WEAR_COL])
        if w is None or pd.isna(w):
            continue
        iid = str(r[IMAGE_ID_COL])
        if iid not in id2emb:
            continue
        wears.append(float(w))
        embs.append(id2emb[iid])

    n = len(wears)
    if n < 1:
        return []

    K = MAX_INDEX_GAP_K
    if K is not None and K < 0:
        return []

    pairs: List[Pair] = []

    for i in range(n):
        Ei = embs[i]
        wi = wears[i]

        if K is None:
            for j in range(n):
                Ej = embs[j]
                wj = wears[j]
                pairs.append(Pair(E_ref=Ei, E_cur=Ej, d_wear=abs(wj - wi), head_idx=head_idx))
        else:
            pairs.append(Pair(E_ref=Ei, E_cur=Ei, d_wear=0.0, head_idx=head_idx))

            left_end = i - K
            for j in range(0, max(0, left_end)):
                Ej = embs[j]
                wj = wears[j]
                pairs.append(Pair(E_ref=Ei, E_cur=Ej, d_wear=abs(wj - wi), head_idx=head_idx))

            right_start = i + K + 1
            for j in range(min(n, right_start), n):
                Ej = embs[j]
                wj = wears[j]
                pairs.append(Pair(E_ref=Ei, E_cur=Ej, d_wear=abs(wj - wi), head_idx=head_idx))

    return pairs


def _reference_pairs_from_df_for_type(
        df: pd.DataFrame,
        id2emb: Dict[str, np.ndarray],
        wear_type: str,
        head_idx: int,
) -> List[Pair]:
    sub = df[df[TYPE_COL] == wear_type].copy()
    if sub.empty:
        return []

    sub = sub[sub[IMAGE_ID_COL].isin(id2emb.keys())].copy()
    if sub.empty:
        return []

    if PAIR_BY_TIME_COL in sub.columns:
        sub = sub.sort_values(PAIR_BY_TIME_COL)
    sub = sub.drop_duplicates(subset=[IMAGE_ID_COL], keep="first").reset_index(drop=True)

    def _to_float(x) -> Optional[float]:
        try:
            return float(x)
        except Exception:
            return None

    wears: List[float] = []
    embs: List[np.ndarray] = []

    for _, r in sub.iterrows():
        w = _to_float(r[WEAR_COL])
        if w is None or pd.isna(w):
            continue
        iid = str(r[IMAGE_ID_COL])
        if iid not in id2emb:
            continue
        wears.append(float(w))
        embs.append(id2emb[iid])

    n = len(wears)
    if n < 1:
        return []

    Eref = embs[0]
    wref = wears[0]

    pairs: List[Pair] = []
    for j in range(n):
        pairs.append(Pair(E_ref=Eref, E_cur=embs[j], d_wear=abs(wears[j] - wref), head_idx=head_idx))

    return pairs


def build_pairs_for_set_multitype(
        set_id: int,
        type_to_head: Dict[str, int],
        *,
        mode: str,
) -> List[Pair]:
    set_dir = PROCESSED_DIR / f"set{set_id}"
    merged_path = set_dir / "merged.csv"
    emb_path = set_dir / "image_embeddings.npz"

    if not merged_path.is_file():
        raise FileNotFoundError(f"Missing {merged_path}")
    if not emb_path.is_file():
        raise FileNotFoundError(f"Missing {emb_path}")

    df = load_merged_csv(merged_path)
    emb_data = load_embeddings_npz(emb_path)
    id2emb = build_imageid_to_embedding(emb_data)

    pairs: List[Pair] = []
    rows_used = 0

    for wear_type, head_idx in type_to_head.items():
        sub = df[df[TYPE_COL] == wear_type]
        rows_used += len(sub)
        if mode == "train":
            pairs.extend(_dense_pairs_from_df_for_type(df, id2emb, wear_type, head_idx))
        elif mode == "ref":
            pairs.extend(_reference_pairs_from_df_for_type(df, id2emb, wear_type, head_idx))
        else:
            raise ValueError(f"Unknown pairing mode: {mode}")

    print(f"[Set{set_id}] rows_used={rows_used} pairs_built={len(pairs)} types={list(type_to_head.keys())}")
    return pairs


def build_pairs_over_sets(
        set_ids: List[int],
        type_to_head: Dict[str, int],
        *,
        mode: str,
) -> List[Pair]:
    all_pairs: List[Pair] = []
    for sid in set_ids:
        all_pairs.extend(build_pairs_for_set_multitype(sid, type_to_head, mode=mode))
    print(f"Total pairs: {len(all_pairs)}")
    return all_pairs


# ----------------------------
# Dataset / Dataloader
# ----------------------------

class WearPairDataset(Dataset):
    def __init__(self, pairs: List[Pair]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        p = self.pairs[idx]
        return (
            torch.tensor(p.E_ref, dtype=torch.float32),
            torch.tensor(p.E_cur, dtype=torch.float32),
            torch.tensor([p.d_wear], dtype=torch.float32),   # (1,)
            torch.tensor(p.head_idx, dtype=torch.long),
        )


# ----------------------------
# Model: Heads only (uses precomputed 2048-D embeddings)
# ----------------------------

class WearNetHead(nn.Module):
    def __init__(self, in_dim: int = EMBED_IN_DIM, embed_dim: int = HEAD_EMBED_DIM):
        super().__init__()
        self.embedding = nn.Linear(in_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


class WearDistanceLoss(nn.Module):
    def forward(self, Z_ref: torch.Tensor, Z_cur: torch.Tensor, d_wear: torch.Tensor) -> torch.Tensor:
        d_embed = torch.norm(Z_cur - Z_ref, dim=1, keepdim=True)  # (B,1)
        return ((d_embed - d_wear) ** 2).mean()


# ----------------------------
# Eval
# ----------------------------

@torch.no_grad()
def evaluate_heads_on_pairs(
        heads: nn.ModuleList,
        pairs: List[Pair],
        num_heads: int,
        batch_size: int = 256,
) -> Dict[str, Dict[str, float]]:
    device = next(heads[0].parameters()).device
    ds = WearPairDataset(pairs)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    sum_se = [0.0 for _ in range(num_heads)]
    sum_ae = [0.0 for _ in range(num_heads)]
    cnt    = [0   for _ in range(num_heads)]

    for E_ref, E_cur, d_wear, head_idx in dl:
        E_ref = E_ref.to(device)
        E_cur = E_cur.to(device)
        d_wear = d_wear.to(device)      # (B,1)
        head_idx = head_idx.to(device)  # (B,)

        for h in range(num_heads):
            mask = (head_idx == h)
            m = int(mask.sum().item())
            if m == 0:
                continue

            Er = E_ref[mask]
            Ec = E_cur[mask]
            dw = d_wear[mask]

            Zr = heads[h](Er)
            Zc = heads[h](Ec)

            pred = torch.norm(Zc - Zr, dim=1, keepdim=True)  # (m,1)
            err = pred - dw

            sum_se[h] += float((err ** 2).sum().item())
            sum_ae[h] += float(err.abs().sum().item())
            cnt[h] += m

    out: Dict[str, Dict[str, float]] = {}
    for h in range(num_heads):
        if cnt[h] == 0:
            out[f"H{h}"] = {"mse": float("nan"), "mae": float("nan"), "n": 0}
        else:
            out[f"H{h}"] = {
                "mse": sum_se[h] / cnt[h],
                "mae": sum_ae[h] / cnt[h],
                "n": float(cnt[h]),
            }
    return out


def load_heads_from_ckpt(ckpt_path: Path) -> Tuple[nn.ModuleList, Dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)

    num_heads = int(ckpt["num_heads"])
    in_dim = int(ckpt["in_dim"])
    embed_dim = int(ckpt["embed_dim"])

    heads = nn.ModuleList([WearNetHead(in_dim=in_dim, embed_dim=embed_dim).to(device) for _ in range(num_heads)])
    for h in range(num_heads):
        heads[h].load_state_dict(ckpt["state_dicts"][h])
    return heads, ckpt


# ----------------------------
# Export head embeddings to ONE CSV (both models, no new columns)
# ----------------------------

@torch.no_grad()
def export_head_embeddings_one_csv(
        *,
        out_path: Path,
        exports: List[Tuple[Path, List[int], Dict[str, int]]],
) -> None:
    """
    Writes ONE CSV containing everything you already export:
      set_id, image_id, image_name, type, head_idx, z0..z{D-1}

    No extra columns are added.
    Exports is a list of:
      (ckpt_path, set_ids, type_to_head)
    """

    if len(exports) == 0:
        print("[Export] No exports provided, skipping.")
        return

    # Ensure all checkpoints have the same embed_dim so the CSV columns match
    embed_dims = set()
    for ckpt_path, _, _ in exports:
        ck = torch.load(ckpt_path, map_location="cpu")
        embed_dims.add(int(ck["embed_dim"]))
    if len(embed_dims) != 1:
        raise RuntimeError(
            f"[Export] Checkpoints have different embed_dim values {sorted(embed_dims)}. "
            "To keep the CSV format unchanged (no new columns), embed_dim must match."
        )
    embed_dim = int(next(iter(embed_dims)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = False
    total_rows = 0

    for ckpt_path, set_ids, type_to_head in exports:
        heads, ckpt = load_heads_from_ckpt(ckpt_path)
        num_heads = int(ckpt["num_heads"])
        device = next(heads[0].parameters()).device

        for sid in set_ids:
            set_dir = PROCESSED_DIR / f"set{sid}"
            merged_path = set_dir / "merged.csv"
            emb_path = set_dir / "image_embeddings.npz"

            if not merged_path.is_file() or not emb_path.is_file():
                print(f"[Export] Set{sid}: missing merged.csv or image_embeddings.npz, skipping")
                continue

            df = load_merged_csv(merged_path)
            emb_data = load_embeddings_npz(emb_path)
            id2emb = build_imageid_to_embedding(emb_data)
            id2name = build_imageid_to_imagename(emb_data)

            df = df[df[TYPE_COL].isin(type_to_head.keys())].copy()
            if df.empty:
                continue

            df = df[df[IMAGE_ID_COL].isin(id2emb.keys())].copy()
            if df.empty:
                continue

            # Stable/time order
            if PAIR_BY_TIME_COL in df.columns:
                df = df.sort_values(PAIR_BY_TIME_COL)
            df = df.drop_duplicates(subset=[IMAGE_ID_COL], keep="first").reset_index(drop=True)

            df["head_idx"] = df[TYPE_COL].map(type_to_head).astype(int)

            rows_out = []

            for h in range(num_heads):
                sub = df[df["head_idx"] == h]
                if sub.empty:
                    continue

                X = np.stack(
                    [id2emb[str(iid)] for iid in sub[IMAGE_ID_COL].astype(str).tolist()],
                    axis=0
                ).astype(np.float32)

                Xt = torch.tensor(X, dtype=torch.float32, device=device)

                bs = 512
                Z_all = []
                for i in range(0, Xt.shape[0], bs):
                    Z_all.append(heads[h](Xt[i:i + bs]).detach().cpu().numpy())
                Z = np.concatenate(Z_all, axis=0)  # (n, embed_dim)

                sub2 = sub.reset_index(drop=True)
                for k in range(len(sub2)):
                    r = sub2.iloc[k]
                    row = {
                        "set_id": sid,
                        "image_id": str(r[IMAGE_ID_COL]),
                        "image_name": id2name.get(str(r[IMAGE_ID_COL]), ""),
                        "wear": float(r[WEAR_COL]),
                        "type": str(r[TYPE_COL]),
                        "head_idx": int(h),
                    }
                    for d in range(embed_dim):
                        row[f"z{d}"] = float(Z[k, d])

                    rows_out.append(row)

            if not rows_out:
                continue

            out_df = pd.DataFrame(rows_out)
            out_df.to_csv(
                out_path,
                index=False,
                mode=("a" if wrote_header else "w"),
                header=(not wrote_header),
            )
            wrote_header = True
            total_rows += len(rows_out)
            print(f"[Export] Appended Set{sid}: {len(rows_out)} rows")

    print(f"[Export] Saved ONE combined file: {out_path} (rows={total_rows})")


# ----------------------------
# Training
# ----------------------------

def train_multhead(
        *,
        exp_name: str,
        pairs: List[Pair],
        num_heads: int,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        lr: float = LR,
        save_dir: Path = MODELS_DIR,
) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n==============================")
    print(f"Experiment: {exp_name}")
    print("Device:", device)
    print("Heads:", num_heads)
    print("==============================")

    if len(pairs) == 0:
        raise RuntimeError(f"[{exp_name}] No pairs built. Check types present, merged.csv, and embeddings.npz.")

    ds = WearPairDataset(pairs)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    heads = nn.ModuleList([WearNetHead().to(device) for _ in range(num_heads)])
    opts = [torch.optim.Adam(head.parameters(), lr=lr) for head in heads]
    criterion = WearDistanceLoss()

    for epoch in range(1, epochs + 1):
        heads.train()
        total_loss = [0.0 for _ in range(num_heads)]
        total_batches = [0 for _ in range(num_heads)]

        for E_ref, E_cur, d_wear, head_idx in dl:
            E_ref = E_ref.to(device)
            E_cur = E_cur.to(device)
            d_wear = d_wear.to(device)      # (B,1)
            head_idx = head_idx.to(device)  # (B,)

            for h in range(num_heads):
                mask = (head_idx == h)
                if mask.sum().item() == 0:
                    continue

                Er = E_ref[mask]
                Ec = E_cur[mask]
                dw = d_wear[mask]

                Zr = heads[h](Er)
                Zc = heads[h](Ec)

                loss = criterion(Zr, Zc, dw)

                opts[h].zero_grad()
                loss.backward()
                opts[h].step()

                total_loss[h] += loss.item()
                total_batches[h] += 1

        parts = []
        for h in range(num_heads):
            if total_batches[h] == 0:
                parts.append(f"H{h}: n/a")
            else:
                parts.append(f"H{h}: {total_loss[h]/total_batches[h]:.4f}")
        print(f"Epoch {epoch:03d} | " + " | ".join(parts))

    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "exp_name": exp_name,
        "in_dim": EMBED_IN_DIM,
        "embed_dim": HEAD_EMBED_DIM,
        "num_heads": num_heads,
        "state_dicts": [h.state_dict() for h in heads],
    }
    out_path = save_dir / f"{exp_name}.pkl"
    torch.save(ckpt, out_path)
    print("Saved model to:", out_path)
    return out_path


# ----------------------------
# Main: run both experiments + eval + export
# ----------------------------

def _print_metrics(tag: str, metrics: Dict[str, Dict[str, float]]) -> None:
    parts = []
    for hk, m in metrics.items():
        parts.append(f"{hk}: mse={m['mse']:.6f} mae={m['mae']:.6f} n={int(m['n'])}")
    print(f"[{tag}] " + " | ".join(parts))


def main():
    # -------------------------
    # Model A: 2-head (FW + AD)
    # -------------------------
    type_to_head_A = {TYPE_FW: 0, TYPE_AD: 1}

    pairs_A_train = build_pairs_over_sets(TRAIN_SETS, type_to_head_A, mode="train")
    pairs_A_val   = build_pairs_over_sets(VAL_SETS,   type_to_head_A, mode="ref")
    pairs_A_test  = build_pairs_over_sets(TEST_SETS,  type_to_head_A, mode="ref")

    ckpt_A = train_multhead(exp_name="wear_heads_FW_AD", pairs=pairs_A_train, num_heads=2)

    heads_A, _ = load_heads_from_ckpt(ckpt_A)
    _print_metrics("ModelA VAL",  evaluate_heads_on_pairs(heads_A, pairs_A_val,  num_heads=2))
    _print_metrics("ModelA TEST", evaluate_heads_on_pairs(heads_A, pairs_A_test, num_heads=2))

    # -------------------------
    # Model B: 1-head (FW+AD) — separate experiment
    # -------------------------
    type_to_head_B = {TYPE_FWAD: 0}

    pairs_B_train = build_pairs_over_sets(FWAD_TRAIN_SETS, type_to_head_B, mode="train")
    pairs_B_val   = build_pairs_over_sets(FWAD_VAL_SETS,   type_to_head_B, mode="ref")
    pairs_B_test  = build_pairs_over_sets(FWAD_TEST_SETS,  type_to_head_B, mode="ref")

    ckpt_B = train_multhead(exp_name="wear_head_FWAD", pairs=pairs_B_train, num_heads=1)

    heads_B, _ = load_heads_from_ckpt(ckpt_B)
    _print_metrics("ModelB VAL",  evaluate_heads_on_pairs(heads_B, pairs_B_val,  num_heads=1))
    _print_metrics("ModelB TEST", evaluate_heads_on_pairs(heads_B, pairs_B_test, num_heads=1))

    # -------------------------
    # ONE combined export (both models) into ONE CSV under processed/
    # -------------------------
    out_csv = PROCESSED_DIR / "head_embeddings_all_sets.csv"

    export_head_embeddings_one_csv(
        out_path=out_csv,
        exports=[
            (ckpt_A, (TRAIN_SETS + VAL_SETS + TEST_SETS), type_to_head_A),
            (ckpt_B, (FWAD_TRAIN_SETS + FWAD_VAL_SETS + FWAD_TEST_SETS), type_to_head_B),
        ],
    )


if __name__ == "__main__":
    main()