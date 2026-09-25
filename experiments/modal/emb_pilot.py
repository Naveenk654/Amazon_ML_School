# Embedding-retrieval PILOT (run in a Modal Notebook with a T4 GPU).
#
# Paste each "# %%" cell into its own notebook cell and run them in order.
# It embeds every TRAIN target (S2+S3) and ~1% of TRAIN S1 with
# intfloat/multilingual-e5-small (MIT), retrieves the top-50 nearest targets
# of the same country for each sampled S1, and writes emb_pilot.parquet
# (s1_id, t_id, cos, rank). Uses train data only; test and labels are not
# sent anywhere. Ground truth is used only to print a recall@k summary.

# %% [1] install
# !pip install -q gdown sentence-transformers pyarrow

# %% [2] download the dataset from Google Drive (same folder used before)
import glob, os, subprocess
os.makedirs("/data", exist_ok=True)
if not glob.glob("/data/**/train_source1.tsv", recursive=True):
    subprocess.run(["gdown", "--folder", "--remaining-ok",
                    "https://drive.google.com/drive/folders/12flLP8tLEpSn0xUo1tK1CZiDBBYShTad",
                    "-O", "/data"], check=True)
TRAIN = os.path.dirname(glob.glob("/data/**/train_source1.tsv", recursive=True)[0])
print("train dir:", TRAIN, os.listdir(TRAIN))

# %% [3] load train data and pick the pilot S1 sample (deterministic: md5(id) % 100 == 7)
import hashlib, time
import numpy as np, pandas as pd
KW = dict(sep="\t", dtype=str, keep_default_na=False)
s1 = pd.read_csv(f"{TRAIN}/train_source1.tsv", **KW)
tg = pd.concat([pd.read_csv(f"{TRAIN}/train_source{k}.tsv", **KW) for k in (2, 3)], ignore_index=True)
pick = s1["entity_id"].map(lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % 100 == 7)
q = s1[pick].reset_index(drop=True)
print("S1:", len(s1), "targets:", len(tg), "pilot S1:", len(q), q["country"].value_counts().to_dict())

def text(df):
    return ("query: " + df["business_name"].str.strip() + " | " + df["business_address"].str.strip()).tolist()

# %% [4] load model (fp16 on GPU)
import torch
from sentence_transformers import SentenceTransformer
assert torch.cuda.is_available(), "No GPU - set GPU = T4 in the compute profile"
model = SentenceTransformer("intfloat/multilingual-e5-small", device="cuda")
model.max_seq_length = 64
model.half()

def enc(texts, bs=1024):
    return model.encode(texts, batch_size=bs, convert_to_tensor=True, normalize_embeddings=True,
                        show_progress_bar=True).half()

# %% [5] embed targets per country, retrieve top-50 for the pilot S1 (takes ~30-60 min on a T4)
K = 50
out, stats = [], {}
for c in sorted(q["country"].unique()):
    t0 = time.time()
    tc = tg[tg["country"] == c].reset_index(drop=True)
    E = enc(text(tc))                                   # (n_targets, 384) fp16 on GPU
    t_emb = time.time() - t0
    qc = q[q["country"] == c].reset_index(drop=True)
    Q = enc(text(qc))
    for lo in range(0, len(qc), 2048):
        sims = Q[lo:lo + 2048] @ E.T                    # cosine (normalized)
        v, i = sims.float().topk(K, dim=1)
        v, i = v.cpu().numpy(), i.cpu().numpy()
        n = len(v)
        out.append(pd.DataFrame({"s1_id": np.repeat(qc["entity_id"].to_numpy()[lo:lo + n], K),
                                 "t_id": tc["entity_id"].to_numpy()[i.ravel()],
                                 "cos": v.ravel().astype(np.float32),
                                 "rank": np.tile(np.arange(K, dtype=np.int16), n)}))
    stats[c] = {"targets": len(tc), "embed_s": round(t_emb, 1), "per_s": round(len(tc) / t_emb)}
    print(c, stats[c])
    del E, Q
    torch.cuda.empty_cache()
R = pd.concat(out, ignore_index=True)
R.to_parquet("/data/emb_pilot.parquet", index=False)
print("saved /data/emb_pilot.parquet", len(R), "rows", round(os.path.getsize("/data/emb_pilot.parquet") / 1e6, 1), "MB")

# %% [6] quick recall summary (train labels, pilot S1 only)
gt = pd.read_csv(f"{TRAIN}/train_ground_truth.tsv", **KW)
gt = gt[gt["source1_entity_id"].isin(q["entity_id"])]
pairs = set((a, b) for a, l in zip(gt["source1_entity_id"], gt["matched_entity_ids"]) for b in l.split(",") if b)
R["y"] = [(a, b) in pairs for a, b in zip(R["s1_id"], R["t_id"])]
for k in (5, 10, 20, 50):
    print(f"recall@{k}: {R.loc[R['rank'] < k, 'y'].sum() / len(pairs):.4f}")
print("true pairs:", len(pairs), "| embed speed:", stats)
