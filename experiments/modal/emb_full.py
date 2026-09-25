# Embedding retrieval FULL RUN (Modal Notebook, 1x GPU). Same model and text as emb_pilot.py.
#
# Paste each "# %%" block into its own notebook cell and run them in order.
# For EVERY S1 of train and test, retrieves the top-K (K=30) nearest S2/S3 records
# of the same country with intfloat/multilingual-e5-small (MIT).
# No labels are used. Test data is embedded only (no fitting on it).
# Embeddings are cached in /data/emb_cache so a restarted kernel resumes.
#
# Output folder /data/emb_out (upload the whole folder to Google Drive):
#   {split}_knn.parquet   s1_row:int32, t_row:int32, cos:float32, rank:int8
#   {split}_s1_ids.parquet / {split}_t_ids.parquet   entity_id per row index
# where s1_row is the row in {split}_source1.tsv and t_row the row in
# concat(source2, source3), both in file order.

# %% [1] install
# !pip install -q gdown sentence-transformers pyarrow

# %% [2] dataset (skips the download if cell 2 of the pilot already fetched it)
import glob, os, shutil, subprocess, time
os.makedirs("/data", exist_ok=True)
if not glob.glob("/data/**/test_source1.tsv", recursive=True):
    subprocess.run(["gdown", "--folder", "--remaining-ok",
                    "https://drive.google.com/drive/folders/12flLP8tLEpSn0xUo1tK1CZiDBBYShTad",
                    "-O", "/data"], check=True)
DIR = {s: os.path.dirname(glob.glob(f"/data/**/{s}_source1.tsv", recursive=True)[0]) for s in ("train", "test")}
print(DIR)

# %% [3] model
import numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer
assert torch.cuda.is_available(), "No GPU attached"
print(torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
model = SentenceTransformer("intfloat/multilingual-e5-small", device="cuda")
model.max_seq_length = 64
model.half()
KW = dict(sep="\t", dtype=str, keep_default_na=False)
os.makedirs("/data/emb_cache", exist_ok=True)
os.makedirs("/data/emb_out", exist_ok=True)

def text(df):
    return ("query: " + df["business_name"].str.strip() + " | " + df["business_address"].str.strip()).tolist()

def emb_cached(name, df):
    """fp16 normalized embeddings, cached on disk as .npy."""
    f = f"/data/emb_cache/{name}.npy"
    if os.path.exists(f):
        return np.load(f)
    t0 = time.time()
    E = model.encode(text(df), batch_size=1024, normalize_embeddings=True, convert_to_numpy=True,
                     show_progress_bar=True).astype(np.float16)
    if shutil.disk_usage("/data").free > 3 * E.nbytes:   # cache only if the disk has room
        np.save(f, E)
    print(f"{name}: {len(df)} rows in {time.time() - t0:.0f}s")
    return E

# %% [4] embed + retrieve, per split and country (~1.5-2.5 h in total on a T4; resumable)
K, QB = 30, 512
for split in ("train", "test"):
    if os.path.exists(f"/data/emb_out/{split}_knn.parquet"):
        print(split, "already done"); continue
    s1 = pd.read_csv(f"{DIR[split]}/{split}_source1.tsv", **KW)
    tg = pd.concat([pd.read_csv(f"{DIR[split]}/{split}_source{k}.tsv", **KW) for k in (2, 3)], ignore_index=True)
    s1[["entity_id"]].to_parquet(f"/data/emb_out/{split}_s1_ids.parquet", index=False)
    tg[["entity_id"]].to_parquet(f"/data/emb_out/{split}_t_ids.parquet", index=False)
    parts = []
    for c in sorted(s1["country"].unique()):
        qi = np.flatnonzero(s1["country"].to_numpy() == c)
        ti = np.flatnonzero(tg["country"].to_numpy() == c)
        if len(ti) == 0:
            continue
        T = torch.from_numpy(emb_cached(f"{split}_t_{c}", tg.iloc[ti])).cuda()
        Q = emb_cached(f"{split}_s1_{c}", s1.iloc[qi])
        k = min(K, len(ti))
        t0 = time.time()
        for lo in range(0, len(qi), QB):
            q = torch.from_numpy(Q[lo:lo + QB]).cuda()
            v, i = (q @ T.T).topk(k, dim=1)
            v, i = v.float().cpu().numpy(), i.cpu().numpy()
            n = len(v)
            parts.append(pd.DataFrame({"s1_row": np.repeat(qi[lo:lo + n], k).astype(np.int32),
                                       "t_row": ti[i.ravel()].astype(np.int32),
                                       "cos": v.ravel().astype(np.float32),
                                       "rank": np.tile(np.arange(k, dtype=np.int8), n)}))
        print(f"{split} {c}: {len(qi)} S1 x {len(ti)} targets searched in {time.time() - t0:.0f}s")
        del T
        torch.cuda.empty_cache()
    R = pd.concat(parts, ignore_index=True)
    R.to_parquet(f"/data/emb_out/{split}_knn.parquet", index=False)
    print(split, "saved", len(R), "rows")

# %% [5] check the output
for f in sorted(os.listdir("/data/emb_out")):
    print(f, round(os.path.getsize(f"/data/emb_out/{f}") / 1e6, 1), "MB")
