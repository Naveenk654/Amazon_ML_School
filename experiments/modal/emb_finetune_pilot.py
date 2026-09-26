# Fine-tuned embedding PILOT (Modal Notebook, 1 GPU; L4/A10G preferred, T4 works).
#
# Paste each "# %%" block into its own notebook cell and run them in order.
#
# 1. Fine-tunes intfloat/multilingual-e5-base (MIT) on TRAIN pairs (S1 text -> true S2/S3
#    text) with in-batch negatives plus one mined HARD negative per pair: a record of the
#    same country whose name starts with the same words but is not a match (e.g. another
#    branch). The loss is contrastive (cached multiple-negatives ranking).
# 2. Evaluates exactly like emb_pilot.py on the same 21,948 pilot S1 (md5(id) % 100 == 7),
#    which are EXCLUDED from fine-tuning. Training uses only S1 with md5(id) % 2 == 0, and the
#    pilot S1 are all odd, so they never enter training.
#    Baseline to beat (e5-small, not fine-tuned): recall@5 0.8549, @10 0.9456, @20 0.9581, @50 0.9685.
# Output: /data/emb_pilot_ft.parquet (same format as emb_pilot.parquet) + the model in /data/ft_e5base.

# %% [1] install
# !pip install -q gdown "sentence-transformers>=3" datasets accelerate pyarrow

# %% [2] dataset (skips the download when /data already has it)
import glob, os, subprocess
os.makedirs("/data", exist_ok=True)
if not glob.glob("/data/**/train_source1.tsv", recursive=True):
    subprocess.run(["gdown", "--folder", "--remaining-ok",
                    "https://drive.google.com/drive/folders/12flLP8tLEpSn0xUo1tK1CZiDBBYShTad",
                    "-O", "/data"], check=True)
TRAIN = os.path.dirname(glob.glob("/data/**/train_source1.tsv", recursive=True)[0])

# %% [3] data + training pairs with hard negatives
import hashlib, re, time
import numpy as np, pandas as pd
KW = dict(sep="\t", dtype=str, keep_default_na=False)
s1 = pd.read_csv(f"{TRAIN}/train_source1.tsv", **KW)
tg = pd.concat([pd.read_csv(f"{TRAIN}/train_source{k}.tsv", **KW) for k in (2, 3)], ignore_index=True)
gt = pd.read_csv(f"{TRAIN}/train_ground_truth.tsv", **KW)
h = s1["entity_id"].map(lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % 100)
pilot = s1[h == 7].reset_index(drop=True)
train_ids = set(s1["entity_id"][h % 2 == 0])           # fine-tuning half; pilot (h == 7) is odd
print("S1", len(s1), "targets", len(tg), "pilot", len(pilot), "fine-tune S1", len(train_ids))

def text(df):
    return ("query: " + df["business_name"].str.strip() + " | " + df["business_address"].str.strip()).tolist()

s1_text = dict(zip(s1["entity_id"], text(s1)))
t_pos = pd.Series(np.arange(len(tg)), index=tg["entity_id"])
t_text = np.array(text(tg), dtype=object)
t_cty = tg["country"].to_numpy()
nkey = lambda s: " ".join(re.sub(r"[^0-9a-z ]", " ", s.lower()).split()[:2])
tkey = (tg["country"] + "|" + tg["business_name"].map(nkey)).to_numpy()
groups = pd.Series(np.arange(len(tg))).groupby(tkey).apply(lambda x: x.to_numpy()).to_dict()
by_cty = {c: np.flatnonzero(t_cty == c) for c in np.unique(t_cty)}
s1c = dict(zip(s1["entity_id"], s1["country"]))
s1n = dict(zip(s1["entity_id"], s1["business_name"]))

rng = np.random.default_rng(13)
N_PAIRS, PER_S1 = 400_000, 2
g = gt[gt["source1_entity_id"].isin(train_ids) & (gt["matched_entity_ids"] != "")]
g = g.sample(frac=1.0, random_state=13)
A, P, Nn = [], [], []
for sid, lst in zip(g["source1_entity_id"], g["matched_entity_ids"]):
    truth = lst.split(",")
    tset = set(t_pos.reindex(truth).dropna().astype(int))
    if not tset:
        continue
    cand = groups.get(s1c[sid] + "|" + nkey(s1n[sid]), by_cty[s1c[sid]])
    for tp in rng.choice(list(tset), size=min(PER_S1, len(tset)), replace=False):
        neg = None
        for _ in range(5):                               # hard negative: same country + name start
            j = int(cand[rng.integers(len(cand))])
            if j not in tset:
                neg = j; break
        if neg is None:                                  # fallback: random record, same country
            pool = by_cty[s1c[sid]]
            neg = int(pool[rng.integers(len(pool))])
        A.append(s1_text[sid]); P.append(t_text[tp]); Nn.append(t_text[neg])
    if len(A) >= N_PAIRS:
        break
print("training triplets:", len(A))
print("example:\n ", A[0], "\n +", P[0], "\n -", Nn[0])

# %% [4] fine-tune (about 20-40 min on a T4, less on L4/A10G)
import torch
from datasets import Dataset
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments, losses)
from sentence_transformers.training_args import BatchSamplers
assert torch.cuda.is_available(), "No GPU attached"
print(torch.cuda.get_device_name(0))
model = SentenceTransformer("intfloat/multilingual-e5-base", device="cuda")
model.max_seq_length = 64
ds = Dataset.from_dict({"anchor": A, "positive": P, "negative": Nn})
loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=64)
args = SentenceTransformerTrainingArguments(
    output_dir="/data/ft_tmp", num_train_epochs=1, per_device_train_batch_size=512,
    learning_rate=2e-5, warmup_ratio=0.1, fp16=True, batch_sampler=BatchSamplers.NO_DUPLICATES,
    logging_steps=50, save_strategy="no", report_to="none", seed=13)
t0 = time.time()
SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss).train()
model.save("/data/ft_e5base")
print(f"fine-tuned in {time.time() - t0:.0f}s, saved /data/ft_e5base")

# %% [5] evaluate like emb_pilot.py: embed all train targets, top-50 for the pilot S1 (about 40-90 min)
model.half()
K, out, stats = 50, [], {}
for c in sorted(pilot["country"].unique()):
    t0 = time.time()
    ti = by_cty[c]
    E = model.encode(list(t_text[ti]), batch_size=1024, convert_to_tensor=True,
                     normalize_embeddings=True, show_progress_bar=True).half()
    qc = pilot[pilot["country"] == c].reset_index(drop=True)
    Q = model.encode(text(qc), batch_size=1024, convert_to_tensor=True, normalize_embeddings=True).half()
    for lo in range(0, len(qc), 512):
        v, i = (Q[lo:lo + 512] @ E.T).float().topk(K, dim=1)
        v, i = v.cpu().numpy(), i.cpu().numpy()
        n = len(v)
        out.append(pd.DataFrame({"s1_id": np.repeat(qc["entity_id"].to_numpy()[lo:lo + n], K),
                                 "t_id": tg["entity_id"].to_numpy()[ti[i.ravel()]],
                                 "cos": v.ravel().astype(np.float32),
                                 "rank": np.tile(np.arange(K, dtype=np.int16), n)}))
    stats[c] = {"targets": len(ti), "embed_s": round(time.time() - t0), "per_s": round(len(ti) / (time.time() - t0))}
    print(c, stats[c])
    del E, Q
    torch.cuda.empty_cache()
R = pd.concat(out, ignore_index=True)
R.to_parquet("/data/emb_pilot_ft.parquet", index=False)

# %% [6] recall on the pilot S1 (compare: 0.8549 / 0.9456 / 0.9581 / 0.9685)
gp = gt[gt["source1_entity_id"].isin(set(pilot["entity_id"]))]
pairs = set((a, b) for a, l in zip(gp["source1_entity_id"], gp["matched_entity_ids"]) for b in l.split(",") if b)
R["y"] = [(a, b) in pairs for a, b in zip(R["s1_id"], R["t_id"])]
for k in (5, 10, 20, 50):
    print(f"recall@{k}: {R.loc[R['rank'] < k, 'y'].sum() / len(pairs):.4f}")
print("true pairs:", len(pairs), "| speed:", stats)
