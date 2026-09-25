import hashlib, os, numpy as np, pandas as pd
from business_entity_resolution.blocking import Blocker
from business_entity_resolution.config import BlockingConfig, work_dir
from business_entity_resolution.pipeline import key_freqs, prepared
from business_entity_resolution.stack import StackModels, score_batch
if __name__ == "__main__":
    wd = work_dir()
    s1, tg = prepared("train")
    s1f, tf = key_freqs(s1, tg)
    rows = np.flatnonzero(s1["entity_id"].map(lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % 100 == 7).to_numpy())
    print("pilot rows", len(rows))
    M = StackModels.load(wd)
    e, keep = score_batch(Blocker(tg, BlockingConfig(hybrid=True)), s1, tg, s1f, tf, rows, M, stage3=True)
    out = pd.DataFrame({"s1_id": s1["entity_id"].to_numpy()[e["s1_row"].to_numpy()],
                        "t_id": tg["entity_id"].to_numpy()[e["t_row"].to_numpy()],
                        "s2": e["s2"].to_numpy(), "cos_name": e["cos_name"].to_numpy(), "cos_addr": e["cos_addr"].to_numpy()})
    out.to_parquet(wd / "pilot_ours.parquet", index=False)
    print("saved", len(out))
