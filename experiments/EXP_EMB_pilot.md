# Embedding retrieval pilot (multilingual-e5-small, MIT)

Run on Modal (T4) with `experiments/modal/emb_pilot.py`. The sample is 21,948 train S1
(md5(id) % 100 == 7; 13,168 US, 8,780 India) against all 10.3M train targets of the same
country. Text: `query: name | address`, fp16, max 64 tokens. Speed: ~6k texts/s.
Our candidates for the same S1: `experiments/scripts/pilot_ours.py`, the production stack
(C blocking + neighbour expansion, stage-3 path).

## Share of true pairs found (76,027 true pairs)
| set | added cands/S1 | recall | recovered of 3,929 missed | precision of the new cands |
|---|---|---|---|---|
| ours (21.55 cands/S1) | – | 0.9483 | – | – |
| ∪ emb top-5 | +1.08 | 0.9686 | 1,540 | 0.065 |
| ∪ emb top-10 | +5.00 | 0.9773 | 2,200 | 0.020 |
| ∪ emb top-20 | +14.19 | 0.9810 | 2,484 | 0.008 |
| ∪ emb top-30 | +23.74 | 0.9830 | 2,633 | 0.005 |
| ∪ emb top-50 | +43.20 | 0.9848 | 2,771 | 0.003 |

By country at top-20: US 0.9630 → 0.9886; India 0.9264 → 0.9696.
Embedding alone: recall@10 0.9456, @20 0.9581, @50 0.9685.

Of our existing candidates, 31.5% appear in the emb top-50, including 98.3% of the true
ones. Among those covered pairs, emb cos AUC is 0.894 (s2: 0.998). So its value is
mainly recall, and a secondary feature for the new candidates.

Recovered pairs look like this: character typos in the name ("Robolihs", "Mendenha1l",
"Teechnial"), a website domain as the name, Hindi/Telugu script names, and dropped or
changed digits in the address.

Next: a full run (`emb_full.py`) over all train + test S1 at K=30, then add the new
candidates to the pipeline and retrain.
