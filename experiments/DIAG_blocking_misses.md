# Blocking-miss diagnostic (M2 frozen at 0.94603)

Target: the 53,846 validation true pairs missing from C's candidates (of
691,564). Each method is an independent top-10 retrieval over the full train
S2+S3 pool within country, for the 200k validation S1. Nothing was fed to the
matcher.
Scripts: `experiments/scripts/blocking_diag.py`, `experiments/scripts/blocking_diag_expand.py`.

The misses break down as: Indic target 13,773 · India 29,745 · empty-address
target 10,501 · common-name S1 24,078 · Latin name-typo 19,242 · Latin other 20,831.

| method | new true beyond C | share of misses | blocking recall (C → 0.9221) | new cands / S1 | precision of new cands | runtime / peak |
|---|---|---|---|---|---|---|
| 1. Indic→Latin transliteration (MIT `indic_transliteration`, ITRANS + phonetic fold, char 3-grams over 753k Indic-name targets) | 409 | 0.8% | 0.9227 | 3.95 | 0.05% | 81 s / 3.1 GB |
| 2. char 3-gram name retrieval (all targets) | 1,422 | 2.6% | 0.9242 | 5.63 | 0.13% | 291 s / 5.4 GB |
| 3. char 3-gram address retrieval | 4,627 | 8.6% | 0.9288 | 4.80 | 0.48% | 522 s / 7.4 GB |
| union of 1–3 | 6,292 | 11.7% | 0.9312 | ~14 | — | — |
| **4. neighbour expansion** (top-5 hybrid name+address neighbours of each M2-predicted target) | **18,261** | **33.9%** | **0.9485** | **1.66** | **5.5%** | 410 s / 11.6 GB |

Overlap with C (share of each method's true pairs that C already had):
translit 76%, char_name 99.4%, char_addr 99.0%, expansion 97.2%.
Overlap among the misses recovered: translit∩char_name 4, translit∩char_addr 11,
char_name∩char_addr 151; expansion∩(translit, char_name, char_addr) = 102, 706, 2,547.

Recovery by category (recovered / missed):
| category | translit | char_name | char_addr | expansion |
|---|---|---|---|---|
| Indic target | 409/13,773 | 4 | 694 | **5,410** |
| India | 409/29,745 | 627 | 1,386 | **9,469** |
| empty-address target | 39/10,501 | 276 | 0 | 202 |
| common-name S1 | 292/24,078 | 27 | 1,381 | **6,008** |
| Latin name-typo | 0/19,242 | 780 | 2,177 | **7,208** |
| Latin other | 0/20,831 | 638 | 1,756 | **5,643** |

Upper bound for expansion: 92.3% of missed pairs belong to an S1 with at least one
M2 prediction; 36,274 missed targets are ≥ 80 similar (27,487 ≥ 90) to an
already-predicted target of the same S1.

Conclusions:
- Query-side methods (1–3) are limited by the top-k budget. For common and typo'd
  names, the true target competes with many similar records elsewhere.
  Transliteration also recovers little: the Indic records the matcher misses are
  mostly ones whose address or name differs anyway.
- Expansion works because S2/S3 carry several near-duplicate records per
  business: records similar to an already-matched record are likely the same
  entity. It is 10–100× more precise per added candidate.
- Empty-address targets (10,501) are not recoverable by any method tested.
