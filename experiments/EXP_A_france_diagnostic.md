# Experiment A — France diagnostic (frozen M3, test predictions, no labels)

Scripts: `experiments/scripts/france_diag.py` (+ ad-hoc twin and recall-proxy checks recorded here).

## Measured France anomalies vs US / India (test)
| | France | India | US |
|---|---|---|---|
| S1 sharing name_key with another S1 | 52.9% | 57.1% | 42.8% |
| S1 sharing name_key **and city** | **23.7%** | 16.9% | 1.3% |
| S1 in a same-name + same-street "twin" group (train: ≤0.03%) | **1.5%** | 0.01% | 0.01% |
| competition vetoes (s2 ≥ 0.65 rejected by M3) per S1 | **0.126** | 0.035 | 0.012 |
| share of vetoes on twin S1 | 14% | 0.1% | 0.5% |
| predictions with a strong competitor | **3.2%** | 0.8% | 0.3% |
| predictions with a same-key competitor | **25.5%** | 16.8% | 13.7% |
| competition-driven accepts (s2 < 0.65, M3 ≥ 0.60) | 3.5% | 4.0% | 4.1% |
| predictions: near-identical name but no shared number | 8.9% | 4.5% | 14.5% |
| predictions: address numbers differ | 11.0% | 30.6% | 36.3% |
| targets predicted for > 1 S1 | 53 | 6 | 2 |
| predictions per S1 / empty rate | 3.30 / 6.0% | 3.26 / 7.4% | 3.55 / 5.6% |

- **Name templates.** France names are generated as `<city> <generic word>` or `<word> <word> <legal form>`.
  Examples: "bordeaux club" ×530, "nantes club" ×486, "lille ecole" ×294. There are only a few dozen
  cities, so name + city is not discriminative in France (unlike the US). Street + number
  must decide.
- **Twins / conflicts.** Examples: two "Lille Ecole" S1 at 7 and 14 Rue des Roses both claim a
  target "LILLE ECOLE | RUE DES ROSES" (no number). Two "Lille Ecole SARL" S1 at the same address
  (Maison des Associations, a shared building). At most one claim can be correct.
- **Address format.** S1 always ends with the region; targets use the region or the département
  (Nord, Gironde, Loire-Atlantique) or the city. 99.6% of France S1 have a house number, targets 93.3%.

## Label-free recall proxy
Targets sharing (country, name_key, a house number) with an S1 but absent from its candidates:
- val: India 3.17 per S1 (0.37% true), US 0.17 per S1 (0.93% true);
- test: France 0.66 per S1, India 3.38, US 0.10.

On val such records are almost never true matches, so there is no evidence of a France recall hole
of this kind.

## Conclusion
France differs structurally: template names embedding the city, a dense same-name population,
twins, and 10× the US veto rate. The cost of its errors cannot be measured without labels, and
no single label-free mechanism is shown to be a large, safely fixable error source. France's
risk is concentrated in the competition stage (vetoes and conflicts). That makes the
competition model's robustness (Experiment B) the relevant, measurable lever.
