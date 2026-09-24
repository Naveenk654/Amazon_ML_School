# Dataset Analysis Checklist

Before choosing a model, answer these from the actual files.

## Structure
- Number of rows in Source 1/2/3
- Columns and dtypes
- Missing values
- Duplicate IDs
- Duplicate records

## Names
- Empty names
- Average/median length
- Token counts
- Common legal suffixes
- Abbreviations
- Punctuation
- Typos
- Transliteration patterns
- Repeated business names

## Addresses
- Missing addresses
- Length distribution
- Numeric components
- PIN/postal-code patterns
- City/state/locality patterns if extractable
- Abbreviations
- Landmark patterns
- Reordered components

## Ground Truth
- Number of Source 1 entities
- Number with zero matches
- Number with one match
- Number with multiple matches
- S1→S2 matches
- S1→S3 matches
- Match multiplicity distribution

## Blocking
Measure:
- candidate count per S1
- candidate recall
- reduction ratio
- false candidate volume

## Matching
Measure:
- positive/negative pair balance
- feature distributions
- hard negatives
- threshold vs precision/recall/F0.5

Do not fill any value until it has been measured from the dataset.
