# Notice

## Scope of the MIT licence

The [MIT licence](LICENSE) covers the source code in `src/`, `vectorrag/`, `analysis/`, `scripts/`, `tests/` and `data/vignettes/*.py`. The data are covered by the terms below.

## ESC guideline text

The rule tables in `data/rule_tables/` and the files derived from them (`data/rule_index/`, `data/vector_index/row_catalog.jsonl`, `data/reference/`, `data/vignettes/output/`, `results/`, `vectorrag/examples/`) reproduce recommendation text, class of recommendation and level of evidence from

> Vrints C, Andreotti F, Koskinas KC, Rossello X, Adamo M, Ainslie J, et al. 2024 ESC Guidelines for the management of chronic coronary syndromes. Eur Heart J. 2024;45(36):3415-537. doi:10.1093/eurheartj/ehae177

This text is the property of the European Society of Cardiology (ESC). It is reproduced here only for non-commercial research, so that the evaluation can be reproduced, and it is not covered by the licences of this repository. The guideline PDF is not included. For any clinical purpose consult the original guideline.

## SNOMED CT

The rule tables, the annotation index (`data/rule_index/`) and the vignette provenance (`data/vignettes/output/snomed_synonym_provenance.json`) contain SNOMED CT concept identifiers and terms. SNOMED CT® is a registered trademark of SNOMED International, and SNOMED CT content is © SNOMED International. Using SNOMED CT requires a licence from SNOMED International or the National Release Centre of a member country. No SNOMED CT release files are included; `src/cardio_graph_core/snomedct/` only reads a release that you load yourself.

## Data created by the authors

The rule encoding (conditions, logic groups, operators, thresholds and actions of each rule), the synthetic vignettes, the reference standard and the predictions in `results/` were created by the authors and are released under the [Creative Commons Attribution 4.0 International licence (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/), without the ESC text and SNOMED CT content named above. The vignettes are synthetic and contain no patient data.

LLM outputs in `results/` were produced with qwen3:32b (Ollama) and gpt-5.6-luna (OpenAI API).

## No medical device

This is research software. It is not a medical device and must not be used for decisions about individual patients.
