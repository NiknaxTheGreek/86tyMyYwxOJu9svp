# Human Resources Candidate Ranking

**Candidate relevance ranking with interpretable NLP, duplicate-safe evaluation, and recruiter feedback**

GitHub repository: `NiknaxTheGreek/86tyMyYwxOJu9svp`

## 1. Executive Summary

This project builds a transparent ranking system for surfacing Human Resources candidates from a small LinkedIn-style dataset. The objective is **ranking rather than classification**: the practical question is not only whether a candidate is relevant, but whether the strongest candidates appear near the top of a recruiter shortlist.

The source data contains **104 candidate rows**, which collapse to **53 unique profile groups** after duplicate-safe grouping. Human relevance is graded on a **0-3 scale**: 0 = irrelevant, 1 = weak/adjacent, 2 = clearly HR-relevant, and 3 = strong/direct alignment with the aspiring/seeking HR search objective. Duplicate copies are retained for traceability but are not counted as additional profile judgements.

The analysis evaluates **192 model configurations** with five-fold GroupKFold at the unique-profile level. All corpus-fitted retrieval statistics are learned inside each training fold: TF-IDF vocabulary/IDF values and BM25 corpus statistics do not use validation-profile text, and the search queries are transformed only after the training representation is fitted.

The selected Stage-1 model uses **normalized text, a hybrid BM25 + word-TF-IDF + character-TF-IDF representation, lexical overlap and HR/intent features, and Ridge alpha = 10**. It achieves mean **NDCG@10 = 0.9628** with standard deviation **0.0621** across the 10 fold-query observations. MAP@10 and MRR are both **1.000**. The best configuration that includes connection count reaches **0.9562 NDCG@10**, so network size is not retained in the final relevance model.

Among direct lexical baselines, character TF-IDF reaches **0.9535 NDCG@10**, BM25 **0.9527**, and word TF-IDF **0.9359**. A paired BM25-versus-word-TF-IDF comparison gives a mean difference of **+0.0168 NDCG@10**, **p = 0.278**, 95% CI **[-0.0161, +0.0498]**, with win/tie/loss **3/6/1** across the 10 matching fold-query observations. The evidence does not establish a clear BM25 advantage.

After the initial ranking, recruiter feedback can refine the current search using Rocchio relevance feedback. The demonstration keeps **65% of the Stage-1 score** and uses **35% feedback**. With source ID 3 starred, that profile moves to Stage-2 rank 1 while the general ranking remains strongly anchored to Stage 1.

## 2. Problem Definition and Objective

The system answers a search-ranking question: **given an HR-oriented recruiter query, in what order should the available candidate profiles be shown?** The two operational queries are:

- `aspiring human resources`
- `seeking human resources`

A ranking score is a **relative relevance score** used to order candidates. It is not a probability of being hired, a measure of employee quality, or a prediction of future job performance.

## 3. Data and Methodology

The original dataset contains candidate ID, job title, location, connection count, and an unused source `fit` field. The labelled dataset adds the project's human relevance grade.

| Item | Project treatment |
|---|---:|
| Source rows | 104 |
| Unique duplicate-safe profile groups | 53 |
| Primary textual field | `job_title` |
| Human relevance scale | 0-3 |
| Binary threshold for MAP/MRR | grades 2-3 are relevant |
| Primary ranking metric | NDCG@10 |

### Relevance labelling

Each unique profile is assessed against the combined aspiring/seeking HR objective.

| Grade | Meaning |
|---:|---|
| 0 | Irrelevant to the HR search objective |
| 1 | Weak, adjacent, or materially mismatched intent |
| 2 | Clearly HR-relevant but not the strongest/direct target |
| 3 | Strong/direct aspiring or seeking HR match |

The unique-profile grade distribution is **17 / 2 / 28 / 6** for grades 0 / 1 / 2 / 3. After propagation to all source rows, the distribution is **27 / 10 / 47 / 20**.

Notebook 01 also checks that the original five source fields are unchanged in the labelled file, all 104 source IDs are unique, all grades are complete and within 0-3, repeated profiles carry one consistent grade, and duplicate grouping yields exactly 53 profile groups.

### Duplicate-safe grouped evaluation

The 104 rows collapse to 53 unique groups using normalized job title, location, and parsed connection count. GroupKFold assigns whole profile groups to folds, so duplicate copies cannot appear on both sides of a train/validation split.

For every fold, text representations are fitted from the training profiles only:

- word TF-IDF: training titles determine vocabulary and IDF;
- character TF-IDF: training titles determine vocabulary and IDF;
- BM25: training titles determine document frequency and average document length;
- validation titles and the two search queries are scored or transformed using those training-fitted statistics.

### Text representation and ranking

Text normalization standardizes casing, punctuation, and abbreviations such as `HR -> human resources` while preserving intent terms such as `aspiring` and `seeking`.

The notebooks compare:

- word TF-IDF;
- character TF-IDF;
- BM25 retrieval scores;
- hybrid retrieval combinations;
- Jaccard similarity and query containment;
- HR-term and intent-term overlap;
- optional connection-count information.

The broad search covers **3 preprocessing variants x 4 representation families x 4 feature bundles x 4 Ridge penalties = 192 configurations**, including **64 Porter2-stemmed variants**.

The highest grouped-CV NDCG@10 is **0.9628**, produced by normalized hybrid retrieval plus overlap/intent features with Ridge `alpha = 10`. That configuration is therefore used for Stage 1. Connection count is excluded because the best connection-inclusive configuration is lower at **0.9562**.

### Recruiter feedback

Stage 2 refines the current search when a recruiter explicitly stars or rejects profiles. Rocchio feedback shifts the query representation toward preferred language and away from rejected language. The demonstration uses source ID 3 as the starred candidate, no rejected candidate, Rocchio parameters `alpha = 1.0`, `beta = 0.75`, `gamma = 0.15`, and a **65% Stage-1 / 35% feedback** score blend.

## 4. Results and Key Findings

| Finding | Result | Interpretation |
|---|---:|---|
| Effective profile sample | 53 unique profiles | Duplicate rows are grouped before model evaluation |
| Broad model search | 192 configurations | Preprocessing, retrieval, feature, and regularization choices are compared systematically |
| Selected Stage-1 model | normalized hybrid + overlap/intent, Ridge alpha = 10 | Best mean grouped-CV NDCG@10 |
| Selected NDCG@10 | 0.9628 +/- 0.0621 | Strong graded ordering near the top of the shortlist |
| Selected MAP@10 / MRR | 1.000 / 1.000 | Relevant profiles appear very early; these metrics saturate on this dataset |
| Best model with connections | 0.9562 NDCG@10 | Connection count does not improve the selected relevance model |
| Character TF-IDF baseline | 0.9535 NDCG@10 | Strongest direct lexical baseline |
| BM25 baseline | 0.9527 NDCG@10 | Similar to character TF-IDF |
| Word TF-IDF baseline | 0.9359 NDCG@10 | Lower than BM25 and character TF-IDF |
| BM25 - word TF-IDF | +0.0168; p = 0.278; CI [-0.0161, +0.0498] | No clear evidence of superiority |
| Stage-1 top profile | ID 99 - Seeking Human Resources Position | Highest full-data Stage-1 score |
| Stage-2 demonstration | ID 3 moves to rank 1 | Explicit feedback personalizes ordering |
| Feedback blend | 65% Stage 1 / 35% feedback | Personalization remains anchored to the base ranker |

### What the ranking metrics mean

- **NDCG@10** asks whether the strongest human-graded profiles are placed near the top of the first ten results. It uses the full 0-3 grades and is the primary metric.
- **MAP@10** asks whether profiles considered relevant (grades 2-3) are consistently placed early in the first ten results.
- **MRR** asks how quickly the first relevant profile appears.

These are offline ranking-quality measures. They do not prove recruiter productivity, hiring quality, or future employee performance.

## 5. Interpretation and Limitations

The evidence supports a transparent lexical ranking system at the current data scale, with explicit recruiter feedback as a controlled personalization layer. The main constraint is the size and richness of the labelled profile set rather than a lack of model complexity.

Important limitations are:

- only **53 unique labelled profiles** support development and evaluation;
- duplicated source rows are not separate profile observations;
- relevance grades contain human judgement uncertainty and no formal inter-rater agreement estimate is available;
- the candidate information is sparse and relies mainly on job-title text plus basic metadata;
- lexical methods can miss deeper semantic equivalence;
- repeated experimentation on a small labelled dataset can overfit model-selection decisions;
- offline ranking metrics do not establish business impact;
- fairness is not established simply because protected attributes are absent, because text, location, and network-related variables may still act as proxies.

The highest-value next steps are to collect more unique labelled profiles, improve annotation consistency, create a prospective holdout set, and measure recruiter behavior in a controlled pilot before introducing more complex semantic models.

## 6. Conclusion

This project provides a complete candidate-ranking workflow: source-data checks, duplicate-safe grouping, graded human relevance labels, interpretable lexical representations, fold-safe grouped model selection, a selected Stage-1 ranker, and controlled recruiter-feedback reranking. The notebooks are the analytical source of truth, and the two PDF reports present the same evidence for technical and business audiences.

---

## Repository Contents

The repository is intentionally flat and contains **exactly 10 files**:

```text
86tyMyYwxOJu9svp/
|-- .gitignore
|-- 01_data_representation.ipynb
|-- 02_grouped_cv_model_selection.ipynb
|-- 03_final_ranking_stage2.ipynb
|-- Business_Recommendations.pdf
|-- README.md
|-- Technical_Report.pdf
|-- potential-talents-labelled.csv
|-- potential-talents.csv
`-- requirements.txt
```

### File guide

| File | Purpose |
|---|---|
| `01_data_representation.ipynb` | Source-data checks, relevance labels, duplicate handling, text normalization, TF-IDF/BM25 foundations, and descriptive outputs |
| `02_grouped_cv_model_selection.ipynb` | Ranking metrics, fold-safe 192-configuration search, representation comparisons, Ridge tuning, connection-count comparison, and baseline statistics |
| `03_final_ranking_stage2.ipynb` | Selected Stage-1 ranking, Rocchio feedback, Stage-2 reranking, and rank-shift analysis |
| `potential-talents.csv` | Original source dataset |
| `potential-talents-labelled.csv` | Source rows with the final human relevance grades |
| `Technical_Report.pdf` | Detailed methodology, experiment design, results, limitations, and interpretation |
| `Business_Recommendations.pdf` | Business implications, recommendations, rollout plan, governance, and metric explanations |
| `README.md` | Project summary, results, limitations, repository map, and reproduction instructions |
| `requirements.txt` | Pinned package versions used to execute the notebooks |
| `.gitignore` | Excludes local notebook checkpoints, caches, and environments |

### Reproduction

1. Install the pinned environment with `pip install -r requirements.txt`.
2. Launch Jupyter from the repository root.
3. Run the notebooks in numerical order: `01` -> `02` -> `03`.

The notebooks are fully executed with outputs embedded. They read the two CSV files by bare filename from the same flat directory. No helper modules, parent-folder discovery, build directories, environment-variable routing, generated result folders, or external figure files are required.
