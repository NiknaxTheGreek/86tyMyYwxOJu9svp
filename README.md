# Human Resources Candidate Ranking

**Candidate relevance ranking with interpretable NLP, duplicate-safe evaluation, and recruiter feedback**

GitHub repository: `NiknaxTheGreek/86tyMyYwxOJu9svp`

## 1. Executive Summary

This project builds a transparent ranking system for surfacing Human Resources candidates from a small LinkedIn-style dataset. The objective is **ranking rather than classification**: the practical question is not only whether a candidate is relevant, but whether the strongest candidates appear near the top of a recruiter’s shortlist.

The source data contains **104 candidate rows**, but exact repetition reduces the independent evidence to **53 unique profile groups**. Human relevance is graded on a **0–3 scale**: 0 = irrelevant, 1 = weak/adjacent, 2 = clearly HR-relevant, and 3 = strong/direct alignment with the aspiring/seeking HR search objective. Duplicate copies are retained for traceability but are not treated as additional human judgements.

The analysis evaluates **192 model configurations** under five-fold duplicate-safe grouped cross-validation. The final retained initial-ranking model is a **text-only Ridge scorer with alpha = 10**, achieving mean **NDCG@10 = 0.9596** with fold standard deviation **0.0325**. MAP@10 and MRR are both **1.000** for this retained model. Adding connection count lowers mean NDCG@10 to **0.9562**, so network size is not retained as a core relevance feature.

A matched BM25-versus-TF-IDF comparison shows only a small average difference in favor of BM25 (**+0.0118 NDCG@10**) with substantial uncertainty: **p = 0.46**, 95% CI **[-0.020, +0.043]**, and win/tie/loss **12/18/10**. The evidence therefore does **not** establish BM25 as superior.

After the initial ranking, recruiter feedback can refine the current search using Rocchio relevance feedback. The refined score keeps **65% of the original ranking** and uses **35% feedback**, allowing personalization without replacing the validated baseline.

## 2. Problem Definition and Objective

The system answers a search-ranking question: **given an HR-oriented recruiter query, in what order should the available candidate profiles be shown?** The two operational queries are:

- `aspiring human resources`
- `seeking human resources`

A ranking score is a **relative relevance score** used to order candidates. It is not a probability of being hired, a measure of employee quality, or a prediction of future job performance.

## 3. Data and Methodology

The original dataset contains candidate ID, job title, location, connection count, and an unused source `fit` field. The labelled dataset adds the project’s human relevance grade.

| Item | Project treatment |
|---|---:|
| Source rows | 104 |
| Unique duplicate-safe profile groups | 53 |
| Primary textual field | `job_title` |
| Human relevance scale | 0–3 |
| Binary threshold for MAP/MRR | grades 2–3 are relevant |
| Primary ranking metric | NDCG@10 |

### Relevance labelling

Each unique profile is assessed against the combined aspiring/seeking HR objective. The rubric is intentionally graded rather than binary:

| Grade | Meaning |
|---:|---|
| 0 | Irrelevant to the HR search objective |
| 1 | Weak, adjacent, or materially mismatched intent |
| 2 | Clearly HR-relevant but not the strongest/direct target |
| 3 | Strong/direct aspiring or seeking HR match |

The grading rules consider the meaning of the full title rather than keyword presence alone. HR-adjacent roles, established senior HR profiles, and generic student/seeking language can therefore receive different grades depending on intent alignment.

The independent profile-level grade distribution is **17 / 2 / 28 / 6** for grades 0 / 1 / 2 / 3. After duplicate propagation to all source rows, the distribution becomes **27 / 10 / 47 / 20**. In particular, six independent grade-3 judgements expand to twenty raw rows; they remain six independent judgements.

### Duplicate-safe evaluation

The 104 rows collapse to 53 unique groups using normalized job title, location, and parsed connection count. All copies of the same profile remain in the same cross-validation fold, preventing duplicate leakage between training and validation.

### Text representation and ranking

Text normalization standardizes casing, punctuation, and abbreviations such as `HR -> human resources` while preserving intent terms such as `aspiring` and `seeking`. The notebooks generate and compare interpretable lexical evidence including:

- word TF-IDF and character TF-IDF;
- BM25 retrieval scores;
- Jaccard similarity and query containment;
- HR-term and intent-term overlap;
- optional connection-count information.

Ridge regression combines these signals into a relative ranking score. The broader search covers **3 preprocessing variants × 4 representation families × 4 feature bundles × 4 Ridge penalties = 192 configurations**, including **64 Porter2-stemmed variants**.

The highest broad-screen NDCG@10 is about **0.9641**, but several configurations are separated by only a few thousandths on 53 independent profiles. The final retained family therefore prioritizes a compact, normalized, interpretable feature set rather than treating a tiny screening difference as a durable winner. Within that retained family, text-only Ridge with `alpha = 10` is best at **0.9596 mean NDCG@10**.

### Recruiter feedback

The first pass produces the general candidate order. A second pass can then refine that order when a recruiter explicitly stars or rejects candidates. Rocchio feedback shifts the search representation toward preferred language and away from rejected language. The refined score remains anchored to the original ranking, using a **65% original / 35% feedback** blend.

## 4. Results and Key Findings

| Finding | Result | Interpretation |
|---|---:|---|
| Effective sample size | 53 unique profiles | Raw duplicates must not be treated as independent evidence |
| Broad model search | 192 configurations | Preprocessing, representation, feature and regularization choices were compared systematically |
| Best broad-screen NDCG@10 | ~0.9641 | Useful sensitivity result, but too close to nearby alternatives to justify a strong winner claim |
| Retained initial ranker | Ridge, alpha = 10, text-only | Compact and interpretable final model |
| Retained NDCG@10 | 0.9596 ± 0.0325 | Strong graded ordering near the top of the shortlist under grouped CV |
| Retained MAP@10 / MRR | 1.000 / 1.000 | Relevant candidates appear early; these metrics saturate and are less discriminating here |
| Adding connection count | NDCG@10 falls to 0.9562 | Connection count is not retained as a relevance feature |
| BM25 − TF-IDF | +0.0118 | Small observed average difference |
| BM25 uncertainty | p = 0.46; CI [-0.020, +0.043] | No evidence of clear BM25 superiority |
| Feedback blend | 65% baseline / 35% feedback | Personalization changes ordering without replacing the validated baseline |

### What the ranking metrics mean

- **NDCG@10** asks whether the strongest human-graded candidates are placed near the top of the first ten results. Because it uses the full 0–3 grades, it is the primary metric.
- **MAP@10** asks whether candidates considered relevant (grades 2–3) are consistently placed early in the first ten results.
- **MRR** asks how quickly the first relevant candidate appears.

These are **offline ranking-quality measures**. They do not prove recruiter productivity, hiring quality, or future employee performance.

## 5. Interpretation and Limitations

The evidence supports a simple conclusion: a transparent lexical ranking system is appropriate for the current data scale, and explicit recruiter feedback is a sensible controlled extension. The main constraint is the amount and quality of independent labelled data rather than a lack of model complexity.

Important limitations are:

- only **53 unique labelled profiles** support development and evaluation;
- duplicated source rows are not independent observations;
- the relevance grades contain human judgement uncertainty and no formal inter-rater agreement estimate is available;
- the available candidate information is sparse and relies mainly on job-title text plus basic metadata;
- lexical methods can miss deeper semantic equivalence;
- repeated experimentation on a small labelled dataset can overfit the evaluation process itself;
- offline ranking metrics do not establish business impact;
- fairness is not established simply because protected attributes are absent, because text, location, and network-related variables may still act as proxies.

The highest-value next steps are to collect more **unique** labelled profiles, improve annotation consistency, create a prospective holdout set, and measure recruiter behavior in a controlled pilot before introducing more complex semantic models.

## 6. Conclusion

This project provides a complete candidate-ranking workflow: duplicate-safe data preparation, graded human relevance labels, interpretable lexical representations, systematic grouped model selection, a compact final ranking model, and controlled recruiter-feedback reranking. The notebooks are the reproducible analytical source of truth; the two PDF reports present the same evidence for technical and business audiences.

---

## Repository Contents

The repository is intentionally flat and contains **exactly 10 files**:

```text
86tyMyYwxOJu9svp/
├── .gitignore
├── 01_data_representation.ipynb
├── 02_grouped_cv_model_selection.ipynb
├── 03_final_ranking_stage2.ipynb
├── Business_Recommendations.pdf
├── README.md
├── Technical_Report.pdf
├── potential-talents-labelled.csv
├── potential-talents.csv
└── requirements.txt
```

### File guide

| File | Purpose |
|---|---|
| `01_data_representation.ipynb` | Data inspection, relevance labelling, duplicate handling, text normalization, TF-IDF/BM25 foundations, and descriptive figures/tables |
| `02_grouped_cv_model_selection.ipynb` | Ranking metrics, 192-configuration search, grouped CV, representation comparisons, Ridge tuning, connection-count test, and statistical comparison |
| `03_final_ranking_stage2.ipynb` | Final initial ranking, Rocchio feedback, refined ranking, and rank-shift analysis |
| `potential-talents.csv` | Original source dataset |
| `potential-talents-labelled.csv` | Source rows with the final human relevance grades |
| `Technical_Report.pdf` | Detailed methodology, experimental design, evidence, results, limitations, and interpretation |
| `Business_Recommendations.pdf` | Plain-language business implications, recommendations, rollout plan, governance, and metric explanations |
| `README.md` | Project summary, key findings, limitations, repository map, and reproduction instructions |
| `requirements.txt` | Pinned package versions used to execute the notebooks |
| `.gitignore` | Excludes local notebook checkpoints, caches, and environments |

### Reproduction

1. Install the pinned environment with `pip install -r requirements.txt`.
2. Launch Jupyter from the repository root.
3. Run the notebooks in numerical order: `01` → `02` → `03`.

The notebooks are fully executed in the repository with outputs embedded. All analytical tables, figures, model-selection results, final rankings, and feedback results used by the reports are reproducible from those notebooks. No helper modules, generated result folders, or external figure files are required for the submitted project.
