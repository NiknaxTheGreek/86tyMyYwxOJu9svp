# Human Resources Candidate Ranking

**Candidate Relevance Ranking with NLP and Interactive Feedback**

Repository: `Wa_BVVFTGqd_OVCYjY`

## 1. Executive Summary

This project develops a transparent two-stage ranking system for surfacing Human Resources candidates from a small LinkedIn-style dataset. The objective is **ranking**, not conventional classification: a recruiter needs the strongest matches to appear near the top of a shortlist, not merely a yes/no label for every profile.

The source dataset contains **104 candidate rows but only 53 unique profile groups** after duplicate handling. Candidate relevance is manually graded on a **0–3 scale** so that direct matches can be distinguished from weaker or adjacent matches. Duplicate copies are retained for traceability but are not treated as additional independent observations.

Stage 1 converts job-title text into interpretable lexical evidence using TF-IDF, BM25, word/character similarity, and overlap features, then combines those signals with a Ridge-based ranking score. The broader experiment evaluated **192 configurations** under five-fold duplicate-safe grouped cross-validation, including **64 stemming variants**. NDCG@10 is the primary ranking metric, with MAP and MRR as supporting measures. The selected compact Stage-1 model is a text-only Ridge model with `alpha = 10`, reaching mean grouped-CV NDCG@10 of approximately **0.960**.

The experimental results do not support declaring BM25 or TF-IDF universally superior on this dataset. Across matched comparisons in the broader search, BM25's mean NDCG@10 advantage was small and statistically inconclusive (`+0.0118`, `p = 0.46`, 95% CI `[-0.020, +0.043]`). Stage 2 adds Rocchio relevance feedback so recruiter star/reject signals can rerank candidates while preserving the original Stage-1 score as a stable baseline.

Overall, the project supports a compact and interpretable candidate-ranking baseline with controlled personalization. Its main constraint is the small number of independent human-labelled profiles. The highest-value next step is therefore **better and broader relevance data**, not model complexity for its own sake.

## 2. Problem Definition and Objective

The system is designed to answer a search question: **given an HR-oriented recruiter query, in what order should the available candidate profiles be presented?** This differs from ordinary classification. Two systems can identify the same relevant candidates but provide very different recruiter experiences if one places them in the first few positions and the other places them near the bottom.

The project uses two related operational queries:

- `aspiring human resources`
- `seeking human resources`

A **query** is the recruiter search expression, a **candidate/profile** is an item being ranked, a **ranking score** is the numerical value used to order profiles, and **top-k** refers to the first `k` ranked results.

The score represents **relative textual relevance to the search objective**. It is not a probability of being hired, a measure of employee quality, a prediction of job performance, or an automated recommendation to accept or reject a candidate.

## 3. Data and Methodology

The original dataset contains candidate ID, job title, location, connection count, and an unused source fit field. The labelled dataset adds the **final human relevance grade** used for modelling and evaluation.

| Item | Final project treatment |
|---|---|
| Source rows | 104 |
| Unique profile groups | 53 |
| Primary textual evidence | Job title |
| Human relevance scale | 0–3 graded relevance |
| Binary threshold for MAP/MRR | Grades 2–3 count as relevant |
| Primary model-selection metric | NDCG@10 |

### Relevance labelling

Relevance was graded rather than reduced immediately to a binary label because the search objective contains meaningful degrees of match. The final rubric is:

| Grade | Meaning | Practical interpretation |
|---:|---|---|
| 0 | Irrelevant | No meaningful alignment with the HR search objective |
| 1 | Weak / adjacent | Some HR-related evidence, but weak alignment with the intended search |
| 2 | Clearly relevant | Strong HR relevance, though not the most direct target match |
| 3 | Highly / directly relevant | Strong HR relevance with direct aspiring/seeking-style intent alignment |

The grading process considers the **meaning of the whole title**, not keyword presence alone. Important boundary cases include HR-adjacent roles, established senior HR professionals, generic `student` or `seeking` language without HR context, and titles containing HR terminology without matching the intended search intent. The Technical Report contains the full rubric, examples, ambiguity/revision logic, and annotation limitations.

Labels are treated at the **unique-profile level** and propagated back to repeated source rows for traceability. This matters statistically: the 104 raw rows represent only 53 independent profile groups. The unique-group grade counts are **17 grade-0, 2 grade-1, 28 grade-2, and 6 grade-3 profiles**. After duplicate propagation, the raw-row counts become **27, 10, 47, and 20** respectively. In particular, six unique grade-3 judgements expand to twenty source rows; those twenty rows are not twenty independent human labels.

### Methodology summary

The experimental workflow is:

**duplicate handling → text normalization → lexical representations → ranking features → grouped model selection → final Stage-1 ranking → Stage-2 relevance feedback**

Text normalization standardizes casing, punctuation, abbreviations such as `HR → human resources`, and other surface variation while preserving important intent words such as `aspiring` and `seeking`. Stage 1 evaluates word and character TF-IDF, cosine similarity, BM25, interpretable overlap features, optional connection-count information, and Ridge regularization. The broader search covered **192 configurations**, including **64 stemming variants**, to compare preprocessing, representation, feature, and model choices under the same duplicate-safe evaluation logic.

Five-fold **GroupKFold** cross-validation keeps every duplicate profile entirely inside one fold so that an identical candidate cannot appear in both training and validation. NDCG@10 is the primary model-selection metric because it rewards both stronger relevance grades and better placement near the top of the list; MAP and MRR provide complementary binary-relevance views.

Stage 2 uses **Rocchio relevance feedback**. Positive and negative recruiter feedback shifts the query representation toward or away from corresponding profile language, and the feedback score is blended with the immutable Stage-1 baseline so personalization changes ordering without recursively replacing the general search logic.

## 4. Results and Key Findings

The experiments were deliberately staged from simple lexical baselines through feature and regularization tuning. The first table shows comparable grouped-CV checkpoints; the second summarizes what the broader preprocessing/model search taught us.

### Comparative quantitative results

| Model / checkpoint | NDCG@10 | MAP@10 | MRR | Interpretation |
|---|---:|---:|---:|---|
| Word TF-IDF lexical baseline | 0.9550 | 1.0000 | 1.0000 | Strong simple baseline; graded ordering still leaves room for improvement |
| Character TF-IDF lexical baseline | 0.9552 | 0.9982 | 1.0000 | Similar top-of-list performance with more tolerance to surface variation |
| BM25 lexical baseline | 0.9527 | 0.9909 | 1.0000 | Competitive lexical retrieval, but not consistently better in this retained split |
| Ridge text-only, `alpha = 1` | 0.9510 | 1.0000 | 1.0000 | Moderate regularization does not produce the best graded ordering |
| **Ridge text-only, `alpha = 10`** | **0.9596** | **1.0000** | **1.0000** | **Selected Stage-1 model** |
| Ridge + connections, `alpha = 10` | 0.9562 | 1.0000 | 1.0000 | Connection count does not improve the selected model |
| Ridge text-only, `alpha = 100` | 0.9547 | 1.0000 | 1.0000 | Stronger regularization slightly reduces mean NDCG@10 |

*Lexical baselines are evaluated on the same five duplicate-safe folds as descriptive ranking checkpoints. The Ridge rows are the retained grouped-CV tuning results. MAP/MRR saturate for several configurations, which is why NDCG@10 is the primary discriminator.*

### Experimental choices and findings

| Experimental question | Evidence | Decision / finding |
|---|---|---|
| Does regularization strength matter? | Text-only Ridge NDCG@10 rises from ~0.932 at very small `alpha` to 0.9596 at `alpha = 10`, then falls to 0.9547 at `alpha = 100` | Tune regularization rather than assume the weakest/strongest penalty is best |
| Does connection count help? | At `alpha = 10`, text-only = 0.9596 vs with-connections = 0.9562 NDCG@10 | Exclude connection count from the selected compact model |
| Does stemming deserve to remain? | 64 stemmed configurations were evaluated within the 192-configuration search | Stemming was treated as an empirical option, not a required preprocessing step, and was not retained in the compact final model |
| Is one lexical family clearly superior? | Across matched broad-search comparisons, BM25 − TF-IDF mean NDCG@10 difference = +0.0118; `p = 0.46`; 95% CI `[-0.020, +0.043]`; 12/18/10 win/tie/loss | No evidence of BM25 superiority; both provide useful lexical evidence |
| Why use NDCG@10 as primary? | MAP/MRR reach or approach 1.0 across many strong configurations while NDCG@10 still separates them | Graded top-of-list ordering provides more resolution than binary retrieval success alone |

### Key findings

| Key finding | Evidence | Interpretation |
|---|---|---|
| **Duplicate structure materially changes the effective sample** | 104 source rows → 53 unique profiles | Raw row count overstates independent evidence; duplicate-safe grouping is required |
| **Relevance is graded and intent-sensitive** | 0–3 rubric; only 6 unique grade-3 profiles despite 20 grade-3 raw rows | Keyword presence alone is insufficient; direct intent alignment differs from merely HR-related text |
| **Broad tuning supports a compact final model** | 192 configurations evaluated; selected text-only Ridge `alpha = 10`, NDCG@10 = 0.9596 | The final method is systematic but remains interpretable and simple |
| **TF-IDF and BM25 are both viable lexical evidence** | Broad paired comparison is statistically inconclusive | Small observed differences should not be turned into unsupported winner claims |
| **Ranking metrics add information that classification-style summaries miss** | MAP/MRR saturate while NDCG@10 continues to separate graded ordering | Evaluation must reward both relevance strength and rank position |
| **Stage-2 feedback enables controlled personalization** | Rocchio reranking changes candidate order while retaining Stage 1 as the baseline | Recruiter preferences can influence ordering without replacing the validated default ranking |
| **Data quality is the main next constraint** | Only 53 unique labelled profiles; annotation uncertainty remains | More independent, carefully labelled examples are a higher-value investment than immediate model complexity |

The labelling exercise itself produced an important methodological finding: **simple keyword rules are not sufficient to define relevance**. A title can be strongly HR-related yet only weakly aligned with an aspiring/seeking search intent, while another profile can be a direct match because the complete title expresses both domain and intent. This is why the project uses graded human relevance rather than automatically derived keyword labels.

Stage 2 demonstrates that explicit recruiter feedback can alter ordering in the intended direction without discarding the general Stage-1 ranking. The feedback mechanism therefore functions as controlled session-level personalization rather than as a replacement for the validated baseline.

## 5. Interpretation and Limitations

The project should be interpreted as a **candidate search and ranking system**, not a candidate-quality model. The strongest evidence supports four conclusions: ranking is the correct formulation for the recruiter-facing objective; graded human relevance and duplicate-safe evaluation are necessary; a transparent lexical model is appropriate for the current data scale; and relevance feedback is a sensible controlled extension for personalization.

The BM25 comparison illustrates why quantitative differences require context. A slightly higher average in one set of comparisons does not by itself establish method superiority. The paired confidence interval crosses zero and the win/tie/loss pattern is mixed, so the defensible interpretation is that both lexical families are useful and their relative performance is not stable enough here to justify a strong winner claim.

The main limitations are:

- only **53 unique labelled profiles** underpin the experiment;
- 104 raw rows do not constitute 104 independent observations;
- relevance grades contain human judgement uncertainty and no formal inter-rater agreement estimate is available;
- the model relies mainly on job-title text and basic metadata rather than full resumes, skills, interviews, or employment outcomes;
- TF-IDF and BM25 are lexical methods and may miss deeper semantic equivalence;
- repeated development on a small labelled dataset can eventually overfit the evaluation process itself;
- offline ranking metrics do not establish recruiter productivity, hiring quality, or real-world utility;
- fairness is not established simply because protected attributes are not explicitly modelled; text, location, and network-related information may still act as proxies;
- ranking scores must not be interpreted as probabilities of candidate suitability or future success.

The highest-value future work is to collect more **unique**, carefully labelled profiles, strengthen annotation consistency, create a prospective holdout set, measure recruiter behavior in a controlled pilot, and only then evaluate richer semantic models against the current transparent baseline.

## 6. Conclusion

The project successfully frames candidate search as a ranking problem, handles duplicate evidence at the correct unit of analysis, uses graded relevance rather than oversimplified binary labels, and evaluates an interpretable two-stage ranking pipeline with leakage-safe grouped cross-validation and ranking-specific metrics. The broad model search supports a compact Stage-1 lexical model, while the comparative analysis shows that small average differences between retrieval methods should be interpreted with statistical uncertainty rather than converted into unsupported winner claims. Stage-2 Rocchio feedback provides a practical mechanism for controlled personalization.

The resulting system is best positioned as a human-supervised recruiter search aid. Further progress should be driven primarily by better relevance labels, more independent candidate examples, and real usage evidence rather than by adding model complexity for its own sake.

---

## Repository Contents

```text
Wa_BVVFTGqd_OVCYjY/
├── 01_data_representation.ipynb
├── 02_grouped_cv_model_selection.ipynb
├── 03_final_ranking_stage2.ipynb
├── Business_Recommendations.pdf
├── README.md
├── Technical_Report.pdf
├── potential-talents.csv
├── potential-talents-labelled.csv
├── requirements.txt
└── .gitignore
```

The three executed notebooks contain the complete analytical workflow and embedded outputs: data preparation and representation, grouped cross-validation/model selection, and the final Stage-1 ranking with Stage-2 relevance feedback. `potential-talents.csv` is the original source dataset; `potential-talents-labelled.csv` is the same candidate table with the final human relevance grade added. The two PDF reports provide the detailed technical analysis and the business-facing recommendations.

**Recommended review order:** `README.md` → `Technical_Report.pdf` → `Business_Recommendations.pdf` → `01_data_representation.ipynb` → `02_grouped_cv_model_selection.ipynb` → `03_final_ranking_stage2.ipynb`.

**Reproduction:** install the pinned dependencies with `pip install -r requirements.txt`, launch Jupyter, and execute the notebooks in numerical order. The repository is intentionally notebook-driven; no helper modules, generated result files, or command-line pipeline are required.
