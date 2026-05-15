# Changelog

## 2026-05-15 — feature/remove-intern-stopwords
- Added common internship-related tokens to domain stopwords (`intern`, `interns`, `internship`, `internships`, `sip`, `sips`, `trainee`, `trainees`, `student`, `students`).
- Implemented post-lemmatization filtering to remove any residual domain stopwords from the cleaned corpus.
- Updated keyword-analysis reporting to reflect the stricter filtering.
- Re-ran preprocessing and downstream analysis (TF-IDF, similarity, NMF topic modelling, classification, sentiment, curriculum–industry gap analysis) to propagate cleaned results.

Notes:
- Changes made in `SIPReports_ToYunLin/basic_eda_sip_reports.ipynb`.
- Branch: `feature/remove-intern-stopwords`.
