# Evaluation metrics

Run experiments on verified annual-report ground truth; do not fill scores with assumed values.

- **Correctness (0–5):** human rating against verified answer; for numerical answers check period, currency, units and rounding.
- **Relevance (0–5):** whether the response answers the question.
- **Verifiability (0–5):** whether citations support the answer.
- **Hit@k:** answerable queries with at least one gold evidence chunk among top-k / all answerable queries.
- **Citation support rate:** supported audited claims / all audited cited claims × 100.
- **Abstention accuracy:** unanswerable questions correctly declined / all unanswerable questions × 100.
- **Latency:** record retrieval, generation and end-to-end time.
- **RBAC leakage rate:** unauthorized tests that expose restricted content / all unauthorized tests × 100; target zero in the defined test set.

Compare LLM-only and RAG using the same question set, model version, prompt settings, and verified source documents.
