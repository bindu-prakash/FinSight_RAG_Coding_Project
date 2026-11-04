# Financial Annual Report RAG Replication-Style Dataset + Project Materials

## Important source/data distinction
This is a **student-created replication-style dataset inspired by**:
Mokashi, A., Puthuparambil, B., Daniel, C., & Hanne, T. (2025). *Analysis of Large Language Models for Company Annual Reports Based on Retrieval-Augmented Generation*. Information, 16(9), 786. https://doi.org/10.3390/info16090786

The article describes ten selected companies and ten evaluation questions. It states that its dataset is available on request from the authors. This package is **not the authors' original dataset**, does not contain their model outputs or their scored answers, and does not reproduce their reported results as your own.

The company names and the ten question themes follow the article's Table 1 and Table 2. Questions have been lightly normalized for clarity (e.g., “net revenue”). Ground-truth answers and page citations are deliberately blank so they can be completed from the selected official annual reports. Do not fill them by guessing or by copying results from the article.

## Files
- `data/metadata/company_source_register.csv`: 10 companies from the article's company table, with official SEC filing search links and fields to record report year/download details.
- `evaluation/question_dataset_100_rows.csv`: 10 questions × 10 companies = 100 company-question items, with blank ground truth and source/page evidence fields.
- `evaluation/prompt_experiment_matrix_300_rows.csv`: three prompt-specificity variants × 100 company-question items; use for prompt design experiments.
- `evaluation/experiment_results_template.csv`: blank run-by-run results template for LLM-only vs RAG.
- `evaluation/rbac_security_test_matrix.csv`: optional security tests for your extended RBAC system.
- `reports/methodology_and_metrics.md`: study design, formulas and reporting guidance.
- `reports/report_outline.md`: report structure.

## Official report retrieval
Use the SEC EDGAR search links in `company_source_register.csv`. Download the relevant official annual report (10-K) PDF or HTML and record:
1. exact fiscal year,
2. source URL and download date,
3. page count and local filename,
4. page/section/table supporting each answer.

The article's table uses a revenue-range selection criterion, but this package does not assume that all companies' fiscal-year ends match. Verify the exact fiscal year used for each report. The original article reports a ten-company qualitative study and should not be treated as a public downloadable answer key.

## Recommended implementation
- PDF extraction: pypdf; for complex tables, verify with the original PDF and consider table-aware extraction.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`.
- Retrieval: cosine similarity; use ChromaDB or FAISS for a larger corpus.
- Local generation: Ollama with a compatible small instruction model (e.g., `qwen2.5:3b` if available on your machine).
- Interface: Streamlit.
- Baseline: same questions and prompts without retrieved context.
- RAG condition: same questions and prompts with retrieved chunks and source metadata.
- Optional security extension: SQLite user/role records and permission checks before retrieval.

## Scoring
The paper uses a 0–5 scale for accuracy, relevance and verifiability. Use this same scale for comparability:
- 0: wholly incorrect / irrelevant / no verifiable support
- 1: mostly incorrect / barely relevant / citations fail
- 2: substantially incomplete or significant errors
- 3: partly correct and useful, but notable omissions or weak evidence
- 4: mostly correct, relevant, supported; minor omissions
- 5: correct, directly relevant and verifiable with precise source support

For financial calculations, check both formula and inputs. For debt-to-equity, state exactly whether debt means interest-bearing debt or total liabilities; do not silently mix definitions. For quick ratio, state the chosen formula and follow the report's classifications.

## Results integrity
The paper's published results are reference findings, not your experimental results. Run your own evaluation, save configuration details, and report measured values only. Do not claim the starter project includes implemented RBAC unless you add and test it.
