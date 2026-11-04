# FinSight RAG — Coding Project

**Project title:** FinSight RAG: Intelligent Financial Report Analysis Using Retrieval-Augmented Generation  
**Team:** Akshitha S, Bindu Prakash, Janya Jetty

This is a runnable student prototype containing:
- Generative AI answer generation through a local Ollama model.
- Retrieval-Augmented Generation (PDF/text extraction, chunks, Sentence Transformer embeddings, cosine-similarity retrieval).
- A bounded agent workflow: question classification → permission-filtered retrieval → evidence gate → answer with source/page references.
- Demo RBAC login and fail-closed document permissions.
- LLM-only baseline button for comparison.
- Basic security tests and evaluation metric definitions.

## 1. Requirements
- Python 3.10 or 3.11 recommended.
- Ollama installed locally.
- Internet is needed the first time to download the embedding model and to install packages.
- Enough memory to load `sentence-transformers/all-MiniLM-L6-v2`.

## 2. Install and run
Open a terminal in this folder:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
ollama pull qwen2.5:3b
ollama serve
```

Open a second terminal, activate the same environment, and run:

```bash
streamlit run src/app.py
```

The app opens in your browser.

## 3. Demo accounts
- `admin` / `admin123`
- `analyst` / `analyst123`
- `viewer` / `viewer123`

These credentials are for local demonstration only. Change them before any shared deployment. This is a classroom prototype, not production authentication.

## 4. Test without real annual reports
Choose **Synthetic smoke-test folder** in the app. These documents are clearly fictional and only test the pipeline.

Example question:
`What was Northstar Example Corporation's revenue for FY2025?`

Expected evidence should include the synthetic warning and a source/page citation. Do not use the synthetic numbers as real company data.

## 5. Add real annual reports
1. Download official reports from the SEC EDGAR or company investor-relations website.
2. Place PDF files in `data/financial_reports/`.
3. Add each exact filename to `config/document_permissions.csv`, e.g.:

```csv
source_name,allowed_roles
Example_Company_2025_10K.pdf,admin;analyst
```

Unknown files are denied to non-admin users. The app filters documents **before** indexing/retrieval, so restricted text is not included in the non-admin retrieval context.

## 6. Important RBAC limitations
The demo credentials are stored as salted PBKDF2 hashes, but this is not a production identity system. Admin can access all files. A real deployment needs a secure secret store/identity provider, TLS, rate limiting, stronger audit logging, document-level access reviews, and independent penetration testing. Do not expose this prototype to the public internet.

## 7. Agentic AI scope
This is a **bounded agent**, not a fully autonomous agent. It classifies the query, retrieves permitted passages, applies an evidence threshold, and calls the LLM. It does not execute arbitrary code or make investment decisions. The ratio question classifier is currently used to broaden retrieval; production-quality deterministic ratio extraction/calculation is a planned extension and should be implemented with validated numeric inputs.

## 8. Testing
```bash
pytest -q
```

Metrics are documented in `evaluation/metrics.md`. Populate ground truth only after verifying annual-report figures and source pages. Compare LLM-only and RAG outputs with identical questions and scoring rules.

## 9. Scientific basis
- Mokashi et al. (2025), *Analysis of Large Language Models for Company Annual Reports Based on Retrieval-Augmented Generation*. https://doi.org/10.3390/info16090786
- Lewis et al. (2020), *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. https://arxiv.org/abs/2005.11401
- Yao et al. (2023), *ReAct: Synergizing Reasoning and Acting in Language Models*. https://arxiv.org/abs/2210.03629

The included question dataset scaffold is separate from the original research paper's dataset. Actual reports and verified ground-truth answers must be added before formal evaluation.
