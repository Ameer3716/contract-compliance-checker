# Contract Compliance Checker

An AI-powered contract compliance checker using RAG (Retrieval-Augmented Generation) to analyze contracts against 15 predefined compliance rules.

## Live Demo

[Try it on Streamlit Cloud](#)

## Features

- Single Rule Check: Analyze compliance for individual rules
- Full Compliance Scan: Run comprehensive checks against all 15 rules
- Visual Results: Interactive dashboard with compliance statistics
- Export Results: Download compliance reports as CSV
- AI-Powered: Uses TinyLlama 1.1B with FAISS vector search

## Tech Stack

- LLM: TinyLlama 1.1B Chat
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- Vector Store: FAISS
- Framework: LangChain
- UI: Streamlit
- Dataset: CUAD (Contract Understanding Atticus Dataset)

## Installation

```bash
git clone https://github.com/Ameer3716/contract-compliance-checker.git
cd contract-compliance-checker
pip install -r requirements.txt
streamlit run app.py
```

## Usage

1. Single Rule Check: Select a specific rule and check compliance
2. Full Compliance Scan: Run all 15 rules at once
3. View Results: Browse and filter previous scan results

## Compliance Rules (15 Categories)

- Termination clauses
- Confidentiality obligations
- Liability limitations
- Governing law
- IP rights
- Payment terms
- Indemnification
- Force majeure
- Non-compete clauses
- Data protection
- Warranty terms
- Assignment provisions
- Amendment procedures
- Insurance requirements
- Audit rights

## Performance

- Initial Load: 2-3 minutes (model loading)
- Single Rule Check: 10-20 seconds
- Full Scan (15 rules): 3-5 minutes

## License

MIT License

## Contact

GitHub: [@Ameer3716](https://github.com/Ameer3716)

---

Note: This tool is for assistance purposes only and should not replace professional legal review.
