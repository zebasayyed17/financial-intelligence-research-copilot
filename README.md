# Financial Intelligence & Research Copilot

A Python + Streamlit prototype for evidence-backed financial report research. It extracts text from annual-report PDFs, creates semantic embeddings, retrieves relevant passages, calculates core KPIs, and flags risk-related disclosures for analyst review.

## Current features
- PDF text extraction with page metadata (PyMuPDF)
- Overlapping text chunks
- Semantic retrieval using `all-MiniLM-L6-v2` embeddings and cosine similarity
- Source and page labels on retrieved evidence
- Revenue growth, net margin, operating margin, and debt/assets calculations
- Keyword-based risk disclosure screening
- Two-report passage retrieval for disclosure comparisons

## Important limitations
- This starter retrieves evidence; it does not yet generate final LLM answers.
- Risk categories are keyword-based and require human review.
- A topic not found in a report is not proof that the risk disappeared.
- This is a research prototype, not investment advice. Verify all figures against original filings.

## Setup (Windows PowerShell)

```powershell
cd financial-intelligence-research-copilot
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run the following for the current terminal only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Use
1. Open the Streamlit local URL shown in the terminal.
2. Upload text-based annual-report PDFs in the sidebar.
3. Click **Save uploaded reports**.
4. Click **Process PDFs and build index**.
5. Ask questions in Research Chat and inspect source/page evidence.

The first embedding run downloads a pretrained model, so it needs internet access. Scanned PDFs may require OCR, which is not included in this starter.

## Suggested initial data
Start with two annual reports from the same company (for example, two consecutive Microsoft or Apple annual reports) so the disclosure comparison has two periods. Download reports from the company's official investor-relations website or SEC EDGAR. Follow each source's applicable terms and retain source attribution.

## Next development milestones
1. Add an LLM provider for grounded synthesis and citations.
2. Add structured SEC Company Facts ingestion for automatically sourced financial metrics.
3. Improve disclosure change detection using section matching and semantic similarity.
4. Build an evaluation set with expected evidence passages; measure Recall@K and citation accuracy.
5. Add tests, charts, deployment configuration, and screenshots.

## Project structure
```text
app.py
src/
  ingestion.py
  retrieval.py
  analytics.py
data/
  reports/
  processed/
requirements.txt
README.md
```
