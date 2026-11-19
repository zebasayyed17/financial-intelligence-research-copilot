# 📊 Financial Intelligence & Research Copilot

### AI-Powered Financial Report Analysis | RAG | Generative AI | Python | Streamlit

An intelligent financial research application designed to help analysts explore corporate annual reports, retrieve relevant financial information, compare disclosures, calculate key financial ratios, and identify potential business risks.

The project combines **Retrieval-Augmented Generation (RAG), semantic search, financial analytics, and Google Gemini** to support evidence-based financial research.

---

## 🚀 Project Overview

Financial analysts often spend significant time reviewing lengthy annual reports, extracting financial information, comparing year-over-year disclosures, and identifying emerging business risks.

The Financial Intelligence & Research Copilot aims to simplify this process through an interactive research environment where users can upload corporate PDF reports, build a searchable document index, retrieve relevant evidence, and generate AI-assisted research insights.

## ✨ Key Features

### 1. Interactive Financial Dashboard
- Financial overview and company selection.
- Key financial metric visualisations.
- Navigation across research and analytical modules.

### 2. AI-Powered Research Chat
- Ask natural-language questions about uploaded annual reports.
- Retrieve relevant passages using semantic search.
- Generate AI-assisted answers using Google Gemini.
- Display supporting source documents and page references.

### 3. Disclosure Change Detector
- Compare disclosures across two annual reports.
- Retrieve relevant passages from different reporting periods.
- Support analysis of changes in corporate risk disclosures.

### 4. Financial KPI Analyzer
Calculate and analyse important financial indicators, including:
- Revenue Growth
- Net Profit Margin
- Operating Margin
- Debt-to-Assets Ratio

### 5. Risk Intelligence
- Screen financial disclosures for risk-related topics.
- Identify keywords associated with cybersecurity, supply chain, regulation, and market risks.
- Present results for further analyst review.

### 6. Investment Thesis & SWOT Analysis
- Generate AI-assisted investment research notes.
- Explore investment horizons and risk profiles.
- Present a structured SWOT analysis covering strengths, weaknesses, opportunities, and threats.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Application | Streamlit |
| Data Processing | Pandas, NumPy |
| PDF Processing | PyMuPDF |
| Embeddings | Sentence Transformers |
| Semantic Retrieval | Cosine Similarity |
| Generative AI | Google Gemini API |
| Visualisation | Plotly |
| Environment | Python, PowerShell |

---

## ⚙️ How It Works

1. **PDF Ingestion:** Upload corporate annual reports and extract text with page-level metadata using PyMuPDF.
2. **Text Processing:** Divide extracted content into overlapping chunks.
3. **Embedding Generation:** Convert text chunks into semantic vector representations using Sentence Transformers.
4. **Semantic Retrieval:** Retrieve relevant passages based on user queries.
5. **AI-Assisted Analysis:** Use Google Gemini to generate research responses from retrieved evidence.
6. **Financial Analytics:** Calculate financial KPIs and screen disclosures for risk-related topics.
7. **Comparative Research:** Retrieve passages from multiple reports to support disclosure comparisons.

---

## 📁 Project Structure

```text
financial-intelligence-research-copilot/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
└── src/
    ├── ingestion.py
    ├── retrieval.py
    ├── analytics.py
    ├── synthesis.py
    └── recommendations.py
```

---

## 💻 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/zebasayyed17/financial-intelligence-research-copilot.git
```

### 2. Navigate to the project directory

```bash
cd financial-intelligence-research-copilot
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment (Windows)

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure Google Gemini API

Set up a Google Gemini API key and configure it securely as an environment variable before using the Gemini-powered features.

Never upload API keys or `.env` files to GitHub.

### 7. Run the application

```bash
streamlit run app.py
```

---

## 📄 Using the Application

1. Launch the Streamlit application.
2. Open the Document Library.
3. Upload text-based corporate annual-report PDFs.
4. Save the uploaded reports.
5. Process the PDFs and build the semantic search index.
6. Explore Research Chat, Disclosure Change Detector, KPI Analyzer, and Risk Intelligence.

For comparative disclosure analysis, upload at least two annual reports.

The first embedding run requires internet access to download the pretrained model. Scanned PDFs may require OCR, which is not currently included.

---

## ⚠️ Limitations & Considerations

- Risk identification is keyword-based and requires human validation.
- Semantic retrieval quality depends on document quality and chunking.
- AI-generated research outputs should be checked against original filings.
- Some dashboard and SWOT examples are illustrative rather than automatically extracted financial facts.
- This is a research prototype, not a substitute for professional financial analysis or investment advice.

---

## 🔮 Future Enhancements

- Integrate structured financial data from SEC Company Facts.
- Improve disclosure comparison using section-level semantic matching.
- Add automated financial statement extraction.
- Develop an evaluation framework for retrieval accuracy and citation quality.
- Add unit tests and deployment configuration.
- Expand support for additional financial document formats.

---

## 👩‍💻 Author

**Zeba Sayyed**

Data Science & Analytics | Machine Learning | Generative AI

GitHub: [@zebasayyed17](https://github.com/zebasayyed17)

