import os
from google import genai

def generate_investment_recommendation(company_name: str, kpis: dict, risk_count: int, disclosure_summary: str) -> str:
    """Generate an institutional-grade investment research synthesis and recommendation with model fallbacks."""
    
    prompt = f"""You are an elite Senior Financial Analyst and Equity Research Director. 
    Analyze the following extracted data for {company_name} and formulate an institutional investment research note.

    Financial KPIs:
    {kpis}

    Total Risk Passages Flagged in Report: {risk_count}
    Key Disclosure Changes: {disclosure_summary}

    Provide your research note using the following strict structure:
    1. **Executive Investment Thesis**: Summary of the company's financial health and strategic standing.
    2. **Growth & Trend Analysis (Uptrends/Downtrends)**: Evaluate revenue growth, margins, and operational momentum.
    3. **Key Risk Factors**: Highlight vulnerabilities derived from the risk screening.
    4. **Analyst Recommendation**: Clearly state one of the following recommendations: [STRONG BUY / BUY / HOLD / SELL / UNDERPERFORM] with supporting rationale.
    
    Disclaimer: Explicitly state that this is an automated research demonstration for educational capstone purposes and not formal financial advice.
    """

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "⚠️ **API Key Missing:** Set your `GEMINI_API_KEY` environment variable."

    try:
        client = genai.Client(api_key=api_key)
        
        # Try fallback flash models to bypass high-demand bottlenecks
        models_to_try = ['gemini-3.8-flash', 'gemini-3.6-flash', 'gemini-3.5-flash-lite']
        last_error = None
        
        for model_name in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                return response.text
            except Exception as e:
                last_error = e
                continue
                
        return f"⚠️ Error generating recommendation thesis: All fallback models are currently experiencing high demand. Details: {last_error}"

    except Exception as e:
        return f"⚠️ Error initializing Gemini client: {e}"