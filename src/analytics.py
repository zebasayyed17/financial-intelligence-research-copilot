import re

def _pct(value):
    return "N/A" if value is None else f"{value:.2f}%"

def calculate_kpis(revenue_current, revenue_previous, net_income, operating_income, total_assets, total_debt):
    growth = None if revenue_previous == 0 else ((revenue_current - revenue_previous) / revenue_previous) * 100
    net_margin = None if revenue_current == 0 else (net_income / revenue_current) * 100
    op_margin = None if revenue_current == 0 else (operating_income / revenue_current) * 100
    debt_assets = None if total_assets == 0 else (total_debt / total_assets) * 100
    return {
        "Revenue growth YoY": _pct(growth),
        "Net profit margin": _pct(net_margin),
        "Operating margin": _pct(op_margin),
        "Debt / assets": _pct(debt_assets),
    }

RISK_KEYWORDS = {
    "Cybersecurity & data": [
        "cybersecurity", "cyber attack", "cybercrime", "data breach", "privacy",
        "information security", "secure future initiative", "phishing-resistant",
        "multifactor authentication", "vulnerabilities", "threat detection",
        "secure-by-design", "identity protection", "ransomware",
    ],
    "Supply chain": ["supply chain", "supplier", "component shortage", "logistics", "limited availability"],
    "Regulatory & legal": ["regulatory", "regulation", "litigation", "compliance", "government inquiry"],
    "Competition & market": ["competition", "competitive", "market conditions", "pricing pressure"],
    "Macroeconomic": ["inflation", "interest rates", "foreign exchange", "economic uncertainty", "recession"],
}

def detect_risk_topics(chunks):
    rows = []
    for item in chunks:
        text_lower = item["text"].lower()
        for category, keywords in RISK_KEYWORDS.items():
            hits = [word for word in keywords if word in text_lower]
            if hits:
                rows.append({
                    "Risk category": category,
                    "Matched terms": ", ".join(hits),
                    "Source": item.get("source", ""),
                    "PDF page": item.get("page", ""),
                    "Evidence excerpt": item["text"][:500] + ("…" if len(item["text"]) > 500 else ""),
                })
    return rows

_NUMBER_PATTERN = re.compile(r"\b\d[\d,]*(?:\.\d+)?\s*(?:%|percent|million|billion|thousand|\$)?", re.I)
_DETAIL_TERMS = (
    "implemented", "remedied", "reduced", "increased", "expanded", "strengthened",
    "coverage", "monitor", "detect", "response", "assessment", "controls",
    "vulnerability", "authentication", "engineers", "incident", "governance",
    "testing", "audit", "recovery", "resilience", "training", "investigation",
)

def _detail_signals(text: str) -> int:
    lowered = text.lower()
    numeric_signals = len(_NUMBER_PATTERN.findall(lowered))
    action_signals = sum(lowered.count(term) for term in _DETAIL_TERMS)
    return numeric_signals + action_signals

def _topic_groups(text: str) -> set[str]:
    text = text.lower()
    topic_keywords = {
        "cybersecurity": ["cybersecurity", "cyber attack", "cybercrime", "data breach", "information security", "multifactor authentication", "ransomware"],
        "financial_market_risk": ["derivative", "foreign exchange", "interest rate", "market risk", "credit risk"],
        "supply_chain": ["supply chain", "supplier", "component shortage", "logistics"],
        "regulatory_legal": ["regulatory", "regulation", "litigation", "compliance"],
        "competition": ["competition", "competitive", "market share", "pricing pressure"],
        "macroeconomic": ["inflation", "economic uncertainty", "recession"],
    }
    return {topic for topic, keywords in topic_keywords.items() if any(keyword in text for keyword in keywords)}

def classify_disclosure_change(earlier_text: str, later_text: str) -> tuple[str, str]:
    earlier_topics = _topic_groups(earlier_text)
    later_topics = _topic_groups(later_text)

    if earlier_topics and later_topics and not earlier_topics.intersection(later_topics):
        return ("Needs review", "The passages appear to discuss different topics; verify the retrieval match.")

    old_words = max(1, len(earlier_text.split()))
    new_words = max(1, len(later_text.split()))
    old_signals = _detail_signals(earlier_text)
    new_signals = _detail_signals(later_text)
    ratio = new_words / old_words

    if new_signals >= old_signals + 2 and ratio >= 0.75:
        return ("Expanded", "The later passage contains more quantified or operational detail signals.")
    if old_signals >= new_signals + 2 and ratio <= 1.25:
        return ("Reduced", "The later passage contains fewer quantified or operational detail signals.")
    if ratio >= 1.30:
        return ("Expanded", "The later passage is materially longer, suggesting broader disclosure detail.")
    if ratio <= 0.70:
        return ("Reduced", "The later passage is materially shorter, suggesting less detail.")
    return ("Unchanged", "The matched passages have broadly similar length and detail signals.")