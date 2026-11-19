import os
from google import genai

def synthesize_answer(query: str, retrieved_chunks: list) -> str:
    """Generate a grounded financial research answer using Gemini based on retrieved chunks."""
    if not retrieved_chunks:
        return "No relevant chunks found to synthesize."

    # Format the retrieved chunks into a clean context block
    context_blocks = []
    for i, item in enumerate(retrieved_chunks, 1):
        source_name = item.get("source", "Unknown Source")
        page_num = item.get("page", "Unknown")
        text = item.get("text", "")
        context_blocks.append(f"--- Evidence [{i}] | File: {source_name} | Page: {page_num} ---\n{text}\n")

    context_str = "\n".join(context_blocks)

    prompt = f"""You are a professional financial research assistant. Answer the user's question using ONLY the provided evidence passages from corporate annual reports. 
    - Provide a clear, structured, and factual synthesis.
    - Always cite your sources using the format [Source Name, Page X] immediately after claims derived from them.
    - If the retrieved context does not contain enough information to answer the question, state that clearly. Do not invent information.

    User Question: {query}

    Retrieved Evidence:
    {context_str}

    Synthesized Answer:"""

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "⚠️ **API Key Missing:** Please set your `GEMINI_API_KEY` in your environment variables or Streamlit secrets to enable LLM synthesis."

    try:
        client = genai.Client(api_key=api_key)
        
        # Try multiple fallback flash models in case of high demand / 503 errors
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
                
        return f"⚠️ Error generating LLM synthesis: All fallback models are currently experiencing high demand. Details: {last_error}"

    except Exception as e:
        return f"⚠️ Error initializing Gemini client: {e}"