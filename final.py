import google.generativeai as genai
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Plus
import numpy as np
import json
from pathlib import Path
from PIL import Image

# ==================== CONFIGURATION ====================
GEMINI_API_KEY = ""  # Add your Gemini API key
CHROMA_DB_PATH = "./db"
COLLECTION_NAME = "sym_collection"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """You are a medical report analysis assistant. Extract test parameters, identify abnormalities, and describe symptoms concisely."""

# ==================== REPORT ANALYSIS FUNCTIONS ====================
def analyze_report_image(image_path, report_type):
    """Analyze medical report image using Gemini Vision API."""
    print(f"\n{'='*60}")
    print(f"📄 Analyzing {report_type} report...")
    print(f"{'='*60}")
    
    img = Image.open(image_path)
    
    prompt = f"""Extract ALL test parameters from this {report_type} report.

Return ONLY valid JSON (no markdown):
{{
  "parameters": [
    {{
      "name": "Test Name",
      "value": "measured value",
      "referenceRange": "min - max",
      "unit": "unit",
      "status": "normal/low/high"
    }}
  ]
}}

Determine status by comparing value to reference range."""

    model = genai.GenerativeModel('gemini-2.0-flash-exp', system_instruction=SYSTEM_PROMPT)
    
    response = model.generate_content(
        [prompt, img],
        generation_config={
            "temperature": 0.1,
            "max_output_tokens": 4096,
        }
    )
    
    cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()
    
    try:
        parsed_data = json.loads(cleaned_text)
        print(f"✓ Extracted {len(parsed_data['parameters'])} parameters")
        return parsed_data
    except json.JSONDecodeError as e:
        print(f"✗ JSON parsing error: {e}")
        return {"parameters": []}


def get_combined_symptoms(abnormal_params):
    """Get combined symptoms for all abnormalities in 10-15 words."""
    if not abnormal_params:
        return ""
    
    abnormalities_text = "\n".join([
        f"- {p['name']}: {p['status']} ({p['value']} {p['unit']}, Normal: {p['referenceRange']})"
        for p in abnormal_params
    ])
    
    prompt = f"""Abnormalities found:
{abnormalities_text}

List the symptoms from ALL these abnormalities combined.

REQUIREMENTS:
- Use EXACTLY 10-15 words (not more, not less)
- Write ONLY the symptoms directly (e.g., "Fatigue, dizziness, headaches, and shortness of breath")
- NO phrases like "the patient may experience", "symptoms include", etc.
- Focus on symptoms, not disease names
- Be specific and clear

Symptoms:"""
    
    model = genai.GenerativeModel('gemini-2.0-flash-exp', system_instruction=SYSTEM_PROMPT)
    
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.3,
            "max_output_tokens": 100,
        }
    )
    
    symptoms = response.text.strip()
    
    # Ensure it's within 15 words
    words = symptoms.split()
    if len(words) > 15:
        symptoms = ' '.join(words[:15])
    
    return symptoms


# ==================== DISEASE SEARCH FUNCTIONS ====================
def initialize_search_system():
    """Initialize ChromaDB, embeddings, and BM25."""
    print(f"\n{'='*60}")
    print("🔍 Initializing search system...")
    print(f"{'='*60}")
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    
    # Initialize models
    embedding_model = HuggingFaceEmbeddings(
        model_name='Qwen/Qwen3-Embedding-0.6B',
        model_kwargs={"device": "cpu"}
    )
    reranker = CrossEncoder(RERANKER_MODEL)
    
    # Get all documents
    print("Loading documents from ChromaDB...")
    all_data = collection.get(include=['documents', 'metadatas'])
    documents = all_data['documents']
    metadatas = all_data['metadatas']
    ids = all_data['ids']
    print(f"✓ Loaded {len(documents)} documents")
    
    # Initialize BM25
    print("Initializing BM25Plus...")
    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Plus(tokenized_docs)
    print("✓ Search system ready")
    
    return collection, embedding_model, reranker, bm25, documents, metadatas, ids


def search_diseases(query, collection, embedding_model, reranker, bm25, documents, metadatas, ids):
    """Perform hybrid search + reranking to find top 3 diseases."""
    print(f"\n{'='*60}")
    print(f"🔎 Searching for diseases...")
    print(f"{'='*60}")
    print(f"Query: {query}")
    
    # Step 1: Hybrid Search (BM25 + Semantic)
    # BM25 scores
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Normalize BM25 scores
    if bm25_scores.max() > 0:
        bm25_scores_norm = bm25_scores / bm25_scores.max()
    else:
        bm25_scores_norm = bm25_scores
    
    # Semantic search
    query_embedding = embedding_model.embed_query(query)
    semantic_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=len(documents)
    )
    
    # Get semantic scores
    semantic_distances = np.array(semantic_results['distances'][0])
    semantic_scores = 1 / (1 + semantic_distances)
    
    # Normalize semantic scores
    if semantic_scores.max() > semantic_scores.min():
        semantic_scores_norm = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min())
    else:
        semantic_scores_norm = semantic_scores
    
    # Map semantic results back to original document order
    semantic_ids = semantic_results['ids'][0]
    id_to_semantic_score = dict(zip(semantic_ids, semantic_scores_norm))
    semantic_scores_ordered = np.array([id_to_semantic_score.get(doc_id, 0) for doc_id in ids])
    
    # Combine scores (0.6 for BM25, 0.4 for semantic)
    bm25_weight = 0.6
    semantic_weight = 0.4
    hybrid_scores = (bm25_weight * bm25_scores_norm) + (semantic_weight * semantic_scores_ordered)
    
    # Get top-k results (expand to ensure 3 unique names)
    top_k = 10
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
    
    # Expand search until we have at least 3 unique names
    unique_names_in_topk = set()
    for idx in top_indices:
        name = metadatas[idx].get('Name', 'N/A')
        if name != 'N/A':
            unique_names_in_topk.add(name)
    
    while len(unique_names_in_topk) < 3 and top_k < len(documents):
        top_k += 10
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        unique_names_in_topk = set()
        for idx in top_indices:
            name = metadatas[idx].get('Name', 'N/A')
            if name != 'N/A':
                unique_names_in_topk.add(name)
    
    # Prepare hybrid search results
    hybrid_results = []
    for idx in top_indices:
        hybrid_results.append({
            'id': ids[idx],
            'document': documents[idx],
            'metadata': metadatas[idx],
            'name': metadatas[idx].get('Name', 'N/A'),
            'hybrid_score': float(hybrid_scores[idx]),
            'bm25_score': float(bm25_scores_norm[idx]),
            'semantic_score': float(semantic_scores_ordered[idx])
        })
    
    print(f"✓ Found top {top_k} candidates")
    
    # Step 2: Cross-Encoder Reranking
    print("Reranking results...")
    pairs = [[query, result['document']] for result in hybrid_results]
    rerank_scores = reranker.predict(pairs)
    
    # Add rerank scores and sort
    for i, result in enumerate(hybrid_results):
        result['rerank_score'] = float(rerank_scores[i])
    
    sorted_results = sorted(hybrid_results, key=lambda x: x['rerank_score'], reverse=True)
    
    # Get top 3 unique names
    final_results = []
    seen_names = set()
    
    for result in sorted_results:
        name = result['name']
        if name not in seen_names and name != 'N/A':
            seen_names.add(name)
            final_results.append(result)
            if len(final_results) == 3:
                break
    
    print(f"✓ Found top 3 unique diseases")
    return final_results


# ==================== MAIN INTEGRATION FUNCTION ====================
def main():
    """Main integrated function."""
    print("="*60)
    print("🏥 INTEGRATED MEDICAL REPORT ANALYZER & DISEASE SEARCH")
    print("="*60)
    
    # ========== PART 1: ANALYZE BLOOD REPORT ==========
    image_path = "input/report.png"
    report_type = "BLOOD"
    
    if not Path(image_path).exists():
        print("✗ Image file not found!")
        return
    
    try:
        # Extract parameters from report
        data = analyze_report_image(image_path, report_type)
        
        if not data['parameters']:
            print("✗ No parameters extracted")
            return
        
        # Find abnormalities
        abnormal = [p for p in data['parameters'] if p['status'] != 'normal']
        normal_count = len(data['parameters']) - len(abnormal)
        
        print(f"\n{'='*60}")
        print("📊 REPORT SUMMARY")
        print(f"{'='*60}")
        print(f"Total Parameters: {len(data['parameters'])}")
        print(f"Abnormal: {len(abnormal)} | Normal: {normal_count}")
        
        # Get report-based symptoms
        report_symptoms = ""
        if abnormal:
            print(f"\n{'-'*60}")
            print("⚠️ ABNORMALITIES:")
            print(f"{'-'*60}")
            
            for i, p in enumerate(abnormal, 1):
                print(f"{i}. {p['name']}: {p['status'].upper()}")
                print(f"   Value: {p['value']} {p['unit']} (Normal: {p['referenceRange']})")
            
            report_symptoms = get_combined_symptoms(abnormal)
            print(f"\n{'-'*60}")
            print("💡 REPORT-BASED SYMPTOMS:")
            print(f"{'-'*60}")
            print(f"{report_symptoms}")
        else:
            print("\n✓ All parameters normal")
        
        # ========== PART 2: GET PATIENT SYMPTOMS ==========
        print(f"\n{'='*60}")
        print("👤 PATIENT INPUT")
        print(f"{'='*60}")
        
        # Get patient symptoms (you can modify this to accept user input)
        patient_symptoms = "I feel dizzy."
        
        print(f"Patient symptoms: {patient_symptoms}")
        
        # ========== PART 3: COMBINE SYMPTOMS & SEARCH ==========
        # Combine both symptoms
        if report_symptoms and patient_symptoms:
            combined_query = f"{patient_symptoms}. {report_symptoms}"
        elif report_symptoms:
            combined_query = report_symptoms
        elif patient_symptoms:
            combined_query = patient_symptoms
        else:
            print("\n✗ No symptoms to search for")
            return
        
        print(f"\n{'='*60}")
        print("🔗 COMBINED QUERY")
        print(f"{'='*60}")
        print(f"{combined_query}")
        
        # Initialize search system
        collection, embedding_model, reranker, bm25, documents, metadatas, ids = initialize_search_system()
        
        # Search for diseases
        top_diseases = search_diseases(
            combined_query,
            collection,
            embedding_model,
            reranker,
            bm25,
            documents,
            metadatas,
            ids
        )
        
        # ========== PART 4: DISPLAY RESULTS ==========
        print(f"\n{'='*60}")
        print("🎯 TOP 3 PREDICTED DISEASES")
        print(f"{'='*60}")
        
        for i, result in enumerate(top_diseases, 1):
            print(f"\n{i}. {result['name']}")
            print(f"   Rerank Score: {result['rerank_score']:.4f}")
            print(f"   Hybrid Score: {result['hybrid_score']:.4f}")
            print(f"   Description: {result['document'][:150]}...")
        
        # ========== PART 5: SAVE RESULTS ==========
        results = {
            "report_analysis": {
                "total_parameters": len(data['parameters']),
                "abnormal_count": len(abnormal),
                "abnormalities": abnormal,
                "report_symptoms": report_symptoms if abnormal else "All normal"
            },
            "patient_symptoms": patient_symptoms,
            "combined_query": combined_query,
            "top_diseases": [
                {
                    "rank": i + 1,
                    "name": result['name'],
                    "rerank_score": result['rerank_score'],
                    "hybrid_score": result['hybrid_score'],
                    "description": result['document']
                }
                for i, result in enumerate(top_diseases)
            ]
        }
        
        output_path = Path("results2") / "integrated_analysis.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✓ Complete results saved to {output_path}")
        print(f"{'='*60}\n")
        
        # Extract top 3 disease names
        top_3_names = [result['name'] for result in top_diseases]
        print(f"Final Top 3 Diseases: {top_3_names}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
