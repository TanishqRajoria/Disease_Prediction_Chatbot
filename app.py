import streamlit as st
import google.generativeai as genai
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Plus
import numpy as np
import json
from pathlib import Path
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="MedAnalyze Pro",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    /* Main styling */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%);
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .main-subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border-left: 5px solid #667eea;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    .abnormal-card {
        background: linear-gradient(135deg, #fff5f5 0%, #ffe5e5 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid #ef4444;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(239, 68, 68, 0.1);
    }
    
    .abnormal-card strong {
        color: #dc2626;
        font-size: 1.1rem;
    }
    
    .disease-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-top: 4px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .disease-card:hover {
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        transform: translateY(-4px);
    }
    
    .disease-card h3 {
        color: #667eea;
        margin-bottom: 1rem;
        font-size: 1.5rem;
    }
    
    
    /* Symptoms box */
    .symptoms-box {
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #0ea5e9;
        font-size: 1.1rem;
        color: #0c4a6e;
        line-height: 1.6;
        box-shadow: 0 2px 4px rgba(14, 165, 233, 0.1);
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
        border: none;
        padding: 0.8rem 2rem;
        font-size: 1.1rem;
        border-radius: 12px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(102, 126, 234, 0.5);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 12px;
        border: none;
    }
    
    /* Dataframe */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Upload area */
    .uploadedFile {
        border-radius: 12px;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
    }
    
    /* Status badges */
    .status-normal {
        color: #059669;
        background: #d1fae5;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .status-abnormal {
        color: #dc2626;
        background: #fee2e2;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ==================== CONFIGURATION ====================

GEMINI_API_KEY = "AIzaSyCUtVi_PmwF9t40OBKeU5J7vBv4NUxASAU"  # Add your Gemini API key here
CHROMA_DB_PATH = "./db"
COLLECTION_NAME = "sym_collection"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

SYSTEM_PROMPT = """You are a medical report analysis assistant. Extract test parameters, identify abnormalities, and describe symptoms concisely."""

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# ==================== SESSION STATE ====================
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.collection = None
    st.session_state.embedding_model = None
    st.session_state.reranker = None
    st.session_state.bm25 = None
    st.session_state.documents = None
    st.session_state.metadatas = None
    st.session_state.ids = None
    st.session_state.analysis_results = None

# ==================== HELPER FUNCTIONS ====================
@st.cache_resource
def initialize_search_system():
    """Initialize ChromaDB, embeddings, and BM25."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        
        embedding_model = HuggingFaceEmbeddings(
            model_name='Qwen/Qwen3-Embedding-0.6B',
            model_kwargs={"device": "cpu"}
        )
        reranker = CrossEncoder(RERANKER_MODEL)
        
        all_data = collection.get(include=['documents', 'metadatas'])
        documents = all_data['documents']
        metadatas = all_data['metadatas']
        ids = all_data['ids']
        
        tokenized_docs = [doc.lower().split() for doc in documents]
        bm25 = BM25Plus(tokenized_docs)
        
        return collection, embedding_model, reranker, bm25, documents, metadatas, ids
    except Exception as e:
        st.error(f"Error initializing search system: {e}")
        return None, None, None, None, None, None, None


def analyze_report_image(image, report_type):
    """Analyze medical report image using Gemini Vision API."""
    
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

    # model = genai.GenerativeModel('gemini-2.0-flash-exp', system_instruction=SYSTEM_PROMPT)
    model = genai.GenerativeModel('gemini-2.0-flash', system_instruction=SYSTEM_PROMPT)
    response = model.generate_content(
        [prompt, image],
        generation_config={
            "temperature": 0.1,
            "max_output_tokens": 4096,
        }
    )
    
    cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()
    
    try:
        parsed_data = json.loads(cleaned_text)
        return parsed_data
    except json.JSONDecodeError as e:
        st.error(f"JSON parsing error: {e}")
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
    
    model = genai.GenerativeModel('gemini-2.0-flash', system_instruction=SYSTEM_PROMPT)
    
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.3,
            "max_output_tokens": 100,
        }
    )
    
    symptoms = response.text.strip()
    words = symptoms.split()
    if len(words) > 15:
        symptoms = ' '.join(words[:15])
    
    return symptoms


def search_diseases(query, collection, embedding_model, reranker, bm25, documents, metadatas, ids):
    """Perform hybrid search + reranking to find top 3 diseases."""
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    if bm25_scores.max() > 0:
        bm25_scores_norm = bm25_scores / bm25_scores.max()
    else:
        bm25_scores_norm = bm25_scores
    
    query_embedding = embedding_model.embed_query(query)
    semantic_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=len(documents)
    )
    
    semantic_distances = np.array(semantic_results['distances'][0])
    semantic_scores = 1 / (1 + semantic_distances)
    
    if semantic_scores.max() > semantic_scores.min():
        semantic_scores_norm = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min())
    else:
        semantic_scores_norm = semantic_scores
    
    semantic_ids = semantic_results['ids'][0]
    id_to_semantic_score = dict(zip(semantic_ids, semantic_scores_norm))
    semantic_scores_ordered = np.array([id_to_semantic_score.get(doc_id, 0) for doc_id in ids])
    
    bm25_weight = 0.6
    semantic_weight = 0.4
    hybrid_scores = (bm25_weight * bm25_scores_norm) + (semantic_weight * semantic_scores_ordered)
    
    top_k = 10
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
    
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
    
    pairs = [[query, result['document']] for result in hybrid_results]
    rerank_scores = reranker.predict(pairs)
    
    for i, result in enumerate(hybrid_results):
        result['rerank_score'] = float(rerank_scores[i])
    
    sorted_results = sorted(hybrid_results, key=lambda x: x['rerank_score'], reverse=True)
    
    final_results = []
    seen_names = set()
    
    for result in sorted_results:
        name = result['name']
        if name not in seen_names and name != 'N/A':
            seen_names.add(name)
            final_results.append(result)
            if len(final_results) == 3:
                break
    
    return final_results

# ==================== MAIN APP ====================
def main():
    # Header
    st.markdown('<div class="main-header">🏥 MedAnalyze Pro</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Advanced Medical Report Analysis & Disease Prediction System</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📋 How It Works")
        st.markdown("""
        <div style='background: white; padding: 1rem; border-radius: 12px; margin-bottom: 1rem;'>
        <ol style='margin: 0; padding-left: 1.5rem; color: #334155;'>
            <li style='margin-bottom: 0.5rem;'>Upload medical report image</li>
            <li style='margin-bottom: 0.5rem;'>Select report type</li>
            <li style='margin-bottom: 0.5rem;'>Add patient symptoms (optional)</li>
            <li>Get instant AI-powered analysis</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### ✨ Features")
        st.markdown("""
        <div style='background: white; padding: 1rem; border-radius: 12px;'>
        <ul style='margin: 0; padding-left: 1.5rem; color: #334155; list-style-type: none;'>
            <li style='margin-bottom: 0.5rem;'>🔍 Parameter extraction</li>
            <li style='margin-bottom: 0.5rem;'>⚠️ Abnormality detection</li>
            <li style='margin-bottom: 0.5rem;'>💡 Symptom analysis</li>
            <li>🎯 Disease prediction</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.caption("⚕️ For educational purposes only. Consult healthcare professionals for medical advice.")
    
    # Main content
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.markdown('<div class="section-header">📄 Upload Medical Report</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=['png', 'jpg', 'jpeg'], key="report_uploader", label_visibility="collapsed")
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="📋 Uploaded Report", use_container_width=True)
        else:
            st.info("👆 Upload a medical report image to begin analysis")
    
    with col2:
        st.markdown('<div class="section-header">🔧 Configuration</div>', unsafe_allow_html=True)
        
        report_type = st.selectbox(
            "Report Type",
            ["BLOOD", "URINE", "OTHER"],
            help="Select the type of medical report"
        )
        
        patient_symptoms = st.text_area(
            "Patient Symptoms (Optional)",
            placeholder="e.g., Feeling dizzy, tired, and experiencing headaches",
            help="Describe symptoms the patient is experiencing",
            height=120,
            key="patient_symptoms_input"
        )
    
    # Analyze button
    st.markdown("---")
    analyze_col1, analyze_col2, analyze_col3 = st.columns([1, 2, 1])
    with analyze_col2:
        analyze_button = st.button("🔍 Analyze Report", type="primary", use_container_width=True)
    
    if analyze_button:
        if not uploaded_file:
            st.error("❌ Please upload a medical report image")
            return
        
        if not GEMINI_API_KEY:
            st.error("❌ Gemini API key not configured. Please add it to the configuration section.")
            return
        
        # Initialize search system
        if not st.session_state.initialized:
            with st.spinner("🔄 Initializing AI system..."):
                (st.session_state.collection, 
                 st.session_state.embedding_model, 
                 st.session_state.reranker, 
                 st.session_state.bm25, 
                 st.session_state.documents, 
                 st.session_state.metadatas, 
                 st.session_state.ids) = initialize_search_system()
                
                if st.session_state.collection is not None:
                    st.session_state.initialized = True
                    st.success("✅ System initialized successfully!")
                else:
                    st.error("❌ Failed to initialize system")
                    return
        
        # Analyze report
        with st.spinner("🔬 Analyzing medical report..."):
            image = Image.open(uploaded_file)
            data = analyze_report_image(image, report_type)
            
            if not data['parameters']:
                st.error("❌ No parameters extracted from the report")
                return
            
            # Find abnormalities
            abnormal = [p for p in data['parameters'] if p['status'] != 'normal']
            normal_count = len(data['parameters']) - len(abnormal)
            
            # Display results
            st.markdown("---")
            st.markdown('<div class="section-header">📊 Analysis Results</div>', unsafe_allow_html=True)
            
            # Metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Total Parameters", len(data['parameters']), help="Total number of test parameters")
            with metric_col2:
                delta_color = "inverse" if len(abnormal) > 0 else "normal"
                st.metric("Abnormal Findings", len(abnormal), delta=None if len(abnormal) == 0 else f"{len(abnormal)}", delta_color=delta_color)
            with metric_col3:
                st.metric("Normal Results", normal_count, delta=None if normal_count == 0 else f"{normal_count}", delta_color="normal")
            
            # Visualization
            if len(data['parameters']) > 0:
                fig = go.Figure(data=[go.Pie(
                    labels=['Normal', 'Abnormal'],
                    values=[normal_count, len(abnormal)],
                    marker=dict(colors=['#10b981', '#ef4444']),
                    hole=0.4,
                    textinfo='label+percent',
                    textfont=dict(size=14, color='white'),
                    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                )])
                fig.update_layout(
                    title="Parameter Distribution",
                    height=350,
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Display abnormalities
            if abnormal:
                st.markdown('<div class="section-header">⚠️ Abnormal Findings</div>', unsafe_allow_html=True)
                
                for p in abnormal:
                    st.markdown(f"""
                    <div class="abnormal-card">
                        <strong>{p['name']}</strong>
                        <span class="status-abnormal">{p['status'].upper()}</span>
                        <br><br>
                        <small style="color: #64748b;">
                        <strong>Measured Value:</strong> {p['value']} {p['unit']}<br>
                        <strong>Normal Range:</strong> {p['referenceRange']}
                        </small>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Get symptoms from abnormalities
                with st.spinner("💭 Analyzing symptoms..."):
                    report_symptoms = get_combined_symptoms(abnormal)
                    
                st.markdown('<div class="section-header">💡 Associated Symptoms</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="symptoms-box">🩺 {report_symptoms}</div>', unsafe_allow_html=True)
            else:
                st.success("✅ All parameters are within normal range")
                report_symptoms = ""
            
            # Parameters table
            st.markdown('<div class="section-header">📋 Detailed Parameters</div>', unsafe_allow_html=True)
            df = pd.DataFrame(data['parameters'])
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "status": st.column_config.TextColumn("Status", width="small")
                }
            )
            
            # Disease prediction
            if report_symptoms or patient_symptoms:
                st.markdown("---")
                st.markdown('<div class="section-header">🎯 Disease Prediction</div>', unsafe_allow_html=True)
                
                # Combine symptoms
                if report_symptoms and patient_symptoms:
                    combined_query = f"{patient_symptoms}. {report_symptoms}"
                elif report_symptoms:
                    combined_query = report_symptoms
                else:
                    combined_query = patient_symptoms
                
                # Search for diseases
                with st.spinner("🔍 Analyzing symptoms and searching disease database..."):
                    top_diseases = search_diseases(
                        combined_query,
                        st.session_state.collection,
                        st.session_state.embedding_model,
                        st.session_state.reranker,
                        st.session_state.bm25,
                        st.session_state.documents,
                        st.session_state.metadatas,
                        st.session_state.ids
                    )
                
                st.markdown("#### 🏆 Top Predicted Conditions")
                
                # Display top diseases
                for i, disease in enumerate(top_diseases, 1):
                    st.markdown(f"""
                    <div class="disease-card">
                        <h3>#{i} {disease['name']}</h3>
                        <p style="margin-top: 1rem; color: #64748b; line-height: 1.6;">
                        {disease['document'][:300]}{'...' if len(disease['document']) > 300 else ''}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Score comparison chart
                disease_names = [d['name'] for d in top_diseases]
                rerank_scores = [d['rerank_score'] * 100 for d in top_diseases]
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=disease_names,
                        y=rerank_scores,
                        marker=dict(
                            color=rerank_scores,
                            colorscale='Blues',
                            line=dict(color='rgb(102, 126, 234)', width=2)
                        ),
                        text=[f'{score:.1f}%' for score in rerank_scores],
                        textposition='outside'
                    )
                ])
                
                # Save results
                results = {
                    "report_analysis": {
                        "total_parameters": len(data['parameters']),
                        "abnormal_count": len(abnormal),
                        "abnormalities": abnormal,
                        "report_symptoms": report_symptoms if abnormal else "All normal"
                    },
                    "patient_symptoms": patient_symptoms,
                    "top_diseases": [
                        {
                            "rank": i + 1,
                            "name": d['name'],
                            "description": d['document']
                        }
                        for i, d in enumerate(top_diseases)
                    ]
                }
                
                st.session_state.analysis_results = results
                
                # Download results
                st.markdown("---")
                download_col1, download_col2, download_col3 = st.columns([1, 2, 1])
                with download_col2:
                    st.download_button(
                        label="📥 Download Complete Analysis Report",
                        data=json.dumps(results, indent=2),
                        file_name=f"medical_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )

if __name__ == "__main__":
    main()