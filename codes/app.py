"""
Skin Cancer Detection System - Streamlit Web Application
Modern UI with Professional Design
"""
import os
import sys
import json
import streamlit as st
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import MODEL_PATH, METRICS_PATH, CLASS_NAMES, IMAGE_SIZE, PREDICTION_THRESHOLD
from src.preprocessing import preprocess_uploaded_image

# TensorFlow import (lazy loading)
import tensorflow as tf


# Page configuration
st.set_page_config(
    page_title="Skin Cancer Detection System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Modern CSS Styles
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main Header - Glassmorphism */
    .main-header {
        text-align: center;
        padding: 2.5rem 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    /* Glass Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* Info Card */
    .info-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #60a5fa;
        box-shadow: 0 4px 20px rgba(30, 58, 95, 0.3);
    }
    
    .info-card h4 {
        color: #93c5fd;
        margin: 0 0 1rem 0;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .info-card ol {
        color: #e2e8f0;
        margin: 0;
        padding-left: 1.2rem;
    }
    
    .info-card li {
        margin: 0.5rem 0;
        line-height: 1.6;
    }
    
    /* Result Cards */
    .result-card {
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .result-melanoma {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        box-shadow: 0 10px 40px rgba(220, 38, 38, 0.4);
    }
    
    .result-benign {
        background: linear-gradient(135deg, #16a34a 0%, #15803d 100%);
        box-shadow: 0 10px 40px rgba(22, 163, 74, 0.4);
    }
    
    .result-card h2 {
        color: white;
        font-size: 1.8rem;
        margin: 0;
        font-weight: 600;
    }
    
    .result-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    .confidence-score {
        font-size: 3.5rem;
        font-weight: 700;
        color: white;
        margin: 1rem 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    
    .confidence-label {
        color: rgba(255,255,255,0.8);
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Warning Card */
    .warning-card {
        background: linear-gradient(135deg, #78350f 0%, #92400e 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border-left: 4px solid #fbbf24;
        box-shadow: 0 4px 20px rgba(120, 53, 15, 0.3);
    }
    
    .warning-card h4 {
        color: #fde68a;
        margin: 0 0 0.8rem 0;
        font-weight: 600;
    }
    
    .warning-card p {
        color: #fef3c7;
        margin: 0.5rem 0;
        line-height: 1.6;
    }
    
    /* Low Confidence Warning Card */
    .low-confidence-card {
        background: linear-gradient(135deg, #991b1b 0%, #7f1d1d 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border-left: 4px solid #ef4444;
        box-shadow: 0 4px 20px rgba(153, 27, 27, 0.3);
    }
    
    .low-confidence-card h4 {
        color: #fca5a5;
        margin: 0 0 0.8rem 0;
        font-weight: 600;
    }
    
    .low-confidence-card p {
        color: #fee2e2;
        margin: 0.5rem 0;
        line-height: 1.6;
    }
    
    /* Upload Section */
    .upload-section {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 2px dashed #475569;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #667eea;
        box-shadow: 0 0 30px rgba(102, 126, 234, 0.2);
    }
    
    /* Stats Card */
    .stats-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        text-align: center;
        border: 1px solid #334155;
    }
    
    .stats-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #60a5fa;
    }
    
    .stats-label {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Image Container */
    .image-container {
        background: #1e293b;
        border-radius: 16px;
        padding: 1rem;
        border: 1px solid #334155;
    }
    
    .image-label {
        color: #94a3b8;
        font-size: 0.9rem;
        text-align: center;
        margin-top: 0.5rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #64748b;
        border-top: 1px solid #334155;
        margin-top: 2rem;
    }
    
    .footer p {
        margin: 0.3rem 0;
    }
    
    /* Sidebar Styling */
    .sidebar-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    
    .sidebar-header h3 {
        color: white;
        margin: 0;
        font-size: 1.2rem;
    }
    
    .sidebar-stat {
        background: #1e293b;
        border-radius: 10px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border: 1px solid #334155;
    }
    
    .sidebar-stat-label {
        color: #94a3b8;
        font-size: 0.75rem;
        text-transform: uppercase;
    }
    
    .sidebar-stat-value {
        color: #60a5fa;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    /* Progress Bar Custom */
    .custom-progress {
        background: #1e293b;
        border-radius: 10px;
        height: 12px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .custom-progress-bar {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    .progress-melanoma {
        background: linear-gradient(90deg, #ef4444 0%, #dc2626 100%);
    }
    
    .progress-benign {
        background: linear-gradient(90deg, #22c55e 0%, #16a34a 100%);
    }
    
    /* Animations */
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .pulse-animation {
        animation: pulse 2s infinite;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 12px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: #1e293b;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Loads and caches the model."""
    if not os.path.exists(MODEL_PATH):
        return None
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


@st.cache_data
def load_metrics():
    """Loads metrics from JSON file."""
    if not os.path.exists(METRICS_PATH):
        # Default metrics (if file is missing)
        return {
            'test_accuracy': 0.0,
            'test_auc': 0.0,
            'melanoma': {
                'precision': 0.0,
                'recall': 0.0
            }
        }
    
    try:
        with open(METRICS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Metrics could not be loaded: {e}")
        return {
            'test_accuracy': 0.0,
            'test_auc': 0.0,
            'melanoma': {
                'precision': 0.0,
                'recall': 0.0
            }
        }


def predict_image(model, image_array: np.ndarray) -> tuple:
    """Performs prediction on an image."""
    prediction = model.predict(image_array, verbose=0)
    probability = float(prediction[0][0])
    
    # Classification using threshold (default: 0.3 - balanced)
    if probability > PREDICTION_THRESHOLD:
        class_name = "Melanoma"
        confidence = probability * 100
    else:
        class_name = "Benign"
        confidence = (1 - probability) * 100
    
    return class_name, confidence, probability


def display_result(class_name: str, confidence: float, probability: float):
    """Displays prediction result visually."""
    
    if class_name == "Melanoma":
        # Low confidence check (below 50%)
        is_low_confidence = confidence < 50.0
        
        st.markdown(f"""
        <div class="result-card result-melanoma fade-in">
            <div class="result-icon">⚠️</div>
            <h2>Result: MELANOMA</h2>
            <div class="confidence-score">{confidence:.1f}%</div>
            <div class="confidence-label">Confidence Score</div>
        </div>
        """, unsafe_allow_html=True)
        
        if is_low_confidence:
            # Special warning for low confidence
            st.markdown("""
            <div class="low-confidence-card fade-in">
                <h4>⚠️ LOW CONFIDENCE SCORE - SUSPECT RESULT</h4>
                <p>This prediction has a <strong>low confidence score</strong> (below 50%).</p>
                <p>The result may be <strong>suspect</strong> and has a higher chance of false positive.</p>
                <p><strong>IMPORTANT:</strong> Please consult a <strong>dermatologist</strong> for professional evaluation.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Standard warning for high confidence
            st.markdown("""
            <div class="warning-card fade-in">
                <h4>⚠️ WARNING - URGENT NOTICE</h4>
                <p>This lesion has been detected as potential <strong>melanoma</strong>.</p>
                <p>Please consult a <strong>dermatologist</strong> as soon as possible.</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card result-benign fade-in">
            <div class="result-icon">✅</div>
            <h2>Result: BENIGN</h2>
            <div class="confidence-score">{confidence:.1f}%</div>
            <div class="confidence-label">Confidence Score</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.success("✅ This lesion is detected as **benign**.")
    
    # Detailed Analysis
    st.markdown("### 📊 Detailed Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-value">{probability * 100:.1f}%</div>
            <div class="stats-label">Melanoma Probability</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-value">{(1 - probability) * 100:.1f}%</div>
            <div class="stats-label">Benign Probability</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Custom Progress Bar
    st.markdown("#### Probability Distribution")
    
    melanoma_width = probability * 100
    benign_width = (1 - probability) * 100
    
    st.markdown(f"""
    <div style="display: flex; gap: 1rem; align-items: center; margin: 1rem 0;">
        <span style="color: #ef4444; font-weight: 600; width: 80px;">Melanoma</span>
        <div class="custom-progress" style="flex: 1;">
            <div class="custom-progress-bar progress-melanoma" style="width: {melanoma_width}%;"></div>
        </div>
        <span style="color: #94a3b8; width: 50px;">{melanoma_width:.1f}%</span>
    </div>
    <div style="display: flex; gap: 1rem; align-items: center; margin: 1rem 0;">
        <span style="color: #22c55e; font-weight: 600; width: 80px;">Benign</span>
        <div class="custom-progress" style="flex: 1;">
            <div class="custom-progress-bar progress-benign" style="width: {benign_width}%;"></div>
        </div>
        <span style="color: #94a3b8; width: 50px;">{benign_width:.1f}%</span>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Renders sidebar content."""
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h3>🔬 System Info</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🤖 Model Details")
        
        st.markdown("""
        <div class="sidebar-stat">
            <div class="sidebar-stat-label">Model Architecture</div>
            <div class="sidebar-stat-value">MobileNetV2</div>
        </div>
        <div class="sidebar-stat">
            <div class="sidebar-stat-label">Transfer Learning</div>
            <div class="sidebar-stat-value">ImageNet</div>
        </div>
        <div class="sidebar-stat">
            <div class="sidebar-stat-label">Input Size</div>
            <div class="sidebar-stat-value">224 x 224</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 📊 Performance")
        
        # Load metrics from JSON
        metrics = load_metrics()
        
        accuracy = metrics.get('test_accuracy', 0.0) * 100
        auc_score = metrics.get('test_auc', 0.0) * 100
        melanoma_precision = metrics.get('melanoma', {}).get('precision', 0.0) * 100
        melanoma_recall = metrics.get('melanoma', {}).get('recall', 0.0) * 100
        
        st.markdown(f"""
        <div class="sidebar-stat">
            <div class="sidebar-stat-label">Accuracy</div>
            <div class="sidebar-stat-value">{accuracy:.1f}%</div>
        </div>
        <div class="sidebar-stat">
            <div class="sidebar-stat-label">AUC Score</div>
            <div class="sidebar-stat-value">{auc_score:.1f}%</div>
        </div>
        <div class="sidebar-stat">
            <div class="sidebar-stat-label">Precision (Melanoma)</div>
            <div class="sidebar-stat-value">{melanoma_precision:.1f}%</div>
        </div>
        <div class="sidebar-stat">
            <div class="sidebar-stat-label">Recall (Melanoma)</div>
            <div class="sidebar-stat-value">{melanoma_recall:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Last updated date
        last_updated = metrics.get('last_updated', 'Unknown')
        if last_updated != 'Unknown':
            st.caption(f"📅 Last updated: {last_updated}")
        
        st.markdown("---")
        
        st.markdown("### 📁 Dataset")
        st.info("**HAM10000**\n\n~10,000 dermoscopic images")
        
        st.markdown("---")
        
        st.markdown("### ℹ️ About")
        st.caption("""
        This system is a semester project developed for the Medical Informatics course.
        
        **Technologies:**
        - TensorFlow / Keras
        - Streamlit
        - Python
        """)


def main():
    """Main application function."""
    
    # Sidebar
    render_sidebar()
    
    # Main Header
    st.markdown("""
    <div class="main-header">
        <h1>🔬 Skin Cancer Detection System</h1>
        <p>AI-Powered Dermoscopic Image Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("❌ **Model not found!**")
        st.markdown("""
        <div class="warning-card">
            <h4>⚠️ Model File Not Available</h4>
            <p>Please train the model first:</p>
            <p><code>python src/train.py</code></p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    # Ana içerik - 2 kolon
    col_left, col_right = st.columns([1, 1], gap="large")
    
    with col_left:
        # Info card
        st.markdown("""
        <div class="info-card">
            <h4>📋 How to Use?</h4>
            <ol>
                <li>Upload a dermoscopic skin image below</li>
                <li>Click <strong>\"🔬 Analyze\"</strong></li>
                <li>AI will analyze the image</li>
                <li>Result and confidence score will be displayed</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Image upload
        st.markdown("### 📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Select a dermoscopic skin image",
            type=['jpg', 'jpeg', 'png'],
            help="For best results, use high-quality dermoscopic images.",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            st.markdown("""
            <div class="image-container">
            """, unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown(f"""
            <div class="image-label">📷 Uploaded Image ({image.size[0]}x{image.size[1]})</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col_right:
        if uploaded_file is not None:
            # Processed image
            st.markdown("### 🔍 Processed Image")
            
            processed_preview = image.resize(IMAGE_SIZE)
            
            st.markdown("""
            <div class="image-container">
            """, unsafe_allow_html=True)
            st.image(processed_preview, use_container_width=True)
            st.markdown(f"""
            <div class="image-label">Model Input ({IMAGE_SIZE[0]}x{IMAGE_SIZE[1]})</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Analyze button
            if st.button("🔬 Analyze", type="primary", use_container_width=True):
                with st.spinner("🔄 Analyzing image..."):
                    uploaded_file.seek(0)
                    processed_image = preprocess_uploaded_image(uploaded_file)
                    class_name, confidence, probability = predict_image(model, processed_image)
                
                # Sonuç
                st.markdown("---")
                display_result(class_name, confidence, probability)
        else:
            # Placeholder
            st.markdown("### 🔍 Preview")
            st.markdown("""
            <div style="
                background: #1e293b;
                border: 2px dashed #475569;
                border-radius: 16px;
                padding: 4rem 2rem;
                text-align: center;
                color: #64748b;
            ">
                <p style="font-size: 3rem; margin: 0;">📷</p>
                <p style="margin: 1rem 0 0 0;">Preview will appear here<br>after uploading an image</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Warning card
    st.markdown("""
    <div class="warning-card">
        <h4>⚠️ IMPORTANT WARNING</h4>
        <p>This system is <strong>for educational and research purposes only</strong>. 
        It must <strong>not</strong> be used for medical diagnosis or treatment decisions.</p>
        <p>If you have any concern about a skin lesion, please consult a 
        <strong>dermatologist or healthcare professional</strong>.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p><strong>Medical Informatics Course</strong> - Term Project</p>
        <p>MobileNetV2 + Transfer Learning | HAM10000 Dataset | TensorFlow</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
