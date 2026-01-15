"""
Pneumonia Detection from Chest X-Ray Images - Streamlit Frontend
================================================================
A web application for detecting pneumonia from chest X-ray images
using a trained ResNet18 PyTorch model.
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os

# Configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pneumonia_results", "pneumonia_model_best.pth")
IMG_SIZE = 224
CLASSES = ['NORMAL', 'PNEUMONIA']

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection Scanner",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .result-normal {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        box-shadow: 0 10px 40px rgba(17, 153, 142, 0.3);
    }
    .result-pneumonia {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        box-shadow: 0 10px 40px rgba(235, 51, 73, 0.3);
    }
    .confidence-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    .upload-section {
        border: 3px dashed #667eea;
        border-radius: 1rem;
        padding: 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
        transition: all 0.3s ease;
    }
    .upload-section:hover {
        border-color: #764ba2;
        background: rgba(118, 75, 162, 0.1);
    }
    .stats-card {
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    .metric-label {
        color: #6c757d;
        font-size: 0.9rem;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained PyTorch model."""
    # Create model architecture (matching training script)
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    
    # Load trained weights
    if os.path.exists(MODEL_PATH):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    else:
        return None, None


def preprocess_image(image):
    """Preprocess image for model input."""
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return transform(image).unsqueeze(0)


def predict(model, device, image):
    """Make prediction on the image."""
    with torch.no_grad():
        image_tensor = preprocess_image(image).to(device)
        output = model(image_tensor).item()
        
        # output > 0.5 means PNEUMONIA (class 1)
        prediction = CLASSES[1] if output > 0.5 else CLASSES[0]
        confidence = output if output > 0.5 else 1 - output
        
        return prediction, confidence, output


def main():
    # Header
    st.markdown('<h1 class="main-header">🫁 Pneumonia Detection Scanner</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a chest X-ray image to detect pneumonia using AI</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/lung.png", width=80)
        st.markdown("## About This Model")
        st.markdown("---")
        
        st.markdown("### 📊 Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", "89.26%")
            st.metric("Precision", "86.62%")
        with col2:
            st.metric("Recall", "97.95%")
            st.metric("F1-Score", "91.94%")
        
        st.markdown("---")
        st.markdown("### 🔧 Technical Details")
        st.markdown("""
        - **Architecture**: ResNet18
        - **Framework**: PyTorch
        - **Input Size**: 224×224
        - **Training**: Transfer Learning
        - **AUC Score**: 0.9683
        """)
        
        st.markdown("---")
        st.markdown("### ⚠️ Disclaimer")
        st.warning("""
        This tool is for educational purposes only. 
        Always consult a qualified healthcare professional 
        for medical diagnosis.
        """)
    
    # Load model
    model, device = load_model()
    
    if model is None:
        st.error(f"❌ Model not found at: {MODEL_PATH}")
        st.info("Please ensure the trained model file exists in the pneumonia_results folder.")
        return
    
    # Main content
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📤 Upload X-Ray Image")
        uploaded_file = st.file_uploader(
            "Drag and drop or click to upload",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a chest X-ray image (JPG, JPEG, or PNG format)"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-Ray Image", use_container_width=True)
    
    with col2:
        st.markdown("### 🔍 Analysis Results")
        
        if uploaded_file is not None:
            with st.spinner("🔄 Analyzing X-ray image..."):
                prediction, confidence, raw_output = predict(model, device, image)
            
            # Display result
            if prediction == "NORMAL":
                st.markdown(f"""
                <div class="result-normal">
                    ✅ {prediction}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-pneumonia">
                    ⚠️ {prediction}
                </div>
                """, unsafe_allow_html=True)
            
            # Confidence display
            st.markdown(f"""
            <div class="confidence-box">
                <div style="font-size: 0.9rem; opacity: 0.9;">Confidence Score</div>
                <div style="font-size: 2.5rem; font-weight: 700;">{confidence*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Probability breakdown
            st.markdown("#### 📈 Probability Distribution")
            prob_pneumonia = raw_output
            prob_normal = 1 - raw_output
            
            st.markdown(f"**Normal**: {prob_normal*100:.1f}%")
            st.progress(prob_normal)
            
            st.markdown(f"**Pneumonia**: {prob_pneumonia*100:.1f}%")
            st.progress(prob_pneumonia)
            
            # Additional info
            st.markdown("---")
            st.markdown("#### ℹ️ Interpretation Guide")
            if prediction == "NORMAL":
                st.success("""
                The model predicts this X-ray shows **normal** lung appearance. 
                No signs of pneumonia were detected.
                """)
            else:
                st.error("""
                The model predicts signs of **pneumonia** in this X-ray. 
                Please consult a healthcare professional for proper diagnosis.
                """)
        else:
            st.info("👆 Upload a chest X-ray image to begin analysis")
            
            # Example placeholder
            st.markdown("---")
            st.markdown("#### 💡 Tips for Best Results")
            st.markdown("""
            - Use high-quality chest X-ray images
            - Ensure the image is well-lit and clear
            - Frontal (PA or AP) views work best
            - Avoid images with text overlays
            """)


if __name__ == "__main__":
    main()
