import streamlit as st
import joblib
import cv2
import numpy as np
from skimage.feature import hog

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Dog/Cat Image Classifier",
    layout="wide"
)

# ---------------- CUSTOM CSS (SPOTIFY VIBE) ----------------
st.markdown("""
<style>
/* Background */
body {
    background-color: #0b0c10;
}

/* Main container */
.main {
    background: linear-gradient(180deg, #0b0c10, #111827);
    color: white;
}

/* Titles */
h1, h2, h3 {
    color: #3b82f6;
}

/* Buttons */
.stButton>button {
    background-color: #1f2937;
    color: white;
    border-radius: 10px;
    border: 1px solid #3b82f6;
}
.stButton>button:hover {
    background-color: #3b82f6;
    color: black;
}

/* Progress bars */
.stProgress > div > div {
    background-color: #3b82f6;
}

/* File uploader */
.stFileUploader {
    border: 1px dashed #3b82f6;
    padding: 10px;
    border-radius: 10px;
}

/* Cards */
.card {
    padding: 20px;
    border-radius: 15px;
    background: #111827;
    box-shadow: 0px 0px 20px rgba(59,130,246,0.2);
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")
IMG_SIZE = 64

# ---------------- HEADER ----------------
st.markdown("""
<div class="card">
    <h1>Dog/Cat Image Classifier</h1>
    <p>Upload an image and get instant prediction with confidence scores.</p>
</div>
""", unsafe_allow_html=True)

st.write("")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])


# ---------------- FEATURE EXTRACTION ----------------
def extract_features(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    return hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True
    )


# ---------------- MAIN LOGIC ----------------
if uploaded_file is not None:

    col1, col2 = st.columns([1, 1])

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # LEFT SIDE (IMAGE)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📸 Uploaded Image")
        st.image(img_rgb, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # FEATURE + PREDICTION
    features = extract_features(img)
    proba = model.predict_proba([features])[0]

    cat_prob = float(proba[0])
    dog_prob = float(proba[1])

    # RIGHT SIDE (RESULT)
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎯 Prediction")

        if cat_prob > dog_prob:
            st.success("🐱 Cat Detected")
        else:
            st.success("🐶 Dog Detected")

        st.write("### Confidence Levels")

        st.progress(cat_prob)
        st.caption(f"Cat: {cat_prob*100:.2f}%")

        st.progress(dog_prob)
        st.caption(f"Dog: {dog_prob*100:.2f}%")

        st.markdown('</div>', unsafe_allow_html=True)

    # INFO SECTION
    with st.expander("⚙️ How it works"):
        st.markdown("""
        - Image → Grayscale  
        - Feature Extraction using HOG  
        - SVM Model Prediction  
        - Confidence Scores Displayed  
        """)