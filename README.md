#  Cat vs Dog Image Classification using SVM + HOG

An **interactive Machine Learning web application** that classifies images as either a **Cat or a Dog**.

This project uses **Support Vector Machine (SVM)** with **HOG (Histogram of Oriented Gradients)** feature extraction to analyze image structure and perform classification.

The model is deployed using a **Streamlit web app** where users can upload images and get real-time predictions with confidence scores.

---

# Project Highlights

*  Image upload and real-time prediction
*  HOG feature extraction for image analysis
*  SVM classification model
*  Confidence score output for predictions
*  Interactive Streamlit web interface
*  Lightweight and fast performance

---

#  Machine Learning Approach

This project follows a classical ML pipeline for image classification.

## Workflow:

1. Load dataset (cats and dogs images)
2. Preprocess images (resize + grayscale conversion)
3. Extract features using HOG (Histogram of Oriented Gradients)
4. Train SVM classifier
5. Evaluate model performance
6. Deploy using Streamlit

---

## Algorithm Used:

* Support Vector Machine (SVM)
* Kernel: RBF (Radial Basis Function)

---

#  Features

## Image Classification

Users can upload an image and the model predicts:

*  Cat
*  Dog

---

##  Confidence Score

The model provides probability scores for each class:

* Cat probability (%)
* Dog probability (%)

---

##  Real-Time Prediction

Instant prediction results after image upload.

---

#  Tech Stack

* Python
* OpenCV
* Scikit-learn
* Scikit-image (HOG)
* NumPy
* Streamlit
* Joblib

---

#  How to Run the Project

## 1 Clone the Repository

```bash
git clone https://github.com/keerthanarajr/SCT_ML_3.git
cd SCT_ML_3
```

---

## 2 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3 Train the Model

```bash
python train_model.py
```

After training, the model will be saved as:

```
models/svm_model.pkl
```

---

## 4 Run the Application

```bash
streamlit run app.py
```

The app will open in your browser automatically.

---

#  Application Features

*  Upload image
*  Preview image
* Predict Cat or Dog
* Show confidence scores

---

# Project Objective

The goal is to demonstrate how **classical machine learning can be used for image classification tasks**.

It helps understand:

* Feature extraction techniques (HOG)
* SVM classifier behavior
* Real-time ML deployment using Streamlit

---

# Limitations

* Not 100% accurate (SVM limitation)
* Sensitive to image quality and lighting
* May misclassify similar-looking images

---

#  Future Improvements

* Upgrade to CNN for higher accuracy
* Add batch image prediction
* Improve dataset quality
* Deploy online (Streamlit Cloud / HuggingFace)

---

# Support

If you like this project, give it a star on GitHub P

