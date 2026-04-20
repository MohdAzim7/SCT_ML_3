import os
import cv2
import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from skimage.feature import hog

DATASET_PATH = "datasets/train"
IMG_SIZE = 64
MAX_IMAGES = 1500

data = []
labels = []

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

print("Loading data...")

for category in ["cats", "dogs"]:
    folder = os.path.join(DATASET_PATH, category)
    label = 0 if category == "cats" else 1

    files = os.listdir(folder)[:MAX_IMAGES]

    for file in files:
        path = os.path.join(folder, file)
        img = cv2.imread(path)

        if img is None:
            continue

        data.append(extract_features(img))
        labels.append(label)

X = np.array(data)
y = np.array(labels)

print("Dataset size:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = SVC(
    kernel="rbf",
    C=3,
    gamma="scale",
    probability=True
)

model.fit(X_train, y_train)

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)

joblib.dump(model, "model.pkl")

print("Model saved!")