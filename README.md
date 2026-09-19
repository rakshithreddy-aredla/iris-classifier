# Iris Flower Classifier 🌸

Multi-class classification model that identifies iris flower species (Setosa, Versicolor, Virginica) from sepal/petal measurements. Compares **5 different algorithms** and visualizes the decision boundary.

## 🎯 Why this project matters

The Iris dataset is the "hello world" of machine learning — but this project goes beyond a single model. It demonstrates **model comparison** and **model selection**, which is how real ML engineers actually work: try multiple algorithms, measure with cross-validation, and pick the best.

## 🔧 How it works

1. **Data** - 150 iris flowers, 4 measurements (sepal/petal length & width), 3 species
2. **Model comparison** - 5-fold cross-validation on 5 algorithms:
   - K-Nearest Neighbors
   - Decision Tree
   - Logistic Regression
   - Support Vector Machine
   - Random Forest
3. **Best model** - Re-train on full train split, evaluate on held-out test set
4. **Visualize** - Plot the decision boundary of the winning model

## 📊 Results

**Model Comparison (5-fold CV accuracy):**
```
K-Nearest Neighbors    0.9733  ← best
Decision Tree          0.9733
Logistic Regression    0.9733
Support Vector Machine 0.9667
Random Forest          0.9667
```

**Final test accuracy: 97.78%** with perfect recall on Setosa.

## 📷 Visualization

The script saves a `decision_boundary.png` showing how the model separates the three species in 2D space.

## 🧠 ML Concepts Covered

- Multi-class classification
- Cross-validation (comparing models honestly)
- Hyperparameter intuition
- Confusion matrix & per-class precision/recall
- Decision boundary visualization
- Why no single model is always "best" — need to compare

## 🚀 How to run

```bash
pip install -r requirements.txt
python iris_classifier.py
```

## 🏗️ Project Structure

```
03-iris-classifier/
├── iris_classifier.py      # Main script
├── decision_boundary.png   # Generated visualization
├── requirements.txt
└── README.md
```
