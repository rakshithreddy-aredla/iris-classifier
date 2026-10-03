# Iris classification, five ways

Every iris tutorial fits one model to one dataset and calls it done. This one fits five, scores them honestly with cross-validation, and picks a winner.

```
KNN                 0.9733
Decision Tree       0.9733
Logistic Regression 0.9733
SVM                 0.9667
Random Forest       0.9667
```

Three models tie at the top, which is the more interesting result: on four features and 150 samples, iris is saturated. The dataset separates cleanly and the interesting structure runs out before the models do.

The winning model is retrained on the full training split and evaluated once on the held-out test set — **97.78%**, with perfect recall on Setosa. The script also writes `decision_boundary.png` showing how the classes actually separate in 2D.

```bash
pip install -r requirements.txt
python iris_classifier.py
```

## Files

```
iris_classifier.py      # CV comparison, final fit, evaluation, plot
decision_boundary.png   # generated
requirements.txt
```