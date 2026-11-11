import csv
import os

import numpy as np
import pandas as pd
import graphviz

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Create outputs folder
os.makedirs("outputs", exist_ok=True)


# 1. Load Dataset
data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset loaded successfully!")
print("Dataset Shape:", X.shape)


# 2. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# 3. Train Decision Tree
decision_tree = DecisionTreeClassifier(random_state=42)

decision_tree.fit(X_train, y_train)

y_train_pred = decision_tree.predict(X_train)
y_test_pred = decision_tree.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("\nDecision Tree Training Accuracy:", train_accuracy * 100)
print("Decision Tree Testing Accuracy:", test_accuracy * 100)


# 4. Analyze Overfitting
print("\nDecision Tree Overfitting Analysis")

depths = [1, 2, 3, 4, 5, 10, 15, 20, None]

for depth in depths:

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    print(
        f"Depth: {depth} | "
        f"Training Accuracy: {model.score(X_train, y_train)*100:.2f}% | "
        f"Testing Accuracy: {model.score(X_test, y_test)*100:.2f}%"
    )


# 5. Visualize Decision Tree
dot_data = export_graphviz(
    decision_tree,
    out_file=None,
    feature_names=data.feature_names,
    class_names=data.target_names,
    filled=True,
    rounded=True,
    special_characters=True
)

graph = graphviz.Source(dot_data)

graph.render(
    "outputs/decision_tree",
    format="png",
    cleanup=True
)

print("\nDecision Tree visualization saved!")


# 6. Train Random Forest
random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest.fit(X_train, y_train)

rf_train_pred = random_forest.predict(X_train)
rf_test_pred = random_forest.predict(X_test)

rf_train_accuracy = accuracy_score(y_train, rf_train_pred)
rf_test_accuracy = accuracy_score(y_test, rf_test_pred)

print("\nRandom Forest Training Accuracy:", rf_train_accuracy * 100)
print("Random Forest Testing Accuracy:", rf_test_accuracy * 100)


# 7. Model Comparison
print("\nMODEL ACCURACY COMPARISON")

print(f"Decision Tree: {test_accuracy*100:.2f}%")
print(f"Random Forest: {rf_test_accuracy*100:.2f}%")


# 8. Feature Importance
feature_importance = random_forest.feature_importances_

sorted_features = sorted(
    zip(data.feature_names, feature_importance),
    key=lambda x: x[1],
    reverse=True
)

with open("outputs/feature_importance.csv", "w", newline="") as file:

    writer = csv.writer(file)
    writer.writerow(["Feature", "Importance"])
    writer.writerows(sorted_features)

print("\nFeature importance saved!")


# 9. Cross Validation
cv_scores = cross_val_score(
    random_forest,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\nCross Validation Scores:", cv_scores)
print("Average Cross Validation Accuracy:", cv_scores.mean() * 100)


# Save Cross Validation Results
with open("outputs/cross_validation_results.csv", "w", newline="") as file:

    writer = csv.writer(file)
    writer.writerow(["Fold", "Accuracy"])

    for i, score in enumerate(cv_scores, start=1):
        writer.writerow([f"Fold {i}", score])

    writer.writerow(["Average Accuracy", cv_scores.mean()])


# 10. Final Summary
print("\n===================================")
print(" FINAL PROJECT SUMMARY")
print("===================================")

print(f"Decision Tree Accuracy: {test_accuracy*100:.2f}%")
print(f"Random Forest Accuracy: {rf_test_accuracy*100:.2f}%")
print(f"Cross Validation Accuracy: {cv_scores.mean()*100:.2f}%")

print("\nProject execution completed successfully!")