# Decision Trees and Random Forests

## Internship Task 5 – ElevateLabs

### Project Overview
This project implements Decision Tree and Random Forest classification algorithms using the Breast Cancer Wisconsin dataset.

The objective is to understand decision tree learning, analyze overfitting, visualize decision trees, compare model performance, and evaluate feature importance.

### Dataset
- Dataset: Breast Cancer Wisconsin Dataset
- Total Samples: 569
- Total Features: 30
- Classes: Malignant and Benign

### Technologies Used
- Python
- Scikit-learn
- Graphviz
- NumPy
- Pandas
- Jupyter Notebook

### Implementation
1. Loaded and explored the dataset.
2. Split the dataset into training and testing sets.
3. Trained a Decision Tree Classifier.
4. Evaluated training and testing accuracy.
5. Analyzed overfitting using different tree depths.
6. Visualized the Decision Tree using Graphviz.
7. Trained a Random Forest Classifier.
8. Compared Decision Tree and Random Forest accuracy.
9. Calculated feature importance.
10. Performed 5-Fold Cross-Validation.

### Results

| Model | Testing Accuracy |
|---|---:|
| Decision Tree | 91.23% |
| Random Forest | 95.61% |

Average 5-Fold Cross-Validation Accuracy: 95.61%

### Top Important Features
- worst area
- worst concave points
- worst radius
- mean concave points
- worst perimeter

### Project Structure

```text
ElevateLabs_Task5_Decision_Trees_Random_Forests/
│
├── task5_decision_trees.ipynb
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── outputs/
    ├── decision_tree.png
    ├── feature_importance.csv
    └── cross_validation_results.csv
```

### How to Run

1. Clone or download this repository.
2. Install the required libraries:

```bash
pip install -r requirements.txt
```

3. Install Graphviz separately and configure its PATH.
4. Open `task5_decision_trees.ipynb` in Jupyter Notebook or VS Code.
5. Select the appropriate Python kernel.
6. Execute the notebook cells sequentially.

### Conclusion

The Random Forest model achieved 95.61% testing accuracy, outperforming the Decision Tree model, which achieved 91.23%.

This project demonstrates the implementation and evaluation of tree-based machine learning algorithms.