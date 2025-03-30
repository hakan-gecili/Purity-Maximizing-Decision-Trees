# Purity-Maximizing-Decision-Trees

### Decision Tree Implementation Based on "A New Splitting Criterion for Better Interpretable Trees"

This repository contains an implementation of a decision tree algorithm, inspired by the article **"A New Splitting Criterion for Better Interpretable Trees"**. The focus of the algorithm is to improve the interpretability of decision trees by utilizing a novel splitting criterion that emphasizes both purity and interpretability.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Performance Metrics](#performance-metrics)
- [Tree Visualization](#tree-visualization)
- [Contributing](#contributing)
- [License](#license)

## Overview

The algorithm in this repository implements a decision tree classifier with an innovative splitting criterion, derived from the concepts discussed in the article [A New Splitting Criterion for Better Interpretable Trees](https://ieeexplore.ieee.org/document/9054987). The splitting criterion is based on **purity measures** combined with a **lambda parameter**, offering a more interpretable and robust model. The tree construction process is recursive, and nodes are split based on the best lambda and threshold.

## Features

- **Custom Splitting Criterion**: Implements the splitting criterion based on the inverse Gini index, as discussed in the article.

- **Tree Construction**: A decision tree is built recursively using the custom criterion, which allows for improved interpretability.

- **Performance Metrics**: The model evaluates various metrics such as **accuracy**, **precision**, **recall**, **F1 score**, **MCC**, and **Cohen Kappa score**.

- **AUC-ROC and AUC-PR Plotting**: The performance of the model is visualized using **AUC-ROC** and **AUC-PR** curves.

- **Rule Extraction**: Extracts decision rules from leaf nodes to root nodes, connected with AND logic.

## Installation

To install this repository and use the code, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/decision-tree-interpretability.git
    ```
2. Install Required Libraries:

   a. pandas

   b. numpy

   c. scikit-learn

   d. matplotlib

   e. graphviz

   f. jupyter (for visualization)

## Usage

Once the repository is installed, you can use the DecisionTree class to train and evaluate the model.

## Example

```python
from decision_tree import DecisionTree
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# Load data and split into train test
dt = load_breast_cancer()
X = pd.DataFrame(dt.data, columns= list(dt.feature_names))
y = pd.Series(dt.target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

# Initialize a DecisionTree object
tree = DecisionTree(max_depth= 5)

# Train the tree
tree.fit(X_train, y_train)

# Make predictions
predictions = tree.predict(X_test)

# Evaluate performance
metrics_df = tree.performance_metrics(X_test, y_test)

# Extract rules
rules = tree.extract_rules()
for rule in rules:
    print(rule)

# Plot tree
graph = Digraph(format='png')
_ = tree.plot_tree(graph=graph)
display(graph)

```

## Performance Metrics

The following metrics are calculated and displayed:

- **Accuracy**: The proportion of correct predictions.

- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.

- **Recall**: The ratio of correctly predicted positive observations to all observations in actual class.

- **F1 Score**: The weighted average of precision and recall.

- **MCC (Matthews Correlation Coefficient)**: Measures the quality of binary classifications.

- **Cohen's Kappa Score**: Measures the agreement between two raters.

- **AUC-ROC**: The area under the receiver operating characteristic curve.

- **AUC-PR**: The area under the precision-recall curve.

These metrics are printed and visualized with plots.

## Tree Visualization

The plot_tree() method visualizes the decision tree using Graphviz. It generates a graphical representation of the decision tree structure, which is useful for understanding how the model makes decisions.

## Example Output:

A tree structure showing decision nodes and leaf nodes with predictions.
AUC-ROC curve and Precision-Recall curve for model performance evaluation.
Contributing

## Contributing 

Contributions are welcome! If you'd like to contribute to this project, feel free to open issues or create pull requests.

### How to contribute:

1. Fork the repository.

2. Create a branch (git checkout -b feature-branch).

3. Commit your changes (git commit -am 'Add new feature').

4. Push to the branch (git push origin feature-branch).

5. Open a pull request.

## License

This project is licensed under the MIT License
