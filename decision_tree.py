import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef,
                             cohen_kappa_score, precision_recall_curve, auc, roc_curve)
import matplotlib.pyplot as plt 
from graphviz import Digraph

class TreeNode:
    def __init__ (self, feature=None, threshold=None, left=None, right=None, best_lambda = None, value=None,):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.best_lambda = best_lambda 

class DecisionTree:
    def __init__(self, max_depth=5, min_instance_ratio=0.01, candidate_lambdas = np.logspace(-2, 0, num=10, base=10.0)):
        self.max_depth = max_depth
        self.min_instance_ratio = min_instance_ratio
        self.candidate_lambdas = candidate_lambdas
        self.graph = None

    def fit(self,X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.total_observations = len(y_train)
        self.root = self.build_tree(X_train, y_train)

    # Define the purity measure (inverse Gini index)
    # Equation 7 on page 4
    def purity_measure(self, S):
        total_instances = len(S)
        class_counts = S.value_counts()
        if total_instances == 0:
            return 0
        return sum((class_counts/total_instances) ** 2)

    # Define the splitting criterion based on purity measure and lambda parameter
    # Equation 8 on page 4
    def splitting_criterion(self, S, Sl_v, Sr_v, lambda_value):
        purity_left = self.purity_measure(Sl_v)
        purity_right = self.purity_measure(Sr_v)
        sample_ratio_left = len(Sl_v) / len(S)
        sample_ratio_right = len(Sr_v) / len(S)
        score_left = purity_left * (sample_ratio_left ** lambda_value)
        score_right = purity_right * (sample_ratio_right ** lambda_value)
        return max(score_left, score_right)

    # find the best split feature, split point and lambda
    # Described in Algorithm 1 on page 6
    def find_split_point(self, X, y):
        best_lambda = None
        lambda_star = None
        best_split_point = None
        best_split_point_star = None
        best_score = -float('inf')
        
        prev_score = None
        prev_drop = None
        i = 0

        # iterate through candidate lambda values
        for lambda_val in self.candidate_lambdas:
            best_split_point = None
            current_best_score = -float('inf')

            # iterate through all columns
            for col in X.columns:
                #unique_values = X[col].unique()
                values = np.sort(X[col].unique())
                thresholds = (values[:-1] + values[1:]) / 2 # midpoints
                # test all unique values in the current column
                
                for split_point in thresholds:
                    Sl_v = y[X[col] <= split_point]
                    Sr_v = y[X[col] > split_point]
                
                    if len(Sl_v) == 0 or len(Sr_v) == 0:
                        continue # Skip invalit split
                    # Calculate the split score for this col and threshold and lambda
                
                    score = self.splitting_criterion(y, Sl_v, Sr_v, lambda_val)
                    # if the score is better than the best score so far, assign the current score to best score
                    if score > current_best_score:
                        current_best_score = score
                        best_split_point = (col, split_point)
                        #best_lambda = lambda_val

            # Evaluate Drop behavior
            # if it is the first column assing current scores to best score  
            if i == 0:
                prev_score = current_best_score
                lambda_star = lambda_val
                best_split_point_star = best_split_point 
                continue
            elif i == 1:
                drop_1 = prev_score - current_best_score
                prev_score = current_best_score
                lambda_star = lambda_val
                best_split_point_star = best_split_point
            else:
                drop_2 = prev_score - current_best_score
                if drop_2 < drop_1: # drop decreased -> stop
                    break
                else:
                    drop_1 = drop_2
                    prev_score = current_best_score
                    lambda_star = lambda_val
                    best_split_point_star = best_split_point
            i += 1
        return lambda_star, best_split_point_star

    def majority_class(self, y):
        """Returns the most common class in y"""
        return np.bincount(y).argmax()

    def build_tree (self, X, y, depth = 0):
        """Recursive function to construct the decision tree"""
        if len(set(y)) == 1 or depth > self.max_depth or len(y) == 0:
            return TreeNode(value = self.majority_class(y) if len(y) > 0 else 0)
        
        best_lambda, (feature, threshold) = self.find_split_point(X, y)
        print(f"best_lambda: {best_lambda}, feature: {feature}, threshold: {threshold} Depth: {depth}")

        if feature is None:
            return TreeNode(value = self.majority_class(y))
        
        left_mask = X[feature] <= threshold
        right_mask = ~left_mask

        if sum(left_mask) == 0 or (len(left_mask) / self.total_observations < self.min_instance_ratio):
            return TreeNode(value = self.majority_class(y))
        else:
            left_subtree = self.build_tree(X[left_mask], y[left_mask], depth+1)
        
        if sum(right_mask) == 0 or (len(right_mask) / self.total_observations < self.min_instance_ratio):
            return TreeNode(value = self.majority_class(y))
        else:
            right_subtree = self.build_tree(X[right_mask], y[right_mask], depth+1)
        
        return TreeNode(feature, threshold, left_subtree, right_subtree, best_lambda)

    def predict_samples(self, node , x):
        """Predict single sample using the tree"""
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self.predict_samples(node.left, x)
        else:
            return self.predict_samples(node.right, x)

    def predict(self, X):
        """Predict multiple samples."""
        return np.array([self.predict_samples(self.root, x) for _, x in X.iterrows()])
    
    def performance_metrics(self, X_test, y_test):
        """Compute and print various performance metrics."""
        
        # Get predictions
        y_pred = self.predict(X_test)

        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        mcc = matthews_corrcoef(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)

        # Compute ROC-AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)

        # Compute Precision-Recall AUC
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred)
        pr_auc = auc(recall_curve, precision_curve)

        # Create a DataFrame with all metrics
        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "MCC", "Cohen Kappa", "AUC-ROC", "AUC-PR"],
            "Value": [accuracy, precision, recall, f1, mcc, kappa, roc_auc, pr_auc]
        })

        # Print the metrics
        print(metrics_df)

        # Plot AUC-ROC Curve
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')

        # Plot Precision-Recall Curve
        plt.subplot(1, 2, 2)
        plt.plot(recall_curve, precision_curve, color='red', lw=2, label=f'PR AUC = {pr_auc:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')

        plt.show()

        return metrics_df

    def extract_rules(self, node=None, path=None, rules=None):
        if node is None:
            node = self.root
        if path is None:
            path = []
        if rules is None:
            rules = []
                
        # If it's a leaf node, save the rule
        if node.value is not None:
            rule = " & ".join(path) + f" => Prediction: {node.value}"
            rules.append(rule)
            return
        
        # Traverse left
        if node.left:
            self.extract_rules(node.left, path + [f" {node.feature} <= {node.threshold}"], rules)
        
        # Traverse right
        if node.right:
            self.extract_rules(node.right, path + [f" {node.feature} > {node.threshold}"], rules)
        
        return rules

    def plot_tree(self, node=None, graph=None, node_counter=[0]):
        if node is None:
            node = self.root        
        if graph is None:
            graph = Digraph(format='png')
        
        # Unique ID for the current node
        node_id = str(node_counter[0])
        node_counter[0] += 1

        if node.value is not None:  # Leaf node
            graph.node(node_id, label=f"Prediction: {node.value}", shape="box", style="filled", fillcolor="lightblue")
        else:  # Decision node
            graph.node(node_id, label=f"{node.feature} <= {node.threshold}?")
        
        # Connect left child (<= threshold)
        if node.left is not None:
            left_id = self.plot_tree(node.left, graph, node_counter)
            graph.edge(node_id, left_id, label="Y")

        # Connect right child (> threshold)
        if node.right is not None:
            right_id = self.plot_tree(node.right, graph, node_counter)
            graph.edge(node_id, right_id, label="N")
        
        return node_id  # Return the node ID for recursive linking