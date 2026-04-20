import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

def train_decision_tree(X_train, y_train, max_depth=5, random_state=42):
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def get_feature_importances(model, feature_names):
    importances = model.feature_importances_
    importance_dict = dict(zip(feature_names, importances))
    sorted_importance_dict = dict(
        sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)
    )
    return sorted_importance_dict
def train_balanced_forest(X_train, y_train, X_test, y_test,
                          n_estimators=100, random_state=42):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
        max_depth=5
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0)
    }

    return metrics
if __name__ == "__main__":
    df = pd.read_csv("data/telecom_churn.csv")
    features = ["tenure", "monthly_charges", "total_charges",
                "num_support_calls", "senior_citizen", "has_partner",
                "has_dependents", "contract_months"]
    X = df[features]
    y = df["churned"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Task 1
    tree = train_decision_tree(X_train, y_train)
    if tree:
        print(f"Decision tree trained, depth={tree.get_depth()}")

    # Task 2
    if tree:
        importances = get_feature_importances(tree, features)
        if importances:
            print(f"Top features: {list(importances.items())[:3]}")

    # Task 3
    metrics = train_balanced_forest(X_train, y_train, X_test, y_test)
    if metrics:
        print(f"Balanced RF: {metrics}")