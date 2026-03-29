"""
ensemble_models.py

This module provides functions to build various ensemble models using scikit-learn.
Available models include Random Forest, Gradient Boosting, and Voting Classifier.
"""

from typing import Any, Dict, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def build_random_forest(X: Any, y: Any, n_estimators: int = 100, test_size: float = 0.2, random_state: int = None) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Build and train a Random Forest classifier.

    Parameters:
    X : Any
        Feature dataset.
    y : Any
        Target labels.
    n_estimators : int
        The number of trees in the forest (default is 100).
    test_size : float
        Proportion of the dataset to include in the test split (default is 0.2).
    random_state : int
        Controls the randomness of the estimator (default is None).

    Returns:
    Tuple[RandomForestClassifier, Dict[str, Any]]
        The trained Random Forest model and its performance metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    return model, {'accuracy': accuracy}


def build_gradient_boosting(X: Any, y: Any, n_estimators: int = 100, learning_rate: float = 0.1, test_size: float = 0.2, random_state: int = None) -> Tuple[GradientBoostingClassifier, Dict[str, Any]]:
    """
    Build and train a Gradient Boosting classifier.

    Parameters:
    X : Any
        Feature dataset.
    y : Any
        Target labels.
    n_estimators : int
        The number of boosting stages to be run (default is 100).
    learning_rate : float
        The learning rate shrinks the contribution of each tree (default is 0.1).
    test_size : float
        Proportion of the dataset to include in the test split (default is 0.2).
    random_state : int
        Controls the randomness of the estimator (default is None).

    Returns:
    Tuple[GradientBoostingClassifier, Dict[str, Any]]
        The trained Gradient Boosting model and its performance metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, {'accuracy': accuracy}


def build_voting_classifier(models: Dict[str, ClassifierMixin], X: Any, y: Any, test_size: float = 0.2, voting: str = 'hard', random_state: int = None) -> Tuple[VotingClassifier, Dict[str, Any]]:
    """
    Build and train a Voting Classifier using the provided models.

    Parameters:
    models : Dict[str, ClassifierMixin]
        A dictionary where keys are model names and values are classifier instances.
    X : Any
        Feature dataset.
    y : Any
        Target labels.
    test_size : float
        Proportion of the dataset to include in the test split (default is 0.2).
    voting : str
        Voting mechanism to use ('hard' or 'soft', default is 'hard').
    random_state : int
        Controls the randomness of the estimator (default is None).

    Returns:
    Tuple[VotingClassifier, Dict[str, Any]]
        The trained Voting Classifier model and its performance metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    voting_classifier = VotingClassifier(estimators=list(models.items()), voting=voting)
    voting_classifier.fit(X_train, y_train)

    predictions = voting_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return voting_classifier, {'accuracy': accuracy}