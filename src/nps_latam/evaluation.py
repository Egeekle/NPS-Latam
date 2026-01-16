from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score

def get_model_metrics(model, X_train, y_train, X_valid, y_valid):
    """
    Calculates performance metrics for a model on training and validation sets.
    Returns AUC, Accuracy, and F1 score for both sets.
    """
    y_pred_train = model.predict(X_train)
    y_pred_valid = model.predict(X_valid)

    # Handle cases where model might not have predict_proba (e.g. SVM without probability=True)
    if hasattr(model, "predict_proba"):
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_valid_proba = model.predict_proba(X_valid)[:, 1]
        auc_train = roc_auc_score(y_train, y_train_proba)
        auc_valid = roc_auc_score(y_valid, y_valid_proba)
    else:
        # Fallback if no probabilities are available
        auc_train = None
        auc_valid = None

    acc_train = accuracy_score(y_train, y_pred_train)
    f1_train = f1_score(y_train, y_pred_train)

    acc_valid = accuracy_score(y_valid, y_pred_valid)
    f1_valid = f1_score(y_valid, y_pred_valid)

    return auc_train, auc_valid, acc_train, acc_valid, f1_train, f1_valid

def get_cv_metrics(model, X, y, cv=5, scoring='f1'):
    """
    Calculates Cross-Validation metrics.
    """
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return cv_scores.mean(), cv_scores.std()
