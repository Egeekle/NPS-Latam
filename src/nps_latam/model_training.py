from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
import pandas as pd

def create_logreg_pipeline(max_iter=1000, random_state=42):
    """
    Creates a standard Logistic Regression pipeline with scaling.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(max_iter=max_iter, random_state=random_state))
    ])

def run_rfecv_selection(X_train, y_train, cv=5, scoring='roc_auc', min_features=1):
    """
    Runs Recursive Feature Elimination with Cross-Validation (RFECV) using RandomForest.
    Returns the fitted selector and the ranked feature dataframe.
    """
    rfecv = RFECV(
        estimator=RandomForestClassifier(n_estimators=20, n_jobs=-1, random_state=42),
        cv=cv,
        scoring=scoring,
        min_features_to_select=min_features,
        n_jobs=-1
    )
    
    rfecv.fit(X_train, y_train)
    
    ranking_df = pd.DataFrame({
        'feature': X_train.columns,
        'selected': rfecv.support_,
        'ranking': rfecv.ranking_
    })
    
    ranking_df.sort_values(by='ranking', inplace=True)
    ranking_df.reset_index(drop=True, inplace=True)
    ranking_df.index += 1
    ranking_df.index.name = 'N°'
    
    return rfecv, ranking_df

def apply_feature_selection(selector, *datasets):
    """
    Applies the fitted RFECV selector to multiple datasets (X_train, X_valid, etc.).
    Returns DataFrame versions of the transformed data.
    """
    # Get selected feature names
    # Assuming selector was fitted on a dataframe with columns, but RFECV doesn't store feature_names_in_ robustly in all versions
    # We rely on the support mask
    
    # We need the original columns from the first dataset to map back
    if not datasets:
        return []
        
    X_ref = datasets[0]
    selected_cols = X_ref.columns[selector.support_]
    
    transformed_dfs = []
    for X in datasets:
        X_trans = selector.transform(X)
        X_trans_df = pd.DataFrame(X_trans, columns=selected_cols, index=X.index)
        transformed_dfs.append(X_trans_df)
        
    return tuple(transformed_dfs)
