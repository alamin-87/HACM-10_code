import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from scipy import stats


def train_chbm(X, y, feature_names, seed=42):
    """
    Trains the C-HBM as logistic regression, reporting odds ratios with
    Wald p-values. Features are standardized so odds ratios are comparable.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=seed)
    model.fit(X_scaled, y)

    auc_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')

    coefs = model.coef_[0]
    odds_ratios = np.exp(coefs)

    probs = model.predict_proba(X_scaled)[:, 1]
    W = np.diag(probs * (1 - probs))
    X_design = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])
    try:
        cov = np.linalg.inv(X_design.T @ W @ X_design)
        std_errs = np.sqrt(np.diag(cov))[1:]
        z_scores = coefs / std_errs
        p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
    except np.linalg.LinAlgError:
        std_errs = np.full_like(coefs, np.nan)
        z_scores = np.full_like(coefs, np.nan)
        p_values = np.full_like(coefs, np.nan)

    results = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefs,
        'odds_ratio': odds_ratios,
        'std_error': std_errs,
        'z_score': z_scores,
        'p_value': p_values,
    })
    return model, scaler, results, auc_scores
