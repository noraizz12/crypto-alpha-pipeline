from typing import List, Dict, Optional, Union

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from lib.util.util import log_and_raise

MIN_COEFF = 1e-10  # Minimum coefficient value to consider non-zero

# Base features from bar data used in classification
BASE_CLASSIFICATION_BAR_FEATURES = [
    'last_funding_rate_mean',
    'logret',
    'dvolume',
    'high_trade',
    'low_trade',
    'index_mid_price_diff_mean',
    'vwap',
    'logret_resid_wgtmkt',
    'logret_funding_adj'
]


# Minimum observations required for classifier training
MIN_CLASSIFICATION_OBS = 3500

# Minimum observations required for regression fitting
MIN_FITTING_OBS = 1000

# Minimum coefficient magnitude to consider non-zero
MIN_COEFF = 1e-8

def get_dep(return_type: str, lag: int, horizon: int) -> str:
    """Get the dependent variable column name for a given return type and lag.

    This function maps return type specifications to the corresponding forward
    return column names in the dataset, handling various types of returns
    including raw, market-residualized, and funding-adjusted variants.

    Args:
        return_type: Type of return to use as dependent variable. Options:
            - 'raw': Raw forward returns
            - 'resid_eq': Equal-weighted market residualized returns
            - 'resid_wgt': Volume-weighted market residualized returns
            - 'funding_adj_raw': Funding-adjusted raw returns
            - 'funding_adj_resid_eq': Funding-adjusted eq-weighted residuals
            - 'funding_adj_resid_wgt': Funding-adjusted vol-weighted residuals
        lag: Number of periods forward (0-indexed, will be converted to 1-indexed)
        horizon: Time horizon in minutes

    Returns:
        Column name for the specified forward return

    Raises:
        Exception: If return_type is not recognized

    Example:
        >>> get_dep('raw', 0, 60)
        'y_raw1_60'
        >>> get_dep('resid_wgt', 2, 1440)
        'y_resid_wgtmkt3_1440'
    """
    lag = lag + 1

    if return_type == "resid_wgt":
        dep = f'y_resid_wgtmkt{lag}_{horizon}'
    elif return_type == "resid_eq":
        dep = f'y_resid_eqmkt{lag}_{horizon}'
    elif return_type == 'raw':
        dep = f'y_raw{lag}_{horizon}'
    elif return_type == 'funding_adj_resid_wgt':
        dep = f'y_funding_adj_resid_wgtmkt{lag}_{horizon}'
    elif return_type == "funding_adj_resid_eq":
        dep = f'y_funding_adj_resid_eqmkt{lag}_{horizon}'
    elif return_type == 'funding_adj_raw':
        dep = f'y_funding_adj_raw{lag}_{horizon}'
    else:
        raise log_and_raise(f"Unknown return type {return_type}")

    return dep


def make_classification_bar_features(horizons: List[int]) -> List[str]:
    """Generate list of bar-based features for classification across horizons.

    Creates horizon-specific feature names by combining base classification
    features with each requested horizon. These features are derived from
    OHLCV bar data and used in the classifier.

    Args:
        horizons: List of time horizons in minutes

    Returns:
        List of feature names like 'last_funding_rate_mean_60', 'logret_1440', etc.
    """
    features = []
    for fld in BASE_CLASSIFICATION_BAR_FEATURES:
        features += [f"{fld}_{hh}" for hh in horizons]
    return features


def extract_feature_importances(
    classifier: Union[Pipeline, RandomForestClassifier, LinearSVC],
    only_nonzero: bool = False
) -> Dict[str, float]:
    """Extract feature importances/coefficients from a classifier.
    
    Handles both RandomForestClassifier (feature_importances_) and 
    LinearSVC (coef_) models, whether standalone or in a Pipeline.
    
    Args:
        classifier: A fitted classifier or Pipeline containing a classifier
        only_nonzero: If True, only return features with non-zero importance
        
    Returns:
        Dictionary mapping feature names to their importance values.
        For RandomForest: importance values sum to 1.0
        For LinearSVC: raw coefficient values
        
    Raises:
        ValueError: If classifier type is not supported or not fitted
    """
    # Extract the actual estimator from Pipeline if needed
    estimator = classifier
    if isinstance(classifier, Pipeline):
        # Get the final step (should be the classifier)
        estimator = classifier.steps[-1][1]
    
    # Get feature names
    feature_names = None
    if hasattr(estimator, 'feature_names_in_'):
        feature_names = estimator.feature_names_in_
    elif isinstance(classifier, Pipeline):
        # Try to get from pipeline or any step
        if hasattr(classifier, 'feature_names_in_'):
            feature_names = classifier.feature_names_in_
        else:
            for step_name, step in classifier.steps:
                if hasattr(step, 'feature_names_in_'):
                    feature_names = step.feature_names_in_
                    break
    
    # Extract importances/coefficients based on estimator type
    if isinstance(estimator, RandomForestClassifier):
        if not hasattr(estimator, 'feature_importances_'):
            raise ValueError("RandomForestClassifier is not fitted")
        importances = estimator.feature_importances_
        
    elif isinstance(estimator, LinearSVC):
        if not hasattr(estimator, 'coef_'):
            raise ValueError("LinearSVC is not fitted")
        # Handle both binary and multiclass cases
        coef = estimator.coef_[0] if estimator.coef_.ndim > 1 else estimator.coef_
        importances = coef
        
    else:
        raise ValueError(f"Unsupported classifier type: {type(estimator).__name__}")
    
    # Create the feature importance dictionary
    result = {}
    for i, importance in enumerate(importances):
        if only_nonzero and np.abs(importance) < MIN_COEFF:
            continue
        
        if feature_names is not None and i < len(feature_names):
            feature_name = feature_names[i]
        else:
            feature_name = f"feature_{i}"
        
        result[feature_name] = float(importance)
    
    return result

