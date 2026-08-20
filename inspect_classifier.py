#!/usr/bin/env python3
"""
Inspect classifier pipelines to extract features and their importances.

Usage: python inspect_classifier.py <classifier_file.joblib>
"""

import argparse
import sys
from pathlib import Path
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from lib.fits.fit_util import extract_feature_importances


def inspect_classifier(classifier_path: str) -> None:
    """Load and inspect a classifier pipeline to extract features and weights."""
    
    # Load the classifier
    try:
        classifier = joblib.load(classifier_path)
    except Exception as e:
        print(f"Error loading classifier: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"\nClassifier loaded from: {classifier_path}")
    print("=" * 80)
    
    # Extract classifier type
    if isinstance(classifier, Pipeline):
        print(f"Pipeline steps: {[name for name, _ in classifier.steps]}")
        estimator = classifier.steps[-1][1]
        estimator_name = classifier.steps[-1][0]
        print(f"Final estimator: {estimator_name} ({type(estimator).__name__})")
    else:
        estimator = classifier
        print(f"Direct classifier: {type(estimator).__name__}")
    
    print("-" * 80)
    
    # Extract feature importances using the function from fit_util
    try:
        feature_importances = extract_feature_importances(classifier, only_nonzero=False)
    except ValueError as e:
        print(f"Error extracting features: {e}")
        print("=" * 80)
        return
    
    # Sort features by absolute importance
    sorted_features = sorted(
        feature_importances.items(), 
        key=lambda x: abs(x[1]), 
        reverse=True
    )
    
    # Display summary statistics
    n_features = len(feature_importances)
    n_nonzero = sum(1 for _, val in feature_importances.items() if val != 0)
    
    print(f"\nTotal features: {n_features}")
    print(f"Non-zero features: {n_nonzero}")
    if n_features > 0:
        print(f"Sparsity: {(1 - n_nonzero/n_features):.2%}")
    
    # Display top features
    print("\nTop 20 features by absolute importance/coefficient:")
    print("-" * 65)
    
    if isinstance(estimator, RandomForestClassifier):
        print(f"{'Rank':<6} {'Feature Name':<40} {'Importance':<12}")
        print("-" * 65)
        for i, (name, importance) in enumerate(sorted_features[:20], 1):
            if importance > 0:
                print(f"{i:<6} {name:<40} {importance:.6f}")
    else:
        print(f"{'Rank':<6} {'Feature Name':<40} {'Coefficient':<15}")
        print("-" * 65)
        for i, (name, coef) in enumerate(sorted_features[:20], 1):
            if coef != 0:
                print(f"{i:<6} {name:<40} {coef:>14.6f}")
    
    # Calculate and display statistics
    if n_nonzero > 0:
        values = [val for _, val in feature_importances.items() if val != 0]
        print("\nStatistics (non-zero features only):")
        print(f"  Mean: {np.mean(values):.6f}")
        print(f"  Std:  {np.std(values):.6f}")
        print(f"  Max:  {np.max(values):.6f}")
        print(f"  Min:  {np.min(values):.6f}")
        if isinstance(estimator, RandomForestClassifier):
            print(f"  Sum:  {np.sum(values):.6f}")
    
    # Display classifier-specific parameters
    if isinstance(estimator, RandomForestClassifier):
        print(f"\nRandomForestClassifier parameters:")
        print(f"  n_estimators: {estimator.n_estimators}")
        print(f"  max_depth: {estimator.max_depth}")
        print(f"  min_samples_split: {estimator.min_samples_split}")
        print(f"  min_samples_leaf: {estimator.min_samples_leaf}")
    elif isinstance(estimator, LinearSVC):
        print(f"\nLinearSVC parameters:")
        print(f"  C (regularization): {estimator.C}")
        print(f"  penalty: {estimator.penalty}")
        print(f"  dual: {estimator.dual}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Inspect classifier pipelines to extract features and their importances'
    )
    parser.add_argument(
        'classifier_file',
        help='Path to the classifier .joblib file'
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.classifier_file).exists():
        print(f"Error: File not found: {args.classifier_file}", file=sys.stderr)
        sys.exit(1)
    
    inspect_classifier(args.classifier_file)


if __name__ == '__main__':
    main()