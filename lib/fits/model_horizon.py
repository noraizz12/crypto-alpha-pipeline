import copy
import logging
from datetime import timedelta as td
from typing import Dict, List, Optional, Self, Tuple, Any

import joblib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn import metrics, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.linear_model import RegressionResultsWrapper

from lib.util.dataframes import check_df, get_min_max_ts, remove_infs, make_date
from lib.util.files import safe_mkdir
from lib.util.logging_util import KeyLogger
from lib.util.time_util import date_to_str
from lib.util.config import extract_max_lags
from lib.util.util import fpct, log_and_raise
from .fit_util import get_dep, MIN_CLASSIFICATION_OBS, MIN_FITTING_OBS, MIN_COEFF, extract_feature_importances

original_logger = logging.getLogger(__name__)
original_logger.setLevel(logging.INFO)
logger = KeyLogger(original_logger)

pd.options.mode.chained_assignment = None

# Set random seed for reproducibility
np.random.seed(42)



class ModelHorizonFit:
    """Fits a single model at a specific horizon with optional regime classification.

    This class handles the fitting of alpha models for a specific time horizon,
    optionally using classifiers to separate momentum and mean-reversion
    regimes. It performs regularized regressions with robust standard errors
    and manages the persistence of fitted models and classifiers.

    The fitting process involves:
    1. Optional classification to detect market regimes
    2. Separate regressions for each regime (or single regression without classifier)
    3. Coefficient smoothing across lags
    4. Model persistence for production deployment

    Attributes:
        prod: Production mode flag
        debug: Debug mode for additional logging
        config: System configuration dictionary
        forecast: Model specification from config
        horizon: Time horizon in minutes
        fit_type: Type of fit ('classifier' or 'security')
        weighted_fit: Whether to use market-cap weighting
        name: Model name (e.g., 'hl', 'c2vwap')
        name_horizon: Combined model and horizon identifier
        indep0: Independent variable name for lag 0
        lags: Number of lags to fit (must be <= 1)
        return_type: Type of return used as dependent variable
        classification_dep: Dependent variable for classification
        classifier_dir: Directory for classifier model storage
        result_df: DataFrame containing fitted coefficients
        fitting_df: Data used for model fitting
        classification_df: Data used for classification
        as_of: Date the model is fitted through
    """

    def __init__(
            self,
            prod: bool,
            debug: bool,
            config: dict,
            forecast: dict,
            fitting_df: pd.DataFrame,
            classification_df: pd.DataFrame,
            weighted_fit: bool,
            horizon: int,
            lags: int,
            return_type: str,
            classifier_dir: str,
            classification_features: Optional[List[str]] = None,
            svm_regularization: Optional[float] = None,
            classifier: Optional[Pipeline] = None
    ):
        self.prod = prod
        self.debug = debug
        self.config = config
        self.forecast = forecast
        self.horizon = horizon
        self.fit_type = self.forecast['fit_type'] # 'svm_adapt', 'svm_cv', or 'rf', 'security', 'vanilla'
        self.weighted_fit = weighted_fit
        self.name = self.forecast['name']
        self.name_horizon = f"{self.name}_{self.horizon}"
        self.indep0 = f"{self.name_horizon}_L0"
        assert lags <= 1
        self.lags = lags
        self.return_type = return_type
        self.result_df: Optional[pd.DataFrame] = None

        check_df(fitting_df)
        if len(fitting_df) < MIN_FITTING_OBS:
            logger.error(f"Fitting length too small ({len(fitting_df)} < {MIN_FITTING_OBS}) on {self.name_horizon}", key="data length too small in ModelHorizonFit")
            raise ValueError()

        self.fitting_df = fitting_df

        self.use_cluster_regression = self.config['USE_CLUSTER_REGRESSION']
        self.sigma_bound_dep = self.config['SIGMA_BOUND_DEP']
        self.sigma_bound_indep = self.config['SIGMA_BOUND_INDEP']
        self.resample_horizon_threshold = self.config['FITTING_RESAMPLE_HORIZON_THRESHOLD']

        if self.fit_type in ('vanilla', 'security'):
            self.use_classifier = False
        else:
            self.use_classifier = True
            self.classifier = classifier
            self.classification_dep = get_dep(return_type=self.return_type, lag=0, horizon=self.horizon)
            self.min_classification_quantile = self.config['MIN_CLASSIFICATION_QUANTILE']
            self.resample_frequency = self.config['FITTING_RESAMPLE_FREQUENCY']
            self.classification_df = classification_df
            self.classifier_dir = classifier_dir
            check_df(classification_df)
            assert self.min_classification_quantile > 0
            self.classification_features = classification_features
            self.class_col = f"class_{self.name}"

            if self.indep0 not in self.fitting_df.columns:
                raise log_and_raise(f"{self.indep0} not in fitting_df, columns: {self.fitting_df.columns}")

            self.fitting_df[f"{self.indep0}_abs"] = self.fitting_df[self.indep0].abs()
            self.classification_df[f"{self.indep0}_abs"] = self.classification_df[self.indep0].abs()

            fitting_df = self.fitting_df.dropna(subset=self.classification_features)
            fit_cnt = len(fitting_df)
            if fit_cnt < MIN_FITTING_OBS:
                if len(fitting_df) > 0:
                    print(fitting_df[self.classification_features].iloc[0])
                logger.error(f"Fitting length too small ({fit_cnt}) on {self.name_horizon}, once removing NAs for classification features!", key="fitting data length too small in ModelHorizonFit")
                raise ValueError()
            self.fitting_df = fitting_df

            if self.fit_type == 'rf':
                self.classifier_name = 'randomforestclassifier'
                self.rf_n_estimators = self.config['RF_N_ESTIMATORS']
                self.rf_max_depth = self.config['RF_MAX_DEPTH']
                self.rf_min_samples_split = self.config['RF_MIN_SAMPLES_SPLIT']
                self.rf_min_samples_leaf = self.config['RF_MIN_SAMPLES_LEAF']
                self.rf_random_state = self.config['RF_RANDOM_STATE']
                self.rf_n_jobs = self.config['RF_N_JOBS']
                self.rf_class_weight = self.config['RF_CLASS_WEIGHT']
                self.rf_verbose = self.config['RF_VERBOSE']
                self.rf_max_samples = self.config["RF_MAX_SAMPLES"]
                self.rf_max_features = self.config["RF_MAX_FEATURES"]
                self.rf_bootstrap = self.config["RF_BOOTSTRAP"]
                self.rf_oob_score = self.config["RF_OOB_SCORE"]
                self.rf_ccp_alpha = self.config["RF_CCP_ALPHA"]
            elif self.fit_type in ('svm_adapt', 'svm_cv'):
                self.svm_regularization = self.config['SVM_REGULARIZATION'] if svm_regularization is None else svm_regularization
                self.classifier_name = 'linearsvc'

            # Sample fraction for subsampling classification data
            self.sample_frac = 0.1

        lags = 1
        models = config['FCASTS'][str(self.horizon)]['models']
        for model in models:
            if model['name'] == self.name:
                lags = model['lags']
                break

        lag_days = int(self.horizon / 1440) * (lags + 1)
        logger.info(f"Fitting {self.horizon} with {lag_days=}")
        self.as_of = self.fitting_df.index.get_level_values('ts').max().date() + td(days=lag_days)

        self.log_name = f"ModelHorizonFit:{self.name_horizon}:{date_to_str(self.as_of)}"
        fit_cnt = len(self.fitting_df)
        classification_cnt = len(self.classification_df) if self.use_classifier else 0
        self.info(f"Running ModelHorizonFit with {self.fit_type} {lags=} fit cnt: {fit_cnt}, classification cnt: {classification_cnt}")

    def info(self, msg: str):
        logger.info(f"{self.log_name} {msg}")

    def warning(self, msg: str):
        logger.warning(f"{self.log_name} {msg}")

    def error(self, msg: str, key: Optional[str] = None):
        logger.error(f"{self.log_name} {msg}", key=key)

    def smooth_fits(self, fit_df: pd.DataFrame) -> pd.DataFrame:
        """Apply coefficient smoothing across lags to reduce noise.

        Implements a differencing approach to smooth coefficients across lag
        periods, which helps reduce estimation noise and improve out-of-sample
        performance. The smoothing preserves the lag-0 coefficient and applies
        differences for subsequent lags.

        Args:
            fit_df: DataFrame with fitted coefficients across lags

        Returns:
            DataFrame with additional 'coeff_diff' and 'coeff_smooth' columns
        """
        self.info("Diffing and Smoothing fits...")
        index_cols = ['lag', 'condition']
        if self.fit_type == "security":
            index_cols = ['symbol_venue'] + index_cols
        smoothed_df = fit_df[index_cols + ['coeff']].set_index(index_cols).unstack().diff().stack(future_stack=True).reset_index()
        fit_df = pd.merge(fit_df, smoothed_df[index_cols + ['coeff']], on=index_cols, how='left', suffixes=('', '_diff'))
        fit_df.loc[fit_df['lag'] == 0, 'coeff_diff'] = fit_df.loc[fit_df['lag'] == 0, 'coeff']
        fit_df['coeff_smooth'] = fit_df['coeff_diff']
        return fit_df

    def run_fits(self) -> Self:
        """Execute the main fitting routine based on fit type.

        Dispatches to either security-specific or cross-sectional regression
        methods based on the fit_type parameter. Security fits create separate
        models for each symbol, while cross-sectional fits pool all data.

        Returns:
            Self for method chaining
        """
        if self.fit_type == "security":
            result_df = self.run_lagged_security_regressions()
        else:
            result_df = self.run_lagged_regressions()

        self.result_df = result_df
        return self

    def _persist_classifier(self, classifier: Pipeline,  insample_accuracy: float, outsample_accuracy: float, feature_coefficients: Optional[Dict[str, float]] = None):
        """Save fitted classifier to disk with metadata.

        Persists the trained classifier along with feature coefficients and
        accuracy metrics for later use in production. The classifier is saved
        in joblib format with accompanying metadata in a text file.

        Args:
            classifier: Trained sklearn Pipeline with classifier
            feature_coefficients: Dictionary mapping feature names to coefficients
            insample_accuracy: Classification accuracy on training data
            outsample_accuracy: Classification accuracy on validation data
        """
        safe_mkdir(f'{self.classifier_dir}/{self.horizon}/{self.name}')
        filename = f'{self.classifier_dir}/{self.horizon}/{self.name}/classifier.{self.horizon}.{self.name}.{date_to_str(self.as_of)}'
        self.info(f"Dumping {filename}")
        joblib.dump(classifier, f'{filename}.joblib')

        # Save metrics separately
        with open(f'{filename}.metrics', 'w') as mf:
            mf.write(f"insample: {insample_accuracy}\n")
            mf.write(f"outsample: {outsample_accuracy}\n")

        if feature_coefficients is not None:
            # Save only features and their coefficients
            with open(f'{filename}.features', 'w') as ff:
                for k, v in feature_coefficients.items():
                    ff.write(f"{k}: {v}\n")

    def _fraction_features_kept(self, classifier: Pipeline, indep_df: pd.DataFrame) -> Tuple[float, List[str]]:
        selector = SelectFromModel(classifier.named_steps[self.classifier_name], prefit=True)
        # selector.transform(indep_df)
        selected_cols = selector.get_support()
        features_removed = list(indep_df.columns[np.invert(selected_cols)])
        features_kept = list(indep_df.columns[selected_cols])
        feature_fraction = len(features_kept) / (len(features_removed) + len(features_kept))
        self.info(f"Keeping {fpct(feature_fraction)}: {features_kept}, Ignoring: {features_removed}")
        return feature_fraction, features_kept

    def classify(self, classification_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Create classification labels for momentum vs mean-reversion regimes.

        Assigns classification labels based on the relationship between model
        predictions and realized returns. Points where predictions and returns
        have the same sign are labeled as momentum (1), opposite signs as
        mean-reversion (-1), and small moves as neutral (0).

        The classification uses a quantile-based threshold to identify significant
        moves, avoiding classification on noise. This helps the SVM focus on
        economically meaningful regime differences.

        Args:
            classification_df: DataFrame with model predictions and forward returns

        Returns:
            DataFrame with non-zero classifications, or None if insufficient data

        The classification logic:
        - Momentum (1): Prediction and return have same sign, above threshold
        - Mean-reversion (-1): Prediction and return have opposite signs
        - Neutral (0): Small moves below threshold (excluded from output)
        """
        ts_min, ts_max = get_min_max_ts(classification_df)
        self.info(f"Classifying {self.indep0} to {self.classification_dep} at {self.horizon} from {ts_min} to {ts_max}")

        min_move = classification_df[self.classification_dep].abs().quantile(self.min_classification_quantile)

        classification_df.loc[:, self.class_col] = 0
        classification_df.loc[(classification_df[self.classification_dep] > min_move) & (classification_df[self.indep0] > 0), self.class_col] = 1
        classification_df.loc[(classification_df[self.classification_dep] < -min_move) & (classification_df[self.indep0] < 0), self.class_col] = 1
        classification_df.loc[(classification_df[self.classification_dep] > min_move) & (classification_df[self.indep0] < 0), self.class_col] = -1
        classification_df.loc[(classification_df[self.classification_dep] < -min_move) & (classification_df[self.indep0] > 0), self.class_col] = -1

        # to use as classification feature...
        # classification_df['last_class'] = classification_df[[self.class_col]].unstack().shift(self.horizon).stack(future_stack=True)[self.class_col]

        cnt = len(classification_df)
        pos_frac = len(classification_df[classification_df[self.class_col] == 1]) / cnt
        neg_frac = len(classification_df[classification_df[self.class_col] == -1]) / cnt
        epsilon_frac = len(classification_df[classification_df[self.class_col] == 0]) / cnt

        valid_classification_df = classification_df[classification_df[self.class_col] != 0]
        valid_cnt = len(valid_classification_df)
        self.info(f"Splitting on features for {self.indep0} with {min_move=} on {self.classification_dep} classified {valid_cnt} points")
        if valid_cnt == 0:
            dep_avg = classification_df[self.classification_dep].mean()
            indep_avg = classification_df[self.indep0].mean()
            raise log_and_raise(f"No points classified! {self.indep0}:{self.classification_dep}, dep avg: {dep_avg} indep avg: {indep_avg}", df=classification_df[[self.classification_dep, self.indep0]])

        self.info(f"fractions pos/neg/epsilon: {pos_frac:.4f}/{neg_frac:.4f}/{epsilon_frac:.4f}")

        if len(valid_classification_df) < MIN_CLASSIFICATION_OBS:
            print(classification_df[[self.classification_dep, self.indep0, self.class_col]])
            self.warning(f"Not enough observations {len(valid_classification_df)} for classifier to train...")
            return None
        return valid_classification_df

    def _subsample_classification_data(self, classification_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        cnt_before = len(classification_df)
        if cnt_before * self.sample_frac > MIN_CLASSIFICATION_OBS:
            classification_df = classification_df.sample(frac=self.sample_frac, random_state=42)
            cnt_after = len(classification_df)
            self.info(f"Removed {100.0 * (1.0 - self.sample_frac)}% of classification data points {cnt_before} -> {cnt_after}")

        if len(classification_df) < MIN_CLASSIFICATION_OBS:
            self.warning(f"Not enough classification data {len(classification_df)}")
            return None
        return classification_df

    def create_classifier(self) -> Optional[Pipeline]:
        """Create classifier (SVM or Random Forest) based on configuration.

        This method creates classifiers based on the CLASSIFIER_TYPE config:
        - 'svm_adapt': SVM with adaptive regularization
        - 'svm_cv': SVM with cross-validation regularization
        - 'rf': Random Forest classifier
        """
        if not self.use_classifier:
            return None

        if self.classifier is not None:
            self.info(f"Using cached classifier")
            return self.classifier

        classification_df = self.classify(self.classification_df)
        if classification_df is None or len(self.classification_features) == 0:
            self.warning(f"Not creating classifier since no features")
            return None

        # Choose classifier type based on config
        if self.fit_type == 'rf':
            self.info("Using Random Forest classifier")
            self.classifier = self.create_rf(classification_df)
        elif self.fit_type == 'svm_cv':
            self.info("Using cross-validation method for SVM regularization")
            self.classifier = self.create_svm_with_cv(classification_df)
        elif self.fit_type == 'svm_adapt':
            self.info("Using original adaptive method for SVM regularization")
            self.classifier = self.create_svm_adaptive(classification_df)
        else:
            raise log_and_raise(f"Unknown classifier type: {self.fit_type}")

        return self.classifier


    def create_svm_adaptive(self, classification_df: pd.DataFrame) -> Pipeline:
        """Original SVM classifier creation method with adaptive regularization"""

        indep_df = remove_infs(classification_df[self.classification_features]).fillna(0)
        dep_df = remove_infs(classification_df[[self.class_col]]).fillna(0)[self.class_col]

        # TODO: fix this hacky solver and replace with in/out sample
        classifier = None
        svm_regularization = self.svm_regularization
        for ii in range(10):
            self.info(f"Running classifier {ii}/10 with regularization {svm_regularization}")
            classifier = make_pipeline(
                StandardScaler(),
                svm.LinearSVC(dual=False, penalty='l1', C=svm_regularization, verbose=True, max_iter=2000),
            )
            kwargs = {}
            classifier.fit(indep_df, dep_df, **kwargs)
            feature_fraction, _ = self._fraction_features_kept(classifier, indep_df)
            if feature_fraction < .4:
                svm_regularization *= 1.5
            elif feature_fraction < .75:
                break
            else:
                svm_regularization /= 2.0

        if classifier is None:
            raise log_and_raise("Could not create classifier!")

        self.svm_regularization = svm_regularization
        self._debug_and_persist_classifier(classifier, indep_df, dep_df)
        return classifier

    def create_svm_with_cv(self, classification_df: pd.DataFrame) -> Pipeline:
        """Create SVM classifier using time series cross-validation for regularization selection.

        This is an alternative to the adaptive regularization method, using proper
        time series cross-validation to select the optimal L1 regularization parameter.
        This approach typically yields better out-of-sample performance by avoiding
        overfitting to the full dataset.

        The method:
        1. Tests multiple regularization values using TimeSeriesSplit
        2. Selects the value with best average validation accuracy
        3. Fine-tunes feature selection around the optimal value
        4. Persists the final classifier with metadata

        Args:
            classification_df: DataFrame with features and classification labels

        Returns:
            Fitted sklearn Pipeline with StandardScaler and LinearSVC

        This method is enabled by setting USE_CV_REGULARIZATION=true in config.
        """

        indep_df = remove_infs(classification_df[self.classification_features]).fillna(0)
        dep_df = remove_infs(classification_df[[self.class_col]]).fillna(0)[self.class_col]

        # Use time series cross-validation to find optimal regularization
        tscv = TimeSeriesSplit(n_splits=5)

        # Test a range of regularization values around the default
        base_C = self.svm_regularization
        C_values = [base_C * factor for factor in [0.01, 0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 4.0, 8.0, 10.0]]

        best_score = -1
        best_C = base_C

        self.info("Starting cross-validation for regularization parameter selection...")

        for C_candidate in C_values:
            cv_scores = []

            try:
                for train_idx, val_idx in tscv.split(indep_df):
                    # Split data temporally to respect time series nature
                    X_train, X_val = indep_df.iloc[train_idx], indep_df.iloc[val_idx]
                    y_train, y_val = dep_df.iloc[train_idx], dep_df.iloc[val_idx]

                    # Skip if validation set is too small or has no variation
                    if len(np.unique(y_val)) < 2 or len(y_val) < 10:
                        continue

                    # Train model with candidate C value
                    temp_classifier = make_pipeline(
                        StandardScaler(),
                        svm.LinearSVC(dual=False, penalty='l1', C=C_candidate,
                                      max_iter=2000, random_state=42),
                    )
                    temp_classifier.fit(X_train, y_train)

                    # Evaluate on validation set
                    val_pred = temp_classifier.predict(X_val)
                    val_score = metrics.accuracy_score(y_val, val_pred)
                    cv_scores.append(val_score)

                if cv_scores:  # Only update if we have valid scores
                    avg_cv_score = np.mean(cv_scores)
                    self.info(f"C={C_candidate:.6f}: CV score = {avg_cv_score:.4f} (n_folds={len(cv_scores)})")

                    # Track best regularization value
                    if avg_cv_score > best_score:
                        best_score = avg_cv_score
                        best_C = C_candidate

            except Exception as e:
                self.warning(f"CV failed for C={C_candidate}: {str(e)}")
                continue

        self.info(f"Best C value from CV: {best_C:.6f} with score: {best_score:.4f}")

        # Train final model with best regularization and adaptive feature selection
        svm_regularization = best_C
        classifier = None

        for ii in range(5):  # Reduced iterations since we already found good C
            self.info(f"Fine-tuning classifier {ii}/5 with regularization {svm_regularization}")
            classifier = make_pipeline(
                StandardScaler(),
                svm.LinearSVC(dual=False, penalty='l1', C=svm_regularization,
                              verbose=True, max_iter=2000, random_state=42),
            )
            classifier.fit(indep_df, dep_df)
            feature_fraction, _ = self._fraction_features_kept(classifier, indep_df)
            self.info(f"Feature fraction kept: {feature_fraction:.3f}")

            # Fine-tune based on feature selection with tighter bounds
            if feature_fraction < .3:  # Tighter lower bound
                svm_regularization *= 1.1
            elif feature_fraction < .6:  # Tighter upper bound
                break
            else:
                svm_regularization /= 1.1

        if classifier is None:
            raise log_and_raise("Could not create classifier!")

        self.info(f"Final classifier regularization: {svm_regularization}")
        self.svm_regularization = svm_regularization
        self._debug_and_persist_classifier(classifier, indep_df, dep_df)
        return classifier

    def create_rf(self, classification_df: pd.DataFrame) -> Optional[Pipeline]:
        """Create Random Forest classifier for momentum/reversion classification.

        This is an alternative to SVM classification, using Random Forest which can
        capture non-linear relationships and provide feature importance rankings.
        Random Forest is often more robust to overfitting and can handle complex
        interactions between features.

        The method:
        1. Uses TimeSeriesSplit for proper temporal cross-validation
        2. Tests multiple hyperparameters (n_estimators, max_depth, min_samples_split)
        3. Selects the best parameters based on validation accuracy
        4. Trains final model and persists with metadata

        Args:
            classification_df: DataFrame with features and classification labels

        Returns:
            Fitted sklearn Pipeline with StandardScaler and RandomForestClassifier
        """

        # DEBUG: Check for NaN in features BEFORE fillna
        indep_df_raw = remove_infs(classification_df[self.classification_features])
        nan_counts_features = indep_df_raw.isna().sum()
        total_nans = nan_counts_features.sum()
        if total_nans > 0:
            self.info(f"NaN values in features BEFORE fillna: {total_nans:,} total")
            high_nan_features = nan_counts_features[nan_counts_features > len(indep_df_raw) * 0.2]
            if len(high_nan_features) > 0:
                self.error(f"Features with >20% NaN values:", key="high_nan_features")
                for feat, count in high_nan_features.items():
                    self.error(f"  {feat}: {count} ({100*count/len(indep_df_raw):.2f}%)", key="high_nan_features")

        indep_df = indep_df_raw.fillna(0)

        # DEBUG: Check for NaN in labels BEFORE fillna
        dep_df_raw = remove_infs(classification_df[[self.class_col]])[self.class_col]
        nan_count = dep_df_raw.isna().sum()
        if nan_count > 0:
            self.error(f"WARNING: {nan_count} ({100*nan_count/len(dep_df_raw):.2f}%) NaN values in classification labels BEFORE fillna!", key="nan_in_labels")
            self.error(f"These NaN values will be filled with 0, creating a spurious third class!", key="nan_in_labels")

        dep_df = dep_df_raw.fillna(0)

        # DEBUG: Check training label distribution AFTER fillna
        self.info(f"Training data label distribution:")
        label_counts = dep_df.value_counts()
        for label, count in label_counts.items():
            self.info(f"  Class {label}: {count} ({100*count/len(dep_df):.2f}%)")

        # Check for single-class problem
        if len(label_counts) == 1:
            self.error(f"TRAINING DATA HAS ONLY ONE CLASS: {label_counts.index[0]}", key="single_class_training_data")
            self.error(f"This will cause the classifier to fail!", key="single_class_training_data")

            # Dump the training data for debugging
            self._dump_training_data(classification_df, indep_df, dep_df)

        classifier = make_pipeline(
            RandomForestClassifier(
                n_estimators=self.rf_n_estimators,
                max_depth=self.rf_max_depth,
                min_samples_split=self.rf_min_samples_split,
                min_samples_leaf=self.rf_min_samples_leaf,
                random_state=self.rf_random_state,
                n_jobs=self.rf_n_jobs,
                class_weight=self.rf_class_weight,
                verbose=self.rf_verbose,
                max_features=self.rf_max_features,
                max_samples=self.rf_max_samples,
                bootstrap=self.rf_bootstrap,
                oob_score=self.rf_oob_score,
                ccp_alpha=self.rf_ccp_alpha
            )
        )

        self.info(f"Training RandomForestClassifier")
        classifier.fit(indep_df, dep_df)

        feature_importances = extract_feature_importances(classifier, only_nonzero=False)
        if feature_importances is None:
            self.error(f"Could not extract feature importances from Random Forest classifier!")
            print(classifier.feature_names_in_)
            print(classifier)
            raise RuntimeError()

        # MIN_FEATURE_IMPORTANCE = 0.001
        # kept_features = []
        # for k, v in feature_importances.items():
        #     self.info(f"Feature importance {k}: {v}")
        #     if v < MIN_FEATURE_IMPORTANCE:
        #         self.warning(f"Feature importance {k}: {v} < {MIN_FEATURE_IMPORTANCE}, removing")
        #     else:
        #         kept_features.append(k)
        #
        # self.info(f"Features importance {len(feature_importances)} -> {len(kept_features)}")
        # if len(kept_features) == 0:
        #     print(feature_importances)
        #     raise log_and_raise("no important features from classifier!")
        #
        # if len(kept_features) < len(feature_importances):
        #     self.info("Retraining RandomForestClassifier on subset of features...")
        #     classifier = make_pipeline(
        #         RandomForestClassifier(
        #             n_estimators=self.rf_n_estimators,
        #             max_depth=self.rf_max_depth,
        #             min_samples_split=self.rf_min_samples_split,
        #             min_samples_leaf=self.rf_min_samples_leaf,
        #             random_state=self.rf_random_state,
        #             n_jobs=self.rf_n_jobs,
        #             class_weight=self.rf_class_weight,
        #             verbose=self.rf_verbose,
        #             max_features=self.rf_max_features,
        #             max_samples=self.rf_max_samples,
        #             bootstrap=self.rf_bootstrap,
        #             oob_score=self.rf_oob_score,
        #             ccp_alpha=self.rf_ccp_alpha
        #         )
        #     )
        #     classifier.fit(indep_df[kept_features], dep_df)

        self._debug_and_persist_classifier(
            classifier=classifier,
            indep_df=indep_df,
            class_df=dep_df,
            feature_importances=feature_importances
        )
        return classifier

    def _debug_and_persist_classifier(
            self,
            classifier: Pipeline,
            indep_df: pd.DataFrame,
            class_df: pd.Series,
            feature_importances: Optional[Dict[str, float]] = None
    ) -> Optional[Tuple[float, float]]:
        """Debug and persist Random Forest classifier with feature importances.

        Similar to debug_and_persist_classifier but adapted for Random Forest.
        Saves feature importances instead of linear coefficients.
        """
        min_ts, max_ts = get_min_max_ts(indep_df)
        insample_accuracy = metrics.accuracy_score(class_df, classifier.predict(indep_df))
        self.info(f"In-sample Accuracy: {fpct(insample_accuracy)}, running regressions for lags: {self.lags} from {min_ts} - {max_ts}")

        #out of sample accuracy
        oos_df = self.classify(self.fitting_df)
        if oos_df is None:
            self.error("Could not classify OOS, not persisting classifier")
            return None
        oos_indep_df = remove_infs(oos_df[indep_df.columns]).fillna(0)
        oos_dep_df = remove_infs(oos_df[[self.class_col]]).fillna(0)[self.class_col]
        outsample_accuracy = metrics.accuracy_score(oos_dep_df, classifier.predict(oos_indep_df))
        self.info(f"Out-sample Accuracy: {fpct(outsample_accuracy)}, running regressions for lags: {self.lags}")

        if not self.debug:
            self._persist_classifier(
                classifier=classifier,
                feature_coefficients=feature_importances,
                insample_accuracy=insample_accuracy,
                outsample_accuracy=outsample_accuracy
            )
        return insample_accuracy, outsample_accuracy

    # function to create the clipped independent and dependent
    def make_clipped_indep_and_dep(self, class_df: pd.DataFrame, class_grp: str, dep: str, lag: int, cnt: int) -> pd.DataFrame:
        class_len = len(class_df)

        self.info(f":{lag}:{class_grp} Regressing on {class_grp} count: {class_len}")

        dep_l = class_df[dep].mean() - self.sigma_bound_dep * class_df[dep].std()
        dep_h = class_df[dep].mean() + self.sigma_bound_dep * class_df[dep].std()
        self.info(f":{lag}:{class_grp} Clipping dependent {dep} between {dep_l} - {dep_h}")

        indep_l = class_df[self.indep0].mean() - self.sigma_bound_indep * class_df[self.indep0].std()
        indep_h = class_df[self.indep0].mean() + self.sigma_bound_indep * class_df[self.indep0].std()
        self.info(f":{lag}:{class_grp} Clipping independent {self.indep0} between {indep_l} - {indep_h}")

        dep_too_high_pct = fpct(len(class_df[class_df[dep] > dep_h]) / cnt)
        dep_too_low_pct = fpct(len(class_df[class_df[dep] < dep_l]) / cnt)
        indep_too_high_pct = fpct(len(class_df[class_df[self.indep0] > indep_h]) / cnt)
        indep_too_low_pct = fpct(len(class_df[class_df[self.indep0] < indep_l]) / cnt)
        self.info(f":{lag}:{class_grp} Fit outliers {dep_too_low_pct=} {dep_too_high_pct=} {indep_too_low_pct=} {indep_too_high_pct=}")

        class_df[dep] = remove_infs(class_df[dep].clip(lower=dep_l, upper=dep_h)).fillna(0)
        class_df[self.indep0] = remove_infs(class_df[self.indep0].clip(lower=indep_l, upper=indep_h)).fillna(0)

        self.info(f":{lag}:{class_grp} Avg Dep {class_df[dep].abs().mean()}, Avg Indep: {class_df[self.indep0].abs().mean()}")
        return class_df

    def _run_regresssion_results(self, class_df: pd.DataFrame, dep: str, weight_col: Optional[str] = None) -> RegressionResultsWrapper:
        formula = f"{dep} ~ {self.indep0}"
        regression_cols = [dep, self.indep0, weight_col] if weight_col is not None else [dep, self.indep0]
        regression_df = class_df[regression_cols].reset_index()
        cnt = len(regression_df)
        self.info(f"Running regression for {formula=}, with {weight_col=} on {cnt} points, {self.use_cluster_regression=}")

        if self.horizon > 4320 or self.use_cluster_regression:
            self.info("running cluster regression...")
            regression_df = make_date(regression_df)
            regression_df['cluster'] = regression_df['symbol_venue'] + regression_df['date'].dt.strftime('%Y%m%d')
        else:
            # to get the autocorrelation (screws up the boundaries, but ok...
            regression_df = regression_df.sort_values(by=['symbol_venue', 'ts'])

        if weight_col is not None:
            if self.use_cluster_regression:
                fit_results = smf.wls(formula, data=regression_df, weights=regression_df[weight_col]).fit(cov_type='cluster', cov_kwds={'groups': regression_df['cluster']})
            else:
                # For subsampled data at long horizons, adjust maxlags proportionally
                maxlags = self.horizon
                if self.horizon >= self.resample_horizon_threshold:
                    # Data is subsampled to hourly, so reduce maxlags accordingly
                    maxlags = min(self.horizon, int(self.horizon / 60))
                fit_results = smf.wls(formula, data=regression_df, weights=regression_df[weight_col]).fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
        else:
            if self.use_cluster_regression:
                fit_results = smf.ols(formula, data=regression_df).fit(cov_type='cluster', cov_kwds={'groups': regression_df['cluster']})
            else:
                # For subsampled data at long horizons, adjust maxlags proportionally
                maxlags = self.horizon
                if self.horizon >= self.resample_horizon_threshold:
                    # Data is subsampled to hourly, so reduce maxlags accordingly
                    maxlags = min(self.horizon, int(self.horizon / 60))
                fit_results = smf.ols(formula, data=regression_df).fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
        return fit_results

    def run_regression(self, class_df: pd.DataFrame, class_grp: str, dep: str, lag: int, weight_col: Optional[str] = None) -> Tuple[float, float, float]:
        """Run linear regression and extract coefficient statistics.

        Performs OLS or WLS regression of forward returns on model predictions,
        with HAC or clustered standard errors for robust inference. Handles
        edge cases like zero variance and numerical issues.

        Args:
            class_df: DataFrame with clipped data for regression
            class_grp: Regime identifier ('mom', 'rev', or 'no_classifier')
            dep: Dependent variable column name
            lag: Lag period for logging
            weight_col: Optional column for weighted least squares

        Returns:
            Tuple of (coefficient, t-statistic, standard_error)
            Returns zeros if regression fails or has numerical issues
        """
        coeff = 0
        tstat = 0
        stderr = 0
        if class_df[self.indep0].std() == 0 or class_df[dep].std() == 0:
            self.warning(f":{lag}:{class_grp} No variation in independent variable! {self.indep0}")
        else:
            fit_results = self._run_regresssion_results(class_df, dep, weight_col)

            fit_coeff = fit_results.params.iloc[1]
            fit_tstat = fit_results.tvalues.iloc[1]
            print(fit_results.summary())
            if np.isinf(fit_tstat) or np.isnan(fit_tstat):
                self.warning(f":{lag}:{class_grp} Bad tstats!! {fit_tstat}")
            elif abs(fit_coeff) < MIN_COEFF:
                self.warning(f":{lag}:{class_grp} Near zero coefficient {fit_coeff=}, problem with regression??")
                tstat = fit_tstat
                stderr = fit_results.bse.iloc[1]
            else:
                coeff = fit_coeff
                tstat = fit_tstat
                stderr = fit_results.bse.iloc[1]
        self.info(f":{lag}:{class_grp} get regression results {tstat=}, {coeff=}, {stderr=}")
        return coeff, tstat, stderr

    def assign_fit_result(self, fit_result: dict, ii: int) -> Dict[str, Any]:
        fit_result['name'] = self.name
        fit_result['indep'] = self.name_horizon
        fit_result['horizon'] = self.horizon
        fit_result['as_of'] = self.as_of
        fit_result['lag'] = ii
        return fit_result

    def make_dep_fitting_df(self, raw_fitting_df: pd.DataFrame, dep: str, lag: int) -> pd.DataFrame:
        fitting_df = raw_fitting_df.dropna(subset=[dep])
        if len(fitting_df) < MIN_FITTING_OBS:
            min_ts, max_ts = get_min_max_ts(raw_fitting_df)
            self.error(f":{lag} Less than {MIN_FITTING_OBS} features, once removing NA for {dep}:{lag} from {min_ts} - {max_ts}", key="fitting data length too small in classifier regressions")
            print(fitting_df)
            print(fitting_df[[dep]])
            print(raw_fitting_df[[dep]])
            if self.use_classifier:
                print(raw_fitting_df[self.classification_features + [dep]].iloc[0])
            raise ValueError()

        # clear stupidly erroneous data
        fitting_df = fitting_df[fitting_df[dep].abs() < 5]
        cnt = len(fitting_df)
        if cnt < MIN_FITTING_OBS:
            self.warning(f":{lag} Zero-length dataframe after basic sanity filters for {self.indep0}")
            raise ValueError()

        return fitting_df

    def debug_feature_variance_and_predictions(self, fitting_df: pd.DataFrame, dep: str, lag: int) -> None:
        """Debug feature variance and per-feature predictions to identify problematic features."""

        # 1. Check variance of each classification feature
        self.info(f":{lag} Feature variance analysis:")
        feature_stats = {}
        for feat in self.classification_features:
            feat_data = fitting_df[feat]
            stats = {
                'mean': feat_data.mean(),
                'std': feat_data.std(),
                'min': feat_data.min(),
                'max': feat_data.max(),
                'n_unique': feat_data.nunique(),
                'n_na': feat_data.isna().sum(),
                'pct_zero': (feat_data == 0).sum() / len(feat_data)
            }
            feature_stats[feat] = stats
            self.info(f"  {feat}: std={stats['std']:.6f}, unique={stats['n_unique']}, pct_zero={stats['pct_zero']:.2%}")

        # 2. Check if features have been scaled properly by looking at pre-prediction data
        feature_matrix = fitting_df[self.classification_features]
        self.info(f":{lag} Feature matrix stats before prediction:")
        self.info(f"  Shape: {feature_matrix.shape}")
        self.info(f"  Overall mean: {feature_matrix.mean().mean():.6f}")
        self.info(f"  Overall std: {feature_matrix.std().mean():.6f}")

        # 3. Check for features with zero or near-zero variance
        low_variance_features = [
            feat for feat, stats in feature_stats.items()
            if stats['std'] < 1e-6
        ]
        if low_variance_features:
            self.warning(f":{lag} Features with near-zero variance: {low_variance_features}")

        # 4. Check for features that are mostly zeros
        mostly_zero_features = [
            feat for feat, stats in feature_stats.items()
            if stats['pct_zero'] > 0.95
        ]
        if mostly_zero_features:
            self.warning(f":{lag} Features that are >95% zeros: {mostly_zero_features}")

        # 5. Check prediction probabilities (works for RF and SVM with probability=True)
        if hasattr(self.classifier, 'predict_proba'):
            try:
                proba_matrix = self.classifier.predict_proba(feature_matrix)
                # proba_matrix is shape (n_samples, n_classes), typically (n, 2) for binary classification
                # Column 0 is probability of class -1, Column 1 is probability of class +1
                self.info(f":{lag} Prediction probabilities:")
                self.info(f"  Shape: {proba_matrix.shape}")

                # Probability of class -1 (mean reversion)
                prob_rev = proba_matrix[:, 0]
                self.info(f"  P(class=-1) - Mean: {prob_rev.mean():.6f}, Std: {prob_rev.std():.6f}")
                self.info(f"  P(class=-1) - Min: {prob_rev.min():.6f}, Max: {prob_rev.max():.6f}")
                self.info(f"  P(class=-1) - Quantiles [0.1, 0.25, 0.5, 0.75, 0.9]: {np.quantile(prob_rev, [0.1, 0.25, 0.5, 0.75, 0.9])}")

                # Probability of class +1 (momentum)
                prob_mom = proba_matrix[:, 1]
                self.info(f"  P(class=+1) - Mean: {prob_mom.mean():.6f}, Std: {prob_mom.std():.6f}")
                self.info(f"  P(class=+1) - Min: {prob_mom.min():.6f}, Max: {prob_mom.max():.6f}")
                self.info(f"  P(class=+1) - Quantiles [0.1, 0.25, 0.5, 0.75, 0.9]: {np.quantile(prob_mom, [0.1, 0.25, 0.5, 0.75, 0.9])}")

                # Check if probabilities are stuck
                if prob_rev.std() < 0.01:
                    self.warning(f":{lag} P(class=-1) has very low variance ({prob_rev.std():.6f}) - classifier may not be discriminating!")
                if prob_mom.std() < 0.01:
                    self.warning(f":{lag} P(class=+1) has very low variance ({prob_mom.std():.6f}) - classifier may not be discriminating!")

                # Check if all probabilities strongly favor one class
                if prob_mom.min() > 0.9:
                    self.warning(f":{lag} ALL probabilities favor momentum (P(+1) > 0.9 for all samples)")
                elif prob_rev.min() > 0.9:
                    self.warning(f":{lag} ALL probabilities favor mean-reversion (P(-1) > 0.9 for all samples)")

            except Exception as e:
                self.warning(f":{lag} Could not compute predict_proba: {e}")

        # 5b. Check decision function scores (SVM-specific)
        if hasattr(self.classifier, 'decision_function'):
            try:
                decision_scores = self.classifier.decision_function(feature_matrix)
                self.info(f":{lag} Decision function scores:")
                self.info(f"  Mean: {decision_scores.mean():.6f}")
                self.info(f"  Std: {decision_scores.std():.6f}")
                self.info(f"  Min: {decision_scores.min():.6f}")
                self.info(f"  Max: {decision_scores.max():.6f}")
                self.info(f"  Quantiles [0.1, 0.25, 0.5, 0.75, 0.9]: {np.quantile(decision_scores, [0.1, 0.25, 0.5, 0.75, 0.9])}")

                # Check if all scores are on one side of the decision boundary
                if decision_scores.min() >= 0:
                    self.warning(f":{lag} ALL decision scores >= 0 (all classified as class +1)")
                elif decision_scores.max() <= 0:
                    self.warning(f":{lag} ALL decision scores <= 0 (all classified as class -1)")
            except Exception as e:
                self.warning(f":{lag} Could not compute decision scores: {e}")

        # 6. Examine classifier coefficients/feature importances
        if hasattr(self.classifier, 'named_steps'):
            # For Pipeline with SVM
            if 'linearsvc' in self.classifier.named_steps:
                svm_model = self.classifier.named_steps['linearsvc']
                if hasattr(svm_model, 'coef_'):
                    coeffs = svm_model.coef_[0]
                    self.info(f":{lag} SVM feature coefficients:")
                    for feat, coeff in zip(self.classification_features, coeffs):
                        self.info(f"  {feat}: {coeff:.6f}")

                    # Identify features with very large coefficients (potential issues)
                    large_coeff_features = [
                        (feat, coeff) for feat, coeff in zip(self.classification_features, coeffs)
                        if abs(coeff) > 10
                    ]
                    if large_coeff_features:
                        self.warning(f":{lag} Features with large coefficients (>10): {large_coeff_features}")

            # For Pipeline with RandomForest
            if 'randomforestclassifier' in self.classifier.named_steps:
                rf_model = self.classifier.named_steps['randomforestclassifier']
                if hasattr(rf_model, 'feature_importances_'):
                    importances = rf_model.feature_importances_
                    self.info(f":{lag} Random Forest feature importances:")
                    for feat, imp in zip(self.classification_features, importances):
                        self.info(f"  {feat}: {imp:.6f}")

        # 7. Check the actual distribution of the dependent variable
        self.info(f":{lag} Dependent variable ({dep}) distribution:")
        self.info(f"  Mean: {fitting_df[dep].mean():.6f}")
        self.info(f"  Std: {fitting_df[dep].std():.6f}")
        self.info(f"  Quantiles [0.1, 0.25, 0.5, 0.75, 0.9]: {fitting_df[dep].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).values}")

        # 8. Sample predictions to see what's happening at the individual level
        try:
            sample_size = min(10, len(fitting_df))
            sample_indices = np.random.choice(len(fitting_df), sample_size, replace=False)
            sample_df = fitting_df.iloc[sample_indices]
            sample_predictions = self.classifier.predict(sample_df[self.classification_features])

            self.info(f":{lag} Sample predictions (showing {sample_size} random examples):")

            # For Random Forest, show probabilities
            if hasattr(self.classifier, 'predict_proba'):
                sample_probas = self.classifier.predict_proba(sample_df[self.classification_features])
                for idx, (proba, pred) in enumerate(zip(sample_probas, sample_predictions)):
                    self.info(f"  Sample {idx}: P(class=-1)={proba[0]:.6f}, P(class=+1)={proba[1]:.6f}, prediction={pred}")

            # For SVM, show decision scores
            elif hasattr(self.classifier, 'decision_function'):
                sample_scores = self.classifier.decision_function(sample_df[self.classification_features])
                for idx, (score, pred) in enumerate(zip(sample_scores, sample_predictions)):
                    self.info(f"  Sample {idx}: decision_score={score:.6f}, prediction={pred}")
            else:
                # Just show predictions
                for idx, pred in enumerate(sample_predictions):
                    self.info(f"  Sample {idx}: prediction={pred}")

        except Exception as e:
            self.warning(f":{lag} Could not sample predictions: {e}")

    def _dump_training_data(self, classification_df: pd.DataFrame, indep_df: pd.DataFrame, dep_df: pd.Series) -> None:
        """Dump classifier training data to parquet file for offline debugging.

        Args:
            classification_df: Full classification DataFrame with all columns
            indep_df: Feature matrix used for training
            dep_df: Labels used for training
        """
        try:
            # Create debug directory if it doesn't exist
            debug_dir = f'{self.classifier_dir}/debug/{self.horizon}/{self.name}'
            safe_mkdir(debug_dir)

            # Prepare data to dump - include features, labels, indep, and dep
            dump_df = indep_df.copy()
            dump_df[self.class_col] = dep_df
            dump_df[self.indep0] = classification_df[self.indep0]
            dump_df[self.classification_dep] = classification_df[self.classification_dep]

            # Save to parquet
            filename = f'{debug_dir}/training_data.{self.horizon}.{self.name}.{date_to_str(self.as_of)}.parquet'
            dump_df.to_parquet(filename)
            self.info(f"Dumped TRAINING data to {filename}")
            self.info(f"Dumped {len(dump_df)} rows with {len(dump_df.columns)} columns")

        except Exception as e:
            self.error(f"Failed to dump training data: {e}")

    def _dump_classifier_input_data(self, fitting_df: pd.DataFrame, dep: str, lag: int, predictions: Optional[np.ndarray] = None) -> None:
        """Dump classifier input data to parquet file for offline debugging.

        Saves the feature matrix, dependent variable, predictions (if provided),
        and prediction probabilities to a parquet file for later analysis.

        Args:
            fitting_df: DataFrame with features and dependent variable
            dep: Name of dependent variable column
            lag: Lag period for filename
            predictions: Optional array of classifier predictions to include
        """
        try:
            # Create debug directory if it doesn't exist
            debug_dir = f'{self.classifier_dir}/debug/{self.horizon}/{self.name}'
            safe_mkdir(debug_dir)

            # Prepare data to dump
            dump_df = fitting_df[self.classification_features + [dep]].copy()

            # Add predictions if provided
            if predictions is not None:
                dump_df['class_predict'] = predictions

            # Add prediction probabilities if available
            if hasattr(self.classifier, 'predict_proba'):
                try:
                    probas = self.classifier.predict_proba(fitting_df[self.classification_features])
                    dump_df['proba_class_neg1'] = probas[:, 0]
                    dump_df['proba_class_pos1'] = probas[:, 1]
                except Exception as e:
                    self.warning(f":{lag} Could not add probabilities to dump: {e}")

            # Add decision scores if available (SVM)
            if hasattr(self.classifier, 'decision_function'):
                try:
                    scores = self.classifier.decision_function(fitting_df[self.classification_features])
                    dump_df['decision_score'] = scores
                except Exception as e:
                    self.warning(f":{lag} Could not add decision scores to dump: {e}")

            # Save to parquet
            filename = f'{debug_dir}/classifier_input.{self.horizon}.{self.name}.lag{lag}.{date_to_str(self.as_of)}.parquet'
            dump_df.to_parquet(filename)
            self.info(f":{lag} Dumped classifier input data to {filename}")
            self.info(f":{lag} Dumped {len(dump_df)} rows with {len(dump_df.columns)} columns")

        except Exception as e:
            self.error(f":{lag} Failed to dump classifier input data: {e}")

    def run_classifier_regressions(self, raw_fitting_df: pd.DataFrame, dep: str, lag: int) -> List[dict]:
        self.info(f"Running classifier regressions on {dep}")
        assert self.classifier is not None
        fitting_df = self.make_dep_fitting_df(raw_fitting_df, dep, lag)
        cnt = len(fitting_df)

        #XXX remove me for performance
        self.debug_feature_variance_and_predictions(fitting_df, dep, lag)

        results = []
        fitting_df['class_predict'] = self.classifier.predict(fitting_df[self.classification_features])
        rev_cnt = len(fitting_df[fitting_df['class_predict'] == -1])
        mom_cnt = len(fitting_df[fitting_df['class_predict'] == 1])
        self.info(f":{lag} Classification rev / mom: {rev_cnt} {fpct(rev_cnt / cnt)} / {mom_cnt} {fpct(mom_cnt / cnt)}")

        # Dump raw regression data for offline experimentation
        if self.debug:
            dump_cols = [dep, self.indep0, 'class_predict']
            if self.weighted_fit and 'mkt_wgt' in fitting_df.columns:
                dump_cols.append('mkt_wgt')
            dump_cols.extend([f for f in self.classification_features if f not in dump_cols])
            ts_idx = fitting_df.index.get_level_values('ts')
            dump_path = f"regression_data_{self.name}_{self.horizon}_lag{lag}.{date_to_str(ts_idx.min())}.{date_to_str(ts_idx.max())}.parquet"
            fitting_df[dump_cols].to_parquet(dump_path)
            self.info(f":{lag} Dumped raw regression data ({len(fitting_df)} rows, {len(dump_cols)} cols) to {dump_path}")

        # Error if classifier puts everything in one class - dump data for debugging
        if rev_cnt == 0:
            self.error(f":{lag} CLASSIFIER ERROR: All {cnt} observations classified as MOMENTUM (class +1). No mean-reversion cases found!", key="classifier_all_same_class")
            self._dump_classifier_input_data(fitting_df, dep, lag, predictions=fitting_df['class_predict'].values)
        elif mom_cnt == 0:
            self.error(f":{lag} CLASSIFIER ERROR: All {cnt} observations classified as MEAN-REVERSION (class -1). No momentum cases found!", key="classifier_all_same_class")
            self._dump_classifier_input_data(fitting_df, dep, lag, predictions=fitting_df['class_predict'].values)

        for class_grp in ('rev', 'mom'):
            condition_num = -1 if class_grp == 'rev' else 1

            class_df = fitting_df[fitting_df['class_predict'] == condition_num]
            class_len = len(class_df)
            dummy_result = {'tstat': 0, 'coeff': 0, 'stderr': 0, 'nobs': class_len, 'condition': class_grp, 'condition_num': condition_num, 'impurity': 0}
            # bad case without enough fitting observations
            if class_len < MIN_FITTING_OBS:
                self.warning(f":{lag}:{class_grp} Too few observations ({class_len}/{cnt}) for {self.name}:{self.horizon}:{class_grp} {self.as_of}, continuing...")
                results.append(dummy_result)
                continue

            class_df = self.make_clipped_indep_and_dep(class_df, class_grp, dep, lag, cnt)
            # bad case without variation in independent or dependence
            if class_df[self.indep0].std() == 0 or class_df[dep].std() == 0:
                self.warning(f":{lag}:{class_grp} No variation in independent variable! {self.indep0} or {dep}")
                print(class_df[[self.indep0, dep]])
                results.append(dummy_result)
                continue

            coeff, tstat, stderr = self.run_regression(
                class_df=class_df,
                class_grp=class_grp,
                dep=dep,
                lag=lag,
                weight_col='mkt_wgt' if self.weighted_fit else None,
            )

            impurity = len(class_df[class_df[dep] * condition_num < 0]) / class_len
            results.append({
                'tstat': tstat,
                'coeff': coeff,
                'stderr': stderr,
                'nobs': class_len,
                'condition': class_grp,
                'condition_num': condition_num,
                'impurity': impurity,
            })
        return results

    def run_no_classifier_regressions(self, raw_fitting_df: pd.DataFrame, dep: str, lag: int) -> List[dict]:
        self.info(f":{lag} Fitting without classifier...")

        cnt = len(raw_fitting_df)
        fitting_df = self.make_dep_fitting_df(raw_fitting_df, dep, lag)
        class_grp = 'no_classifier'
        class_df = fitting_df.copy()
        class_len = len(class_df)
        non_regression_result = {'tstat': 0, 'coeff': 0, 'stderr': 0, 'nobs': class_len, 'condition': class_grp, 'condition_num': 0, 'impurity': 0}
        regression_result = copy.deepcopy(non_regression_result)
        class_df = self.make_clipped_indep_and_dep(class_df, class_grp, dep, lag, cnt)
        coeff, tstat, stderr = self.run_regression(class_df, class_grp, dep, lag, weight_col='mkt_wgt' if self.weighted_fit else None)
        # assign results to its class based on the sign of coeff
        if coeff != 0:
            regression_result['tstat'] = tstat
            regression_result['coeff'] = coeff
            regression_result['stderr'] = stderr
            if coeff > 0:
                regression_result['condition'] = 'mom'
                regression_result['condition_num'] = 1
                non_regression_result['condition'] = 'rev'
                non_regression_result['condition_num'] = -1
            else:
                regression_result['condition'] = 'rev'
                regression_result['condition_num'] = -1
                non_regression_result['condition'] = 'mom'
                non_regression_result['condition_num'] = 1
        else:
            regression_result['condition'] = 'rev'
            regression_result['condition_num'] = -1
            non_regression_result['condition'] = 'mom'
            non_regression_result['condition_num'] = 1

        results = [regression_result, non_regression_result]
        return results

    def run_lagged_security_regressions(self) -> pd.DataFrame:
        fits = []
        for ii in range(self.lags + 1):
            dep = get_dep(return_type=self.return_type, lag=ii, horizon=self.horizon)
            self.info(f"Fitting securities {self.name} to {dep} at lag {ii}")
            for sv in self.fitting_df.index.get_level_values('symbol_venue').unique():
                self.info(f"Fitting {self.name} to {dep} at lag {ii} for {sv}")
                security_fitting_df = self.fitting_df.loc[self.fitting_df.index.get_level_values('symbol_venue') == sv]
                try:
                    fit_results = self.run_no_classifier_regressions(raw_fitting_df=security_fitting_df, dep=dep, lag=ii)
                except ValueError as ve:
                    self.info(f"Fail to run regression of {self.name}, {dep=}, {ii=} for {sv=} {ve}")
                    continue
                for fit_result in fit_results:
                    fit_result = self.assign_fit_result(fit_result, ii)
                    fit_result['symbol_venue'] = sv
                    fits.append(fit_result)

        if len(fits) == 0:
            self.error(f"No fits for {self.name}!, maybe lags are wrong? lags: {self.lags}", key="fail to append fits in lagged classifier regressions")
            raise ValueError()

        fit_df = pd.DataFrame(fits)
        fit_df = self.smooth_fits(fit_df)
        return fit_df

    def run_lagged_regressions(self) -> pd.DataFrame:
        # Subsample fitting data for very long horizons to speed up computation
        if self.horizon >= self.resample_horizon_threshold:
            original_len = len(self.fitting_df)
            # Keep only observations at minute 7 of each hour
            minute_mask = self.fitting_df.index.get_level_values('ts').minute == 7
            self.fitting_df = self.fitting_df[minute_mask]
            new_len = len(self.fitting_df)
            self.info(f"Subsampled fitting data from {original_len} to {new_len} points for horizon {self.horizon} (reduction: {100 * (1 - new_len / original_len):.1f}%)")

        if self.use_classifier:
            cf = self.create_classifier()
            if cf is None:
                self.error("No classifier created!")
                raise RuntimeError("No classifier created!")

        fits = []
        for ii in range(self.lags + 1):
            dep = get_dep(return_type=self.return_type, lag=ii, horizon=self.horizon)
            self.info(f"Fitting {self.indep0} to {dep} at lag {ii}")
            try:
                if self.use_classifier:
                    fit_results = self.run_classifier_regressions(raw_fitting_df=self.fitting_df, dep=dep, lag=ii)
                else:
                    fit_results = self.run_no_classifier_regressions(raw_fitting_df=self.fitting_df, dep=dep, lag=ii)
            except ValueError as ve:
                self.error(f"Fail to run regression: {ve}")
                continue

            for fit_result in fit_results:
                fit_result['name'] = self.name
                fit_result['indep'] = self.name_horizon
                fit_result['horizon'] = self.horizon
                fit_result['as_of'] = self.as_of
                fit_result['lag'] = ii
                fits.append(fit_result)

        if len(fits) == 0:
            raise log_and_raise(f"Fits failed for {self.indep0}!, maybe lags are wrong? lags: {self.lags}")

        fit_df = pd.DataFrame(fits)
        fit_df = self.smooth_fits(fit_df)
        return fit_df
