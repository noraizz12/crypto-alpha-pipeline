import logging
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

from lib.alpha.features import classify_features
from lib.data.data_news import NEWS_START_DATE
from lib.util.dataframes import remove_infs, log_col_summary, set_index, round_col_to_zero, cols_to_list_str
from lib.calcs.calc_returns import calculate_weighted_mkt_return
from lib.util.time_util import date_to_start_dt
from lib.util.util import unique_list

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MIN_ALPHA = 1e-10


class Models:
    def __init__(self, config: dict):
        self.config = config
        self.feature_sigma_bound = self.config['FEATURE_SIGMA_BOUND']

        self.calc_map = {
            'pca': {'func': self.calculate_pca_resid, 'base_features': ['pca_resid_HORIZON']},
            'hl': {'func': self.calculate_hl_ma, 'base_features': ['close_trade', 'high_trade_HORIZON', 'low_trade_HORIZON']},
            'rr': {'func': self.calculate_rr, 'base_features': ['logret_HORIZON']},
            'vadj': {'func': self.calculate_vadj_ma, 'base_features': ['dvolume_HORIZON_lz', 'logret_HORIZON', 'logret_HORIZON_trstd', 'news_event_decayed_HORIZON_lz']},
            'c2vwap': {'func': self.calculate_c2vwap_ma, 'base_features': ['close_mid', 'vwap_HORIZON']},
            'ba': {'func': self.calculate_ba_ma, 'base_features': ['ba_imbal_HORIZON', 'logret_resid_wgtmkt_HORIZON']},
            'bal': {'func': self.calculate_bal_ma, 'base_features': ['ba_imbal_HORIZON']},
            'slz': {'func': self.calculate_slz, 'base_features': ['logret_HORIZON_lz_cz']},
            'ip': {'func': self.calculate_ip, 'base_features': ['index_mid_price_diff_mean_HORIZON']},
            'badj': {'func': self.calculate_badj, 'base_features': ['logret_funding_adj_HORIZON', 'beta_1440']},
            'rsi': {'func': self.calculate_rsi, 'base_features': ['rsi_HORIZON']},
            'oi': {'func': self.calculate_oi, 'base_features': ['open_interest_HORIZON_dp', 'logret_HORIZON_lz']},
            'grev': {'func': self.calculate_grev, 'base_features': ['logret_funding_adj_HORIZON']},
            'oracle': {'func': self.calculate_oracle, 'base_features': []},
            'goober': {'func': self.calculate_goober, 'base_features': []},
        }

    def extract_models_features(self, models: List[dict]) -> Tuple[List[str], List[str]]:
        features = []
        for model in models:
            new_features = self.calc_map[model['name']]['base_features']
            logger.info(f"Adding features for {model['name']}: {new_features}")
            features += new_features
        features = unique_list(features)
        bar_features, feature_features = classify_features(features)
        return bar_features, feature_features

    @staticmethod
    def _calc_eqmkt(df: pd.DataFrame, base: str) -> pd.DataFrame:
        new_df = df[['ts', base]].dropna().groupby('ts', observed=False).mean().reset_index()
        df = pd.merge(df, new_df, on='ts', how='left', suffixes=('', '_mkt'))
        return df

    def calc_ma(self, df: pd.DataFrame, base: str) -> pd.DataFrame:
        ma_name = f"{base}_ma"
        df = self._calc_eqmkt(df, base)
        df[ma_name] = df[base] - df[f'{base}_mkt']
        df = round_col_to_zero(df, ma_name)
        del df[f'{base}_mkt']
        return df

    @staticmethod
    def _make_std_feature(df: pd.DataFrame, base: str) -> pd.DataFrame:
        ma_name = f"{base}_z"
        new_df = df[['ts', base]].dropna().groupby('ts', observed=False).std().reset_index()
        df = pd.merge(df, new_df, on='ts', how='left', suffixes=('', '_mkt'))
        df[ma_name] = df[base] / df[f'{base}_mkt']
        del df[f'{base}_mkt']
        return df

    def _make_bounded_feature(self, df: pd.DataFrame, base: str, sigma_bound: Optional[float] = None) -> pd.DataFrame:
        if sigma_bound is None:
            sigma_bound = self.feature_sigma_bound
        bnd_name = f"{base}_B"
        log_col_summary(df, base)
        bounds_df = df.groupby('ts', observed=False).agg({base: ["mean", "std"]})
        bounds_df.columns = bounds_df.columns.droplevel()
        df = pd.merge(df, bounds_df, on='ts', how='left', suffixes=('', '_bnd'))
        df[bnd_name] = df[base].clip((df[base] - df['mean']) - sigma_bound * df['std'], (df[base] - df['mean']) + sigma_bound * df['std'])
        del df['mean']
        del df['std']
        return df

    def calculate(self, df: pd.DataFrame, indep: str, horizon: int) -> pd.DataFrame:
        assert 'ts' in df.columns
        logger.info(f"Calculating model {indep}")
        df = self.calc_map[indep]['func'](df, horizon)
        return df

    def calculate_pca_resid(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Calculate PCA residual-based forecast model.
        
        The PCA residual represents idiosyncratic returns not explained by
        common market factors. When a stock deviates significantly from its
        factor model prediction, it may revert back to the model.
        
        Args:
            df: DataFrame with PCA residual features
            horizon: Time horizon in minutes
            
        Returns:
            DataFrame with PCA model forecast column
        """
        assert 'ts' in df.columns
        name = 'pca'
        base_col = f'pca_resid_{horizon}'
        model_col = f'{name}_{horizon}'
        lagged_model_col = f'{model_col}_L0'
        
        # Check if PCA residual column exists
        if base_col not in df.columns:
            logger.warning(f"PCA residual column {base_col} not found, setting model to 0")
            df[lagged_model_col] = 0.0
            return df
        
        # Use the PCA residual directly as the signal
        # Negative residual (underperforming factors) suggests positive alpha
        # Positive residual (outperforming factors) suggests negative alpha
        df[model_col] = -df[base_col]  # Negative sign for mean reversion
        
        # Apply bounds to prevent extreme values
        df = self._make_bounded_feature(df, model_col)
        
        # Calculate market-adjusted version
        df = self.calc_ma(df, f"{model_col}_B")
        
        # Create final lagged feature
        df[lagged_model_col] = df[f"{model_col}_B_ma"].fillna(0)
        
        # Zero out very small values
        df.loc[df[lagged_model_col].abs() < MIN_ALPHA, lagged_model_col] = 0.0
        
        log_col_summary(df, lagged_model_col)
        
        # Clean up intermediate columns
        for col in [model_col, f"{model_col}_B", f"{model_col}_B_ma"]:
            if col in df.columns:
                del df[col]
        
        return df

    def calculate_grev(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        name = 'grev'
        model_col = f'{name}_{horizon}'
        lagged_model_col = f'{model_col}_L0'
        ret_col = f'logret_funding_adj_{horizon}'

        df[model_col] = np.float32(0.0)
        df['cat_cnt'] = 0

        cat_cnt = 0
        for col in df.columns:
            if not col.startswith("category_gmci"):
                continue

            in_cat_idx = df[col] > 0
            in_cat_df = df[in_cat_idx]
            if len(in_cat_df) == 0:
                logger.info(f"Skipping category {col} with no symbols in category")
                continue

            uniq_symbols = in_cat_df['symbol_venue'].unique()
            cat_len = len(uniq_symbols)
            if cat_len <= 5:
                logger.info(f"Skipping category {col} with {cat_len} symbols: {uniq_symbols}")
                continue

            logger.info(f"Calculating GREV for category {col}")
            cat_cnt += 1

            # Calculate category mean within each timestamp
            cat_mean_df = df[in_cat_idx].groupby('ts', observed=False)[ret_col].mean().reset_index()
            cat_mean_df = cat_mean_df.rename(columns={ret_col: f'{col}_mean'})
            df = pd.merge(df, cat_mean_df, on='ts', how='left')

            # Subtract category mean only for symbols in this category
            df.loc[in_cat_idx, model_col] += df.loc[in_cat_idx, ret_col] - df.loc[in_cat_idx, f'{col}_mean']
            df.loc[in_cat_idx, 'cat_cnt'] += 1

            overall_mean = df.loc[in_cat_idx, ret_col].mean()
            logger.info(f"GREV Category {col} cnt: {cat_len} mean: {overall_mean}")

            del df[f'{col}_mean']

        logger.info(f"GREV final category cnt: {cat_cnt}")

        df.loc[df['cat_cnt'] > 0, model_col] = (df[model_col] / df['cat_cnt']).astype(np.float32)
        del df['cat_cnt']
        df = self._make_bounded_feature(df, model_col)
        df[lagged_model_col] = df[f"{model_col}_B"]
        df.loc[df[lagged_model_col].abs() < MIN_ALPHA, lagged_model_col] = 0.0
        return df

    def calculate_rsi(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        name = 'rsi'
        model_col = f'{name}_{horizon}'
        lagged_model_col = f'{model_col}_L0'

        df[model_col] = df[f'rsi_{horizon}']
        df = self._make_bounded_feature(df, model_col)
        df = self.calc_ma(df, f"{model_col}_B")
        df[lagged_model_col] = df[f"{model_col}_B_ma"].fillna(0)
        df.loc[df[lagged_model_col].abs() < MIN_ALPHA, lagged_model_col] = 0.0

        log_col_summary(df, lagged_model_col)
        del df[model_col]
        return df

    def calculate_oi(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        name = 'oi'
        model_col = f'{name}_{horizon}'
        lagged_model_col = f'{model_col}_L0'

        df[model_col] = df[f'open_interest_{horizon}_dp'] * df[f'logret_{horizon}_lz']
        log_col_summary(df, f'open_interest_{horizon}_dp')
        log_col_summary(df, f'logret_{horizon}_lz')
        df = self._make_bounded_feature(df, model_col)
        df = self.calc_ma(df, f"{model_col}_B")
        df[lagged_model_col] = df[f"{model_col}_B_ma"].fillna(0)
        df.loc[df[lagged_model_col].abs() < MIN_ALPHA, lagged_model_col] = 0.0

        log_col_summary(df, lagged_model_col)
        del df[model_col]
        return df

    def calculate_hl_ma(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        name = 'hl'
        model_col = f'{name}_{horizon}'
        lagged_model_col = f'{model_col}_L0'

        df[model_col] = remove_infs(df['close_trade'] / np.sqrt(df[f'high_trade_{horizon}'] * df[f'low_trade_{horizon}']))
        df = self._make_bounded_feature(df, model_col)
        df = self.calc_ma(df, f"{model_col}_B")
        df[lagged_model_col] = df[f"{model_col}_B_ma"].fillna(0)
        df.loc[df[lagged_model_col].abs() < MIN_ALPHA, lagged_model_col] = 0.0

        log_col_summary(df, lagged_model_col)
        del df[model_col]
        return df

    def calculate_vadj_ma(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        name = 'vadj'
        model_col = f'{name}_{horizon}'
        lagged_model_col = f'{model_col}_L0'

        df[model_col] = df[f'dvolume_{horizon}_lz'] * df[f'logret_{horizon}'] / df[f'logret_{horizon}_trstd']

        #news doesn't start until 2024...
        news_col = f'news_event_decayed_{horizon}_lz'
        if news_col in df.columns:
            news_start_idx = df['ts'] > date_to_start_dt(NEWS_START_DATE)
            if df.loc[news_start_idx, news_col].abs().sum() == 0:
                logger.error(f"All values in {news_col} are zero, skipping multiplication for {model_col}.")
            else:
                df.loc[news_start_idx, model_col] = df.loc[news_start_idx, model_col] * df.loc[news_start_idx, news_col]

        df = self._make_bounded_feature(df, base=model_col)
        df = self.calc_ma(df, f"{model_col}_B")
        df[lagged_model_col] = df[f'{model_col}_B_ma'].fillna(0)

        df.loc[df[lagged_model_col].abs() < MIN_ALPHA, lagged_model_col] = 0.0
        log_col_summary(df, lagged_model_col)
        return df

    def calculate_badj(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        name = "badj"
        model_col = f'{name}_{horizon}'
        lagged_model_col = f'{model_col}_L0'

        df, mkt_col = calculate_weighted_mkt_return(df=set_index(df), ret_col='logret_funding_adj', frequency=horizon)
        df = df.reset_index()
        df[model_col] = df[f'logret_funding_adj_{horizon}'] - df['beta_1440'] * df[mkt_col]
        df = round_col_to_zero(df, model_col)
        log_col_summary(df, f'logret_funding_adj_{horizon}')
        log_col_summary(df, 'beta_1440')
        log_col_summary(df, mkt_col)
        df = self._make_bounded_feature(df, model_col)
        df = self.calc_ma(df, f"{model_col}_B")
        df[lagged_model_col] = df[f'{model_col}_B_ma'].fillna(0)
        df.loc[df[lagged_model_col].abs() < MIN_ALPHA, lagged_model_col] = 0.0
        log_col_summary(df, lagged_model_col)
        del df[model_col]
        return df

    def calculate_c2vwap_ma(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        name = "c2vwap"
        model_col = f'{name}_{horizon}'
        lagged_model_col = f'{model_col}_L0'

        df[model_col] = np.log(df['close_mid'] / df[f'vwap_{horizon}'])
        df = self._make_bounded_feature(df, model_col)
        df = self.calc_ma(df, f"{model_col}_B")
        df[lagged_model_col] = df[f'{model_col}_B_ma'].fillna(0)
        df.loc[df[lagged_model_col].abs() < MIN_ALPHA, lagged_model_col] = 0.0
        log_col_summary(df, lagged_model_col)
        del df[model_col]
        return df

    def calculate_ba_ma(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        name = 'ba'
        model_col = f'{name}_{horizon}'
        lagged_model_col = f'{model_col}_L0'

        df[model_col] = df[f'ba_imbal_{horizon}'] * df[f'logret_resid_wgtmkt_{horizon}']
        log_col_summary(df, f'ba_imbal_{horizon}')
        log_col_summary(df, f'logret_resid_wgtmkt_{horizon}')
        df = self._make_bounded_feature(df, model_col)
        df = self.calc_ma(df, f"{model_col}_B")
        df[lagged_model_col] = df[f'{model_col}_B_ma'].fillna(0)
        df.loc[df[lagged_model_col].abs() < MIN_ALPHA, lagged_model_col] = 0.0

        log_col_summary(df, lagged_model_col)
        del df[model_col]
        return df

    def calculate_bal_ma(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Calculate ba-level model: standalone order book imbalance signal.

        Uses ba_imbal (log bid/ask size ratio) directly as a predictor of
        forward returns. Positive ba_imbal (more bids than asks) predicts
        positive forward returns.

        Args:
            df: DataFrame with ba_imbal feature
            horizon: Time horizon in minutes

        Returns:
            DataFrame with bal model forecast column
        """
        name = 'bal'
        model_col = f'{name}_{horizon}'
        lagged_model_col = f'{model_col}_L0'

        df[model_col] = df[f'ba_imbal_{horizon}']
        log_col_summary(df, f'ba_imbal_{horizon}')
        df = self._make_bounded_feature(df, model_col)
        df = self.calc_ma(df, f"{model_col}_B")
        df[lagged_model_col] = df[f'{model_col}_B_ma'].fillna(0)
        df.loc[df[lagged_model_col].abs() < MIN_ALPHA, lagged_model_col] = 0.0

        log_col_summary(df, lagged_model_col)
        del df[model_col]
        return df

    def calculate_slz(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        name = 'slz'
        model_col = f'{name}_{horizon}'
        lagged_model_col = f'{model_col}_L0'

        df[model_col] = df[f'logret_{horizon}_lz_cz']
        df = self._make_bounded_feature(df, base=model_col)
        df[lagged_model_col] = df[f"{model_col}_B"].fillna(0)
        df.loc[df[lagged_model_col].abs() < MIN_ALPHA, lagged_model_col] = 0.0

        log_col_summary(df, lagged_model_col)
        return df

    def calculate_rr(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        name = 'rr'
        model_col = f'{name}_{horizon}'
        lagged_model_col = f'{model_col}_L0'

        df[model_col] = df[f'logret_{horizon}']
        df = self._make_bounded_feature(df, base=model_col)
        df[lagged_model_col] = df[f"{model_col}_B"].fillna(0)
        df.loc[df[lagged_model_col].abs() < MIN_ALPHA, lagged_model_col] = 0.0

        log_col_summary(df, lagged_model_col)
        return df

    def calculate_ip(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        name = 'ip'
        model_col = f'{name}_{horizon}'
        lagged_model_col = f'{model_col}_L0'

        df[model_col] = df[f'index_mid_price_diff_mean_{horizon}']
        df = self._make_bounded_feature(df, base=model_col)
        df[lagged_model_col] = df[f"{model_col}_B"].fillna(0)
        df.loc[df[lagged_model_col].abs() < MIN_ALPHA, lagged_model_col] = 0.0

        log_col_summary(df, lagged_model_col)
        return df

    @staticmethod
    def calculate_oracle(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        name = 'oracle'
        model_col = f'{name}_{horizon}'
        lagged_model_col = f'{model_col}_L0'

        df[lagged_model_col] = df[f'y_raw1_{horizon}']
        log_col_summary(df, lagged_model_col)
        return df

    @staticmethod
    def calculate_goober(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        name = 'goober'
        model_col = f'{name}_{horizon}'
        lagged_model_col = f'{model_col}_L0'

        df[lagged_model_col] = np.random.random(df.shape[0]) * 2 - 1
        log_col_summary(df, lagged_model_col)
        return df
