import json
from collections import defaultdict
from datetime import datetime as dt
from datetime import timedelta as td, timezone
from typing import Optional, Tuple, List, Dict, Any, Union

from .time_util import time_str_to_dts
from .directory import CONFIG_DIR
from .util import logger, unique_list, PROD


def get_config(config_file: Optional[str] = None) -> Tuple[str, dict]:
    """Load configuration from JSON file.
    
    Args:
        config_file: Path to config file. If None, uses default config/config.json
        prod: Production flag (currently unused, kept for API compatibility)
        
    Returns:
        Tuple of (config_file_path, config_dict) where:
            config_file_path: Full path to loaded config file
            config_dict: Parsed configuration dictionary
    """
    if config_file is None:
        logger.info("No file specified, using default config!")
        config_file = CONFIG_DIR
        if PROD:
            config_file += "/config.json"
        else:
            config_file += "/config_dev.json"

    logger.info(f"Loading config {config_file}")
    config = json.load(open(config_file))
    return config_file, config


def override_config_value(config: dict, key: str, value: Union[str, Any]) -> dict:
    """Override a specific value in the config dictionary.
    
    Automatically parses string values to appropriate types:
    - Integers: "123" -> 123
    - Floats: "123.45" -> 123.45
    - Booleans: "true"/"false" -> True/False (case insensitive)
    - None/null: "none"/"null" -> None (case insensitive)
    - Lists: "[1,2,3]" -> [1, 2, 3] (using JSON parsing)
    - Dicts: '{"a": 1}' -> {"a": 1} (using JSON parsing)
    - Otherwise keeps as string
    
    Args:
        config: Configuration dictionary to modify
        key: Key to override (supports nested keys with dot notation, e.g., "FCASTS.15.models")
        value: Value to set (can be string that will be parsed, or already parsed value)
        
    Returns:
        Modified config dictionary (note: modifies in-place and returns the same dict)
        
    Raises:
        KeyError: If a parent key in a nested path doesn't exist
        ValueError: If JSON parsing fails for list/dict values
    """
    # If value is already parsed (not a string), use it directly
    if not isinstance(value, str):
        parsed_value = value
    else:
        # Parse string value to appropriate type
        value = value.strip()
        
        # Try JSON parsing first (handles lists, dicts, numbers, booleans, null)
        try:
            parsed_value = json.loads(value)
        except json.JSONDecodeError:
            # Not valid JSON, try other formats
            
            # Check for boolean strings
            if value.lower() in ['true', 'false']:
                parsed_value = value.lower() == 'true'
            # Check for none/null strings
            elif value.lower() in ['none', 'null']:
                parsed_value = None
            else:
                # Keep as string
                parsed_value = value
    
    # Handle nested keys (e.g., "FCASTS.15.models")
    if '.' in key:
        keys = key.split('.')
        current = config
        
        # Navigate to the parent of the final key
        for k in keys[:-1]:
            if k not in current:
                raise KeyError(f"Key '{k}' not found in config at path: {'.'.join(keys[:keys.index(k)+1])}")
            current = current[k]
        
        # Set the final value
        final_key = keys[-1]
        current[final_key] = parsed_value
        logger.info(f"Config override: {key} = {parsed_value} (type: {type(parsed_value).__name__})")
    else:
        # Simple key
        config[key] = parsed_value
        logger.info(f"Config override: {key} = {parsed_value} (type: {type(parsed_value).__name__})")
    
    return config


def get_factors(config: dict) -> List[str]:
    """Extract list of factor names from configuration.
    
    Args:
        config: Configuration dictionary containing FACTOR_SIGMAS
        
    Returns:
        List of factor names (e.g., ['dollar_exposure', 'dvolume_1440_trmean_cz'])
    """
    return list(config['FACTOR_SIGMAS'].keys())


def extract_security_fit_horizon_model(config: dict, horizon: int, model_name: str) -> bool:
    """Check if a specific model uses per-security fitting.
    
    Args:
        config: Configuration dictionary containing FCASTS
        horizon: Forecast horizon in minutes (e.g., 15, 60, 1440)
        model_name: Name of the model (e.g., 'hl', 'c2vwap', 'slz')
        
    Returns:
        True if the model uses security-specific fitting, False otherwise
    """
    model_security_fit = False
    for fcast in config['FCASTS'][str(horizon)]['models']:
        if fcast['name'] == model_name and 'security_fit' in fcast:
            model_security_fit = fcast['security_fit']
            break
    return model_security_fit


def extract_tree_features_horizon_model(config: dict, horizon: int, model_name: str) -> List[str]:
    """Extract feature list for a specific model and horizon.
    
    Args:
        config: Configuration dictionary containing FCASTS
        horizon: Forecast horizon in minutes (e.g., 15, 60, 1440)
        model_name: Name of the model (e.g., 'hl', 'c2vwap', 'slz')
        
    Returns:
        List of feature names with MODEL_HORIZON placeholders replaced by actual model/horizon
        (e.g., 'MODEL_HORIZON_alpha' becomes 'hl_15_alpha')
    """
    logger.info(f"Getting features for horizon {horizon}, {model_name}")
    model_features_list = config['FCASTS'][str(horizon)]['features']
    for fcast in config['FCASTS'][str(horizon)]['models']:
        if fcast['name'] == model_name and 'features' in fcast:
            logger.info(f"Update {horizon} {model_name} feature list from {model_features_list} to {fcast['features']}")
            model_features_list = fcast['features']
            break
    model_features_list = [feature.replace('MODEL_HORIZON', f'{model_name}_{horizon}') if feature.startswith('MODEL_HORIZON') else feature for feature in model_features_list]
    return model_features_list


def extract_all_tree_features_horizon(config: dict, horizon: int) -> Tuple[List[str], List[str]]:
    """Extract all features for a given horizon, separating horizon and model features.
    
    Extracts features for a given horizon from the config:
    1. Horizon features: All features that will be loaded from features/bars, which do not start with 'MODEL_HORIZON'
    2. Model features: Features that start with 'MODEL_HORIZON'. 'MODEL_HORIZON' will be replaced with '{model_name}_{horizon}' when used
    
    Args:
        config: Configuration dictionary containing FCASTS
        horizon: Forecast horizon in minutes (e.g., 15, 60, 1440)
        
    Returns:
        Tuple of (horizon_features, model_features) where:
            horizon_features: List of standard features from bars/features files
            model_features: List of MODEL_HORIZON placeholder features
    """
    logger.info(f"Getting features for horizon {horizon}")
    features_list = config['FCASTS'][str(horizon)]['features']
    for fcast in config['FCASTS'][str(horizon)]['models']:
        model_features_list = fcast.get('features', [])
        logger.info(f"Get {horizon} {fcast['name']} model-specific feature list {model_features_list}")
        features_list = unique_list(features_list + model_features_list)
    model_features = [hf for hf in features_list if hf.startswith('MODEL_HORIZON')]
    horizon_features = [feature for feature in features_list if feature not in model_features]
    return horizon_features, model_features


def extract_horizons(config: dict) -> List[int]:
    """Extract all forecast horizons from configuration.
    
    Args:
        config: Configuration dictionary containing FCASTS
        
    Returns:
        Sorted list of horizon values in minutes (e.g., [15, 60, 120, 360, 720, 1440])
    """
    horizons = sorted([int(h) for h in config['FCASTS'].keys()])
    return horizons


def extract_tree_features(config: dict, horizons: Optional[List[int]] = None) -> List[str]:
    """Extract all unique features across specified horizons.
    
    Args:
        config: Configuration dictionary containing FCASTS
        horizons: List of horizons to extract features for. If None, uses all horizons
        
    Returns:
        List of unique feature names across all specified horizons (excludes MODEL_HORIZON features)
    """
    required_features = set()
    if horizons is None:
        horizons = config['FCASTS'].keys()

    for horizon in horizons:
        horizon_features, _ = extract_all_tree_features_horizon(config, horizon)
        for hf in horizon_features:
            required_features.add(hf)
    return list(required_features)


def extract_models(config: dict, horizons: Optional[List[int]]) -> List[str]:
    """Extract unique model names across specified horizons.
    
    Args:
        config: Configuration dictionary containing FCASTS
        horizons: List of horizons to extract models for. If None, includes all horizons
        
    Returns:
        List of unique model names (e.g., ['hl', 'c2vwap', 'slz', 'vadj'])
    """
    models = set()
    for horizon, fcasts in config['FCASTS'].items():
        if horizons is not None and int(horizon) not in horizons:
            continue
        for fcast in fcasts['models']:
            models.add(fcast['name'])
    return list(models)


def extract_horizon_models(config: dict, horizons: Optional[List[int]] = None, models: Optional[List[str]] = None, exclude_short_term: bool = False, exclude_zero_weight: bool = False) -> List[Tuple[int, str]]:
    """Extract (horizon, model) pairs based on filtering criteria.
    
    Args:
        config: Configuration dictionary containing FCASTS
        horizons: List of horizons to include. If None, includes all horizons
        models: List of model names to include. If None, includes all models
        exclude_short_term: If True, excludes horizons in SHORT_TERM_MODEL_HORIZONS
        exclude_zero_weight: If True, excludes models with weight = 0
        
    Returns:
        List of (horizon, model_name) tuples (e.g., [(15, 'hl'), (60, 'c2vwap')])
    """
    horizon_models = []
    for horizon, fcasts in config['FCASTS'].items():
        horizon = int(horizon)
        if horizons is not None and horizon not in horizons:
            continue
        if exclude_short_term and horizon in config['SHORT_TERM_MODEL_HORIZONS']:
            continue
        for fcast in fcasts['models']:
            model = fcast['name']
            weight = fcast['weight']
            if exclude_zero_weight and weight == 0:
                continue
            if models is None or model in models:
                horizon_models.append((int(horizon), model))
    return horizon_models


def extract_max_lags(config: dict) -> Dict[int, int]:
    """Extract maximum lag value for each horizon across all models.
    
    Args:
        config: Configuration dictionary containing FCASTS
        
    Returns:
        Dictionary mapping horizon to maximum lag value
        (e.g., {15: 2, 60: 3, 1440: 5})
    """
    ret = defaultdict(int)
    for horizon, fcasts in config['FCASTS'].items():
        horizon = int(horizon)
        for fcast in fcasts['models']:
            model_lag = fcast['lags']
            if model_lag > ret[horizon]:
                ret[horizon] = model_lag
    return ret


def extract_reopt_times(config: dict, now: Optional[dt] = None) -> List[dt]:
    """Extract reoptimization times from config with offset applied.
    
    Args:
        config: Configuration dictionary containing REOPT_TIMES and OPT_OFFSET_MINS
        now: Current datetime for computing times. If None, uses dt.now(timezone.utc)
        
    Returns:
        Sorted list of timezone-aware UTC datetime objects representing reoptimization times 
        for today and tomorrow, with OPT_OFFSET_MINS added to each time
    """
    if now is None:
        now = dt.now(timezone.utc)
    time_strs = config["REOPT_TIMES"]
    offset = int(config['OPT_OFFSET_MINS'])

    times = time_str_to_dts(time_strs, now)
    # extend one more day for reopt times
    times += [tt + td(days=1) for tt in times]
    times = [tt + td(minutes=offset) for tt in times]
    return sorted(times)


def extract_model_alpha_list(config: dict, horizons: Optional[List[int]] = None, models: Optional[List[str]] = None) -> List[str]:
    """Extract list of alpha column names for specified horizons and models.
    
    Args:
        config: Configuration dictionary containing FCASTS
        horizons: List of horizons to include. If None, includes all horizons
        models: List of model names to include. If None, includes all models
        
    Returns:
        List of alpha column names (e.g., ['alpha_hl_15', 'alpha_c2vwap_60'])
    """
    horizon_models = extract_horizon_models(config, horizons, models)
    model_alpha_list = []
    for horizon, model in horizon_models:
        model_alpha_list.append(f'alpha_{model}_{horizon}')
    return model_alpha_list

def extract_compute_features(config: dict, horizon: int, prod: bool) -> List[str]:
    ftype = "prod" if prod else "non_prod"
    features = config["FEATURES"].get(str(horizon), {}).get(ftype, [])
    features = [ff.replace('HORIZON', str(horizon)) for ff in features]
    return features

def extract_feature_lookback_periods(config: dict, horizon: Optional[int]) -> int:
    if horizon is None:
        logger.info(f"Using default feature lookback periods")
        lookback_periods = config['DEFAULT_FEATURE_LOOKBACK_PERIODS']
    elif horizon in [1, 15, 60, 120, 360, 720, 1440]:
        lookback_periods = config['DEFAULT_FEATURE_LOOKBACK_PERIODS']
    elif horizon == 4320:
        lookback_periods = 30
    elif horizon == 10080:
        lookback_periods = 12
    elif horizon == 43200:
        lookback_periods = 3
    else:
        logger.error(f"Invalid frequency {horizon}")
        raise RuntimeError()

    return lookback_periods