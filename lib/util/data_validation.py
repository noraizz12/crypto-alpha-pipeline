"""Data validation utilities for market data quality checks.

This module provides functions for detecting common data quality issues
in cryptocurrency market data:
- Duplicate detection
- Gap/missing data detection
- Stale data detection
- Timestamp validation
- Value bounds checking

Example:
    from lib.util.data_validation import DataValidator

    validator = DataValidator(staleness_threshold_secs=60)
    report = validator.validate_bars(bars_df)

    if not report.is_valid:
        print(report.summary())
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class ValidationIssue:
    """Represents a single data validation issue."""
    issue_type: str
    severity: str  # 'error', 'warning', 'info'
    description: str
    affected_rows: int = 0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Summary of all validation issues found in a dataset."""
    issues: List[ValidationIssue] = field(default_factory=list)
    rows_checked: int = 0
    columns_checked: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_valid(self) -> bool:
        """True if no errors found (warnings are OK)."""
        return not any(issue.severity == 'error' for issue in self.issues)

    @property
    def error_count(self) -> int:
        """Number of error-level issues."""
        return sum(1 for issue in self.issues if issue.severity == 'error')

    @property
    def warning_count(self) -> int:
        """Number of warning-level issues."""
        return sum(1 for issue in self.issues if issue.severity == 'warning')

    def summary(self) -> str:
        """Generate a human-readable summary of the report."""
        lines = [
            f"Validation Report ({self.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')})",
            f"Rows checked: {self.rows_checked:,}",
            f"Columns checked: {self.columns_checked}",
            f"Status: {'PASS' if self.is_valid else 'FAIL'}",
            f"Errors: {self.error_count}, Warnings: {self.warning_count}",
            ""
        ]

        if self.issues:
            lines.append("Issues found:")
            for issue in self.issues:
                lines.append(f"  [{issue.severity.upper()}] {issue.issue_type}: {issue.description}")
                if issue.affected_rows > 0:
                    lines.append(f"    Affected rows: {issue.affected_rows:,}")

        return "\n".join(lines)


class DataValidator:
    """Validates market data for common quality issues.

    Attributes:
        staleness_threshold_secs: Maximum age before data is considered stale
        max_price_change_pct: Maximum allowed price change between bars
        min_volume: Minimum valid volume (zero volume often indicates issues)
        expected_frequency: Expected time between consecutive bars
    """

    def __init__(
        self,
        staleness_threshold_secs: int = 300,
        max_price_change_pct: float = 50.0,
        min_volume: float = 0.0,
        expected_frequency: str = '1min'
    ):
        self.staleness_threshold_secs = staleness_threshold_secs
        self.max_price_change_pct = max_price_change_pct
        self.min_volume = min_volume
        self.expected_frequency = pd.Timedelta(expected_frequency)

    def validate_bars(self, df: pd.DataFrame, symbol_col: str = 'symbol_venue') -> ValidationReport:
        """Run all validations on a bar DataFrame.

        Args:
            df: DataFrame with bar data (must have timestamp index)
            symbol_col: Column name containing symbol identifier

        Returns:
            ValidationReport with all issues found
        """
        report = ValidationReport(
            rows_checked=len(df),
            columns_checked=len(df.columns)
        )

        if df.empty:
            report.issues.append(ValidationIssue(
                issue_type='empty_data',
                severity='error',
                description='DataFrame is empty'
            ))
            return report

        # Run all validation checks
        report.issues.extend(self._check_duplicates(df, symbol_col))
        report.issues.extend(self._check_gaps(df, symbol_col))
        report.issues.extend(self._check_timestamps(df))
        report.issues.extend(self._check_price_bounds(df))
        report.issues.extend(self._check_volume_bounds(df))
        report.issues.extend(self._check_crossed_prices(df))
        report.issues.extend(self._check_null_values(df))

        return report

    def check_staleness(self, latest_ts: datetime, reference_ts: Optional[datetime] = None) -> ValidationIssue:
        """Check if data is stale based on timestamp.

        Args:
            latest_ts: Most recent timestamp in the data
            reference_ts: Reference time to compare against (defaults to now)

        Returns:
            ValidationIssue if stale, None otherwise
        """
        if reference_ts is None:
            reference_ts = datetime.now(timezone.utc)

        # Ensure both are timezone-aware
        if latest_ts.tzinfo is None:
            latest_ts = latest_ts.replace(tzinfo=timezone.utc)

        age_secs = (reference_ts - latest_ts).total_seconds()

        if age_secs > self.staleness_threshold_secs:
            return ValidationIssue(
                issue_type='stale_data',
                severity='warning',
                description=f'Data is {age_secs:.0f}s old (threshold: {self.staleness_threshold_secs}s)',
                details={'age_seconds': age_secs, 'latest_timestamp': str(latest_ts)}
            )
        return None

    def _check_duplicates(self, df: pd.DataFrame, symbol_col: str) -> List[ValidationIssue]:
        """Check for duplicate rows based on timestamp and symbol."""
        issues = []

        if symbol_col in df.columns:
            # Check duplicates per symbol
            duplicates = df.reset_index().duplicated(subset=['ts', symbol_col], keep=False)
            dup_count = duplicates.sum()
        else:
            # Check duplicates by timestamp only
            duplicates = df.index.duplicated(keep=False)
            dup_count = duplicates.sum()

        if dup_count > 0:
            issues.append(ValidationIssue(
                issue_type='duplicate_rows',
                severity='error',
                description=f'Found {dup_count} duplicate timestamp entries',
                affected_rows=dup_count
            ))

        return issues

    def _check_gaps(self, df: pd.DataFrame, symbol_col: str) -> List[ValidationIssue]:
        """Check for missing time periods in the data."""
        issues = []

        if symbol_col in df.columns:
            # Check gaps per symbol
            for symbol in df[symbol_col].unique():
                symbol_df = df[df[symbol_col] == symbol]
                gaps = self._find_gaps_in_series(symbol_df.index)
                if gaps:
                    issues.append(ValidationIssue(
                        issue_type='data_gaps',
                        severity='warning',
                        description=f'{symbol}: Found {len(gaps)} gap(s) in time series',
                        affected_rows=len(gaps),
                        details={'symbol': symbol, 'gap_count': len(gaps)}
                    ))
        else:
            gaps = self._find_gaps_in_series(df.index)
            if gaps:
                issues.append(ValidationIssue(
                    issue_type='data_gaps',
                    severity='warning',
                    description=f'Found {len(gaps)} gap(s) in time series',
                    affected_rows=len(gaps)
                ))

        return issues

    def _find_gaps_in_series(self, index: pd.DatetimeIndex) -> List[Tuple[datetime, datetime]]:
        """Find gaps in a datetime index."""
        if len(index) < 2:
            return []

        sorted_idx = index.sort_values()
        diffs = sorted_idx.to_series().diff()

        # Find gaps larger than expected frequency (with tolerance)
        tolerance = self.expected_frequency * 1.5
        gap_mask = diffs > tolerance

        gaps = []
        for i, is_gap in enumerate(gap_mask):
            if is_gap and i > 0:
                gaps.append((sorted_idx[i-1], sorted_idx[i]))

        return gaps

    def _check_timestamps(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate timestamp values."""
        issues = []

        # Check for future timestamps
        now = datetime.now(timezone.utc)
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            future_mask = df.index > now
        else:
            future_mask = pd.to_datetime(df.index, utc=True) > now

        future_count = future_mask.sum()
        if future_count > 0:
            issues.append(ValidationIssue(
                issue_type='future_timestamps',
                severity='error',
                description=f'Found {future_count} timestamps in the future',
                affected_rows=future_count
            ))

        # Check for very old timestamps (before 2017 - crypto futures didn't exist)
        min_valid_date = pd.Timestamp('2017-01-01', tz='UTC')
        try:
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                old_mask = df.index < min_valid_date
            else:
                old_mask = pd.to_datetime(df.index, utc=True) < min_valid_date

            old_count = old_mask.sum()
            if old_count > 0:
                issues.append(ValidationIssue(
                    issue_type='invalid_old_timestamps',
                    severity='error',
                    description=f'Found {old_count} timestamps before 2017',
                    affected_rows=old_count
                ))
        except Exception:
            pass  # Skip if comparison fails

        return issues

    def _check_price_bounds(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check for invalid price values."""
        issues = []
        price_cols = ['open', 'high', 'low', 'close', 'close_mid', 'close_trade']

        for col in price_cols:
            if col not in df.columns:
                continue

            # Check for negative prices
            neg_mask = df[col] < 0
            neg_count = neg_mask.sum()
            if neg_count > 0:
                issues.append(ValidationIssue(
                    issue_type='negative_price',
                    severity='error',
                    description=f'Column {col}: {neg_count} negative values',
                    affected_rows=neg_count
                ))

            # Check for zero prices (usually invalid for traded assets)
            zero_mask = df[col] == 0
            zero_count = zero_mask.sum()
            if zero_count > 0:
                issues.append(ValidationIssue(
                    issue_type='zero_price',
                    severity='warning',
                    description=f'Column {col}: {zero_count} zero values',
                    affected_rows=zero_count
                ))

        return issues

    def _check_volume_bounds(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check for invalid volume values."""
        issues = []
        volume_cols = ['volume', 'dvolume']

        for col in volume_cols:
            if col not in df.columns:
                continue

            # Check for negative volume
            neg_mask = df[col] < 0
            neg_count = neg_mask.sum()
            if neg_count > 0:
                issues.append(ValidationIssue(
                    issue_type='negative_volume',
                    severity='error',
                    description=f'Column {col}: {neg_count} negative values',
                    affected_rows=neg_count
                ))

        return issues

    def _check_crossed_prices(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check for crossed OHLC prices (high < low, etc.)."""
        issues = []

        if 'high' in df.columns and 'low' in df.columns:
            crossed_mask = df['high'] < df['low']
            crossed_count = crossed_mask.sum()
            if crossed_count > 0:
                issues.append(ValidationIssue(
                    issue_type='crossed_high_low',
                    severity='error',
                    description=f'Found {crossed_count} bars where high < low',
                    affected_rows=crossed_count
                ))

        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # Open/close should be within high/low range
            open_violation = (df['open'] > df['high']) | (df['open'] < df['low'])
            close_violation = (df['close'] > df['high']) | (df['close'] < df['low'])

            open_count = open_violation.sum()
            close_count = close_violation.sum()

            if open_count > 0:
                issues.append(ValidationIssue(
                    issue_type='open_outside_range',
                    severity='error',
                    description=f'Found {open_count} bars where open is outside high/low range',
                    affected_rows=open_count
                ))

            if close_count > 0:
                issues.append(ValidationIssue(
                    issue_type='close_outside_range',
                    severity='error',
                    description=f'Found {close_count} bars where close is outside high/low range',
                    affected_rows=close_count
                ))

        return issues

    def _check_null_values(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check for null/NaN values in critical columns."""
        issues = []
        critical_cols = ['open', 'high', 'low', 'close', 'volume']

        for col in critical_cols:
            if col not in df.columns:
                continue

            null_count = df[col].isna().sum()
            if null_count > 0:
                pct = (null_count / len(df)) * 100
                severity = 'error' if pct > 5 else 'warning'
                issues.append(ValidationIssue(
                    issue_type='null_values',
                    severity=severity,
                    description=f'Column {col}: {null_count} null values ({pct:.1f}%)',
                    affected_rows=null_count
                ))

        return issues


def validate_websocket_sequence(
    messages: List[Dict[str, Any]],
    sequence_field: str = 'E'
) -> Tuple[bool, List[int]]:
    """Check for gaps in WebSocket message sequence numbers.

    Args:
        messages: List of WebSocket message dictionaries
        sequence_field: Field name containing sequence number (Binance uses 'E' for event time)

    Returns:
        Tuple of (is_valid, list_of_missing_sequences)
    """
    if not messages:
        return True, []

    sequences = sorted(msg.get(sequence_field, 0) for msg in messages)
    missing = []

    for i in range(1, len(sequences)):
        if sequences[i] - sequences[i-1] > 1:
            # Found a gap
            for seq in range(sequences[i-1] + 1, sequences[i]):
                missing.append(seq)

    return len(missing) == 0, missing


def check_data_freshness(
    df: pd.DataFrame,
    max_age_seconds: int = 60,
    ts_column: Optional[str] = None
) -> bool:
    """Quick check if data is fresh enough.

    Args:
        df: DataFrame to check
        max_age_seconds: Maximum allowed age in seconds
        ts_column: Column containing timestamps (uses index if None)

    Returns:
        True if data is fresh, False if stale
    """
    if df.empty:
        return False

    if ts_column:
        latest_ts = pd.to_datetime(df[ts_column].max())
    else:
        latest_ts = pd.to_datetime(df.index.max())

    if latest_ts.tzinfo is None:
        latest_ts = latest_ts.replace(tzinfo=timezone.utc)

    age = (datetime.now(timezone.utc) - latest_ts).total_seconds()
    return age <= max_age_seconds
