"""Unit tests for wait_for_tardis_ready.py"""
import unittest
from datetime import date
from unittest.mock import patch, MagicMock

import requests

from wait_for_tardis_ready import get_exported_until, is_data_ready, wait_for_data


class TestGetExportedUntil(unittest.TestCase):
    """Tests for get_exported_until function."""

    @patch('wait_for_tardis_ready.requests.get')
    def test_successful_api_call(self, mock_get: MagicMock) -> None:
        """Test successful API response parsing."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'datasets': {
                'exportedUntil': '2025-12-10T00:00:00.000Z'
            }
        }
        mock_get.return_value = mock_response

        result = get_exported_until()

        self.assertEqual(result, date(2025, 12, 10))

    @patch('wait_for_tardis_ready.requests.get')
    def test_api_request_failure(self, mock_get: MagicMock) -> None:
        """Test handling of API request failure."""
        mock_get.side_effect = requests.RequestException("Connection error")

        result = get_exported_until()

        self.assertIsNone(result)

    @patch('wait_for_tardis_ready.requests.get')
    def test_missing_datasets_field(self, mock_get: MagicMock) -> None:
        """Test handling of missing datasets field."""
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_get.return_value = mock_response

        result = get_exported_until()

        self.assertIsNone(result)

    @patch('wait_for_tardis_ready.requests.get')
    def test_missing_exported_until_field(self, mock_get: MagicMock) -> None:
        """Test handling of missing exportedUntil field."""
        mock_response = MagicMock()
        mock_response.json.return_value = {'datasets': {}}
        mock_get.return_value = mock_response

        result = get_exported_until()

        self.assertIsNone(result)


class TestIsDataReady(unittest.TestCase):
    """Tests for is_data_ready function."""

    @patch('wait_for_tardis_ready.get_exported_until')
    def test_data_ready_when_exported_after_target(self, mock_get_exported: MagicMock) -> None:
        """Test returns True when exportedUntil > target_date."""
        mock_get_exported.return_value = date(2025, 12, 10)

        result = is_data_ready(target_date=date(2025, 12, 9))

        self.assertTrue(result)

    @patch('wait_for_tardis_ready.get_exported_until')
    def test_data_not_ready_when_exported_equals_target(self, mock_get_exported: MagicMock) -> None:
        """Test returns False when exportedUntil == target_date."""
        mock_get_exported.return_value = date(2025, 12, 10)

        result = is_data_ready(target_date=date(2025, 12, 10))

        self.assertFalse(result)

    @patch('wait_for_tardis_ready.get_exported_until')
    def test_data_not_ready_when_exported_before_target(self, mock_get_exported: MagicMock) -> None:
        """Test returns False when exportedUntil < target_date."""
        mock_get_exported.return_value = date(2025, 12, 9)

        result = is_data_ready(target_date=date(2025, 12, 10))

        self.assertFalse(result)

    @patch('wait_for_tardis_ready.get_exported_until')
    def test_data_not_ready_when_api_fails(self, mock_get_exported: MagicMock) -> None:
        """Test returns False when API call fails."""
        mock_get_exported.return_value = None

        result = is_data_ready(target_date=date(2025, 12, 9))

        self.assertFalse(result)


class TestWaitForData(unittest.TestCase):
    """Tests for wait_for_data function."""

    @patch('wait_for_tardis_ready.time.sleep')
    @patch('wait_for_tardis_ready.is_data_ready')
    def test_returns_immediately_when_ready(
        self, mock_is_ready: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """Test returns True immediately when data is ready."""
        mock_is_ready.return_value = True

        result = wait_for_data(
            target_date=date(2025, 12, 9),
            poll_interval=1,
            max_wait_time=10
        )

        self.assertTrue(result)
        self.assertEqual(mock_is_ready.call_count, 1)
        # 1 buffer sleep (2 min) after data ready
        self.assertEqual(mock_sleep.call_count, 1)

    @patch('wait_for_tardis_ready.time.sleep')
    @patch('wait_for_tardis_ready.is_data_ready')
    def test_polls_until_ready(
        self, mock_is_ready: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """Test polls multiple times until data is ready."""
        mock_is_ready.side_effect = [False, False, True]

        result = wait_for_data(
            target_date=date(2025, 12, 9),
            poll_interval=1,
            max_wait_time=100
        )

        self.assertTrue(result)
        self.assertEqual(mock_is_ready.call_count, 3)
        # 2 poll sleeps + 1 buffer sleep (2 min) after data ready
        self.assertEqual(mock_sleep.call_count, 3)

    @patch('wait_for_tardis_ready.time.sleep')
    @patch('wait_for_tardis_ready.is_data_ready')
    def test_timeout_when_never_ready(
        self, mock_is_ready: MagicMock, _mock_sleep: MagicMock
    ) -> None:
        """Test returns False when timeout exceeded."""
        mock_is_ready.return_value = False

        # Use a very short max_wait_time to trigger timeout quickly
        # Real time.time() will elapse fast enough in this tight loop
        result = wait_for_data(
            target_date=date(2025, 12, 9),
            poll_interval=1,
            max_wait_time=0  # Immediate timeout
        )

        self.assertFalse(result)
        # Should have checked at least once before timeout
        self.assertGreaterEqual(mock_is_ready.call_count, 1)


if __name__ == '__main__':
    unittest.main()
