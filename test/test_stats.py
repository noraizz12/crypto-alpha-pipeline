import unittest
from datetime import datetime as dt, timedelta as td, timezone
from unittest.mock import Mock, MagicMock, patch, call
import numpy as np
import pandas as pd

from lib.sim.sim_util import (
    calc_return_metrics, get_return_metrics_str
)
from lib.util.util import PNL_START_DATE
from lib.pnl.fill_pnl_breakdown import PNL_BREAKDOWN_FEATURES
from lib.pnl.fill_pnl_symbol import (
    PNL_DF_COLUMNS, DAILY_PNL_DF_COLUMNS,
    CalcMultiSymbolFillPnl, parse_fill_data
)
from lib.pnl.pnl_util import DEFAULT_COMMISSION_ASSET_SYMBOL_VENUE, get_commission_px_dict, calculate_top_drawdowns
from lib.pnl import FillBreakdown, FillPnl
from lib.trader.trading import Side, Fill
from lib.util.time_util import date_str_to_dt, today, yesterday
from lib.util.util import TRADING_START_DT


def init_calc_symbol_venue(calc: CalcMultiSymbolFillPnl, symbol_venue: str = 'BTCUSDT_binance-futures'):
    """Helper to initialize a symbol venue for CalcMultiSymbolFillPnl tests."""
    empty_bars = pd.DataFrame().set_index(pd.MultiIndex.from_tuples([], names=['ts', 'symbol_venue']))
    calc._get_initial_symbol_venues(None, empty_bars, None, None)
    # Manually add symbol_venue since we have no data
    calc.symbol_venues = [symbol_venue]
    calc.last_processed_fill_idx[symbol_venue] = -1
    calc.last_processed_funding_idx[symbol_venue] = -1
    calc.realized_pnl[symbol_venue] = 0.0
    calc.funding_income[symbol_venue] = 0.0
    calc.fees[symbol_venue] = 0.0
    calc.fees_usd[symbol_venue] = 0.0
    calc.fill_cnt[symbol_venue] = 0
    calc.dollars_traded[symbol_venue] = 0.0
    calc.dollars_case[Side.BUY][symbol_venue] = 0.0
    calc.dollars_case[Side.SELL][symbol_venue] = 0.0
    # Initialize inventory arrays
    calc.inventory_case[Side.BUY][symbol_venue] = np.array(
        [], dtype=[('price', float), ('quantity', float), ('ts', 'datetime64[ns]')]
    )
    calc.inventory_case[Side.SELL][symbol_venue] = np.array(
        [], dtype=[('price', float), ('quantity', float), ('ts', 'datetime64[ns]')]
    )


class TestFillBreakdown(unittest.TestCase):
    """Test cases for FillBreakdown class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.start_dt = dt(2024, 1, 1, tzinfo=timezone.utc)
        self.end_dt = dt(2024, 1, 31, tzinfo=timezone.utc)
        
    def test_init_default_dates(self):
        """Test FillBreakdown initialization with default dates."""
        mock_today = dt(2024, 2, 1, tzinfo=timezone.utc)
        with patch('lib.pnl.fill_pnl_breakdown.today', return_value=mock_today), \
             patch('lib.pnl.fill_pnl_breakdown.TRADING_START_DT', dt(2024, 1, 1, tzinfo=timezone.utc)):
            fb = FillBreakdown()
            self.assertEqual(fb.start, dt(2024, 1, 1, tzinfo=timezone.utc))
            self.assertEqual(fb.end, mock_today)
            self.assertIsNone(fb.fills_df)
            self.assertIsNone(fb.features_df)
            self.assertEqual(fb.features_dfs, [])
            self.assertEqual(fb.features_name_list, [])
            self.assertEqual(fb.bars_name_list, [])
            self.assertIsNotNone(fb.data_loader)
            
    def test_init_custom_dates(self):
        """Test FillBreakdown initialization with custom dates."""
        fb = FillBreakdown(start=self.start_dt, end=self.end_dt)
        self.assertEqual(fb.start, self.start_dt)
        self.assertEqual(fb.end, self.end_dt)
        
    def test_init_invalid_dates(self):
        """Test FillBreakdown initialization with invalid date range."""
        with self.assertRaises(AssertionError):
            FillBreakdown(start=self.end_dt, end=self.start_dt)
            
    def test_load_fills(self):
        """Test loading fills data."""
        fb = FillBreakdown(self.start_dt, self.end_dt)
        fills_df = pd.DataFrame({
            'symbol_venue': ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures'],
            'date': [self.start_dt, self.start_dt],
            'realized_pnl': [100.0, -50.0],
            'expanding': [True, False]
        })
        fb.load_fills(fills_df)
        pd.testing.assert_frame_equal(fb.fills_df, fills_df)
        
    @patch('lib.data.dataloader.DataLoader')
    def test_load_bars_data(self, mock_data_loader):
        """Test loading bars data."""
        fb = FillBreakdown(self.start_dt, self.end_dt)
        fb.data_loader = mock_data_loader
        
        # Test successful load
        mock_bars_df = pd.DataFrame({
            'ts': [self.start_dt],
            'symbol_venue': ['BTCUSDT_binance-futures'],
            'dvolume_1440': [1000000.0]
        })
        mock_bars_df = mock_bars_df.set_index(['ts', 'symbol_venue'])
        fb.data_loader.load_bars.return_value = mock_bars_df
        
        result = fb._load_bars_data(self.start_dt.date(), self.end_dt.date(), ['dvolume_1440'])
        
        # Should have date column added
        self.assertIn('date', result.columns)
        fb.data_loader.load_bars.assert_called_once_with(
            horizon=1440, 
            start_date=self.start_dt.date(), 
            end_date=self.end_dt.date(), 
            cols=['dvolume_1440']
        )
        
    @patch('lib.data.dataloader.DataLoader')
    def test_load_bars_data_fallback(self, mock_data_loader):
        """Test loading bars data with fallback to previous day."""
        fb = FillBreakdown(self.start_dt, self.end_dt)
        fb.data_loader = mock_data_loader.return_value
        
        # First call returns None, second call returns data
        mock_bars_df = pd.DataFrame({
            'ts': [self.start_dt - td(days=1)],
            'symbol_venue': ['BTCUSDT_binance-futures'],
            'dvolume_1440': [1000000.0]
        })
        mock_bars_df = mock_bars_df.set_index(['ts', 'symbol_venue'])
        fb.data_loader.load_bars.side_effect = [None, mock_bars_df]
        
        result = fb._load_bars_data(self.start_dt.date(), self.end_dt.date())
        
        # Date should be shifted forward by 1 day
        expected_date = (self.start_dt - td(days=1)).date()
        self.assertEqual(result['date'].iloc[0].date(), expected_date)
        
    def test_win_ratio_empty_fills(self):
        """Test win_ratio with empty fills."""
        fb = FillBreakdown(self.start_dt, self.end_dt)
        fb.fills_df = pd.DataFrame()
        
        win_ratio, num_profit, num_contracting, num_trades, gain_per_fill, loss_per_fill = fb.win_ratio()
        
        self.assertEqual(win_ratio, 0)
        self.assertEqual(num_profit, 0)
        self.assertEqual(num_contracting, 0)
        self.assertEqual(num_trades, 0)
        self.assertEqual(gain_per_fill, 0)
        self.assertEqual(loss_per_fill, 0)
        
    def test_win_ratio_with_fills(self):
        """Test win_ratio calculation with fills data."""
        fb = FillBreakdown(self.start_dt, self.end_dt)
        fb.fills_df = pd.DataFrame({
            'date': pd.to_datetime([self.start_dt] * 5),
            'expanding': [False, False, False, True, True],
            'realized_pnl': [100, -50, 200, 0, 0]
        })
        
        win_ratio, num_profit, num_contracting, num_trades, gain_per_fill, loss_per_fill = fb.win_ratio()
        
        self.assertEqual(num_trades, 5)
        self.assertEqual(num_contracting, 3)
        self.assertEqual(num_profit, 2)
        self.assertAlmostEqual(win_ratio, 2/3)
        self.assertAlmostEqual(gain_per_fill, 150.0)  # (100 + 200) / 2
        self.assertAlmostEqual(loss_per_fill, -50.0)
        
    def test_win_ratio_with_start_date(self):
        """Test win_ratio with start date filter."""
        fb = FillBreakdown(self.start_dt, self.end_dt)
        filter_date = self.start_dt + td(days=1)
        fb.fills_df = pd.DataFrame({
            'date': pd.to_datetime([self.start_dt, filter_date, filter_date]),
            'expanding': [False, False, False],
            'realized_pnl': [100, -50, 200]
        })
        
        win_ratio, num_profit, num_contracting, num_trades, gain_per_fill, loss_per_fill = fb.win_ratio(start_date=filter_date.date())
        
        self.assertEqual(num_trades, 2)  # Only trades from filter_date onwards
        self.assertEqual(num_contracting, 2)
        self.assertEqual(num_profit, 1)
        self.assertAlmostEqual(win_ratio, 0.5)

    def test_win_ratio_none_fills(self) -> None:
        """Test win_ratio with fills_df = None (trading stopped)."""
        fb = FillBreakdown(self.start_dt, self.end_dt)
        fb.fills_df = None

        win_ratio, num_profit, num_contracting, num_trades, gain_per_fill, loss_per_fill = fb.win_ratio()

        self.assertEqual(win_ratio, 0)
        self.assertEqual(num_profit, 0)
        self.assertEqual(num_contracting, 0)
        self.assertEqual(num_trades, 0)
        self.assertEqual(gain_per_fill, 0)
        self.assertEqual(loss_per_fill, 0)

    @patch('lib.pnl.fill_pnl_breakdown.make_quintile')
    def test_pnl_by_col(self, mock_make_quintile):
        """Test pnl_by_col method."""
        fb = FillBreakdown(self.start_dt, self.end_dt)
        
        # Set up test data
        fb.fills_df = pd.DataFrame({
            'date': [self.start_dt.date()] * 3,
            'symbol_venue': ['BTCUSDT_binance-futures'] * 3,
            'realized_pnl': [100, -50, 200]
        })
        
        fb.features_df = pd.DataFrame({
            'date': [self.start_dt.date()] * 3,
            'symbol_venue': ['BTCUSDT_binance-futures'] * 3,
            'dvolume_1440': [1000000, 2000000, 3000000]
        })
        
        # Mock quintile calculation
        mock_make_quintile.return_value = pd.DataFrame({
            'date': [self.start_dt.date()] * 3,
            'symbol_venue': ['BTCUSDT_binance-futures'] * 3,
            'realized_pnl': [100, -50, 200],
            'dvolume_1440': [1000000, 2000000, 3000000],
            'dvolume_1440_quintile': [1, 3, 5]
        })
        
        result = fb.pnl_by_col('dvolume_1440')
        
        self.assertIsNotNone(result)
        self.assertIn('realized_pnl', result.columns)
        mock_make_quintile.assert_called_once()
        
    def test_pnl_by_col_missing_data(self):
        """Test pnl_by_col with missing data."""
        fb = FillBreakdown(self.start_dt, self.end_dt)
        fb.fills_df = None
        fb.features_df = None
        
        result = fb.pnl_by_col('dvolume_1440')
        self.assertIsNone(result)
        
    def test_get_pnl_breakdowns(self):
        """Test get_pnl_breakdowns method."""
        fb = FillBreakdown(self.start_dt, self.end_dt)
        
        # Mock pnl_by_col to return simple DataFrames
        test_df = pd.DataFrame({'realized_pnl': [100, 200]})
        fb.pnl_by_col = Mock(return_value=test_df)
        
        # Test with specific column
        result = fb.get_pnl_breakdowns(col='test_col')
        
        self.assertIn('test_col', result)
        self.assertEqual(len(result), 1)
        fb.pnl_by_col.assert_called_once_with('test_col', by_date=False, merge_on_ts=False)
        
    def test_get_pnl_breakdowns_cumulative(self):
        """Test get_pnl_breakdowns with cumulative PnL."""
        fb = FillBreakdown(self.start_dt, self.end_dt)
        
        # Create multi-index DataFrame for testing
        test_df = pd.DataFrame({
            'date': [self.start_dt.date()] * 2,
            'quintile': [1, 2],
            'realized_pnl': [100, 200]
        }).set_index(['date', 'quintile'])
        
        fb.pnl_by_col = Mock(return_value=test_df)
        
        result = fb.get_pnl_breakdowns(use_cum_pnl=True)
        
        # Should process all PNL_BREAKDOWN_FEATURES
        self.assertEqual(len(result), len(PNL_BREAKDOWN_FEATURES))
        

class TestFillPnl(unittest.TestCase):
    """Test cases for FillPnl class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.start_dt = dt(2024, 1, 1, tzinfo=timezone.utc)
        self.end_dt = dt(2024, 1, 31, tzinfo=timezone.utc)
        self.config = {
            'SYMBOL_UNIVERSE': ['BTCUSDT', 'ETHUSDT'],
            'DYNAMIC_UNIVERSE': False
        }  # Minimal config for testing
        
    @patch('lib.pnl.fill_pnl.DataLoader')
    def test_init(self, mock_data_loader):
        """Test FillPnl initialization."""
        # Mock load_prebar_files to return a valid DataFrame
        mock_bars_df = pd.DataFrame({
            'close': [100.0, 101.0],
            'vwap': [100.5, 100.5]
        }, index=pd.MultiIndex.from_tuples([
            (dt(2024, 1, 1, tzinfo=timezone.utc), 'BTCUSDT_binance'),
            (dt(2024, 1, 2, tzinfo=timezone.utc), 'BTCUSDT_binance')
        ], names=['ts', 'symbol_venue']))
        mock_data_loader.return_value.load_prebar_files.return_value = mock_bars_df

        fp = FillPnl(config=self.config, start=self.start_dt, end=self.end_dt)

        self.assertEqual(fp.start_dt, self.start_dt)
        self.assertEqual(fp.end_dt, self.end_dt)
        self.assertIsNone(fp.fills_df)
        self.assertIsNone(fp.positions_df)
        self.assertIsNone(fp.fundings_df)

        # Check DataLoader was initialized
        mock_data_loader.assert_called_once()
        
    def test_init_default_dates(self):
        """Test FillPnl initialization with default dates."""
        mock_now = dt(2024, 2, 1, tzinfo=timezone.utc)

        # Mock load_prebar_files to return a valid DataFrame
        mock_bars_df = pd.DataFrame({
            'close': [100.0],
            'vwap': [100.5]
        }, index=pd.MultiIndex.from_tuples([
            (dt(2024, 1, 1, tzinfo=timezone.utc), 'BTCUSDT_binance')
        ], names=['ts', 'symbol_venue']))

        with patch('lib.pnl.fill_pnl.dt') as mock_dt, \
             patch('lib.pnl.fill_pnl.DataLoader') as mock_data_loader, \
             patch('lib.pnl.fill_pnl.TRADING_START_DT', dt(2024, 1, 1, tzinfo=timezone.utc)):
            mock_dt.now.return_value = mock_now
            mock_data_loader.return_value.load_prebar_files.return_value = mock_bars_df
            fp = FillPnl(config={'SYMBOL_UNIVERSE': [], 'DYNAMIC_UNIVERSE': False})

            self.assertEqual(fp.start_dt, dt(2024, 1, 1, tzinfo=timezone.utc))
            self.assertEqual(fp.end_dt, mock_now)

        
    # test_load_fills removed - load_fills method and fill_loader no longer exist
    # test_load_fills_none removed - load_fills method and fill_loader no longer exist
        

class TestCalcMultiSymbolFillPnl(unittest.TestCase):
    """Test cases for CalcMultiSymbolFillPnl class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.start_dt = dt(2024, 1, 1, tzinfo=timezone.utc)
        self.end_dt = dt(2024, 1, 31, tzinfo=timezone.utc)
        self.calc = CalcMultiSymbolFillPnl(self.start_dt, self.end_dt)
        
        
    def test_init(self):
        """Test CalcMultiSymbolFillPnl initialization."""
        self.assertEqual(self.calc.start_dt, self.start_dt)
        self.assertEqual(self.calc.end_dt, self.end_dt)
        self.assertEqual(self.calc.fills_dict, {})
        self.assertEqual(self.calc.commission_px_dict, {})
        self.assertEqual(self.calc.bars_dict, {})
        self.assertEqual(self.calc.symbol_venues, [])
        
    def test_get_commission_px_dict(self):
        """Test commission price extraction."""
        # Create test data
        close_mid_df = pd.DataFrame({
            'ts': [self.start_dt, self.start_dt],
            'symbol_venue': ['BNBUSDT_binance-futures', 'BTCUSDT_binance-futures'],
            'close_mid': [300.0, 50000.0]
        }).set_index(['ts', 'symbol_venue'])
        
        result = get_commission_px_dict(close_mid_df)
        
        self.assertIn(self.start_dt, result)
        self.assertEqual(result[self.start_dt]['BNB'], 300.0)
        self.assertEqual(result[self.start_dt]['USDT'], 1.0)
        
    def test_get_initial_symbol_venues(self):
        """Test symbol venue extraction and initialization."""
        fills_df = pd.DataFrame({'symbol_venue': ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures']})
        bars_df = pd.DataFrame({
            'ts': [self.start_dt],
            'symbol_venue': ['BTCUSDT_binance-futures'],
            'close_mid': [50000.0]
        }).set_index(['ts', 'symbol_venue'])
        
        self.calc._get_initial_symbol_venues(fills_df, bars_df, None, None)
        
        self.assertIn('BTCUSDT_binance-futures', self.calc.symbol_venues)
        self.assertIn('ETHUSDT_binance-futures', self.calc.symbol_venues)
        self.assertEqual(len(self.calc.symbol_venues), 2)
        
        # Check initialization of tracking dicts
        for sv in self.calc.symbol_venues:
            self.assertEqual(self.calc.last_processed_fill_idx[sv], -1)
            self.assertEqual(self.calc.realized_pnl[sv], 0.0)
            self.assertEqual(self.calc.fees[sv], 0.0)
            
    def test_parse_fill_data(self):
        """Test parsing fill data from numpy array."""
        fill_data = np.array([
            self.start_dt,  # ts
            'BTCUSDT',      # symbol
            0.1,            # fill_qty
            50000.0,        # fill_px
            'BUY',          # side
            0.005,          # commission
            'BNB'           # commission_asset
        ])
        
        fill = parse_fill_data(fill_data)
        
        self.assertEqual(fill.symbol, 'BTCUSDT')
        self.assertEqual(fill.side, Side.BUY)
        self.assertEqual(fill.px, 50000.0)
        self.assertEqual(fill.qty, 0.1)
        self.assertEqual(fill.commission, 0.005)
        self.assertEqual(fill.commission_asset, 'BNB')
        
    def test_match_fills_with_inventory_expanding(self):
        """Test matching fills for expanding positions."""
        init_calc_symbol_venue(self.calc, 'BTCUSDT_binance-futures')
        
        # Create expanding fill
        fill = Fill(
            symbol='BTCUSDT',
            side=Side.BUY,
            px=50000.0,
            qty=1.0,
            exch_ts=self.start_dt,
            recv_ts=self.start_dt,
            commission=0.0,
            commission_asset='USDT'
        )
        
        self.calc._match_fills_with_inventory(fill, 'BTCUSDT_binance-futures')
        
        # Check inventory updated
        self.assertEqual(len(self.calc.inventory_case[Side.BUY]['BTCUSDT_binance-futures']), 1)
        self.assertEqual(self.calc.inventory_case[Side.BUY]['BTCUSDT_binance-futures'][0]['quantity'], 1.0)
        self.assertEqual(self.calc.inventory_case[Side.BUY]['BTCUSDT_binance-futures'][0]['price'], 50000.0)
        
        # Check fill recorded
        self.assertEqual(len(self.calc.fills_pnl_list), 1)
        self.assertTrue(self.calc.fills_pnl_list[0]['expanding'])
        self.assertEqual(self.calc.fills_pnl_list[0]['realized_pnl'], 0.0)
        
    def test_match_fills_with_inventory_contracting(self):
        """Test matching fills for contracting positions."""
        init_calc_symbol_venue(self.calc, 'BTCUSDT_binance-futures')
        
        # Set up initial long position with proper datetime type
        self.calc.inventory_case[Side.BUY]['BTCUSDT_binance-futures'] = np.array(
            [(50000.0, 1.0, pd.Timestamp(self.start_dt))], 
            dtype=[('price', float), ('quantity', float), ('ts', 'datetime64[ns]')]
        )
        self.calc.inventory_case[Side.SELL]['BTCUSDT_binance-futures'] = np.array(
            [], dtype=[('price', float), ('quantity', float), ('ts', 'datetime64[ns]')]
        )
        
        # Create contracting fill (sell)
        fill = Fill(
            symbol='BTCUSDT',
            side=Side.SELL,
            px=51000.0,
            qty=0.5,
            exch_ts=self.start_dt + td(hours=1),
            recv_ts=self.start_dt + td(hours=1),
            commission=0.0,
            commission_asset='USDT'
        )
        
        self.calc._match_fills_with_inventory(fill, 'BTCUSDT_binance-futures')
        
        # Check realized PnL
        expected_pnl = (51000.0 - 50000.0) * 0.5
        self.assertEqual(self.calc.realized_pnl['BTCUSDT_binance-futures'], expected_pnl)
        
        # Check remaining inventory
        self.assertEqual(len(self.calc.inventory_case[Side.BUY]['BTCUSDT_binance-futures']), 1)
        self.assertEqual(self.calc.inventory_case[Side.BUY]['BTCUSDT_binance-futures'][0]['quantity'], 0.5)
        
    def test_calculate_position_info_at_bar(self):
        """Test position info calculation."""
        init_calc_symbol_venue(self.calc, 'BTCUSDT_binance-futures')
        
        # Set up positions - use pandas Timestamp
        self.calc.inventory_case[Side.BUY]['BTCUSDT_binance-futures'] = np.array(
            [(50000.0, 1.0, pd.Timestamp(self.start_dt))], 
            dtype=[('price', float), ('quantity', float), ('ts', 'datetime64[ns]')]
        )
        
        # Calculate at current price
        current_price = 52000.0
        unrealized_pnl, position, age = self.calc._calculate_position_info_at_bar(
            'BTCUSDT_binance-futures', current_price, self.start_dt + td(days=5), calculate_position_age=True
        )
        
        self.assertEqual(unrealized_pnl, 2000.0)  # (52000 - 50000) * 1.0
        self.assertEqual(position, 1.0)
        self.assertEqual(age, 5)  # 5 days old
        

    def test_update_pnl_from_new_fills(self):
        """Test processing new fills and updating PnL."""
        init_calc_symbol_venue(self.calc, 'BTCUSDT_binance-futures')
        
        # Set up fill data
        fill_array = np.array([
            [self.start_dt, 'BTCUSDT', 1.0, 50000.0, 'BUY', 5.0, 'USDT'],
            [self.start_dt + td(hours=1), 'BTCUSDT', 0.5, 51000.0, 'SELL', 2.5, 'USDT']
        ])
        self.calc.fills_dict['BTCUSDT_binance-futures'] = fill_array
        self.calc.commission_px_dict[self.start_dt] = {'USDT': 1.0}
        
        # Process first fill
        self.calc._update_pnl_from_new_fills('BTCUSDT_binance-futures', self.start_dt)
        
        self.assertEqual(self.calc.last_processed_fill_idx['BTCUSDT_binance-futures'], 0)
        self.assertEqual(self.calc.fees['BTCUSDT_binance-futures'], 5.0)
        self.assertEqual(self.calc.fees_usd['BTCUSDT_binance-futures'], 5.0)
        
    def test_update_pnl_from_fundings(self):
        """Test processing funding income."""
        init_calc_symbol_venue(self.calc, 'BTCUSDT_binance-futures')
        
        # Set up funding data
        funding_array = np.array([
            [self.start_dt, 'BTCUSDT', 10.0],
            [self.start_dt + td(hours=8), 'BTCUSDT', -5.0]
        ])
        self.calc.funding_income_dict['BTCUSDT_binance-futures'] = funding_array
        
        # Process funding up to first timestamp
        self.calc._update_pnl_from_fundings('BTCUSDT_binance-futures', self.start_dt)
        
        self.assertEqual(self.calc.last_processed_funding_idx['BTCUSDT_binance-futures'], 0)
        self.assertEqual(self.calc.funding_income['BTCUSDT_binance-futures'], 10.0)
        
    def test_calculate_pnl_performance_metrics(self):
        """Test full PnL calculation pipeline."""
        # Set up data
        bars_df = pd.DataFrame({
            'ts': [self.start_dt, self.start_dt + td(hours=1)],
            'symbol_venue': ['BTCUSDT_binance-futures'] * 2,
            'close_mid': [50000.0, 51000.0]
        }).set_index(['ts', 'symbol_venue'])

        fills_df = pd.DataFrame({
            'ts': [self.start_dt],
            'symbol_venue': ['BTCUSDT_binance-futures'],
            'symbol': ['BTCUSDT'],
            'fill_qty': [1.0],
            'fill_px': [50000.0],
            'side': ['BUY'],
            'commission': [5.0],
            'commission_asset': ['USDT']
        })

        self.calc.load_data(fills_df, bars_df, None, None, update=False)

        # Run calculation
        result = self.calc.calculate_pnl_performance_metrics(self.start_dt)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)  # Two bar timestamps
        self.assertIn('realized_daily', result.columns)
        self.assertIn('unrealized_daily', result.columns)
        self.assertIn('qty', result.columns)
        
    def test_aggregate_pnl_timeslice(self):
        """Test aggregating PnL by timestamp."""
        pnl_df = pd.DataFrame({
            'ts': [self.start_dt, self.start_dt],
            'symbol_venue': ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures'],
            'total_pnl_cumulative': [100, 50],
            'notional': [10000, -5000],
            'dollars_buy_cumulative': [10000, 0],
            'dollars_sell_cumulative': [0, -5000],
            'fees_usd_cumulative': [5, 2.5],
            'funding_income_cumulative': [2, 1]
        })
        
        result = self.calc.aggregate_pnl_timeslice(pnl_df)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result['pnl'].iloc[0], 150)
        self.assertEqual(result['long'].iloc[0], 10000)
        self.assertEqual(result['short'].iloc[0], -5000)
        
    def test_aggregate_pnl_performance_metrics(self):
        """Test aggregating PnL to daily level."""
        pnl_df = pd.DataFrame({
            'ts': [self.start_dt, self.start_dt + td(hours=12)],
            'symbol_venue': ['BTCUSDT_binance-futures'] * 2,
            'date': [self.start_dt] * 2,
            'realized_daily': [100, 200],
            'unrealized_daily': [50, 100],
            'total_pnl_daily': [150, 300],
            'notional': [10000, 10000],
            'fees_daily': [5, 10],
            'fees_usd_daily': [5, 10],
            'funding_income_daily': [2, 4],
            'fill_cnt_daily': [1, 2],
            'dollars_traded_daily': [10000, 20000],
            'dollars_buy_daily': [10000, 20000],
            'dollars_sell_daily': [0, 0]
        }).set_index(['ts', 'symbol_venue'])
        
        result = self.calc.aggregate_pnl_performance_metrics(pnl_df)
        
        self.assertEqual(len(result), 1)
        self.assertIn('return_daily', result.columns)
        self.assertIn('sharpe_lifetime', result.columns)
        self.assertIn('total_pnl_lifetime', result.columns)
        
    def test_calculate_top_drawdowns(self):
        """Test drawdown calculation."""
        daily_pnl_df = pd.DataFrame({
            'date': pd.date_range(self.start_dt, periods=10, freq='D'),
            'total_pnl_lifetime': [0, 100, 200, 150, 100, 50, 150, 250, 200, 300],
            'notional_abs_daily': [10000] * 10
        }).set_index('date')
        
        result = calculate_top_drawdowns(daily_pnl_df)
        
        self.assertLessEqual(len(result), 3)
        self.assertIn('start_date', result.columns)
        self.assertIn('end_date', result.columns)
        self.assertIn('dollar_loss', result.columns)
        self.assertIn('percent_loss', result.columns)
        

class TestReturnMetrics(unittest.TestCase):
    """Test cases for return metrics functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample PnL data
        self.pnl_df = pd.DataFrame({
            'pnl': [0, 100, 150, 140, 200],
            'notional': [10000, 10000, 10000, 10000, 10000],
            'long': [10000, 10000, 10000, 10000, 10000],
            'short': [0, 0, 0, 0, 0],
            'traded_long': [5000, 3000, 4000, 2000, 6000],
            'traded_short': [0, 0, 0, 0, 0],
            'fees_usd': [0, 5, 10, 15, 20],
            'funding_income': [0, 2, 4, 6, 8]
        })
        
    def test_calc_return_metrics(self):
        """Test calc_return_metrics function."""
        metrics = calc_return_metrics(self.pnl_df, daily_scaler=1)
        
        self.assertEqual(metrics['cum_pnl'], 200)
        self.assertAlmostEqual(metrics['cum_ret'], 0.02, places=4)  # 200/10000
        self.assertEqual(metrics['max_drawdown'], -10)  # 140 - 150
        self.assertAlmostEqual(metrics['max_drawdown_perc'], -0.001, places=4)  # -10/10000
        self.assertEqual(metrics['cum_fees'], 20)
        self.assertEqual(metrics['cum_funding'], 8)
        self.assertGreater(metrics['annualized_sharpe'], 0)
        
    def test_calc_return_metrics_with_fill_breakdown(self):
        """Test calc_return_metrics with FillBreakdown."""
        fb = Mock()
        fb.win_ratio.return_value = (0.6, 60, 100, 150, 50.0, -30.0)
        
        metrics = calc_return_metrics(self.pnl_df, pnl_fillbreakdown=fb)
        
        self.assertEqual(metrics['win_ratio'], 0.6)
        self.assertEqual(metrics['num_profit_trades'], 60)
        self.assertEqual(metrics['num_contracting_trades'], 100)
        self.assertEqual(metrics['num_trades'], 150)
        self.assertEqual(metrics['gain_per_fill'], 50.0)
        self.assertEqual(metrics['loss_per_fill'], -30.0)
        
    def test_get_return_metrics_str(self):
        """Test get_return_metrics_str formatting."""
        result = get_return_metrics_str(self.pnl_df, 'test_strategy', daily_scaler=1)
        
        self.assertIn('test_strategy:annualized_sharpe:', result)
        self.assertIn('test_strategy:lifetime_pnl:$200', result)
        self.assertIn('test_strategy:max_drawdown:$-10', result)
        
    # test_get_latest_pnl_records_dict removed - function no longer exists


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.start_dt = dt(2024, 1, 1, tzinfo=timezone.utc)
        self.end_dt = dt(2024, 1, 31, tzinfo=timezone.utc)
        
    def test_fill_breakdown_runall(self):
        """Test FillBreakdown runall method."""
        fb = FillBreakdown(self.start_dt, self.end_dt)
        fb.load_features_df = Mock()
        fb.win_ratio = Mock(return_value=(0.6, 60, 100, 150, 50.0, -30.0))
        fb.get_pnl_breakdowns = Mock(return_value={'test_col': pd.DataFrame({'pnl': [100]})})
        
        # Capture print output
        with patch('builtins.print') as mock_print:
            fb.runall()
            
        fb.load_features_df.assert_called_once()
        fb.win_ratio.assert_called_once()
        fb.get_pnl_breakdowns.assert_called_once()
    def test_calc_multi_symbol_get_initial_inventory_with_positions(self):
        """Test inventory initialization with existing positions."""
        calc = CalcMultiSymbolFillPnl(self.start_dt, self.end_dt)
        
        positions_df = pd.DataFrame({
            'ts': [self.start_dt],
            'symbol_venue': ['BTCUSDT_binance-futures'],
            'qty': [2.0],
            'cost_basis': [100000.0]
        }).set_index(['ts', 'symbol_venue'])
        
        calc.positions_df = positions_df
        calc.symbol_venues = ['BTCUSDT_binance-futures']
        calc._get_initial_inventory()
        
        # Check long position initialized
        self.assertEqual(len(calc.inventory_case[Side.BUY]['BTCUSDT_binance-futures']), 1)
        self.assertEqual(calc.inventory_case[Side.BUY]['BTCUSDT_binance-futures'][0]['quantity'], 2.0)
        self.assertEqual(calc.inventory_case[Side.BUY]['BTCUSDT_binance-futures'][0]['price'], 50000.0)  # 100000/2
        
    def test_calc_multi_symbol_match_fills_partial_match(self):
        """Test FIFO matching with partial fills."""
        calc = CalcMultiSymbolFillPnl(self.start_dt, self.end_dt)
        init_calc_symbol_venue(calc, 'BTCUSDT_binance-futures')
        
        # Set up initial position with proper dtype
        calc.inventory_case[Side.BUY]['BTCUSDT_binance-futures'] = np.array(
            [(50000.0, 2.0, pd.Timestamp(self.start_dt))], 
            dtype=[('price', float), ('quantity', float), ('ts', 'datetime64[ns]')]
        )
        calc.inventory_case[Side.SELL]['BTCUSDT_binance-futures'] = np.array(
            [], dtype=[('price', float), ('quantity', float), ('ts', 'datetime64[ns]')]
        )
        
        # Sell more than one position but less than total
        fill = Fill(
            symbol='BTCUSDT',
            side=Side.SELL,
            px=51000.0,
            qty=1.5,
            exch_ts=self.start_dt + td(hours=1),
            recv_ts=self.start_dt + td(hours=1),
            commission=0.0,
            commission_asset='USDT'
        )
        
        calc._match_fills_with_inventory(fill, 'BTCUSDT_binance-futures')
        
        # Check remaining inventory
        self.assertEqual(len(calc.inventory_case[Side.BUY]['BTCUSDT_binance-futures']), 1)
        self.assertEqual(calc.inventory_case[Side.BUY]['BTCUSDT_binance-futures'][0]['quantity'], 0.5)
        
        # Check realized PnL
        expected_pnl = (51000.0 - 50000.0) * 1.5
        self.assertEqual(calc.realized_pnl['BTCUSDT_binance-futures'], expected_pnl)
        
    def test_calc_multi_symbol_both_long_short_warning(self):
        """Test warning when both long and short positions exist."""
        calc = CalcMultiSymbolFillPnl(self.start_dt, self.end_dt)
        init_calc_symbol_venue(calc, 'BTCUSDT_binance-futures')
        
        # Set up both long and short positions (shouldn't happen in practice)
        calc.inventory_case[Side.BUY]['BTCUSDT_binance-futures'] = np.array(
            [(50000.0, 1.0, pd.Timestamp(self.start_dt))], 
            dtype=[('price', float), ('quantity', float), ('ts', 'datetime64[ns]')]
        )
        calc.inventory_case[Side.SELL]['BTCUSDT_binance-futures'] = np.array(
            [(51000.0, 0.5, pd.Timestamp(self.start_dt))], 
            dtype=[('price', float), ('quantity', float), ('ts', 'datetime64[ns]')]
        )
        
        with self.assertLogs('lib.pnl.fill_pnl_symbol', level='WARNING') as cm:
            calc._calculate_position_info_at_bar('BTCUSDT_binance-futures', 52000.0, self.start_dt)
            
        self.assertIn('seeing symbol_venue=', cm.output[0])
        
    def test_calc_return_metrics_zero_volatility(self):
        """Test return metrics with zero volatility."""
        pnl_df = pd.DataFrame({
            'pnl': [0, 0, 0, 0, 0],
            'notional': [10000, 10000, 10000, 10000, 10000],
            'long': [10000, 10000, 10000, 10000, 10000],
            'short': [0, 0, 0, 0, 0],
            'traded_long': [0, 0, 0, 0, 0],
            'traded_short': [0, 0, 0, 0, 0],
            'fees_usd': [0, 0, 0, 0, 0],
            'funding_income': [0, 0, 0, 0, 0]
        })
        
        metrics = calc_return_metrics(pnl_df)
        
        self.assertEqual(metrics['annualized_sharpe'], 0)  # Should handle zero volatility
        self.assertEqual(metrics['annualized_risk'], 0)
        
    def test_fill_breakdown_pnl_by_col_error_handling(self):
        """Test pnl_by_col error handling."""
        fb = FillBreakdown(self.start_dt, self.end_dt)
        fb.fills_df = pd.DataFrame({'symbol_venue': ['TEST'], 'ts': [self.start_dt], 'realized_pnl': [100]})
        fb.features_df = pd.DataFrame({'symbol_venue': ['TEST'], 'ts': [self.start_dt], 'invalid_col': [1]})
        
        # Should catch RuntimeError and return None
        with self.assertRaises(RuntimeError):
            fb.pnl_by_col('non_existent_col', by_date=False, merge_on_ts=True)
        
    def test_generate_data_after_init_pos_inclusive(self):
        """Test filtering data after initial position with inclusive flag."""
        calc = CalcMultiSymbolFillPnl(self.start_dt, self.end_dt)
        calc.symbol_venues = ['BTCUSDT_binance-futures']
        
        positions_df = pd.DataFrame({
            'ts': [self.start_dt + td(hours=1)],
            'symbol_venue': ['BTCUSDT_binance-futures']
        }).set_index(['ts', 'symbol_venue'])
        calc.positions_df = positions_df
        
        test_df = pd.DataFrame({
            'ts': [self.start_dt, self.start_dt + td(hours=1), self.start_dt + td(hours=2)],
            'symbol_venue': ['BTCUSDT_binance-futures'] * 3
        })
        
        # Test inclusive=True
        result = calc._generate_data_after_init_pos(test_df, inclusive=True)
        self.assertEqual(len(result), 3)  # Should include position timestamp and after
        
        # Test inclusive=False
        result = calc._generate_data_after_init_pos(test_df, inclusive=False)
        self.assertEqual(len(result), 2)  # Should only include after position timestamp
        
    def test_load_features_df_update_name_list(self):
        """Test updating column name lists."""
        fb = FillBreakdown(self.start_dt, self.end_dt)
        
        mock_bars_df = pd.DataFrame({
            'ts': [self.start_dt],
            'symbol_venue': ['BTCUSDT_binance-futures'],
            'dvolume_1440': [1000000.0],
            'spread_avg_1440': [0.01]
        }).set_index(['ts', 'symbol_venue'])
        
        mock_features_df = pd.DataFrame({
            'ts': [self.start_dt],
            'symbol_venue': ['BTCUSDT_binance-futures'],
            'logret_1440_lz': [0.5],
            'day_of_week': [1]
        }).set_index(['ts', 'symbol_venue'])
        
        fb._load_bars_data = Mock(return_value=mock_bars_df)
        fb._load_features_data = Mock(return_value=mock_features_df)
        
        fb.load_features_df(update_name_list=True)
        
        self.assertIn('dvolume_1440', fb.bars_name_list)
        self.assertIn('spread_avg_1440', fb.bars_name_list)
        self.assertIn('logret_1440_lz', fb.features_name_list)
        self.assertIn('day_of_week', fb.features_name_list)


if __name__ == '__main__':
    unittest.main()