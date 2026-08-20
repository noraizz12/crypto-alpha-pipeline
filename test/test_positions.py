import math
from datetime import datetime as dt, timezone, timedelta
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

from lib.trader.positions import SecPos, PositionRecorder
from lib.pnl.pnl_util import round_dust_position
from lib.trader.trading import Fill
from lib.util.util import ONE_PENNY


class TestSecPos:
    """Test cases for SecPos class"""
    
    def test_init(self):
        """Test SecPos initialization"""
        pos = SecPos("BTCUSDT")
        assert pos.symbol == "BTCUSDT"
        assert pos.abs_dvolume == 0.0
        assert pos.abs_qty == 0.0
        assert pos.qty == 0.0
        assert pos.execution_qty == 0.0
        assert pos.cost_basis == 0.0
        assert pos.fill_cnt == 0
        assert pos.fees == 0.0
        assert pos.mark == 0.0
        assert isinstance(pos.mark_ts, dt)
        assert pos.unrealized_profit is None
        assert pos.vwap is None
    
    def test_add_fill_buy(self):
        """Test adding a buy fill"""
        pos = SecPos("BTCUSDT")
        
        # Create a mock fill
        fill = Mock(spec=Fill)
        fill.symbol = "BTCUSDT"
        fill.qty = 0.1
        fill.px = 50000.0
        fill.side = "BUY"
        fill.commission_usd = 2.5
        fill.exch_ts = dt.now(timezone.utc)
        fill.notional.return_value = 5000.0  # 0.1 * 50000
        fill.signed_qty.return_value = 0.1
        
        pos.add_fill(fill)
        
        assert pos.abs_dvolume == 5000.0
        assert pos.abs_qty == 0.1
        assert pos.qty == 0.1
        assert pos.execution_qty == 0.1
        assert pos.cost_basis == -5000.0  # Negative for long position
        assert pos.fill_cnt == 1
        assert pos.fees == 2.5
        assert pos.mark == 50000.0
        assert pos.mark_ts == fill.exch_ts
    
    def test_add_fill_sell(self):
        """Test adding a sell fill"""
        pos = SecPos("BTCUSDT")
        
        # Create a mock fill
        fill = Mock(spec=Fill)
        fill.symbol = "BTCUSDT"
        fill.qty = 0.1
        fill.px = 50000.0
        fill.side = "SELL"
        fill.commission_usd = 2.5
        fill.exch_ts = dt.now(timezone.utc)
        fill.notional.return_value = -5000.0  # Negative for sell
        fill.signed_qty.return_value = -0.1
        
        pos.add_fill(fill)
        
        assert pos.abs_dvolume == 5000.0
        assert pos.abs_qty == 0.1
        assert pos.qty == -0.1
        assert pos.execution_qty == -0.1
        assert pos.cost_basis == 5000.0  # Positive for short position
        assert pos.fill_cnt == 1
        assert pos.fees == 2.5
    
    def test_add_multiple_fills(self):
        """Test adding multiple fills"""
        pos = SecPos("BTCUSDT")
        
        # First fill - buy
        fill1 = Mock(spec=Fill)
        fill1.symbol = "BTCUSDT"
        fill1.qty = 0.1
        fill1.px = 50000.0
        fill1.side = "BUY"
        fill1.commission_usd = 2.5
        fill1.exch_ts = dt.now(timezone.utc)
        fill1.notional.return_value = 5000.0
        fill1.signed_qty.return_value = 0.1
        
        # Second fill - buy more
        fill2 = Mock(spec=Fill)
        fill2.symbol = "BTCUSDT"
        fill2.qty = 0.2
        fill2.px = 51000.0
        fill2.side = "BUY"
        fill2.commission_usd = 5.1
        fill2.exch_ts = dt.now(timezone.utc) + timedelta(seconds=1)
        fill2.notional.return_value = 10200.0
        fill2.signed_qty.return_value = 0.2
        
        pos.add_fill(fill1)
        pos.add_fill(fill2)
        
        assert pos.abs_dvolume == 15200.0  # 5000 + 10200
        assert math.isclose(pos.abs_qty, 0.3)  # 0.1 + 0.2
        assert math.isclose(pos.qty, 0.3)
        assert pos.cost_basis == -15200.0
        assert pos.fill_cnt == 2
        assert pos.fees == 7.6  # 2.5 + 5.1
        assert pos.mark == 51000.0  # Last fill price
    
    def test_notional(self):
        """Test notional calculation"""
        pos = SecPos("BTCUSDT")
        pos.qty = 0.5
        pos.mark = 50000.0
        
        assert pos.notional() == 25000.0
    
    def test_avg_px(self):
        """Test average price calculation"""
        pos = SecPos("BTCUSDT")
        
        # No trades
        assert pos.avg_px() == 0.0
        
        # With trades
        pos.abs_dvolume = 15000.0
        pos.abs_qty = 0.3
        assert pos.avg_px() == 50000.0
    
    def test_update_px_valid(self):
        """Test updating price with valid value"""
        pos = SecPos("BTCUSDT")
        original_ts = pos.mark_ts
        
        pos.update_px(51000.0)
        assert pos.mark == 51000.0
        assert pos.mark_ts > original_ts
    
    def test_update_px_invalid(self):
        """Test updating price with invalid value"""
        pos = SecPos("BTCUSDT")
        pos.mark = 50000.0
        
        # Should not update with negative or zero price
        pos.update_px(0)
        assert pos.mark == 50000.0
        
        pos.update_px(-100)
        assert pos.mark == 50000.0
    
    def test_refresh_qty(self):
        """Test refreshing quantity"""
        pos = SecPos("BTCUSDT")
        pos.qty = 0.1
        
        # Close values - no log
        pos.refresh_qty(0.10000001)
        assert pos.qty == 0.10000001
        
        # Different values - should log
        with patch('lib.trader.positions.logger') as mock_logger:
            pos.refresh_qty(0.2)
            assert pos.qty == 0.2
            mock_logger.info.assert_called_once()
    
    def test_refresh_cost_basis(self):
        """Test refreshing cost basis"""
        pos = SecPos("BTCUSDT")
        pos.cost_basis = -5000.0
        
        # Close values - no log
        pos.refresh_cost_basis(-5000.0001)
        assert pos.cost_basis == -5000.0001
        
        # Different values - should log
        with patch('lib.trader.positions.logger') as mock_logger:
            pos.refresh_cost_basis(-6000.0)
            assert pos.cost_basis == -6000.0
            mock_logger.info.assert_called_once()
    
    def test_reset_avg_cost(self):
        """Test resetting average cost"""
        pos = SecPos("BTCUSDT")
        pos.abs_dvolume = 10000.0
        pos.abs_qty = 0.2
        pos.execution_qty = 0.2
        
        pos.reset_avg_cost()
        
        assert pos.abs_dvolume == 0
        assert pos.abs_qty == 0
        assert pos.execution_qty == 0
    
    def test_get_list_for_df_row(self):
        """Test getting data for DataFrame row"""
        pos = SecPos("BTCUSDT")
        pos.qty = 0.1
        pos.cost_basis = -5000.0
        pos.mark = 50000.0
        pos.mark_ts = dt(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        row = pos.get_list_for_df_row()

        assert row[0] == "BTCUSDT"
        assert row[1] == 0.1
        assert row[2] == -5000.0
        assert row[3] == 50000.0
        assert row[4] == pos.mark_ts
        assert row[5] == 5000.0  # notional


class TestPositionRecorder:
    """Test cases for PositionRecorder class"""
    
    def test_init_prod_mode(self):
        """Test initialization in production mode"""
        with patch('lib.trader.positions.beginning_of_day') as mock_bod:
            mock_bod.return_value = dt(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
            
            recorder = PositionRecorder(prod=True)
            
            assert recorder.prod is True
            assert recorder.start == mock_bod.return_value
            assert isinstance(recorder.end, dt)
            assert not recorder.secs
    
    def test_init_non_prod_mode(self):
        """Test initialization in non-production mode with custom dates"""
        start = dt(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        end = dt(2024, 1, 1, 16, 0, 0, tzinfo=timezone.utc)
        
        recorder = PositionRecorder(prod=False, start=start, end=end)
        
        assert recorder.prod is False
        assert recorder.start == start
        assert recorder.end == end
    
    def test_get_security_new(self):
        """Test getting a new security"""
        recorder = PositionRecorder(prod=False)
        
        sec = recorder.get_security("BTCUSDT")
        
        assert isinstance(sec, SecPos)
        assert sec.symbol == "BTCUSDT"
        assert "BTCUSDT" in recorder.secs
        assert recorder.secs["BTCUSDT"] == sec
    
    def test_get_security_existing(self):
        """Test getting an existing security"""
        recorder = PositionRecorder(prod=False)
        
        sec1 = recorder.get_security("BTCUSDT")
        sec1.qty = 0.1
        
        sec2 = recorder.get_security("BTCUSDT")
        
        assert sec1 is sec2
        assert sec2.qty == 0.1
    
    def test_refresh_qty(self):
        """Test refreshing quantity through recorder"""
        recorder = PositionRecorder(prod=False)
        
        recorder.refresh_qty("BTCUSDT", 0.5)
        
        assert "BTCUSDT" in recorder.secs
        assert recorder.secs["BTCUSDT"].qty == 0.5
    
    def test_add_fill(self):
        """Test adding fill through recorder"""
        recorder = PositionRecorder(prod=False)
        
        fill = Mock(spec=Fill)
        fill.symbol = "BTCUSDT"
        fill.qty = 0.1
        fill.px = 50000.0
        fill.side = "BUY"
        fill.commission_usd = 2.5
        fill.exch_ts = dt.now(timezone.utc)
        fill.notional.return_value = 5000.0
        fill.signed_qty.return_value = 0.1
        
        recorder.add_fill(fill)
        
        assert "BTCUSDT" in recorder.secs
        assert recorder.secs["BTCUSDT"].qty == 0.1
    
    def test_get_symbols(self):
        """Test getting list of symbols"""
        recorder = PositionRecorder(prod=False)
        
        recorder.get_security("BTCUSDT")
        recorder.get_security("ETHUSDT")
        
        symbols = recorder.get_symbols()
        
        assert len(symbols) == 2
        assert "BTCUSDT" in symbols
        assert "ETHUSDT" in symbols
    
    def test_get_pos(self):
        """Test getting list of positions"""
        recorder = PositionRecorder(prod=False)
        
        btc = recorder.get_security("BTCUSDT")
        eth = recorder.get_security("ETHUSDT")
        
        positions = recorder.get_pos()
        
        assert len(positions) == 2
        assert btc in positions
        assert eth in positions
    
    def test_get_qty(self):
        """Test getting quantity for a symbol"""
        recorder = PositionRecorder(prod=False)
        
        # Non-existent symbol
        assert recorder.get_qty("BTCUSDT") == 0.0
        
        # Existing symbol
        recorder.refresh_qty("BTCUSDT", 0.5)
        assert recorder.get_qty("BTCUSDT") == 0.5
    
    def test_reset_avg_cost(self):
        """Test resetting average cost for all positions"""
        recorder = PositionRecorder(prod=False)
        
        btc = recorder.get_security("BTCUSDT")
        btc.abs_dvolume = 5000.0
        btc.abs_qty = 0.1
        
        eth = recorder.get_security("ETHUSDT")
        eth.abs_dvolume = 3000.0
        eth.abs_qty = 1.0
        
        recorder.reset_avg_cost()
        
        assert btc.abs_dvolume == 0
        assert btc.abs_qty == 0
        assert eth.abs_dvolume == 0
        assert eth.abs_qty == 0
    
    @patch('lib.trader.positions.date_to_str')
    @patch('lib.trader.positions.today_date')
    @patch('lib.trader.positions.dir_manager')
    @patch('lib.trader.positions.dt_to_str')
    def test_dump_positions(self, mock_dt_to_str, mock_dir_manager, mock_today_date, mock_date_to_str):
        """Test dumping positions to file"""
        mock_dt_to_str.return_value = "20240101_1200"
        mock_date_to_str.return_value = "20240101"
        mock_today_date.return_value = dt(2024, 1, 1, tzinfo=timezone.utc)
        mock_dir_manager.POSITION_DIR = "/tmp/positions"
        
        recorder = PositionRecorder(prod=False)
        
        # Add some positions
        btc = recorder.get_security("BTCUSDT")
        btc.qty = 0.1
        btc.cost_basis = -5000.0
        btc.mark = 50000.0
        
        eth = recorder.get_security("ETHUSDT")
        eth.qty = 1.0
        eth.cost_basis = -3000.0
        eth.mark = 3000.0
        
        # Mock DataFrame.to_parquet
        with patch.object(pd.DataFrame, 'to_parquet') as mock_to_parquet:
            recorder.dump_positions()
            
            mock_to_parquet.assert_called_once()
            call_args = mock_to_parquet.call_args[0]
            assert call_args[0] == "/tmp/positions/20240101/pos.20240101_1200.parquet"


class TestRoundDustPosition:
    """Test cases for round_dust_position function"""
    
    @patch('lib.pnl.pnl_util.load_exchange_info')
    def test_round_dust_position_basic(self, mock_load_exchange_info):
        """Test basic dust position rounding"""
        # Create test data
        round_dt = dt(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        # Create multi-index DataFrame
        timestamps = [round_dt, round_dt, round_dt]
        symbols = ['BTCUSDT', 'SHIBUSDT', 'ETHUSDT']
        index = pd.MultiIndex.from_arrays([timestamps, symbols], names=['ts', 'symbol_venue'])
        
        df = pd.DataFrame({
            'mark_price': [50000.0, 0.00001, 3000.0],
            'qty': [0.000001, 100.0, 0.00001],  # BTC and ETH are dust
            'notional': [0.05, 0.001, 0.03],  # Less than min_notional
        }, index=index)
        
        # Mock exchange info
        exch_info = pd.DataFrame({
            'quantityPrecision': [5, 0, 4]
        }, index=['BTCUSDT', 'SHIBUSDT', 'ETHUSDT'])
        mock_load_exchange_info.return_value = exch_info
        
        # Run function
        result = round_dust_position(df, round_dt)
        
        # Check results - BTC and ETH should be rounded to 0
        assert result.loc[(round_dt, 'BTCUSDT'), 'qty'] == 0
        assert result.loc[(round_dt, 'BTCUSDT'), 'notional'] == 0
        assert result.loc[(round_dt, 'SHIBUSDT'), 'qty'] == 100.0  # Not dust
        assert result.loc[(round_dt, 'ETHUSDT'), 'qty'] == 0
        assert result.loc[(round_dt, 'ETHUSDT'), 'notional'] == 0
    
    @patch('lib.pnl.pnl_util.load_exchange_info')
    def test_round_dust_position_no_exchange_info(self, mock_load_exchange_info):
        """Test dust rounding when exchange info is not available"""
        mock_load_exchange_info.return_value = None
        
        round_dt = dt(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        # Create multi-index DataFrame
        timestamps = [round_dt]
        symbols = ['BTCUSDT']
        index = pd.MultiIndex.from_arrays([timestamps, symbols], names=['ts', 'symbol_venue'])
        
        df = pd.DataFrame({
            'mark_price': [50000.0],
            'qty': [0.000001],
            'notional': [0.05],
        }, index=index)
        
        # Run function - should use ONE_PENNY as default
        result = round_dust_position(df, round_dt)
        
        # With ONE_PENNY (0.01), notional of 0.05 should not be dust
        assert result.loc[(round_dt, 'BTCUSDT'), 'qty'] == 0.000001
        assert result.loc[(round_dt, 'BTCUSDT'), 'notional'] == 0.05
    
    @patch('lib.pnl.pnl_util.load_exchange_info')
    def test_round_dust_position_with_position_age(self, mock_load_exchange_info):
        """Test dust rounding with position age adjustment"""
        mock_load_exchange_info.return_value = None
        
        round_dt = dt(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        # Create multi-index DataFrame
        timestamps = [round_dt]
        symbols = ['BTCUSDT']
        index = pd.MultiIndex.from_arrays([timestamps, symbols], names=['ts', 'symbol_venue'])
        
        df = pd.DataFrame({
            'mark_price': [50000.0],
            'qty': [0.0000001],
            'notional': [0.005],  # Less than ONE_PENNY
            'position_age': [10],
        }, index=index)
        
        # Run function with position age adjustment
        result = round_dust_position(df, round_dt, adjust_pos_age=True)
        
        # Check dust position is rounded
        assert result.loc[(round_dt, 'BTCUSDT'), 'qty'] == 0
        assert result.loc[(round_dt, 'BTCUSDT'), 'notional'] == 0
        assert pd.isna(result.loc[(round_dt, 'BTCUSDT'), 'position_age'])
    
    @patch('lib.pnl.pnl_util.load_exchange_info')
    def test_round_dust_position_custom_columns(self, mock_load_exchange_info):
        """Test dust rounding with custom column names"""
        mock_load_exchange_info.return_value = None
        
        round_dt = dt(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        # Create multi-index DataFrame with custom columns
        timestamps = [round_dt]
        symbols = ['BTCUSDT']
        index = pd.MultiIndex.from_arrays([timestamps, symbols], names=['ts', 'symbol_venue'])
        
        df = pd.DataFrame({
            'price': [50000.0],
            'position': [0.0000001],
            'value': [0.005],
        }, index=index)
        
        # Run function with custom columns
        result = round_dust_position(
            df, 
            round_dt,
            px_col='price',
            qty_col='position',
            pos_col='value'
        )
        
        # Check dust position is rounded
        assert result.loc[(round_dt, 'BTCUSDT'), 'position'] == 0
        assert result.loc[(round_dt, 'BTCUSDT'), 'value'] == 0
