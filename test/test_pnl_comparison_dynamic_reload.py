"""
Unit tests for PnL Comparison App dynamic sim reload functionality.

Tests that the sim report dropdown dynamically refreshes to show new simulations
without requiring a manual restart.
"""
import os
import shutil
import tempfile
from unittest.mock import patch

import pytest

from reports.pnl_comparison_app import PnlComparisonApp


class TestPnlComparisonDynamicReload:
    """Test dynamic sim reload functionality"""

    @pytest.fixture
    def temp_simcomp_dir(self):
        """Create a temporary simcomp directory for testing"""
        temp_dir = tempfile.mkdtemp()
        simcomp_dir = os.path.join(temp_dir, "simcomp")
        os.makedirs(simcomp_dir)
        yield simcomp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_sim_dir(self, temp_simcomp_dir):
        """Mock SIM_DIR to use temp directory"""
        with patch('reports.pnl_comparison_app.SIM_DIR', os.path.dirname(temp_simcomp_dir)):
            yield temp_simcomp_dir

    def create_test_sim(self, simcomp_dir: str, sim_name: str):
        """Helper to create a test sim directory with required files"""
        sim_path = os.path.join(simcomp_dir, sim_name)
        os.makedirs(sim_path, exist_ok=True)

        # Create required pnl.calculator.csv file
        pnl_file = os.path.join(sim_path, "pnl.calculator.csv")
        with open(pnl_file, 'w', encoding='utf-8') as f:
            f.write("ts,pnl\n")
            f.write("2025-11-20,1000\n")

        return sim_path

    def test_get_simcomp_sims_returns_empty_when_no_sims(self, mock_sim_dir):
        """Test that get_simcomp_sims returns empty lists when no sims exist"""
        sims_1d, sims_7d = PnlComparisonApp.get_simcomp_sims()
        assert sims_1d == []
        assert sims_7d == []

    def test_get_simcomp_sims_finds_single_sim(self, mock_sim_dir):
        """Test that get_simcomp_sims finds a single 1d simulation"""
        self.create_test_sim(mock_sim_dir, "sim_comp_1d_20251120")

        sims_1d, sims_7d = PnlComparisonApp.get_simcomp_sims()

        assert len(sims_1d) == 1
        assert "sim_comp_1d_20251120" in sims_1d
        assert sims_7d == []

    def test_get_simcomp_sims_finds_multiple_sims(self, mock_sim_dir):
        """Test that get_simcomp_sims finds multiple simulations split by type"""
        self.create_test_sim(mock_sim_dir, "sim_comp_1d_20251118")
        self.create_test_sim(mock_sim_dir, "sim_comp_1d_20251119")
        self.create_test_sim(mock_sim_dir, "sim_comp_7d_20251120")

        sims_1d, sims_7d = PnlComparisonApp.get_simcomp_sims()

        assert len(sims_1d) == 2
        assert len(sims_7d) == 1
        assert "sim_comp_1d_20251118" in sims_1d
        assert "sim_comp_1d_20251119" in sims_1d
        assert "sim_comp_7d_20251120" in sims_7d

    def test_get_simcomp_sims_sorts_reverse_order(self, mock_sim_dir):
        """Test that sims are sorted in reverse order (most recent first)"""
        self.create_test_sim(mock_sim_dir, "sim_comp_1d_20251118")
        self.create_test_sim(mock_sim_dir, "sim_comp_1d_20251120")
        self.create_test_sim(mock_sim_dir, "sim_comp_1d_20251119")

        sims_1d, _ = PnlComparisonApp.get_simcomp_sims()

        assert sims_1d[0] == "sim_comp_1d_20251120"  # Most recent first
        assert sims_1d[1] == "sim_comp_1d_20251119"
        assert sims_1d[2] == "sim_comp_1d_20251118"

    def test_get_simcomp_sims_ignores_dirs_without_pnl_file(self, mock_sim_dir):
        """Test that directories without pnl.calculator.csv are ignored"""
        # Create sim WITH pnl file
        self.create_test_sim(mock_sim_dir, "sim_comp_1d_20251120")

        # Create sim WITHOUT pnl file
        invalid_sim_path = os.path.join(mock_sim_dir, "sim_comp_1d_20251119")
        os.makedirs(invalid_sim_path)
        # No pnl.calculator.csv created

        sims_1d, _ = PnlComparisonApp.get_simcomp_sims()

        assert len(sims_1d) == 1
        assert "sim_comp_1d_20251120" in sims_1d
        assert "sim_comp_1d_20251119" not in sims_1d

    def test_dynamic_reload_picks_up_new_sim(self, mock_sim_dir):
        """Test that calling get_simcomp_sims() picks up newly added sims"""
        # Initial state: 2 sims
        self.create_test_sim(mock_sim_dir, "sim_comp_1d_20251118")
        self.create_test_sim(mock_sim_dir, "sim_comp_1d_20251119")

        sims_1d, _ = PnlComparisonApp.get_simcomp_sims()
        assert len(sims_1d) == 2

        # Add new sim (simulates sim completing)
        self.create_test_sim(mock_sim_dir, "sim_comp_1d_20251120")

        # Refresh sim list (simulates callback refresh)
        sims_1d, _ = PnlComparisonApp.get_simcomp_sims()

        # Verify new sim appears
        assert len(sims_1d) == 3
        assert "sim_comp_1d_20251120" in sims_1d
        assert sims_1d[0] == "sim_comp_1d_20251120"  # Most recent first

    def test_dynamic_reload_removes_deleted_sim(self, mock_sim_dir):
        """Test that calling get_simcomp_sims() removes deleted sims"""
        # Initial state: 3 sims
        self.create_test_sim(mock_sim_dir, "sim_comp_1d_20251118")
        sim_to_delete = self.create_test_sim(mock_sim_dir, "sim_comp_1d_20251119")
        self.create_test_sim(mock_sim_dir, "sim_comp_1d_20251120")

        sims_1d, _ = PnlComparisonApp.get_simcomp_sims()
        assert len(sims_1d) == 3

        # Delete middle sim
        shutil.rmtree(sim_to_delete)

        # Refresh sim list
        sims_1d, _ = PnlComparisonApp.get_simcomp_sims()

        # Verify deleted sim is gone
        assert len(sims_1d) == 2
        assert "sim_comp_1d_20251119" not in sims_1d
        assert "sim_comp_1d_20251118" in sims_1d
        assert "sim_comp_1d_20251120" in sims_1d
