"""Base class for all Dash report applications."""

import logging
import logging.config
import os
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime as dt
from datetime import timezone

from dash import Dash, html, dcc

from lib.util.logging_util import get_logging_config
from lib.util.directory import dir_manager

# Base dashboard configuration
REPORT_KILL_FILE = f"{dir_manager.TRADING_DIR}/reports.kill"

# Common table properties used by multiple dashboard applications
COMMON_TABLE_PROPS = {
    'sort_action': "native",
    'style_table': {'overflowX': 'auto', 'width': '100%', 'minWidth': '100%'},
    'fixed_columns': {'headers': True, 'data': 1},
    'style_data_conditional': [
        {
            'if': {'filter_query': '{missing_data} eq 1'},
            'backgroundColor': '#ffcccc',
            'color': 'black',
        },
        {
            'if': {'filter_query': '{no_trade} eq 1'},
            'backgroundColor': '#ff9999',
            'color': 'black',
        },
    ],
}

logger = logging.getLogger(__name__)


class BaseDashApp(ABC):
    """Base class for individual Dash report applications."""
    
    def __init__(self, app_name: str, port: int, interval_secs: int = 60, debug: bool = False):
        """
        Initialize base Dash application.
        
        Args:
            app_name: Name of the application
            port: Port to run the application on
            interval_secs: Refresh interval in seconds
            debug: Whether to run in debug mode
        """
        self.app_name = app_name
        self.port = port
        self.interval_secs = interval_secs
        self.debug = debug
        
        # Configure logging
        logging.config.dictConfig(get_logging_config(f"{app_name.lower().replace(' ', '_')}_server"))
        
        # Initialize Dash app
        self.app = Dash(app_name, suppress_callback_exceptions=True)
        
        # Thread safety for refresh operations
        self.refresh_lock = threading.Lock()
        
        # Setup kill file monitoring
        self.setup_kill_file_monitoring()

        logger.info(f"Initialized {app_name} on port {port}")
    
    def setup_kill_file_monitoring(self):
        """Monitor kill file for shutdown signal."""
        def check_kill_file():
            if os.path.isfile(REPORT_KILL_FILE):
                try:
                    logger.info(f"{REPORT_KILL_FILE} found, shutting down {self.app_name}...")
                    os._exit(0)
                except Exception as e:
                    logger.error(f"Failed to shut down on kill file: {e}")
            
            # Schedule next check
            timer = threading.Timer(50, check_kill_file)
            timer.daemon = True
            timer.start()
        
        # Start monitoring
        timer = threading.Timer(50, check_kill_file)
        timer.daemon = True
        timer.start()
    
    def create_header(self):
        """Create a simple header for the application."""
        return html.Div([
            html.Div([
                html.H1(self.app_name, style={
                    'display': 'inline-block',
                    'marginRight': '30px',
                }),
                html.Div([
                    html.A("Trading", href="http://localhost:8050", target="_blank", 
                          style={'margin': '0 10px', 'color': '#007BFF'}),
                    html.A("Historical", href="http://localhost:8051", target="_blank",
                          style={'margin': '0 10px', 'color': '#007BFF'}),
                    html.A("Slippage", href="http://localhost:8052", target="_blank",
                          style={'margin': '0 10px', 'color': '#007BFF'}),
                    html.A("Fits", href="http://localhost:8053", target="_blank",
                          style={'margin': '0 10px', 'color': '#007BFF'}),
                    html.A("Simulation", href="http://localhost:8054", target="_blank",
                          style={'margin': '0 10px', 'color': '#007BFF'}),
                    html.A("Execution", href="http://localhost:8058", target="_blank",
                          style={'margin': '0 10px', 'color': '#007BFF'}),
                ], style={
                    'display': 'inline-block',
                    'fontSize': '14px',
                }),
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'space-between',
            }),
        ], style={
            'backgroundColor': '#f8f9fa',
            'padding': '15px',
            'marginBottom': '20px',
            'borderBottom': '2px solid #dee2e6',
        })
    
    def create_loading_overlay(self):
        """Create loading overlay component."""
        return html.Div([
            html.Div([
                dcc.Loading(
                    id="main-loading",
                    children=[html.Div(id="loading-dummy")],
                    type="cube",
                    color="#007BFF",
                    style={'transform': 'scale(2)', 'marginBottom': '30px'}
                ),
                html.H2(f"Loading {self.app_name}", 
                    style={'color': '#007BFF', 'textAlign': 'center', 'fontSize': '24px'}),
                html.P("Please wait while we prepare your data...", 
                    style={'color': '#666', 'textAlign': 'center', 'fontSize': '16px', 'marginTop': '20px'}),
            ], style={
                'position': 'absolute',
                'top': '50%',
                'left': '50%',
                'transform': 'translate(-50%, -50%)',
                'textAlign': 'center'
            })
        ], id='loading-overlay', style={
            'position': 'fixed',
            'top': 0,
            'left': 0,
            'width': '100%',
            'height': '100%',
            'backgroundColor': 'rgba(255, 255, 255, 0.95)',
            'zIndex': 9999,
            'display': 'none' if self.debug else 'block'
        })
    
    def handle_refresh(self, updater_func, report_name: str):
        """
        Common refresh handling logic with thread safety.
        
        Args:
            updater_func: Function to call for updating data
            report_name: Name of the report being refreshed
            
        Returns:
            Tuple of (timestamp, disabled_state, message, style)
        """
        lock_acquired = self.refresh_lock.acquire(blocking=True, timeout=1.0)
        if not lock_acquired:
            return (
                time.time(), 
                False, 
                "Another refresh is in progress. Please try again later.", 
                {'color': 'orange'}
            )
        
        start_timer = time.time()
        try:
            # Refresh data
            # Note: resource_manager was removed in refactoring
            # Each app now handles its own data refresh through updater_func
            updater_func()

            update_msg = f"Data Updated at {dt.now(timezone.utc)}"
            update_style = {'color': 'green'}
            logger.info(f"Finished refreshing {report_name} in {time.time() - start_timer:.2f} seconds")
            return (time.time(), False, update_msg, update_style)
            
        except Exception as e:
            update_msg = f"Data Update Failed: {str(e)}"
            update_style = {'color': 'red'}
            logger.error(f"{update_msg}")
            return (time.time(), False, update_msg, update_style)
            
        finally:
            self.refresh_lock.release()
    
    @abstractmethod
    def setup_layout(self):
        """Setup the layout for the application. Must be implemented by child classes."""
        pass
    
    @abstractmethod
    def register_callbacks(self):
        """Register callbacks for the application. Must be implemented by child classes."""
        pass
    
    def run(self, debug: bool = None):
        """Run the Dash application."""
        if debug is None:
            debug = self.debug
        
        logger.info(f"Starting {self.app_name} on port {self.port} (debug={debug})")
        self.app.run(debug=debug, port=self.port, host='0.0.0.0')