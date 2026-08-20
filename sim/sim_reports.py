import argparse
import glob
import json
import logging.config
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from dash.dash_table import DataTable, Format, FormatTemplate
from deepdiff import DeepDiff

from lib.pnl import FillBreakdown
from lib.calcs.calc_returns import calc_factor_return
from lib.calcs.calcs import Calcs
from lib.data import load_sim_data
from lib.sim.sim_util import calc_return_metrics
from lib.util import get_sim_dirs
from lib.util.config import get_config, get_factors
from lib.util.dataframes import clip_col_by_iqr, make_symbol_venue, merge_on_index
from lib.util.directory import CONFIG_DIR, SIM_DIR, dir_manager
from lib.util.logging_util import get_logging_config
from lib.util.time_util import to_datetime
from lib.util.util import unique_list

logging.config.dictConfig(get_logging_config("sim_reports"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

JSON = Union[Dict[str, Any], List[Any]]
FMT_MONEY = FormatTemplate.money(2)
FMT_PERCENT = FormatTemplate.percentage(2)

SUMMARY_COLS = [
    {'id': 'name', 'name': 'Simulation Name', 'type': 'text'},
    {'id': 'sharpe', 'name': 'Sharpe Ratio', 'type': 'numeric', 'format': Format.Format(precision=2, scheme=Format.Scheme.fixed)},
    {'id': 'pnl', 'name': 'PnL', 'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'ret_annualized', 'name': 'Annualized Return', 'type': 'numeric', 'format': FMT_PERCENT},
    {'id': 'risk_annualized', 'name': 'Annualized Risk', 'type': 'numeric', 'format': FMT_PERCENT},
    {'id': 'win_ratio', 'name': 'Win Ratio', 'type': 'text'},
    {'id': 'profit_trades', 'name': 'Profit Trades', 'type': 'text'},
    {'id': 'gain_per_fill', 'name': 'Avg Gain per Trade', 'type': 'text'},
    {'id': 'loss_per_fill', 'name': 'Avg Loss per Trade', 'type': 'text'},
    {'id': 'mdd', 'name': 'Max Drawdown ($)', 'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'mdd_perc', 'name': 'Max Drawdown %', 'type': 'numeric', 'format': FMT_PERCENT},
    {'id': 'avg_notional', 'name': 'Avg. Portfolio Notional', 'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'daily_trading_volume', 'name': 'Avg. Trading Volume', 'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'daily_turnover', 'name': 'Avg. Daily Turnover', 'type': 'numeric', 'format': Format.Format(precision=1, scheme=Format.Scheme.fixed)},
    {'id': 'total_trading_fees', 'name': 'Cum Trading Fees', 'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'trading_fees', 'name': 'Avg. Daily Trading Fees', 'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'trading_fees_bps', 'name': 'Avg. Daily Trading Fees (bps)', 'type': 'numeric', 'format': Format.Format(precision=2, scheme=Format.Scheme.fixed)},
    {'id': 'total_funding', 'name': 'Cum Funding', 'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'funding', 'name': 'Avg. Daily Funding', 'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'funding_bps', 'name': 'Avg. Daily Funding (bps)', 'type': 'numeric', 'format': Format.Format(precision=2, scheme=Format.Scheme.fixed)},
]


def diff_to_table(json_name, comp_json_name, diff):
    def serialize(value):
        """Ensure all values are JSON-serializable."""
        if isinstance(value, dict):
            return json.dumps(value, indent=2)  # Convert dictionaries to a JSON string
        if isinstance(value, (list, tuple)):
            return str(value)  # Convert lists/tuples to a string
        return value  # Leave other values (str, int, float) as they are

    table_data = []
    if not diff:
        # If there's no difference, return a "No differences" message
        return [{"Key": "No differences", f"{json_name}": "-", f"{comp_json_name}": "-"}]

    # Handle value changes
    if "values_changed" in diff:
        for change in diff["values_changed"]:
            row = {
                "Key": str(change.path()),  # Extract the full path
                f"{json_name}": serialize(change.t1),
                f"{comp_json_name}": serialize(change.t2),
            }
            table_data.append(row)

    # Handle removed items
    if "iterable_item_removed" in diff:
        for change in diff["iterable_item_removed"]:
            row = {
                "Key": str(change.path()),  # Extract the full path
                f"{json_name}": serialize(change.t1),
                f"{comp_json_name}": "Removed",
            }
            table_data.append(row)

    # Handle added items
    if "iterable_item_added" in diff:
        for change in diff["iterable_item_added"]:
            row = {
                "Key": str(change.path()),  # Extract the full path
                f"{json_name}": "Added",
                f"{comp_json_name}": serialize(change.t2),
            }
            table_data.append(row)

    # Handle removed dict items
    if "dictionary_item_removed" in diff:
        for change in diff["dictionary_item_removed"]:
            row = {
                "Key": str(change.path()),  # Extract the full path
                f"{json_name}": serialize(change.t1),
                f"{comp_json_name}": "Removed",
            }
            table_data.append(row)

    # Handle added dict items
    if "dictionary_item_added" in diff:
        for change in diff["dictionary_item_added"]:
            row = {
                "Key": str(change.path()),  # Extract the full path
                f"{json_name}": "Added",
                f"{comp_json_name}": serialize(change.t2),
            }
            table_data.append(row)

    return table_data


class SimReports:
    def __init__(self, config: dict, port: int = 8063, sim_dir: Optional[str] = None):
        self.port = port
        self.app = Dash()
        self.sim_dir = sim_dir if sim_dir is not None else SIM_DIR

        self.sim_names = get_sim_dirs(self.sim_dir)
        self.sim_names = sorted(self.sim_names, key=lambda sim_name: os.path.getmtime(f"{self.sim_dir}/{sim_name}"), reverse=True)
        logger.info(self.sim_names)
        if len(self.sim_names) == 0:
            logger.error("No sim directories found")
            exit()

        self.single_sim_names = {}
        self.single_sim_keys = []
        self.sim_family_keys = {}
        self.extract_single_sim()

        self.json_dirs = [file for file in os.listdir(CONFIG_DIR) if file.endswith(".json")]
        self.sim_cache = {}
        self.sim_family = None

        self.current_sim_name = None
        self.current_sim_df = None
        self.current_sim_pnl_df = None
        self.current_json = None
        self.current_sim_summary_dict = {}
        self.current_sim_pnl_breakdown_df = None

        self.comp_sim_name = None
        self.comp_sim_df = None
        self.comp_sim_pnl_df = None
        self.comp_json = None
        self.comp_sim_summary_dict = {}

        self.pnl_fill_breakdown = None
        self.factors = []
        self.family_factors = []
        self.horizon = 1440
        self.features = ['dvolume_1440_trmean']
        self.config = config
        self.exchange_fees = self.config['EXCHANGE_FEES']
        self.setup_page()

    def run(self):
        logger.info("Running page")
        self.app.run(debug=False, port=self.port)

    def extract_single_sim(self):
        self.single_sim_names = {}
        self.single_sim_keys = []
        self.sim_family_keys = {}
        for sim_name in self.sim_names:
            sim_dir = f"{self.sim_dir}/{sim_name}"

            file_pattern = f"{sim_dir}/pnl*.calculator.csv"
            pnl_files = glob.glob(file_pattern)

            self.sim_family_keys[sim_name] = []
            for pnl_file in pnl_files:
                pnl_file_splits = pnl_file.split('.')
                model_case = None
                horizon_case = None
                config_grid_case = None
                alpha_condition_case = None
                for ii in range(1, len(pnl_file_splits)):
                    if pnl_file_splits[ii].startswith('model'):
                        model_case = pnl_file_splits[ii].split('_')[1]
                    elif pnl_file_splits[ii].startswith('horizon'):
                        horizon_case = pnl_file_splits[ii].split('_')[1]
                    elif pnl_file_splits[ii].startswith('conf'):
                        config_grid_case = '_'.join(pnl_file_splits[ii].split('_')[1:])
                    if pnl_file_splits[ii].startswith('condition'):
                        alpha_condition_case = pnl_file_splits[ii].split('_')[1]
                parts = [
                    sim_name,
                    f".{model_case}" if model_case is not None else "",
                    f".{horizon_case}" if horizon_case is not None else "",
                    f".{config_grid_case}" if config_grid_case is not None else "",
                    f".{alpha_condition_case}" if alpha_condition_case is not None else "",
                ]
                sim_key = "".join(parts)
                sim_tuple = (sim_name, model_case, horizon_case, config_grid_case, alpha_condition_case)
                self.single_sim_names[sim_key] = sim_tuple
                self.single_sim_keys.append(sim_key)
                self.sim_family_keys[sim_name].append(sim_tuple)

    def load_sim(
            self,
            sim_name: Optional[str] = None,
            model: Optional[str] = None,
            horizon: Optional[str] = None,
            config_grid: Optional[str] = None,
            alpha_condition: Optional[str] = None,
        ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[JSON], Dict, Optional[pd.DataFrame]]:

        if sim_name is None:
            logger.info(f"Loading {sim_name} failed")
            return None, None, None, {}, None
        if f"{sim_name}.{model}.{horizon}.{config_grid}.{alpha_condition}" in self.sim_cache:
            logger.info(f"Loading {sim_name} from cache")
            return self.sim_cache[f"{sim_name}.{model}.{horizon}.{config_grid}.{alpha_condition}"]

        logger.info(f"Loading {sim_name} from scratch")
        sim_dir = f"{self.sim_dir}/{sim_name}"

        pnl_file = "pnl"
        if model is not None:
            pnl_file += f".model_{model}"
        if horizon is not None:
            pnl_file += f".horizon_{horizon}"
        if config_grid is not None:
            pnl_file += f".conf_{config_grid}"
        if alpha_condition is not None:
            pnl_file += f".condition_{alpha_condition}"

        sim_pnl_df = pd.read_csv(f"{sim_dir}/{pnl_file}.calculator.csv", index_col=0)
        sim_pnl_df['ts'] = to_datetime(sim_pnl_df['ts'])
        sim_pnl_df['notional'] = sim_pnl_df['long'] - sim_pnl_df['short']
        sim_pnl_df['cum_trading_volume'] = (sim_pnl_df['traded_long'] + sim_pnl_df['traded_short'].abs()).cumsum()
        if 'fees_usd' not in sim_pnl_df.columns:
            sim_pnl_df['fees_usd'] = self.exchange_fees * (sim_pnl_df['traded_long'] + sim_pnl_df['traded_short'].abs()).cumsum()
        if 'funding_income' not in sim_pnl_df.columns:
            sim_pnl_df['funding_income'] = 0
        sim_pnl_df['pnl_diff'] = sim_pnl_df['pnl'].diff()
        sim_pnl_df['period_return'] = sim_pnl_df['pnl_diff'] / sim_pnl_df['notional']
        sim_pnl_df = clip_col_by_iqr(sim_pnl_df, 'period_return')
        sim_pnl_df['hour_of_day'] = sim_pnl_df['ts'].dt.hour
        sim_pnl_df['day_of_week'] = sim_pnl_df['ts'].dt.day_name()

        sim_df = load_sim_data(sim_name, model, horizon, config_grid, alpha_condition)
        sim_df['executed_dollars_abs'] = sim_df['executed_dollars'].abs()

        json_data = None
        try:
            json_path = f"{sim_dir}/config.json"
            with open(json_path, 'r') as file:
                json_data = json.load(file)
        except:
            pass

        try:
            sim_summary_df = pd.read_csv(f'{sim_dir}/summary.txt', sep=':', names=['sim_name', 'metrics', 'value'])
            sim_summary_df = sim_summary_df.loc[sim_summary_df['sim_name'] == pnl_file]
            sim_summary_dict = dict(zip(sim_summary_df.metrics, sim_summary_df.value))
        except:
            sim_summary_dict = {}

        try:
            sim_pnl_breakdown_df = pd.read_csv(f"{sim_dir}/{pnl_file}.breakdown.csv", index_col=0)
            sim_pnl_breakdown_df['date'] = to_datetime(sim_pnl_breakdown_df['date'])
            sim_pnl_breakdown_df['ts'] = to_datetime(sim_pnl_breakdown_df['ts'])
        except Exception as e:
            sim_pnl_breakdown_df = None

        self.sim_cache[f"{sim_name}.{model}.{horizon}.{config_grid}.{alpha_condition}"] = (sim_df, sim_pnl_df, json_data, sim_summary_dict, sim_pnl_breakdown_df)
        return sim_df, sim_pnl_df, json_data, sim_summary_dict, sim_pnl_breakdown_df

    def setup_page(self):
        logger.info("Setting up page...")

        self.app.layout = html.Div([
            dcc.Location(id='url', refresh=False),
            html.Div(
                [
                    dcc.Link(
                        'Go to Two Sim Comparison', href='/page-1',
                        style={'font-size': '20px', 'font-weight': 'bold', 'marginTop': '20px', 'display': 'block'}),
                    html.Br(),
                    dcc.Link(
                        'Go to Sim Family Comparsion', href='/page-2',
                        style={'font-size': '20px', 'font-weight': 'bold', 'display': 'block'}),
                ],
                style={
                    'top': '50px',
                    'left': '20px',
                    'zIndex': 1000,
                },
            ),
            html.Div(id='page-content', style={'margin-left': '0px', 'margin-top': '50px'}),
        ])

        self.page_1_layout = html.Div([
            html.H1(children='Two Sim Reports', style={'textAlign': 'center', 'marginBottom': '20px'}),
            html.H3(children='Select two single sim to compare', style={'textAlign': 'center'}),
            html.Button("Refresh sims", id="update-btn2", n_clicks=0, style={"margin-top": "10px"}),
            dcc.Dropdown(id='sim-dropdown-selection'),
            dcc.Dropdown(id='sim-comp-dropdown-selection'),
            html.Div(id='sim-updates', style={'marginBottom': '20px'}),
            html.Div([
                html.H3(children='Compare summary table', style={'textAlign': 'center'}),
                DataTable(
                id='summary-table',
                style_table={'overflowX': 'auto'},
                columns=SUMMARY_COLS,
                page_size=20,
                sort_action="native"),
            ], style={'marginBottom': '20px'}),

            html.Button("Compare JSONs", id="compare-btn", n_clicks=0, style={"margin-top": "10px"}),
            html.Div(
                DataTable(
                    id='diff-output',
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "left", "whiteSpace": "normal"},
                    style_data={"height": "auto"},
                ),
                style={'marginBottom': '20px'},
            ),

            dcc.Graph(id='pnl-comp-graph', style={'marginBottom': '20px'}),
            dcc.Graph(id='notional-comp-graph', style={'marginBottom': '20px'}),
            dcc.Graph(id='volume-comp-graph', style={'marginBottom': '20px'}),
            dcc.Graph(id='funding-comp-graph', style={'marginBottom': '20px'}),
            dcc.Graph(id='utility-comp-graph', style={'marginBottom': '20px'}),

            dcc.Graph(id='pnl-by-security-graph', style={'marginBottom': '20px'}),
            dcc.Graph(id='traded-dollars-by-security-graph', style={'marginBottom': '20px'}),
            dcc.Graph(id='pnl-by-security-diff-graph', style={'marginBottom': '20px'}),

            html.Div([
                html.H3("Return Distribution by Time Period", style={'textAlign': 'center'}),
                dcc.Graph(id='sim-dow-return-figure'),
                dcc.Graph(id='sim-hour-return-figure'),
            ], style={'marginBottom': '20px'}),

            html.H3(children='Pnl Breakdown by feature', style={'textAlign': 'center', 'marginBottom': '20px'}),
            dcc.Dropdown(id='pnl-breakdown-selection', style={'marginBottom': '20px'}),
            dcc.Graph(id='pnl-breakdown-graph', style={'marginBottom': '20px'}),

            html.H3(children='Select factor', style={'textAlign': 'center'}),
            dcc.Dropdown(options=self.factors, id='sim-factor-dropdown-selection', style={"margin-top": "10px"}),
            dcc.Graph(id='sim-factor-return-graph', style={'marginBottom': '20px'}),
            dcc.Graph(id='sim-port-factor-exposure-graph', style={'marginBottom': '20px'}),
            dcc.Graph(id='sim-port-factor-return-graph', style={'marginBottom': '20px'}),
            dcc.Graph(id='sim-port-factor-pnl-graph', style={'marginBottom': '20px'}),
        ])
        self.page_2_layout = html.Div([
            html.H1(children='Sim Family Reports', style={'textAlign': 'center', 'marginBottom': '20px'}),
            html.H3(children='Select single sim family to compare', style={'textAlign': 'center'}),
            html.Button("Refresh sims", id="update-btn", n_clicks=0, style={"margin-top": "10px"}),
            dcc.Dropdown(id='sim-family-dropdown-selection', style={"margin-top": "10px"}),
            html.Div([
                html.H3(children='Compare family summary table', style={'textAlign': 'center'}),
                DataTable(
                id='summary-family-table',
                style_table={'overflowX': 'auto'},
                columns=SUMMARY_COLS,
                page_size=20,
                sort_action="native"),
            ], style={'marginBottom': '20px'}),

            dcc.Graph(id='pnl-family-comp-graph', style={'marginBottom': '20px'}),
            dcc.Graph(id='notional-family-comp-graph', style={'marginBottom': '20px'}),
            dcc.Graph(id='volume-family-comp-graph', style={'marginBottom': '20px'}),
            dcc.Graph(id='funding-family-comp-graph', style={'marginBottom': '20px'}),
            html.H3(children='Select factor', style={'textAlign': 'center'}),
            dcc.Dropdown(options=self.family_factors, id='factor-dropdown-selection', style={"margin-top": "10px"}),
            dcc.Graph(id='factor-return-graph', style={'marginBottom': '20px'}),
            dcc.Graph(id='port-factor-exposure-graph', style={'marginBottom': '20px'}),
            dcc.Graph(id='port-factor-return-graph', style={'marginBottom': '20px'}),
            dcc.Graph(id='port-factor-pnl-graph', style={'marginBottom': '20px'}),

        ])

        self.app.callback(
            Output('page-content', 'children'),
            Input('url', 'pathname'),
        )(self.display_page)

        self.app.callback(
            Output('sim-updates', 'children', allow_duplicate=True),
            Input('sim-dropdown-selection', 'value'),
            prevent_initial_call=True,
        )(self.update_sim_df)

        self.app.callback(
            Output('sim-updates', 'children'),
            Input('sim-comp-dropdown-selection', 'value'),
        )(self.update_sim_comp_df)

        self.app.callback(
            Output('summary-table', 'data'),
            Input('sim-updates', 'children'),
        )(self.get_summary_text)

        self.app.callback(
            Output('pnl-by-security-graph', 'figure'),
            Output('traded-dollars-by-security-graph', 'figure'),
            Input('sim-updates', 'children'),
        )(self.trading_by_security_figure)

        self.app.callback(
            Output('sim-dow-return-figure', 'figure'),
            Output('sim-hour-return-figure', 'figure'),
            Input('sim-updates', 'children'),
        )(self.sim_return_breakdown_figure)

        self.app.callback(
            Output('pnl-by-security-diff-graph', 'figure'),
            Input('sim-updates', 'children'),
        )(self.pnl_diff_by_security_figure)

        self.app.callback(
            Output('pnl-breakdown-selection', 'options'),
            Input('sim-updates', 'children'),
        )(self.update_fill_breakdown_dropdown)

        self.app.callback(
            Output('pnl-breakdown-graph', 'figure'),
            Input('pnl-breakdown-selection', 'value'),
        )(self.get_pnl_breakdown_figure)

        self.app.callback(
            Output('pnl-comp-graph', 'figure'),
            Output('notional-comp-graph', 'figure'),
            Output('volume-comp-graph', 'figure'),
            Output('funding-comp-graph', 'figure'),
            Output('utility-comp-graph', 'figure'),
            Input('sim-updates', 'children'),
        )(self.sim_comp_figures)

        self.app.callback(
            Output('sim-factor-dropdown-selection', 'options'),
            Input('sim-updates', 'children'),
        )(self.update_factor_options)

        self.app.callback(
            Output('sim-factor-return-graph', 'figure'),
            Output('sim-port-factor-exposure-graph', 'figure'),
            Output('sim-port-factor-return-graph', 'figure'),
            Output('sim-port-factor-pnl-graph', 'figure'),
            Input('summary-table', 'data'),
            Input('sim-factor-dropdown-selection', 'value'),
        )(self.sim_factor_return_figure)

        self.app.callback(
            Output("diff-output", "data"),
            Input("compare-btn", "n_clicks"),
        )(self.compare_jsons)

        self.app.callback(
            Output('sim-dropdown-selection', 'options'),
            Output('sim-comp-dropdown-selection', 'options'),
            Input("update-btn2", "n_clicks"),
        )(self.refresh_current_sim_dirs)

        self.app.callback(
            Output('sim-family-dropdown-selection', 'options'),
            Input("update-btn", "n_clicks"),
        )(self.refresh_sim_dirs)

        self.app.callback(
            Output('summary-family-table', 'data'),
            Input('sim-family-dropdown-selection', 'value'),
        )(self.get_family_summary_text)

        self.app.callback(
            Output('pnl-family-comp-graph', 'figure'),
            Output('notional-family-comp-graph', 'figure'),
            Output('volume-family-comp-graph', 'figure'),
            Output('funding-family-comp-graph', 'figure'),
            Input('summary-family-table', 'data'),
        )(self.sim_family_comp_figures)

        self.app.callback(
            Output('factor-dropdown-selection', 'options'),
            Input('summary-family-table', 'data'),
        )(self.update_family_factor_options)

        self.app.callback(
            Output('factor-return-graph', 'figure'),
            Output('port-factor-exposure-graph', 'figure'),
            Output('port-factor-return-graph', 'figure'),
            Output('port-factor-pnl-graph', 'figure'),
            Input('summary-family-table', 'data'),
            Input('factor-dropdown-selection', 'value'),
        )(self.sim_family_factor_return_figure)

    def refresh_current_sim_dirs(self, _) -> Tuple[List, List]:
        self.sim_names = get_sim_dirs(self.sim_dir)
        self.sim_names = sorted(self.sim_names, key=lambda sim_name: os.path.getmtime(f"{self.sim_dir}/{sim_name}"), reverse=True)
        self.extract_single_sim()
        self.current_sim_name = None
        self.current_sim_df = None
        self.current_sim_pnl_df = None
        self.current_json = None
        self.current_sim_summary_dict = {}

        self.comp_sim_name = None
        self.comp_sim_df = None
        self.comp_sim_pnl_df = None
        self.comp_json = None
        self.comp_sim_summary_dict = {}

        return self.single_sim_keys, ['NA'] + self.single_sim_keys

    def refresh_comp_sim_dirs(self, _):
        self.sim_names = get_sim_dirs(self.sim_dir)
        self.sim_names = sorted(self.sim_names, key=lambda sim_name: os.path.getmtime(f"{self.sim_dir}/{sim_name}"), reverse=True)
        self.extract_single_sim()

    def refresh_sim_dirs(self, _):
        self.sim_names = get_sim_dirs(self.sim_dir)
        self.sim_names = sorted(self.sim_names, key=lambda sim_name: os.path.getmtime(f"{self.sim_dir}/{sim_name}"), reverse=True)
        self.extract_single_sim()
        return self.sim_names

    def display_page(self, pathname: str):
        if pathname == '/page-1':
            return self.page_1_layout
        if pathname == '/page-2':
            return self.page_2_layout
        else:
            return html.Div([
                html.H1(children='Please Select Page for Sim Comparison', style={'textAlign': 'center', 'marginBottom': '20px'}),
            ])

    def compare_jsons(self, n_clicks: int):
        if n_clicks == 0:
            return []
        if self.comp_json is not None:
            diff = DeepDiff(self.current_json, self.comp_json, view="tree").to_dict()
        else:
            diff = None
        return diff_to_table(self.current_sim_name, self.comp_sim_name, diff)

    def load_latest_feature_file(self, horizon: int) -> pd.DataFrame:
        feature_files = sorted(glob.glob(f"{dir_manager.FEATURES_DIR}/features_{horizon}_*.parquet"))
        latest_file = feature_files[-1]
        df = pd.read_parquet(latest_file)
        return df

    def update_sim_df(self, sim_key: Optional[str]) -> bool:
        if sim_key is None:
            return False
        sim_name, model, horizon, config_grid, alpha_condition = self.single_sim_names[sim_key]
        sim_df, sim_pnl_df, json_data, sim_summary_dict, sim_pnl_breakdown_df = self.load_sim(sim_name, model, horizon, config_grid, alpha_condition)
        self.current_sim_name = sim_key
        self.current_sim_df = sim_df
        self.current_sim_pnl_df = sim_pnl_df
        self.current_json = json_data
        self.current_sim_summary_dict = sim_summary_dict
        self.current_sim_pnl_breakdown_df = make_symbol_venue(sim_pnl_breakdown_df) if sim_pnl_breakdown_df is not None else None
        self.factors = get_factors(json_data)
        return True

    def update_sim_comp_df(self, sim_key: Optional[str]) -> bool:
        if sim_key is None:
            return False
        if sim_key == 'NA':
            self.comp_sim_name = None
            self.comp_sim_df = None
            self.comp_sim_pnl_df = None
            self.comp_json = None
        else:
            sim_name, model, horizon, config_grid, alpha_condition = self.single_sim_names[sim_key]
            sim_df, sim_pnl_df, json_data, sim_summary_dict, _ = self.load_sim(sim_name, model, horizon, config_grid, alpha_condition)
            self.comp_sim_name = sim_key
            self.comp_sim_df = sim_df
            self.comp_sim_pnl_df = sim_pnl_df
            self.comp_json = json_data
            self.comp_sim_summary_dict = sim_summary_dict
        return True

    def update_factor_options(self, _) -> List:
        return self.factors

    def update_family_factor_options(self, _) -> List:
        return self.family_factors

    def get_summary_text(self, sim_updates: bool) -> str:
        if not sim_updates or self.current_sim_pnl_df is None:
            return [{}]

        current_daily_scaler = (1440 / self.current_json.get('REOPTIMIZE_INTERVAL_MINS', 360)) if self.current_json is not None else 4
        sim_res_dict = calc_return_metrics(self.current_sim_pnl_df, current_daily_scaler)
        if self.comp_sim_pnl_df is not None:
            comp_daily_scaler = (1440 / self.comp_json.get('REOPTIMIZE_INTERVAL_MINS', 360)) if self.comp_json is not None else 4
            sim_comp_res_dict = calc_return_metrics(self.comp_sim_pnl_df, comp_daily_scaler)
        else:
            sim_comp_res_dict = {}
        res = {
            'name': [self.current_sim_name, self.comp_sim_name],
            'pnl': [sim_res_dict['cum_pnl'], sim_comp_res_dict.get('cum_pnl')],
            'ret': [sim_res_dict['cum_ret'], sim_comp_res_dict.get('cum_ret')],
            'ret_annualized': [sim_res_dict['annualized_ret'], sim_comp_res_dict.get('annualized_ret')],
            'risk_annualized': [sim_res_dict['annualized_risk'], sim_comp_res_dict.get('annualized_risk')],
            'sharpe': [sim_res_dict['annualized_sharpe'], sim_comp_res_dict.get('annualized_sharpe')],
            'win_ratio': [self.current_sim_summary_dict.get('win_ratio'), self.comp_sim_summary_dict.get('win_ratio')],
            'profit_trades': [self.current_sim_summary_dict.get('profit_trades'), self.comp_sim_summary_dict.get('profit_trades')],
            'gain_per_fill': [self.current_sim_summary_dict.get('gain_per_fill'), self.comp_sim_summary_dict.get('gain_per_fill')],
            'loss_per_fill': [self.current_sim_summary_dict.get('loss_per_fill'), self.comp_sim_summary_dict.get('loss_per_fill')],
            'mdd': [sim_res_dict['max_drawdown'], sim_comp_res_dict.get('max_drawdown')],
            'mdd_perc': [sim_res_dict['max_drawdown_perc'], sim_comp_res_dict.get('max_drawdown_perc')],
            'avg_notional': [sim_res_dict['avg_notional'], sim_comp_res_dict.get('avg_notional')],
            'daily_trading_volume': [sim_res_dict['avg_trading_volume'], sim_comp_res_dict.get('avg_trading_volume')],
            'daily_turnover': [sim_res_dict['daily_turnover'], sim_comp_res_dict.get('daily_turnover')],
            'total_trading_fees': [sim_res_dict['cum_fees'], sim_comp_res_dict.get('cum_fees')],
            'trading_fees': [sim_res_dict['avg_fees'], sim_comp_res_dict.get('avg_fees')],
            'trading_fees_bps': [sim_res_dict['daily_fees_bps'], sim_comp_res_dict.get('daily_fees_bps')],
            'total_funding': [sim_res_dict['cum_funding'], sim_comp_res_dict.get('cum_funding')],
            'funding': [sim_res_dict['avg_funding'], sim_comp_res_dict.get('avg_funding')],
            'funding_bps': [sim_res_dict['daily_funding_bps'], sim_comp_res_dict.get('daily_funding_bps')],
        }

        data = [dict(zip(res, t)) for t in zip(*res.values())]
        return data

    def get_family_summary_text(self, sim_family: Optional[str]) -> List[Dict]:
        self.sim_family = sim_family
        if sim_family is None:
            return [{}]

        res = defaultdict(list)
        for sim in self.sim_family_keys[sim_family]:
            sim_name, model, horizon, config_grid, alpha_condition = sim
            _, sim_pnl_df, sim_json, sim_summary_dict, _ = self.load_sim(sim_name, model, horizon, config_grid, alpha_condition)
            daily_scaler = (1440 / sim_json.get('REOPTIMIZE_INTERVAL_MINS', 360)) if sim_json is not None else 4
            sim_res_dict = calc_return_metrics(sim_pnl_df, daily_scaler)
            res['name'].append(f"{sim_name}.{model}.{horizon}.{config_grid}.{alpha_condition}")
            res['pnl'].append(sim_res_dict['cum_pnl'])
            res['ret'].append(sim_res_dict['cum_ret'])
            res['ret_annualized'].append(sim_res_dict['annualized_ret'])
            res['risk_annualized'].append(sim_res_dict['annualized_risk'])
            res['sharpe'].append(sim_res_dict['annualized_sharpe'])
            res['win_ratio'].append(sim_summary_dict.get('win_ratio'))
            res['profit_trades'].append(sim_summary_dict.get('profit_trades'))
            res['gain_per_fill'].append(sim_summary_dict.get('gain_per_fill'))
            res['loss_per_fill'].append(sim_summary_dict.get('loss_per_fill'))
            res['mdd'].append(sim_res_dict['max_drawdown'])
            res['mdd_perc'].append(sim_res_dict['max_drawdown_perc'])
            res['avg_notional'].append(sim_res_dict['avg_notional'])
            res['daily_trading_volume'].append(sim_res_dict['avg_trading_volume'])
            res['daily_turnover'].append(sim_res_dict['daily_turnover'])
            res['total_trading_fees'].append(sim_res_dict['cum_fees'])
            res['trading_fees'].append(sim_res_dict['avg_fees'])
            res['trading_fees_bps'].append(sim_res_dict['daily_fees_bps'])
            res['total_funding'].append(sim_res_dict['cum_funding'])
            res['funding'].append(sim_res_dict['avg_funding'])
            res['funding_bps'].append(sim_res_dict['daily_funding_bps'])

        self.family_factors = get_factors(sim_json)

        data = [dict(zip(res, t)) for t in zip(*res.values())]
        return data

    def sim_family_comp_figure(self, fld: str) -> go.Figure:
        fig = go.Figure()
        if self.sim_family is None:
            return fig
        for sim in self.sim_family_keys[self.sim_family]:
            sim_name, model, horizon, config_grid, alpha_condition = sim
            _, sim_pnl_df, _, _, _ = self.load_sim(sim_name, model, horizon, config_grid, alpha_condition)
            if sim_pnl_df is None:
                return fig
            fld_df = sim_pnl_df[['ts', fld]].groupby('ts').sum()

            fig.add_trace(go.Scatter(
                x=fld_df.index,
                y=fld_df[fld],
                mode='lines',
                name=f"{fld}_{sim_name}.{model}.{horizon}.{alpha_condition}",
            ))
        fig.update_layout(
            title=f"{fld.upper()} Time Series by Name, Model, and Horizon",
            xaxis_title="timestamp",
            yaxis_title=fld,
        )
        return fig

    def sim_family_comp_figures(self, _) -> Tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
        return (
            self.sim_family_comp_figure(fld='pnl'),
            self.sim_family_comp_figure(fld='notional'),
            self.sim_family_comp_figure(fld='cum_trading_volume'),
            self.sim_family_comp_figure(fld='funding_income'),
        )

    def sim_factor_return_figure(self, _, factor: str) -> Tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
        factor_ret_fig = go.Figure()
        port_factor_exposure_fig = go.Figure()
        port_factor_ret_fig = go.Figure()
        port_factor_pnl_fig = go.Figure()
        if self.current_sim_name is None or factor not in self.factors:
            return factor_ret_fig, port_factor_exposure_fig, port_factor_ret_fig, port_factor_pnl_fig

        factor_ret_df, port_factor_exposure_df, port_factor_ret_df, port_factor_pnl_df = calc_factor_return(self.current_sim_df, factor)

        port_factor_exposure_fig.add_trace(go.Scatter(
            x=port_factor_exposure_df['ts'],
            y=port_factor_exposure_df[factor],
            mode='lines',
            name=f"{factor}_{self.current_sim_name}",
        ))

        port_factor_ret_fig.add_trace(go.Scatter(
            x=port_factor_ret_df['ts'],
            y=port_factor_ret_df[factor],
            mode='lines',
            name=f"{factor}_{self.current_sim_name}",
        ))

        port_factor_pnl_fig.add_trace(go.Scatter(
            x=port_factor_pnl_df['ts'],
            y=port_factor_pnl_df[factor],
            mode='lines',
            name=f"{factor}_{self.current_sim_name}",
        ))

        factor_ret_fig.add_trace(go.Scatter(
            x=factor_ret_df['ts'],
            y=factor_ret_df[factor],
            mode='lines',
            name=f"{factor}_{self.current_sim_name}",
        ))

        factor_ret_fig.update_layout(
            title=f"{factor.upper()} factor return over time",
            xaxis_title="timestamp",
            yaxis_title=factor,
        )
        if self.comp_sim_name is not None:

            factor_ret_df, port_factor_exposure_df, port_factor_ret_df, port_factor_pnl_df = calc_factor_return(self.comp_sim_df, factor)

            port_factor_exposure_fig.add_trace(go.Scatter(
                x=port_factor_exposure_df['ts'],
                y=port_factor_exposure_df[factor],
                mode='lines',
                name=f"{factor}_{self.comp_sim_name}",
            ))

            port_factor_ret_fig.add_trace(go.Scatter(
                x=port_factor_ret_df['ts'],
                y=port_factor_ret_df[factor],
                mode='lines',
                name=f"{factor}_{self.comp_sim_name}",
            ))

            port_factor_pnl_fig.add_trace(go.Scatter(
                x=port_factor_pnl_df['ts'],
                y=port_factor_pnl_df[factor],
                mode='lines',
                name=f"{factor}_{self.comp_sim_name}",
            ))

        port_factor_exposure_fig.update_layout(
            title=f"{factor.upper()} factor portfoliio exposure over time",
            xaxis_title="timestamp",
            yaxis_title=factor,
        )

        port_factor_ret_fig.update_layout(
            title=f"{factor.upper()} factor portfoliio return over time",
            xaxis_title="timestamp",
            yaxis_title=factor,
        )

        port_factor_pnl_fig.update_layout(
            title=f"{factor.upper()} factor portfoliio pnl over time",
            xaxis_title="timestamp",
            yaxis_title=factor,
        )

        return factor_ret_fig, port_factor_exposure_fig, port_factor_ret_fig, port_factor_pnl_fig

    def sim_family_factor_return_figure(self, _, factor: str) -> Tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
        factor_ret_fig = go.Figure()
        port_factor_exposure_fig = go.Figure()
        port_factor_ret_fig = go.Figure()
        port_factor_pnl_fig = go.Figure()
        if self.sim_family is None or factor not in self.family_factors:
            return factor_ret_fig, port_factor_exposure_fig, port_factor_ret_fig, port_factor_pnl_fig
        for sim in self.sim_family_keys[self.sim_family]:
            sim_name, model, horizon, config_grid, alpha_condition = sim
            sim_df, _, sim_json, _, _ = self.load_sim(sim_name, model, horizon, config_grid, alpha_condition)
            factor_ret_df, port_factor_exposure_df, port_factor_ret_df, port_factor_pnl_df = calc_factor_return(sim_df, factor)

            port_factor_exposure_fig.add_trace(go.Scatter(
                x=port_factor_exposure_df['ts'],
                y=port_factor_exposure_df[factor],
                mode='lines',
                name=f"{factor}_{sim_name}.{model}.{horizon}.{alpha_condition}",
            ))

            port_factor_ret_fig.add_trace(go.Scatter(
                x=port_factor_ret_df['ts'],
                y=port_factor_ret_df[factor],
                mode='lines',
                name=f"{factor}_{sim_name}.{model}.{horizon}.{alpha_condition}",
            ))

            port_factor_pnl_fig.add_trace(go.Scatter(
                x=port_factor_pnl_df['ts'],
                y=port_factor_pnl_df[factor],
                mode='lines',
                name=f"{factor}_{sim_name}.{model}.{horizon}.{alpha_condition}",
            ))

        port_factor_exposure_fig.update_layout(
            title=f"{factor.upper()} factor portfoliio exposure over time",
            xaxis_title="timestamp",
            yaxis_title=factor,
        )

        port_factor_ret_fig.update_layout(
            title=f"{factor.upper()} factor portfoliio return over time",
            xaxis_title="timestamp",
            yaxis_title=factor,
        )

        port_factor_pnl_fig.update_layout(
            title=f"{factor.upper()} factor portfoliio pnl over time",
            xaxis_title="timestamp",
            yaxis_title=factor,
        )

        factor_ret_fig.add_trace(go.Scatter(
            x=factor_ret_df['ts'],
            y=factor_ret_df[factor],
            mode='lines',
            name=f"{factor}_{sim_name}.{model}.{horizon}.{alpha_condition}",
        ))

        factor_ret_fig.update_layout(
            title=f"{factor.upper()} factor return over time",
            xaxis_title="timestamp",
            yaxis_title=factor,
        )

        return factor_ret_fig, port_factor_exposure_fig, port_factor_ret_fig, port_factor_pnl_fig

    def sim_comp_figure(self, sim_updates: bool, flds: List[str]) -> go.Figure:
        fig = go.Figure()

        if not sim_updates or self.current_sim_pnl_df is None:
            logger.info("No dataframe for sim_comp_figure")
        elif self.comp_sim_pnl_df is None and set(flds).issubset(set(self.current_sim_pnl_df.columns)):
            pnl_df = self.current_sim_pnl_df[['ts'] + flds].groupby('ts').sum()
            fig = px.line(pnl_df, x=pnl_df.index, y=pnl_df.columns, title=f"{', '.join(fld.capitalize() for fld in flds)} Single Sim Results")
        elif self.comp_sim_pnl_df is not None and set(flds).issubset(set(self.current_sim_pnl_df.columns)) and set(flds).issubset(set(self.comp_sim_pnl_df.columns)):
            pnl_df = pd.merge(
                self.current_sim_pnl_df[['ts'] + flds].groupby('ts').sum(),
                self.comp_sim_pnl_df[['ts'] + flds].groupby('ts').sum(),
                left_on='ts', right_on='ts', how='outer', suffixes=(f'_{self.current_sim_name}', f'_{self.comp_sim_name}'),
            )
            pnl_df = pnl_df.ffill().fillna(0)
            fig = px.line(
                pnl_df,
                x=pnl_df.index,
                y=pnl_df.columns,
                title=f"{', '.join(fld.capitalize() for fld in flds)} Comparison",
            )
        return fig

    def sim_comp_figures(self, sim_updates: bool) -> Tuple[go.Figure, go.Figure, go.Figure, go.Figure, go.Figure]:
        utility_flds = [
            'expected_utility', 'expected_risk', 'expected_factor_risk', 'expected_resid_risk',
            'expected_return', 'expected_return_bps', 'expected_slippage', 'expected_fees',
        ]
        return (
            self.sim_comp_figure(sim_updates=sim_updates, flds=['pnl']),
            self.sim_comp_figure(sim_updates=sim_updates, flds=['notional']),
            self.sim_comp_figure(sim_updates=sim_updates, flds=['cum_trading_volume']),
            self.sim_comp_figure(sim_updates=sim_updates, flds=['funding_income']),
            self.sim_comp_figure(sim_updates=sim_updates, flds=utility_flds),
        )

    def trading_by_security_figure(self, sim_updates: bool) -> Tuple[go.Figure, go.Figure]:
        pnl_fig = go.Figure()
        traded_dollars_fig = go.Figure()
        if sim_updates and self.current_sim_df is not None:
            max_ts = self.current_sim_df.index.get_level_values('ts').max()
            max_ts_idx = self.current_sim_df.index.get_level_values('ts') == max_ts
            last_pos_df = self.current_sim_df.loc[max_ts_idx].sort_values(by='pnl').reset_index()
            last_pos_df = last_pos_df.loc[last_pos_df.pnl != 0]
            pnl_fig = px.bar(last_pos_df, x='symbol_venue', y='pnl', title=f"PnL By Security of {self.current_sim_name} at {max_ts}")
            avg_execution_df = self.current_sim_df.groupby(['symbol_venue', 'date'])['executed_dollars_abs'].sum().groupby('symbol_venue').mean().sort_values().reset_index()
            avg_execution_df = avg_execution_df.loc[avg_execution_df.executed_dollars_abs != 0]
            traded_dollars_fig = px.bar(avg_execution_df, x='symbol_venue', y='executed_dollars_abs', title=f"Avg Executed Dollars By Security of {self.current_sim_name}")
        return pnl_fig, traded_dollars_fig

    def pnl_diff_by_security_figure(self, sim_updates: bool) -> go.Figure:
        fig = go.Figure()
        if not sim_updates or self.current_sim_df is None or self.comp_sim_df is None:
            return fig
        pos_df = merge_on_index(self.current_sim_df, self.comp_sim_df, suffixes=('_curr', '_comp'), how='inner')
        if pos_df.empty:
            return fig
        pos_df['pnl_diff'] = pos_df['pnl_curr'] - pos_df['pnl_comp']
        max_ts = pos_df.index.get_level_values('ts').max()
        last_pos_df = pos_df[pos_df.index.get_level_values('ts') == max_ts].sort_values(by='pnl_diff').reset_index()
        last_pos_df = last_pos_df.loc[last_pos_df['pnl_diff'] != 0]
        return px.bar(last_pos_df, x='symbol_venue', y='pnl_diff', title=f"PnL Diff By Security of {self.current_sim_name} vs {self.comp_sim_name} at {max_ts}")

    def sim_return_breakdown_figure(self, sim_updates: bool) -> Tuple[go.Figure, go.Figure]:
        fig_dow = go.Figure()
        fig_hod = go.Figure()
        if not sim_updates or self.current_sim_pnl_df is None:
            return fig_dow, fig_hod
        df_dict = {}
        df_dict['sim'] = (self.current_sim_name, self.current_sim_pnl_df)
        if self.comp_sim_df is not None:
            df_dict['sim_comp'] = (self.comp_sim_name, self.comp_sim_pnl_df)

        for sim_name, sim_pnl_df in df_dict.values():
            dow_avg = sim_pnl_df.groupby('day_of_week')['period_return'].mean() * 10000
            dow_avg = dow_avg.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
            fig_dow.add_trace(go.Bar(
                x=dow_avg.index,
                y=dow_avg.values,
                name=sim_name,
            ))
        fig_dow.update_layout(
            title='Average Return by Day of Week',
            xaxis_title='Day of Week',
            yaxis_title='Average Return Bps',
            barmode='group',
        )
        for sim_name, sim_pnl_df in df_dict.values():
            all_hours = pd.Series(index=range(24), data=0.0)
            hod_avg = sim_pnl_df.groupby('hour_of_day')['period_return'].mean() * 10000
            all_hours.update(hod_avg)
            fig_hod.add_trace(go.Bar(
                x=all_hours.index,
                y=all_hours.values,
                name=sim_name,
            ))
        fig_hod.update_layout(
            title='Average Return by Hour of Day',
            xaxis_title='Hour of Day',
            yaxis_title='Average Return Bps',
            barmode='group',
            xaxis={
                "tickmode": 'linear',
                "tick0": 0,
                "dtick": 1,
                "tickvals": list(range(24)),
                "ticktext": [f"{h}" for h in range(24)],
            },
        )

        return fig_dow, fig_hod

    def update_fill_breakdown_dropdown(self, _) -> List[Dict]:
        if self.current_sim_pnl_breakdown_df is None:
            return []
        start_dt = self.current_sim_pnl_breakdown_df['date'].min()
        end_dt = self.current_sim_pnl_breakdown_df['date'].max()
        self.pnl_fill_breakdown = FillBreakdown(start=start_dt, end=end_dt)
        self.pnl_fill_breakdown.load_features_df(cols=None, update_name_list=True)
        current_sim_pnl_breakdown_keys = sorted(unique_list(self.pnl_fill_breakdown.features_name_list + self.pnl_fill_breakdown.bars_name_list))
        options = [{'label': breakdown, 'value': breakdown} for breakdown in current_sim_pnl_breakdown_keys]
        return options

    def get_pnl_breakdown_figure(self, value) -> go.Figure:
        fig = go.Figure()
        if self.current_sim_pnl_breakdown_df is None:
            return fig
        self.pnl_fill_breakdown.load_fills(self.current_sim_pnl_breakdown_df)
        self.pnl_fill_breakdown.load_features_df(cols=[value], update_name_list=False)
        if self.pnl_fill_breakdown.features_df is None:
            return fig

        pnl_breakdown_dict = self.pnl_fill_breakdown.get_pnl_breakdowns(col=value, merge_on_ts=True, use_cum_pnl=True)
        df = pnl_breakdown_dict.get(value, None)
        if df is None:
            return fig
        return px.line(df, x='ts', y='realized_pnl', color=f'{value}_quintile', title=f"Cum. Realized Pnl Quantile by Feature {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sim reports')
    parser.add_argument('-p', '--port', help='port', required=False, type=int, default=8063)
    parser.add_argument('-d', '--dir', help='sim directory', required=False, type=str)
    parser.add_argument('-c', '--config', help='config file', required=False)
    args = vars(parser.parse_args())
    _, config = get_config(args.get('config'))
    port = args['port']
    SimReports(config=config, port=port, sim_dir=args.get('sim_dir')).run()
