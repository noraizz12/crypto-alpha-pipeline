import argparse
import logging.config
import shutil
import sys
import traceback
from datetime import datetime as dt
from datetime import timedelta as td
from typing import Dict, List, Literal, Optional

import pandas as pd

from lib.util.config import get_config
from lib.util.files import make_sim_dir
from lib.util.slack import JobMonitor
from lib.portfolio_optimization import PortfolioOptimizer
from lib.sim.simulate import Simulate
from lib.util.time_util import date_str_to_date, date_to_str, yesterday_date
from lib.util.logging_util import get_logging_config

DUMP_STATE = False
DEBUG_SYMBOL = None

pd.options.display.max_columns = 30
pd.options.display.width = 999
pd.options.display.max_rows = 999
pd.options.mode.chained_assignment = None


def start_sim(
        start_date: dt.date,
        end_date: dt.date,
        config: Dict,
        sim_dir: str,
        init_positions: bool,
        verbose: bool,
        sim_comp: bool,
        skip_missing_alphas: bool,
        horizons: Optional[List],
        models: Optional[List],
        config_grid: Optional[str] = None,
        alpha_condition: Optional[Literal['rev', 'mom']] = None,
        alpha_dir: Optional[str] = None,
        cleanup_checkpoints: bool = False
):
    sim = Simulate(
        config=config,
        sim_dir=sim_dir,
        initialize_positions=init_positions,
        verbose=verbose,
        sim_comp=sim_comp,
        skip_missing_alphas=skip_missing_alphas,
        config_grid=config_grid,
        alpha_condition=alpha_condition,
        alpha_dir=alpha_dir,
        cleanup_checkpoints=cleanup_checkpoints
    )
    if horizons is not None and models is not None:
        for horizon in horizons:
            for model in models:
                try:
                    sim.simulate(start_date=start_date, end_date=end_date, horizons=[horizon], models=[model], weight_override=True)
                except Exception as e:
                    logger.error(traceback.format_exc())
                    logger.error(e)
                    logger.error(f"ERROR Could not simulate {horizon} {model}")
                    continue
    elif horizons is not None:
        for horizon in horizons:
            try:
                sim.simulate(start_date=start_date, end_date=end_date, horizons=[horizon])
            except Exception as e:
                logger.error(traceback.format_exc())
                logger.error(e)
                logger.error(f"ERROR Could not simulate {horizon}")
                continue
    elif models is not None:
        for model in models:
            try:
                sim.simulate(start_date=start_date, end_date=end_date, models=[model])
            except Exception as e:
                logger.error(traceback.format_exc())
                logger.error(e)
                logger.error(f"ERROR Could not simulate {model}")
                continue
    else:
        sim.simulate(start_date=start_date, end_date=end_date)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Data For input into simulation')
    parser.add_argument('-f', '--from', help='from date YYYYMMDD', required=False, type=str, default=None)
    parser.add_argument('-t', '--to', help='to date YYYYMMDD', required=False, type=str, default=None)
    parser.add_argument('-n', '--name', help='name of sim directory', required=False, type=str, default=None)
    parser.add_argument('-d', '--debug-opt', help='file of dumped state', required=False, type=str, default=None)
    parser.add_argument('-z', '--horizons', help='horizons', required=False, type=str)
    parser.add_argument('-m', '--models', help='models', required=False, type=str)
    parser.add_argument('-g', '--grids', help='grids', required=False, type=str)
    parser.add_argument('-c', '--config', help='config file', required=False)
    parser.add_argument('-p', '--init-positions', dest='init_positions', action='store_true')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true')
    parser.add_argument('-s', '--simcomp', dest='sim_comp', action='store_true')
    parser.add_argument('-sl', '--sim-lookback', dest='sim_lookback', required=False, type=int, default=None)
    parser.add_argument('-sa', '--skip-missing-alphas', dest='skip_missing_alphas', action='store_true')
    parser.add_argument('-r', '--revmom', dest='revmom', action='store_true')
    parser.add_argument('-nc', '--notify', help='slack channel to notify', required=False, type=str, default=None)
    parser.add_argument('-q', '--quiet', help='suppress slack notifications', action='store_true', required=False)
    parser.add_argument('--sim-dir', help='custom base directory for simulation output', required=False, type=str, default=None)
    parser.add_argument('-a', '--alpha-dir', help='alpha dir', required=False, type=str, default=None)
    parser.add_argument('--cleanup', dest='cleanup_checkpoints', action='store_true', help='remove checkpoint files after simulation')
    parser.set_defaults(init_positions=False, verbose=False, sim_comp=False, skip_missing_alphas=False, revmom=False, quiet=False, cleanup_checkpoints=False)
    args = vars(parser.parse_args())

    config_file, config = get_config(args.get('config'))

    verbose = args['verbose']
    init_positions = args['init_positions']
    sim_name = args.get('name')
    sim_comp = args['sim_comp']
    sim_lookback = args['sim_lookback']
    skip_missing_alphas = args['skip_missing_alphas']
    revmom = args['revmom']
    notify_channel = args['notify']
    quiet = args['quiet']
    custom_sim_dir = args.get('sim_dir')
    alpha_dir = args.get('alpha_dir')
    cleanup_checkpoints = args['cleanup_checkpoints']

    with JobMonitor("Sim run", notify_channel, quiet=quiet):
        if args['debug_opt'] is not None:
            ts_df = pd.read_csv(args['debug_opt'])
            trade_bot = PortfolioOptimizer(config=config)
            trade_bot.optimize(ts_df)
            sys.exit()

        start_date = end_date = None

        if sim_comp:
            end_date = yesterday_date()
            lookback_days = sim_lookback if sim_lookback is not None else 7
            if lookback_days < 1:
                raise ValueError("lookback_days must be >= 1")
            start_date = end_date - td(days=lookback_days - 1)
            sim_name = f"sim_comp_{lookback_days}d_{date_to_str(end_date)}"
            skip_missing_alphas = True
        elif sim_lookback is not None:
            end_date = yesterday_date()
            start_date = end_date - td(days=sim_lookback)
            sim_name = f"sim_lookback_{date_to_str(end_date)}_{sim_lookback}d"
        else:
            if args['to'] is not None:
                end_date = date_str_to_date(str(args['to']))

            if args['from'] is not None:
                start_date = date_str_to_date(args['from'])

        horizons = args.get('horizons')
        if horizons is not None:
            horizons = [int(h) for h in horizons.split(',')]

        models = args.get('models')
        if models is not None:
            models = list(models.split(','))

        grids = args.get('grids')
        if grids is not None:
            field = grids.split(':')[0]
            grids = list(grids.split(':')[1].split(','))

        sim_dir = make_sim_dir(sim_name, base_dir=custom_sim_dir)

        logging.config.dictConfig(get_logging_config("sim.log", log_dir=sim_dir))
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        shutil.copy(config_file, sim_dir + "/config.json")

        if grids is not None:
            for grid_val in grids:
                config[field] = float(grid_val)
                config_grid = f"conf_{field}_{grid_val}"
                if revmom:
                    start_sim(start_date, end_date, config, sim_dir, init_positions, verbose, sim_comp, skip_missing_alphas, horizons, models, config_grid, 'rev', alpha_dir, cleanup_checkpoints)
                    start_sim(start_date, end_date, config, sim_dir, init_positions, verbose, sim_comp, skip_missing_alphas, horizons, models, config_grid, 'mom', alpha_dir, cleanup_checkpoints)
                else:
                    start_sim(start_date, end_date, config, sim_dir, init_positions, verbose, sim_comp, skip_missing_alphas, horizons, models, config_grid, None, alpha_dir, cleanup_checkpoints)
        else:
            if revmom:
                start_sim(start_date, end_date, config, sim_dir, init_positions, verbose, sim_comp, skip_missing_alphas, horizons, models, alpha_condition='rev', alpha_dir=alpha_dir, cleanup_checkpoints=cleanup_checkpoints)
                start_sim(start_date, end_date, config, sim_dir, init_positions, verbose, sim_comp, skip_missing_alphas, horizons, models, alpha_condition='mom', alpha_dir=alpha_dir, cleanup_checkpoints=cleanup_checkpoints)
            else:
                start_sim(start_date, end_date, config, sim_dir, init_positions, verbose, sim_comp, skip_missing_alphas, horizons, models, config_grid=None, alpha_condition=None, alpha_dir=alpha_dir, cleanup_checkpoints=cleanup_checkpoints)
