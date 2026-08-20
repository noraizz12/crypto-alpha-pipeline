#!/bin/bash
pytest test/test_portfolio_optimization.py::TestOptimizationEndToEnd::test_optimize_regression_snapshot -v -s --log-cli-level=INFO
