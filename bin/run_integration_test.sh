#!/bin/bash
. $ROOT_DIR/src/bin/include.sh
init_env

cd $SRC_DIR
echo "cd to $SRC_DIR"

# Function to print usage information
usage() {
    echo "Usage: $0 -n <name>"
    echo "  name: 'bars', 'sim', 'feature', 'forwards', 'models', 'fits', 'alphas', 'server'"
    exit 1
}

# Initialize variables
name=""

# Parse command line options
while getopts "n:" opt; do
    case $opt in
        n) name="$OPTARG" ;;
        *) usage ;;
    esac
done

# Check if name is provided
if [ -z "$name" ]; then
    echo "Error: Name argument is required."
    usage
fi

# Run the appropriate command based on the name argument
case $name in
    bars)
        echo "Running bars test..."
        python -m pytest -rA -vv test/integration_test_generate_bars.py::test_bar_generation
        ;;
    sim)
        echo "Running sim test..."
        python -m pytest -rA -vv test/integration_test_sim.py::test_simulate_calculation
        ;;
    feature)
        echo "Running feature test..."
        python -m pytest -rA -vv test/integration_test_generate_features.py::test_feature_run
        ;;
    server)
        echo "Running server test..."
        python -m pytest -rA -vv test/integration_test_server.py::test_server_run
        ;;
    models)
        echo "Running models test..."
        python -m pytest -rA -vv test/integration_test_generate_models.py::test_model_run
        ;;
    fits)
        echo "Running fits test..."
        python -m pytest -rA -vv test/integration_test_generate_fits.py::test_fits_run
        ;;
    alphas)
        echo "Running alphas test..."
        python -m pytest -rA -vv test/integration_test_generate_alphas.py::test_alphas_run
        ;;
    forwards)
        echo "Running forwards test..."
        python -m pytest -rA -vv test/integration_test_generate_forwards.py::test_forwards_run
        ;;
    *)
        echo "Error: Invalid name. Use 'bars', 'feature', 'forwards', 'models', 'fits', 'alphas', 'sim', 'server', or 'pnl'."
        usage
        ;;
esac