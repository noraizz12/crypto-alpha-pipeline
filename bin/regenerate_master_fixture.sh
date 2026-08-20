#!/bin/bash
. $ROOT_DIR/src/bin/include.sh
init_env

cd $SRC_DIR
echo "cd to $SRC_DIR"

# Function to print usage information
usage() {
    echo "Usage: $0 -n <name>"
    echo "  name: 'bars', 'sim', 'feature', 'forwards', 'models', 'fits', 'alphas', 'server', or 'pnl'"
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
        # Force regeneration by deleting existing fixtures first
        rm -rf $SRC_DIR/test/fixtures/bars/master/*
        pytest -rA -vv test/integration_test_generate_bars.py::test_generate_master_fixture
        ;;
    sim)
        echo "Running sim test..."
        # Force regeneration by deleting existing fixtures first
        rm -rf $SRC_DIR/test/fixtures/sim/master/*
        pytest -rA -vv test/integration_test_sim.py::test_generate_master_fixture
        ;;
    feature)
        echo "Running feature test..."
        # Force regeneration by deleting existing fixtures first
        rm -rf $SRC_DIR/test/fixtures/features/master/*
        pytest -rA -vv test/integration_test_generate_features.py::test_generate_master_fixture
        ;;
    server)
        echo "Running server test..."
        # Force regeneration by deleting existing fixtures first
        rm -rf $SRC_DIR/test/fixtures/server/master/*
        pytest -rA -vv test/integration_test_server.py::test_generate_master_fixture
        ;;
    pnl)
        echo "Running pnl test..."
        # Force regeneration by deleting existing fixtures first
        rm -rf $SRC_DIR/test/fixtures/pnl/master/*
        pytest -rA -vv test/integration_test_pnl_calculator.py::test_generate_master_fixture
        ;;
    models)
        echo "Running models test..."
        # Force regeneration by deleting existing fixtures first
        rm -rf $SRC_DIR/test/fixtures/models/master/*
        pytest -rA -vv test/integration_test_generate_models.py::test_generate_master_fixture
        ;;
    fits)
        echo "Running fits test..."
        # Force regeneration by deleting existing fixtures first
        echo "Deleting fits fixtures at: $SRC_DIR/test/fixtures/fits/master/"
        find $SRC_DIR/test/fixtures/fits/master/ -type f -name "*.csv" -o -name "*.joblib" -o -name "*.features" -o -name "*.metrics" | xargs rm -f 2>/dev/null || true
        echo "Files remaining after deletion: $(find $SRC_DIR/test/fixtures/fits/master/ -type f 2>/dev/null | wc -l)"
        pytest -rA -vv test/integration_test_generate_fits.py::test_generate_master_fixture
        ;;
    alphas)
        echo "Running alphas test..."
        # Force regeneration by deleting existing fixtures first
        rm -rf $SRC_DIR/test/fixtures/alphas/master/*
        pytest -rA -vv test/integration_test_generate_alphas.py::test_generate_master_fixture
        ;;
    forwards)
        echo "Running forwards test..."
        # Force regeneration by deleting existing fixtures first
        rm -rf $SRC_DIR/test/fixtures/forwards/master/*
        pytest -rA -vv test/integration_test_generate_forwards.py::test_generate_master_fixture
        ;;
    *)
        echo "Error: Invalid name. Use 'bars', 'sim', 'feature', 'forwards', 'models', 'fits', 'alphas', 'server', or 'pnl'."
        usage
        ;;
esac