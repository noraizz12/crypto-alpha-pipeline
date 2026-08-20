#!/bin/bash
. $ROOT_DIR/src/bin/include.sh
init_env

TEST_DIR=$SRC_DIR/test

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default to running all tests
RUN_UNIT=true
RUN_INTEGRATION=true

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -u|--unit)
            RUN_UNIT=true
            RUN_INTEGRATION=false
            ;;
        -i|--integration)
            RUN_UNIT=false
            RUN_INTEGRATION=true
            ;;
        -a|--all)
            RUN_UNIT=true
            RUN_INTEGRATION=true
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -u, --unit        Run only unit tests"
            echo "  -i, --integration Run only integration tests"
            echo "  -a, --all         Run all tests (default)"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
    shift
done

# Function to run unit tests
run_unit_tests() {
    echo -e "${BLUE}================== Running Unit Tests ==================${NC}"

    # Activate virtual environment
    if [ -n "$VIRTUALENV_DIR" ] && [ -f "$VIRTUALENV_DIR/bin/activate" ]; then
        source "$VIRTUALENV_DIR/bin/activate"
    fi
    
    # Add the src directory to PYTHONPATH
    export PYTHONPATH="$SRC_DIR:$PYTHONPATH"
    
    # Change to source directory
    cd $SRC_DIR

    # Run all unit tests
    echo -e "${YELLOW}Running all unit tests...${NC}"
    if python -m pytest test/test_*.py -v --tb=short --ignore=test/integration_test_*.py; then
        echo -e "${GREEN}✓ Unit tests passed successfully.${NC}\n"
        return 0
    else
        echo -e "${RED}✗ Unit tests failed.${NC}"
        return 1
    fi
}

# Function to run integration tests
run_integration_tests() {
    echo -e "${BLUE}================ Running Integration Tests =================${NC}"

    # Activate virtual environment
    if [ -n "$VIRTUALENV_DIR" ] && [ -f "$VIRTUALENV_DIR/bin/activate" ]; then
        source "$VIRTUALENV_DIR/bin/activate"
    fi
    
    # Add the src directory to PYTHONPATH
    export PYTHONPATH="$SRC_DIR:$PYTHONPATH"
    
    # Change to test directory
    cd $TEST_DIR
    
    # Run all integration tests
    echo -e "${YELLOW}Running all integration tests...${NC}"
    python -m pytest -rA -vv \
        integration_test_generate_alphas.py::test_alphas_run \
        integration_test_generate_bars.py::test_bar_generation \
        integration_test_generate_features.py::test_feature_run \
        integration_test_generate_fits.py::test_fits_run \
        integration_test_generate_forwards.py::test_forwards_run \
        integration_test_generate_models.py::test_model_run \
        integration_test_server.py::test_server_run \
        integration_test_sim.py::test_simulate_calculation
    
    return $?
}

# Main execution
echo -e "${BLUE}========== Test Runner ==========${NC}\n"

exit_code=0

if [ "$RUN_UNIT" = true ]; then
    if ! run_unit_tests; then
        exit_code=1
    fi
fi

if [ "$RUN_INTEGRATION" = true ]; then
    if ! run_integration_tests; then
        exit_code=1
    fi
fi

if [ $exit_code -eq 0 ]; then
    echo -e "\n${GREEN}✓ All requested tests passed successfully.${NC}"
else
    echo -e "\n${RED}✗ Some tests failed.${NC}"
fi

exit $exit_code