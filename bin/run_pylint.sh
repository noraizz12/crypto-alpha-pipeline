#!/bin/bash
. $ROOT_DIR/src/bin/include.sh
init_env

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Check if file argument is provided
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: No file specified${NC}"
    echo "Usage: $0 <file_path>"
    echo "Example: $0 lib/data.py"
    exit 1
fi

FILE_PATH="$1"

# Check if file exists
if [ ! -f "$PROJECT_ROOT/$FILE_PATH" ] && [ ! -f "$FILE_PATH" ]; then
    echo -e "${RED}Error: File not found: $FILE_PATH${NC}"
    exit 1
fi

# Determine absolute file path
if [ -f "$PROJECT_ROOT/$FILE_PATH" ]; then
    FULL_PATH="$PROJECT_ROOT/$FILE_PATH"
elif [ -f "$FILE_PATH" ]; then
    FULL_PATH="$FILE_PATH"
fi

# Change to project root directory
cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -n "$VIRTUALENV_DIR" ] && [ -f "$VIRTUALENV_DIR/bin/activate" ]; then
    echo -e "${BLUE}Activating virtual environment ($VIRTUALENV_DIR)...${NC}"
    source "$VIRTUALENV_DIR/bin/activate"
fi

# Check if pylint is installed
if ! command -v pylint &> /dev/null; then
    echo -e "${RED}Error: pylint is not installed${NC}"
    echo "Please install pylint: pip install pylint"
    exit 1
fi

# Get just the filename for display
FILENAME=$(basename "$FULL_PATH")

echo -e "${BLUE}Running Pylint on: ${YELLOW}$FILE_PATH${NC}"
echo "----------------------------------------"

# Run pylint with project configuration
if [ -f ".pylintrc" ]; then
    pylint --rcfile=.pylintrc "$FULL_PATH"
    EXIT_CODE=$?
else
    echo -e "${YELLOW}Warning: .pylintrc not found, using default configuration${NC}"
    pylint "$FULL_PATH"
    EXIT_CODE=$?
fi

echo "----------------------------------------"

# Provide feedback based on exit code
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Pylint passed with no issues!${NC}"
elif [ $EXIT_CODE -lt 32 ]; then
    # Exit codes 1, 2, 4, 8, 16 are for different message types
    # but the file still passes the minimum score
    echo -e "${YELLOW}⚠ Pylint found some issues but passed minimum score${NC}"
else
    echo -e "${RED}✗ Pylint failed - score below threshold${NC}"
fi

# Show score threshold from .pylintrc
if [ -f ".pylintrc" ]; then
    THRESHOLD=$(grep -E "^fail-under=" .pylintrc | cut -d'=' -f2)
    if [ ! -z "$THRESHOLD" ]; then
        echo -e "${BLUE}Minimum score threshold: $THRESHOLD${NC}"
    fi
fi

exit $EXIT_CODE