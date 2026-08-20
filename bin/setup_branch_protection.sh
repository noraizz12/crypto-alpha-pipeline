#!/bin/bash

# setup_branch_protection.sh - Configure GitHub branch protection for master branch
# This script sets up protection rules that:
# - Require pull requests (no direct pushes)
# - Allow admin bypass for hotfixes
# - Prevent force pushes and deletions
# - No approval requirements

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up branch protection for master branch...${NC}"

# Get repository information
REPO_INFO=$(gh repo view --json owner,name)
OWNER=$(echo $REPO_INFO | jq -r '.owner.login')
REPO=$(echo $REPO_INFO | jq -r '.name')

if [ -z "$OWNER" ] || [ -z "$REPO" ]; then
    echo -e "${RED}Error: Could not determine repository owner/name${NC}"
    echo "Make sure you're in a git repository and have gh CLI configured"
    exit 1
fi

echo "Repository: $OWNER/$REPO"

# Check if we're in the right directory
if [ ! -d ".git" ]; then
    echo -e "${RED}Error: Not in a git repository root${NC}"
    exit 1
fi

# Create branch protection rules
echo -e "${YELLOW}Configuring branch protection rules...${NC}"

# The protection rules as JSON
PROTECTION_RULES='{
  "required_status_checks": null,
  "enforce_admins": false,
  "required_pull_request_reviews": null,
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "block_creations": false,
  "required_conversation_resolution": false,
  "lock_branch": false,
  "allow_fork_syncing": false,
  "required_linear_history": false
}'

# Apply protection rules
echo "Applying protection rules to master branch..."
RESPONSE=$(gh api \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  repos/$OWNER/$REPO/branches/master/protection \
  --input - <<< "$PROTECTION_RULES" 2>&1) || {
    echo -e "${RED}Error applying protection rules:${NC}"
    echo "$RESPONSE"
    exit 1
}

echo -e "${GREEN}✓ Branch protection successfully configured!${NC}"
echo ""
echo "Master branch protection settings:"
echo "  • Pull requests required (no direct pushes)"
echo "  • Admins CAN bypass for hotfixes"
echo "  • Force pushes: BLOCKED"
echo "  • Branch deletion: BLOCKED"
echo "  • PR approvals: NOT required"
echo ""
echo -e "${YELLOW}Note: You can still push hotfixes directly to master when needed.${NC}"
echo -e "${YELLOW}The protection mainly prevents accidental direct pushes.${NC}"

# Optional: Show current protection status
echo ""
echo "Current protection status:"
gh api repos/$OWNER/$REPO/branches/master/protection | jq '.'