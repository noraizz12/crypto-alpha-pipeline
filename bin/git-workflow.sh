#!/bin/bash
# Git workflow helper script for stat_arb project

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Ensure we're in the correct directory
cd "$(dirname "$0")/.."

function print_usage() {
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  feature <name>    Create a new feature branch from develop"
    echo "  fix <name>        Create a new fix branch from develop"
    echo "  hotfix <name>     Create a hotfix branch from master (urgent fixes)"
    echo "  sync              Sync your current branch with develop"
    echo "  status            Show branch status and workflow info"
    echo ""
    echo "Examples:"
    echo "  $0 feature add-new-alpha-model"
    echo "  $0 fix correct-volume-calculation"
    echo "  $0 hotfix emergency-trading-halt"
}

function create_feature_branch() {
    local branch_name="feature/$1"
    echo -e "${GREEN}Creating feature branch: $branch_name${NC}"
    
    # Ensure we're on develop and it's up to date
    git checkout develop
    git pull origin develop
    
    # Create and checkout new branch
    git checkout -b "$branch_name"
    
    echo -e "${GREEN}✓ Created branch '$branch_name' from develop${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Make your changes"
    echo "2. git add <files>"
    echo "3. git commit -m \"Your commit message\""
    echo "4. git push -u origin $branch_name"
    echo "5. Create PR to merge into develop"
}

function create_fix_branch() {
    local branch_name="fix/$1"
    echo -e "${GREEN}Creating fix branch: $branch_name${NC}"
    
    # Ensure we're on develop and it's up to date
    git checkout develop
    git pull origin develop
    
    # Create and checkout new branch
    git checkout -b "$branch_name"
    
    echo -e "${GREEN}✓ Created branch '$branch_name' from develop${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Make your fixes"
    echo "2. git add <files>"
    echo "3. git commit -m \"Fix: description\""
    echo "4. git push -u origin $branch_name"
    echo "5. Create PR to merge into develop"
}

function create_hotfix_branch() {
    local branch_name="hotfix/$1"
    echo -e "${YELLOW}⚠️  Creating hotfix branch: $branch_name${NC}"
    echo -e "${YELLOW}Note: Hotfixes should only be used for urgent production issues${NC}"
    
    # Ensure we're on master and it's up to date
    git checkout master
    git pull origin master
    
    # Create and checkout new branch
    git checkout -b "$branch_name"
    
    echo -e "${GREEN}✓ Created branch '$branch_name' from master${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Make your urgent fix"
    echo "2. git add <files>"
    echo "3. git commit -m \"Hotfix: urgent issue description\""
    echo "4. git push -u origin $branch_name"
    echo "5. Create PR to merge into master"
    echo "6. IMPORTANT: Also create PR to merge into develop"
}

function sync_with_develop() {
    local current_branch=$(git branch --show-current)
    
    if [[ "$current_branch" == "master" ]]; then
        echo -e "${RED}Cannot sync master branch with develop${NC}"
        echo "Master should only receive changes via PRs"
        exit 1
    fi
    
    echo -e "${GREEN}Syncing $current_branch with develop...${NC}"
    
    # Fetch latest changes
    git fetch origin develop
    
    # Merge or rebase based on preference
    echo "Choose sync method:"
    echo "1) Merge develop into $current_branch"
    echo "2) Rebase $current_branch onto develop"
    read -p "Select (1 or 2): " choice
    
    case $choice in
        1)
            git merge origin/develop
            echo -e "${GREEN}✓ Merged develop into $current_branch${NC}"
            ;;
        2)
            git rebase origin/develop
            echo -e "${GREEN}✓ Rebased $current_branch onto develop${NC}"
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            exit 1
            ;;
    esac
}

function show_status() {
    echo -e "${GREEN}=== Git Workflow Status ===${NC}"
    echo ""
    
    # Show current branch
    local current_branch=$(git branch --show-current)
    echo -e "Current branch: ${YELLOW}$current_branch${NC}"
    
    # Show branch type
    if [[ "$current_branch" == "master" ]]; then
        echo -e "Branch type: ${RED}Production (protected)${NC}"
    elif [[ "$current_branch" == "develop" ]]; then
        echo -e "Branch type: ${YELLOW}Integration Testing${NC}"
    elif [[ "$current_branch" == feature/* ]]; then
        echo -e "Branch type: ${GREEN}Feature${NC}"
    elif [[ "$current_branch" == fix/* ]]; then
        echo -e "Branch type: ${GREEN}Fix${NC}"
    elif [[ "$current_branch" == hotfix/* ]]; then
        echo -e "Branch type: ${RED}Hotfix (urgent)${NC}"
    fi
    
    echo ""
    echo "Workflow reminder:"
    echo "  feature/* → develop → master"
    echo "  hotfix/* → master (and develop)"
    echo ""
    
    # Show uncommitted changes
    if [[ -n $(git status -s) ]]; then
        echo -e "${YELLOW}You have uncommitted changes:${NC}"
        git status -s
    else
        echo -e "${GREEN}Working tree is clean${NC}"
    fi
}

# Main script logic
case "$1" in
    feature)
        if [[ -z "$2" ]]; then
            echo -e "${RED}Error: Feature name required${NC}"
            print_usage
            exit 1
        fi
        create_feature_branch "$2"
        ;;
    fix)
        if [[ -z "$2" ]]; then
            echo -e "${RED}Error: Fix name required${NC}"
            print_usage
            exit 1
        fi
        create_fix_branch "$2"
        ;;
    hotfix)
        if [[ -z "$2" ]]; then
            echo -e "${RED}Error: Hotfix name required${NC}"
            print_usage
            exit 1
        fi
        create_hotfix_branch "$2"
        ;;
    sync)
        sync_with_develop
        ;;
    status)
        show_status
        ;;
    *)
        print_usage
        exit 1
        ;;
esac