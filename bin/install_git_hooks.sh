#!/usr/bin/env bash
. $ROOT_DIR/src/bin/include.sh
init_env

HOOKS_DIR=$SRC_DIR/hooks
GIT_DIR="$SRC_DIR/.git"
GIT_HOOKS_DIR="$GIT_DIR/hooks"

echo "Using Git directory at: $GIT_DIR"
echo "Installing/Updating hooks in: $GIT_HOOKS_DIR"

install_or_update_hook() {
  local hook_name=$1
  local source_file="$HOOKS_DIR/$hook_name"
  local dest_file="$GIT_HOOKS_DIR/$hook_name"

  if [ -f "$dest_file" ]; then
    if cmp -s "$source_file" "$dest_file"; then
      echo "Hook $hook_name is up to date."
      return
    else
      echo "Updating existing $hook_name hook..."
    fi
  else
    echo "Installing new $hook_name hook..."
  fi

  cp "$source_file" "$dest_file"
  chmod +x "$dest_file"
  sed -i "s|ROOT_DIR=.*|ROOT_DIR=\"$ROOT_DIR\"|" "$dest_file"
  sed -i "s|SRC_DIR=.*|SRC_DIR=\"$SRC_DIR\"|" "$dest_file"
  echo "$hook_name hook installed/updated successfully."
}

mkdir -p "$GIT_HOOKS_DIR"

for hook in "$HOOKS_DIR"/*; do
  if [ -f "$hook" ] && [[ ! "$hook" == *.sample ]]; then
    hook_name=$(basename "$hook")
    install_or_update_hook "$hook_name"
  fi
done

echo "All custom Git hooks have been installed/updated successfully."
echo "ROOT_DIR: $ROOT_DIR"
echo "SRC_DIR: $SRC_DIR"
echo "Custom hooks in $GIT_HOOKS_DIR:"
ls -l "$GIT_HOOKS_DIR" | grep -v '\.sample$'