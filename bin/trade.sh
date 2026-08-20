#!/bin/bash
. $ROOT_DIR/src/bin/include.sh
init_env

MODE="$1"
shift


if [ "$MODE" = "flatten" ]; then
  CMD="--aggression=$*"
elif [ "$MODE" = "manual" ]; then
  # "B ETHUSDT 0.01 @ 2500 [LIMIT GTX]"
  CMD="--order=$*"
elif [ "$MODE" = "cancel" ]; then
  CMD=""
else
  echo "Unknown or no mode $MODE"
  exit 1
fi

python "$SRC_DIR"/trader.py "--mode=$MODE" "$CMD"

