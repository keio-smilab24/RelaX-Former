#!/bin/bash
# usage: ./train.sh -m train -i <id> [-d] [-a <alpha>] [-g <gamma>] [-l <lambda_neg>] [-s <seed>]
# -m: mode (train)
# -i: id of the experiment (wandb id)
# -d: debug mode
# -a: alpha value
# -g: gamma value
# -l: lambda_neg value
# -s: seed value

MODE="train"  # Default mode
ID=""
DEBUG_MODE=false
SEED="46"
ALPHA="0.7"
GAMMA="0.7"
LAMBDA_NEG="0.7"

while getopts "m:i:da:g:l:s:" opt; do
  case $opt in
    m) MODE=$OPTARG;;
    i) ID=$OPTARG;;
    d) DEBUG_MODE=true;;
    a) ALPHA=$OPTARG;;
    g) GAMMA=$OPTARG;;
    l) LAMBDA_NEG=$OPTARG;;
    s) SEED=$OPTARG;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1;;
  esac
done

if [ -z "$ID" ]; then
  echo "You must provide an id with -i option."
  exit 1
fi

echo "seed: $SEED"
echo "alpha: $ALPHA"
echo "gamma: $GAMMA"
echo "lambda_neg: $LAMBDA_NEG"

LOGFILE="log/${ID}.log"
if [ "$DEBUG_MODE" = true ]; then
  LOGFILE="log/debug_${ID}.log"
fi

if [ "$MODE" = "train" ]; then
  echo "Training mode"
  if [ "$DEBUG_MODE" = true ]; then
    echo "Debug mode activated"
    poetry run python src/main.py train --eval_unseen --epoch 2 --bs 128 --lr 1e-4 |& tee -a $LOGFILE
  else
    poetry run python src/main.py train --eval_unseen --epoch 20 --bs 128 --lr 1.0e-4 --seed $SEED --log_wandb --wandb_name $ID --alpha $(printf "%.6f" $ALPHA) --gamma $(printf "%.6f" $GAMMA) --lambda_neg $(printf "%.6f" $LAMBDA_NEG) |& tee -a $LOGFILE
  fi
else
  echo "Invalid mode: $MODE"
  exit 1
fi
