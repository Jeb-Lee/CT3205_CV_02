#!/bin/sh
# Sequential reconstruction of all own sets (focal factor 1.04 from scripts/estimate_focal.py)
cd "$(dirname "$0")/.."
python scripts/run_own.py --set yellow_sign --tag yellow_all --f_factor 1.04 --dense --video > results/own/yellow_all.log 2>&1
python scripts/run_own.py --set "Fire est" --tag fire --f_factor 1.04 --dense --video > results/own/fire.log 2>&1
python scripts/run_own.py --set Snacks --tag snacks --f_factor 1.04 --dense --video > results/own/snacks.log 2>&1
python scripts/run_own.py --set Bottle --tag bottle_classic --f_factor 1.04 --video > results/own/bottle_classic.log 2>&1
python scripts/run_own.py --set Bottle --tag bottle_learned --f_factor 1.04 --detector disk --matcher lightglue --min_pair 150 --video > results/own/bottle_learned.log 2>&1
echo CHAIN_DONE > results/own/chain_done.flag
