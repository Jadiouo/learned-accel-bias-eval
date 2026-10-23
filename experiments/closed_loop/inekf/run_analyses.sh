#!/bin/bash
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
python3 analysis.py run_inekf out_inekf > out_inekf.log 2>&1
python3 analysis.py verify_eskf_ref verify_eskf_ref/out > verify_eskf_ref/analysis.log 2>&1
echo ALLDONE > analyses.done
