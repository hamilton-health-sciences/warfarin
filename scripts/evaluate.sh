#!/bin/bash

LOGS_PATH=$1
TRAIN_DATA_PATH=$2
TEST_DATA_PATH=$3
BEHAVIOR_POLICY_PATH=$4
OUTPUT_PREFIX=$5
TARGET_METRIC=$6
TARGET_MODE=$7
RELY_SUBJIDS_FILENAME=$8
BASELINE_FILENAME=$9
RELY_SASLIBS_PATH=${10}
RELY_DRUGS_PATH=${11}
CLEAN_EVENTS_PATH=${12}

# Check if output prefix already exists, and if so, do not re-process the model
# outputs.
if [ -d $OUTPUT_PREFIX ]; then
    echo $OUTPUT_PREFIX already exists. Will not reproduce model decisions, \
         but will re-run RELY evaluation pipeline.
else
    # Pick the best RL model.
    python3 scripts/evaluate_best_model.py \
        --logs_path $LOGS_PATH \
        --train_data_path $TRAIN_DATA_PATH \
        --data_path $TEST_DATA_PATH \
        --behavior_policy_path $BEHAVIOR_POLICY_PATH \
        --output_prefix $OUTPUT_PREFIX \
        --target_metric $TARGET_METRIC --mode $TARGET_MODE
fi

# Format the output metrics.
python3 scripts/format_metrics.py \
    --metrics_filename $OUTPUT_PREFIX/metrics.json \
    --output_prefix $OUTPUT_PREFIX

# Link the output TTR data to the RELY database.
python3 scripts/link_rely_subj_ids.py \
    --hierarchical_ttr $OUTPUT_PREFIX/hierarchical_ttr.csv \
    --rely_subjid_path $RELY_SUBJIDS_FILENAME \
    --output_path $OUTPUT_PREFIX/hierarchical_ttr_linked.csv

# Generate inputs to TTR & events models.
python3 scripts/process_model_inputs.py \
    --baseline_filename $BASELINE_FILENAME \
    --saslibs_path $RELY_SASLIBS_PATH \
    --model_results_path $OUTPUT_PREFIX \
    --drugs_path $RELY_DRUGS_PATH \
    --events_path $CLEAN_EVENTS_PATH

# Model results.
Rscript scripts/estimate_agreement_models.R \
    $OUTPUT_PREFIX/MLM_threshold.csv \
    $OUTPUT_PREFIX/coxMLM_threshold.csv \
    $OUTPUT_PREFIX/WLS_cent_threshold.csv \
    $OUTPUT_PREFIX/wls_plot_benchmark.pdf \
    > $OUTPUT_PREFIX/agreement_models_benchmark.txt

Rscript scripts/estimate_agreement_models.R \
    $OUTPUT_PREFIX/MLM_RL.csv \
    $OUTPUT_PREFIX/coxMLM_RL.csv \
    $OUTPUT_PREFIX/WLS_cent_RL.csv \
    $OUTPUT_PREFIX/wls_plot_policy.pdf \
    > $OUTPUT_PREFIX/agreement_models_policy.txt
