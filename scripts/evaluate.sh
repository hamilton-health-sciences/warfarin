#!/bin/bash

LOGS_PATH=ray_logs/dbcq
TRAIN_DATA_PATH=data/clean_data/train_data.feather
TEST_DATA_PATH=data/clean_data/test_data.feather
BEHAVIOR_POLICY_PATH=output/behavior_cloner_wis/checkpoint.pt
OUTPUT_PREFIX=output/rely_test_evaluation
RELY_SUBJIDS_FILENAME=data/raw_data/rely_subjids.sas7bdat
BASELINE_FILENAME=data/raw_data/ittr_baseline.csv
RELY_SASLIBS_PATH=data/raw_data/sas_libs
RELY_DRUGS_PATH=data/raw_data/drugs_rely.csv
CLEAN_EVENTS_PATH=data/clean_data/events.feather

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
        --output_prefix $OUTPUT_PREFIX
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
    Benchmark \
    > $OUTPUT_PREFIX/agreement_models_benchmark.txt

Rscript scripts/estimate_agreement_models.R \
    $OUTPUT_PREFIX/MLM_RL.csv \
    $OUTPUT_PREFIX/coxMLM_RL.csv \
    $OUTPUT_PREFIX/WLS_cent_RL.csv \
    $OUTPUT_PREFIX/wls_plot_policy.pdf \
    RL \
    > $OUTPUT_PREFIX/agreement_models_policy.txt
