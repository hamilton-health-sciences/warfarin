#!/bin/bash

# Execute the evaluation pipeline on the test set. Links the test set (as
# processed for RL modeling) back to some original baseline covariates which
# we have access to through the original RE-LY database, and runs the evaluation
# pipeline using the model that performs best in the tuning set.

LOGS_PATH=ray_logs/dbcq
TRAIN_DATA_PATH=data/clean_data/train_data.feather
TEST_DATA_PATH=data/clean_data/test_data.feather
BEHAVIOR_POLICY_PATH=output/behavior_cloner_wis/checkpoint.pt
OUTPUT_PREFIX=output/rely_test_evaluation
RELY_SUBJIDS_FILENAME=data/raw_data/rely_subjids.sas7bdat
BASELINE_FILENAME=data/raw_data/ittr_baseline.csv
RELY_SASLIBS_PATH=data/raw_data/sas_libs
RELY_DRUGS_PATH=data/raw_data/drugs_rely.csv
RAW_EVENTS_PATH=data/raw_data/events.feather
OTHER_PATH=data/raw_data/merged_other.sas7bdat
CLEAN_EVENTS_PATH=data/clean_data/events.feather
SUBSET_IDS_PATH=data/combine_test_ids.txt

# Pick the best RL model.
python3 scripts/evaluate_best_model.py \
    --logs_path $LOGS_PATH \
    --train_data_path $TRAIN_DATA_PATH \
    --data_path $TEST_DATA_PATH \
    --behavior_policy_path $BEHAVIOR_POLICY_PATH \
    --output_prefix $OUTPUT_PREFIX \
    --subset_ids_path $SUBSET_IDS_PATH

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

# Generate input to events NOAC model.
for model in policy threshold maintain random; do
    python3 scripts/process_noac_model_inputs.py \
        --consistency_path ./output/rely_test_evaluation/MLM_${model}.csv \
        --baseline_filename $BASELINE_FILENAME \
        --saslibs_path $RELY_SASLIBS_PATH \
        --rely_subjid_path $RELY_SUBJIDS_FILENAME \
        --events_path $RAW_EVENTS_PATH \
        --drugs_path $RELY_DRUGS_PATH \
        --other_path $OTHER_PATH \
        --output_filename $OUTPUT_PREFIX/coxMLM_noac_${model}.csv
done

# Model results
Rscript scripts/estimate_agreement_models.R \
    $OUTPUT_PREFIX/MLM_threshold.csv \
    $OUTPUT_PREFIX/coxMLM_threshold.csv \
    $OUTPUT_PREFIX/WLS_cent_threshold.csv \
    $OUTPUT_PREFIX/wls_plot_benchmark.pdf \
    $OUTPUT_PREFIX/ttr_table_benchmark.csv \
    $OUTPUT_PREFIX/events_table_benchmark.csv \
    Benchmark \
    > $OUTPUT_PREFIX/agreement_models_benchmark.txt

Rscript scripts/estimate_agreement_models.R \
    none \
    $OUTPUT_PREFIX/coxMLM_noac_threshold.csv \
    none \
    none \
    none \
    $OUTPUT_PREFIX/events_table_noac_benchmark.csv \
    Benchmark \
    > $OUTPUT_PREFIX/agreement_models_noac_benchmark.txt

Rscript scripts/estimate_agreement_models.R \
    $OUTPUT_PREFIX/MLM_policy.csv \
    $OUTPUT_PREFIX/coxMLM_policy.csv \
    $OUTPUT_PREFIX/WLS_cent_policy.csv \
    $OUTPUT_PREFIX/wls_plot_policy.pdf \
    $OUTPUT_PREFIX/ttr_table_policy.csv \
    $OUTPUT_PREFIX/events_table_policy.csv \
    RL \
    > $OUTPUT_PREFIX/agreement_models_policy.txt

Rscript scripts/estimate_agreement_models.R \
    none \
    $OUTPUT_PREFIX/coxMLM_noac_policy.csv \
    none \
    none \
    none \
    $OUTPUT_PREFIX/events_table_noac_policy.csv \
    Policy \
    > $OUTPUT_PREFIX/agreement_models_noac_policy.txt

Rscript scripts/estimate_agreement_models.R \
    $OUTPUT_PREFIX/MLM_maintain.csv \
    $OUTPUT_PREFIX/coxMLM_maintain.csv \
    $OUTPUT_PREFIX/WLS_cent_maintain.csv \
    $OUTPUT_PREFIX/wls_plot_maintain.pdf \
    $OUTPUT_PREFIX/ttr_table_maintain.csv \
    $OUTPUT_PREFIX/events_table_maintain.csv \
    "Always Maintain" \
    > $OUTPUT_PREFIX/agreement_models_maintain.txt

Rscript scripts/estimate_agreement_models.R \
    none \
    $OUTPUT_PREFIX/coxMLM_noac_maintain.csv \
    none \
    none \
    none \
    $OUTPUT_PREFIX/events_table_noac_maintain.csv \
    "Always Maintain" \
    > $OUTPUT_PREFIX/agreement_models_noac_maintain.txt

Rscript scripts/estimate_agreement_models.R \
    $OUTPUT_PREFIX/MLM_random.csv \
    $OUTPUT_PREFIX/coxMLM_random.csv \
    $OUTPUT_PREFIX/WLS_cent_random.csv \
    $OUTPUT_PREFIX/wls_plot_random.pdf \
    $OUTPUT_PREFIX/ttr_table_random.csv \
    $OUTPUT_PREFIX/events_table_random.csv \
    Random \
    > $OUTPUT_PREFIX/agreement_models_random.txt

Rscript scripts/estimate_agreement_models.R \
    none \
    $OUTPUT_PREFIX/coxMLM_noac_random.csv \
    none \
    none \
    none \
    $OUTPUT_PREFIX/events_table_noac_random.csv \
    Random \
    > $OUTPUT_PREFIX/agreement_models_noac_random.txt
