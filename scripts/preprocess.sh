#!/bin/bash

BASELINE_SAS_FILENAME=$1
INR_SAS_FILENAME=$2
EVENTS_SAS_FILENAME=$3
TEST_IDS_PATH=$4

OUTPUT_DIR=./data

mkdir -p output
mkdir -p $OUTPUT_DIR/raw_data $OUTPUT_DIR/clean_data

if [ ! -f $OUTPUT_DIR/raw_data/baseline.feather ]; then
    echo Converting baseline data...
    python3 scripts/convert_sas_to_feather.py \
        --input_filename $BASELINE_SAS_FILENAME \
        --output_filename $OUTPUT_DIR/raw_data/baseline.feather
fi

if [ ! -f $OUTPUT_DIR/raw_data/inr.feather ]; then
    echo Converting INR data...
    python3 scripts/convert_sas_to_feather.py \
        --input_filename $INR_SAS_FILENAME \
        --output_filename $OUTPUT_DIR/raw_data/inr.feather
fi

if [ ! -f $OUTPUT_DIR/raw_data/events.feather ]; then
    echo Converting events data...
    python3 scripts/convert_sas_to_feather.py \
        --input_filename $EVENTS_SAS_FILENAME \
        --output_filename $OUTPUT_DIR/raw_data/events.feather
fi

echo Preprocessing data...
python3 scripts/preprocess_combine_af.py \
    --baseline_path $OUTPUT_DIR/raw_data/baseline.feather \
    --inr_path $OUTPUT_DIR/raw_data/inr.feather \
    --events_path $OUTPUT_DIR/raw_data/events.feather \
    --output_directory $OUTPUT_DIR/clean_data \
    --test_ids_path $TEST_IDS_PATH

echo Auditing processing pipeline...
python3 scripts/audit_combine_preprocessing.py \
    > output/combine_preprocess_audit.txt
