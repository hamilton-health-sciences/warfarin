# Replication

Given access to the COMBINE-AF database, the findings of this study are
end-to-end reproducible.

## Environment and setup

The pipeline and modeling was run in Python 3.9.2.

    $ python3 --version
    Python 3.9.2
    $ git clone git@github.com:hamilton-health-sciences/warfarin.git
    $ cd warfarin
    $ python3 -m pip install -r requirements.txt

Point Python at the cloned directory:

    $ export PYTHONPATH=.

## Preprocessing and modeling

The data preprocessing pipeline and modeling steps are stored in a `dvc`
repository. They can be replicated with:

    $ dvc repro

This will preprocess the data, train the behavior cloners for model selection,
train the reinforcement learning model, and produce the evaluation results.
Component steps can be viewed in `dvc.yaml` and relevant parameters in
`params.yaml`. The command `dvc dag` also provides a visual overview at the
command line.

## Summary statistics

Summary statistics quoted in the tables and throughout the text can be generated
for the primary dataset with:

    $ python3 scripts/compute_summary_statistics.py \
        --merged_data_path ./data/clean_data/merged.feather \
        --raw_baseline_data_path ./data/raw_data/baseline.feather
        --combine_test_ids_path ./data/combine_test_ids.txt \
        --rely_subjids_path ./data/raw_data/rely_subjids.sas7bdat \
        --output_path ./output/summary_statistics

and for the control (NOAC) set with:

    $ python3 scripts/compute_noac_summary_statistics.py \
        --merged_data_path ./data/clean_data/merged.feather \
        --raw_baseline_data_path ./data/raw_data/baseline.feather \
        --raw_events_data_path ./data/raw_data/events.feather \
        --noac_rely_subjids_path ./data/rely_noac_ids.txt \
        --rely_subjids_path ./data/raw_data/rely_Subjids.sas7bdat \
        --output_path ./output/noac_summary_statistics
