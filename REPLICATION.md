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

## Preprocessing

Run the preprocessing steps. This will first convert the input SAS files into
Feather dataframes and store them, then preprocess the data and ready it for RL
modeling, splitting off the test set from the IDs listed in
`./data/test_subject_ids.txt`. This list of IDs was randomly generated during
an earlier phase of the project.

    $ bash scripts/preprocess.sh \
        ./data/raw_data/baseline.sas7bdat \
        ./data/raw_data/inr.sas7bdat \
        ./data/raw_data/events.sas7bdat \
        ./data/test_subject_ids.txt

This will create an audit log in `output/` that can be comparison-checked for
correctness.

## Model training

### Behavioral cloning

When using WIS-estimated value as the model selection metric, a behavioral
cloning model is necessary. To tune it:

    $ python3 scripts/tune_bc.py \
        --train_data `pwd`/data/clean_data/train_data.feather \
        --val_data `pwd`/data/clean_data/val_data.feather \
        --target_metric "val/auroc" \
        --mode max

Evaluations and plots can be accessed in real-time during training through
Tensorboard:

    $ python3 -m tensorboard.main --logdir=./ray_logs/bc

### SMDP-dBCQ

With a GPU available, train and tune the dBCQ model on the development set,
which will attempt to find hyperparameters that maximize the WIS-estimated
value of the policy:

    $ python3 scripts/tune_smdp_dbcq.py \
        --train_data `pwd`/data/clean_data/train_data.feather \
        --val_data `pwd`/data/clean_data/val_data.feather \
        --behavior_policy $PATH_TO_BEHAVIOR_POLICY_MODEL_PT
        --target_metric "val/wis/policy_value" \
        --mode max

where `$PATH_TO_BEHAVIOR_POLICY_MODEL_PT` is the path to the `model.pt` for the
best behavioral cloning checkpoint (TODO: make this easier).

Evaluations and plots can be accessed during training through Tensorboard:

    $ python3 -m tensorboard.main --logdir=./ray_logs/bc
    Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
    TensorBoard 2.6.0 at http://localhost:6006/ (Press CTRL+C to quit)

# Final model selection, evaluation & figures

To run the model selection, evaluation, and plotting routines (currently on
validation data - TODO when opening up test set):

    $ python3 scripts/evaluate_best_model.py \
        --logs_path ./ray_logs/ \
        --data_path ./data/clean_data/val_data.feather \
        --behavior_policy_path $PATH_TO_BEHAVIOR_POLICY_MODEL_PT \
        --output_prefix ./output/val_eval \
        --target_metric val/wis/policy_value --mode max

where `$PATH_TO_BEHAVIOR_POLICY_MODEL_PT` should be the same as above. And to
pretty-print the classification metrics afterward:

    $ python3 scripts/format_metrics.py \
        --metrics_filename ./output/val_eval/metrics.json \
        --output_filename ./output/val_eval/classification.csv
