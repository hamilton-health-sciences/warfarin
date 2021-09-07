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
Feather dataframes and store them, then preprocess the data and create replay
buffers for RL modeling, splitting off the test set from the IDs listed in
`./data/test_subject_ids.txt`. This list of IDs was randomly generated.

    $ bash script/preprocess.sh \
        ./data/raw_data/baseline.sas7bdat \
        ./data/raw_data/inr.sas7bdat
        ./data/raw_data/events.sas7bdat
        ./data/test_subject_ids.txt

## Model training

With a GPU available, train and tune the dBCQ model on the development set:

    $ python3 scripts/tune_smdp_dbcq.py \
        --train_buffer `pwd`/data/replay_buffers/train_data \
        --events_buffer `pwd`/data/replay_buffers/events_data \
        --val_buffer `pwd`/data/replay_buffers/val_data \
        --target_metric val_jindex_good_actions \
        --mode max

Evaluations and plots can be accessed through Tensorboard:

    $ python3 -m tensorboard.main --logdir=./ray_logs
    Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
    TensorBoard 2.6.0 at http://localhost:6006/ (Press CTRL+C to quit)

## Evaluations and figures

    $ bash script/evaluate_combine_af.sh # ...
