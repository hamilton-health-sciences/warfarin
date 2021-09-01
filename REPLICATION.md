# Replication

Given access to the COMBINE-AF database, the findings of this study are
end-to-end reproducible.

## Environment

    $ export PYTHONPATH=.

## Preprocessing

First, rename the input SAS files to `baseline_raw.sas7bdat`, and
`events_raw.sas7bdat`, `inr_raw.sas7bdat` and place them in `data/raw_data`.
Then use the feather conversion script to get these in a more portable format:

    $ python3 scripts/convert_sas_to_feather.py

Next, pre-process the data (described in section XX of the manuscript):

    $ python3 scripts/run_combine_preprocessing.py

Finally, generate the replay buffers, which is the format used as input to the
RL model:

    $ python3 scripts/create_replay_buffers.py --num_actions 7 --incl_hist --buffer_suffix=smdp

## Model tuning

With at least one GPU available, train the dBCQ model:

    $ python3 scripts/tune_smdp_dbcq.py \
          --train_buffer `pwd`/data/replay_buffers/train_data \
          --events_buffer `pwd`/data/replay_buffers/events_data \
          --val_buffer `pwd`/data/replay_buffers/val_data \
          --target_metric val_jindex_good_actions \
          --mode max

## Evaluations and figures

    $ bash script/evaluate_combine_af.sh # ...
