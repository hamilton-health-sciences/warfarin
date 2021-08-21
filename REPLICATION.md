# Replication

Given access to the COMBINE-AF database, the findings of this study are
end-to-end reproducible.

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

## Model training

With a GPU available, train the dBCQ model:

    $ python3 scripts/train_smdp_dBCQ.py --suffix=smdp --num_actions=7 --state_dim=56 \
        --hidden_states=64 --events_batch_size=0 --BCQ_threshold=0.2 --events_batch_size=0 \
        --lr=0.00005 --max_timesteps=1_000_000 --save_folder="./output/dbcq"

## Evaluations and figures

    $ bash script/evaluate_combine_af.sh # ...
