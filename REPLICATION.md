# Replication

Given access to the COMBINE-AF database, the findings of this study are
end-to-end reproducible.

## Preprocessing

First, rename the input SAS files to `baseline_raw.sas7bdat`, and
`events_raw.sas7bdat`, `inr_raw.sas7bdat` and place them in `data/raw_data`.
Then use the feather conversion script to get these in a more portable format:

    $ python3 scripts/convert_sas_to_feather.py

Next, pre-process the data (described in section XX of the manuscript):

    $ python3 scripts/run_combine_preprocessing.py --data_folder ./data/

Finally, generate the replay buffers, which is the format used as input to the
RL model:

    $ python3 scripts/create_replay_buffer.py \
          --input_fn ./data/split_data/train_data.feather \
          --num_actions 7 --incl_hist \
          --output_fn ./data/replay_buffers/train_data \
          --output_events_fn ./data/replay_buffers/events_data \
          --output_normalization ./output/normalization_params.pkl

    $ python3 scripts/create_replay_buffer.py \
          --input_fn ./data/split_data/val_data.feather \
          --num_actions 7 --incl_hist \
          --output_fn ./data/replay_buffers/val_data_buffer.feather \
          --normalization ./output/normalization_params.pkl

    $ python3 scripts/create_replay_buffer.py \
          --input_fn ./data/split_data/test_data.feather \
          --num_actions 7 --incl_hist \
          --output_fn ./data/replay_buffers/test_data_buffer.feather \
          --normalization ./output/normalization_params.pkl

## Model development

    $ python3 script/train # ....

## Evaluations and figures

    $ bash script/evaluate_combine_af.sh # ...
