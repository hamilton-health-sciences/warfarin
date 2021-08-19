#!/bin/bash

cd /dhi_work/wum/warfarin/slurm_logs
# mkdir bcq_$SLURM_JOB_ID


cd /dhi_work/wum/warfarin

python3 -u -m scripts.create_replay_buffers --data_folder="./data/" --state_method=18 --num_actions=7 --buffer_suffix=smdp_most_agg  > /dhi_work/wum/warfarin/slurm_logs/create_replay_buffers_$SLURM_JOB_ID.out


