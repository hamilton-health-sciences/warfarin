#!/bin/bash

cd /dhi_work/wum/warfarin/slurm_logs
# mkdir bcq_$SLURM_JOB_ID


cd /dhi_work/wum/warfarin

python3 -u -m scripts.create_replay_buffers --data_folder="./data/" --incl_hist --num_actions=7 --buffer_suffix=smdp  > /dhi_work/wum/warfarin/slurm_logs/create_replay_buffers_$SLURM_JOB_ID.out


