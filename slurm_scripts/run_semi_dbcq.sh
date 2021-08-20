#!/bin/bash

cd /dhi_work/wum/warfarin/slurm_logs
# mkdir bcq_$SLURM_JOB_ID


cd /dhi_work/wum/warfarin

python3 -u -m scripts.train_smdp_dBCQ --suffix=smdp --num_actions=7 --state_dim=56 --hidden_states=64 --events_batch_size=0 --BCQ_threshold=0.2 --events_batch_size=0 --lr=0.00005 --max_timesteps=1000000 --save_folder="./discrete_BCQ"  > /dhi_work/wum/warfarin/slurm_logs/bcq_job_$SLURM_JOB_ID.out


