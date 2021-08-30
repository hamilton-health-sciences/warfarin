#!/bin/bash

cd /dhi_work/wum/warfarin/slurm_logs
# mkdir bcq_$SLURM_JOB_ID


cd /dhi_work/wum/warfarin

python3 -u -m scripts.run_combine_preprocessing --data_folder="./data/" > /dhi_work/wum/warfarin/slurm_logs/combine_preprocessing_$SLURM_JOB_ID.out


