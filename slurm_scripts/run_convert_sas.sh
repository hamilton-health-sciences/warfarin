#!/bin/bash

cd /dhi_work/wum/warfarin/slurm_logs
# mkdir bcq_$SLURM_JOB_ID


cd /dhi_work/wum/warfarin

python3 -u -m scripts.convert_sas_to_feather --data_folder="./data/" > /dhi_work/wum/warfarin/slurm_logs/convert_sas_$SLURM_JOB_ID.out


