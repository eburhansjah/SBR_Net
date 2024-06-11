#!/bin/bash -l

#$ -P tianlabdl		# Specify SCC project name
#$ -l h_rt=12:00:00		# Hard time limit for the job
#$ -pe omp 4        # Request 4 CPUs
#$ -l gpus=1        # Request 1 GPU
#$ -l gpu_c=6.0     # Specify the minimum GPU compute capability 

#$ -m e				# Notify through email

#$ -N SBR_Net		# Specify job name

#$ -j y				# Merging error and output streams into single file

echo "==========================================================" # Keep track of information related to the current job
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="

source /projectnb/tianlabdl/eburhan/SBR_Net/.venv/bin/activate
module load python3/3.10.12
python main.py