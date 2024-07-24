#!/bin/bash

timestamp=$(date +"%Y%m%d%H%M%S%N")  # %N for nanoseconds

random_suffix=$(($RANDOM % 1000))  # Generates a random number between 0 and 999

jobname="SBR_Net_eburhan_${timestamp}_${random_suffix}"

# Check if the sweep name and project name are provided
if [ -z "$1" ]; then
  echo "Error: Sweep name or project name not provided. Usage: $0 <sweep_name> <project_name>"
  exit 1
fi
sweep_name=$1

# Generating sweep ID
sweep_id_output=$(wandb sweep --name "$sweep_name" --project SBR_Net_eburhan --entity cisl-bu config.yaml 2>&1)

sweep_id=$(echo "$sweep_id_output" | grep -oP '(?<=Creating sweep with ID: )[a-zA-Z0-9]+')

echo "Generated sweep ID: $sweep_id"

# Check if sweep ID was successfully generated
if [ -z "$sweep_id" ]; then
  echo "Failed to generate sweep ID."
  exit 1
fi

# Create the qsub script using a heredoc
qsub_file_path="/projectnb/tianlabdl/eburhan/SBR_Net/sweep.qsub"
cat << EOF > "$qsub_file_path"

#!/bin/bash -l

#$ -P tianlabdl		    # Specify SCC project name
#$ -l h_rt=24:00:00	    # Hard time limit for the job
#$ -pe omp 28           # Request 4 CPUs
#$ -l gpus=1            # Request 1 GPU
#$ -l gpu_c=6.0         # Specify the minimum GPU compute capability 

#$ -m e				    # Notify through email

#$ -j y				    # Merging error and output streams into single file

#$ -t 1-10              # array jobs to define number of nodes/sweep agents to use on the SCC

module load python3/3.10.12
source /projectnb/tianlabdl/eburhan/SBR_Net/.venv/bin/activate

wandb agent --count 1 cisl-bu/SBR_Net_eburhan/$sweep_id        # Using generated sweep ID
EOF

# Assuming your Python script is in the current directory
# Replace "your_sweep_id" with the actual sweep ID
# 'count' specifes the number of runs

qsub -N "${jobname}" -o "/projectnb/tianlabdl/eburhan/SBR_Net/logs/${jobname}.qlog" "$qsub_file_path"