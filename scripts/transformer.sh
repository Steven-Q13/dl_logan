#! /bin/bash
#SBATCH --output=/cfarhomes/squeen0/output_logan/train.out.%j
#SBATCH --error=/cfarhomes/squeen0/output_logan/train.out.%j
#SBATCH --time=70:00:00
#SBATCH --gres=gpu:p6000:4
#SBATCH --account=vulcan
#SBATCH --mem=240000
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger

# Run with: sbatch logan_job.sh
# Check execution with: squeue -u squeen0
module add Python3/3.8.2
module add cuda
srun bash -c "source /cfarhomes/squeen0/dl_logan/venv/bin/activate && python3 /cfarhomes/squeen0/dl_logan/signal_gen/train_transformer.py vulcan 75000 32 5 6 test1"
#Args: Location, num_epochs, batch_size, num_layers, num_heads, version
