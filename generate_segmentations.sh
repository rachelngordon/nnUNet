#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=logs/save_tumor_segmentations.err
#SBATCH --output=logs/save_tumor_segmentations.out
#SBATCH --exclude=''
#SBATCH --gpus-per-node=1
#SBATCH --job-name=save_tumor_segmentations
#SBATCH --mem-per-gpu=80000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --partition=gpuq
#SBATCH --time=1440

# Load Micromamba
source /gpfs/data/karczmar-lab/workspaces/rachelgordon/micromamba/etc/profile.d/micromamba.sh

# Activate your Micromamba environment
micromamba activate segmentation

export nnUNet_raw=/gpfs/data/karczmar-lab/workspaces/rachelgordon/MAMA-MIA/nnUNet/nnunetv2/nnUNet_raw
export nnUNet_preprocessed=/gpfs/data/karczmar-lab/workspaces/rachelgordon/MAMA-MIA/nnUNet/nnunetv2/nnUNet_preprocessed
export nnUNet_results=/gpfs/data/karczmar-lab/workspaces/rachelgordon/MAMA-MIA/nnUNet/nnunetv2/nnUNet_results

nnUNetv2_predict   -i /ess/scratch/scratch1/rachelgordon/zf_data_192_slices/preprocessed_for_seg   -o /ess/scratch/scratch1/rachelgordon/zf_data_192_slices/tumor_segmentations   -d 101   -c 3d_fullres